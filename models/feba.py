"""
F-EBA (Feature-Enhanced Boosting Algorithm).

Follows the pseudocode in the report:

    Step 1: create_ten_models()
    Step 2: train each weak classifier on weighted data; only keep it if
            validation accuracy clears a threshold
    Step 3: evaluate -> accuracy + misclassified indices, used to update
            sample weights
    Step 4: combine_predictions via weighted majority voting
    Step 5: main() ties it all together

Each weak learner is a BiLSTM (optionally fronted by a BERT projection -
see BertHybridLearner). Fits the report's "BERT and BiLSTM are integrated
and function within the model" remark.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .. import config
from .bilstm import BiLSTMClassifier


# ---------------------------------------------------------------------------

@dataclass
class WeakLearner:
    """Bookkeeping wrapper around a trained model + its voting weight."""
    model: nn.Module
    weight: float = 1.0
    val_accuracy: float = 0.0


# ---------------------------------------------------------------------------

class FEBA:
    """
    Feature-Enhanced Boosting Algorithm.

    Holds N weak BiLSTM learners (default 10), each trained against the
    current sample-weight distribution. Learners that don't clear the
    validation threshold are discarded - matching the report's "if
    validation accuracy is greater than the threshold" gate. The surviving
    learners contribute to the final weighted majority vote.
    """

    def __init__(self,
                 input_dim: int,
                 n_models: int       = config.FEBA_N_MODELS,
                 val_threshold: float = config.FEBA_VAL_THRESH,
                 norm: str            = "layer",
                 epochs: int          = config.FEBA_EPOCHS,
                 batch_size: int      = config.FEBA_BATCH_SIZE,
                 lr: float            = config.FEBA_LR,
                 device: str | None   = None):
        self.input_dim    = input_dim
        self.n_models     = n_models
        self.val_threshold = val_threshold
        self.norm         = norm
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.device       = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.learners: list[WeakLearner] = []
        self.classes_: np.ndarray | None = None

    # ---- step 1 ----------------------------------------------------------

    def create_ten_models(self) -> list[BiLSTMClassifier]:
        """Initialises a fresh batch of weak learners. The name is
        historical - it'll create whatever `n_models` is."""
        return [
            BiLSTMClassifier(self.input_dim, num_classes=2, norm=self.norm).to(self.device)
            for _ in range(self.n_models)
        ]

    # ---- step 2 ----------------------------------------------------------

    def train_weak_classifiers(self,
                               X_train, y_train,
                               X_val,   y_val,
                               sample_weights=None):
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train)
        self.classes_ = np.unique(y_train)

        n = X_train.shape[0]
        if sample_weights is None:
            sample_weights = np.full(n, 1.0 / n)

        for model in self.create_ten_models():
            self._train_one(model, X_train, y_train, sample_weights)
            acc, misclassified = self.evaluate_model(model, X_val, y_val)

            if acc > self.val_threshold:
                self.learners.append(WeakLearner(model=model, weight=acc, val_accuracy=acc))
                # re-weight the training set: misclassified samples relative
                # to *this* learner's predictions on the training set
                _, train_miss = self.evaluate_model(model, X_train, y_train)
                sample_weights = self._update_weights(sample_weights, train_miss, acc)

        return self

    def _train_one(self, model, X, y, weights):
        opt = optim.Adam(model.parameters(), lr=self.lr)
        ce  = nn.CrossEntropyLoss(reduction="none")
        ds  = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(weights, dtype=torch.float32),
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        model.train()
        for _ in range(self.epochs):
            for xb, yb, wb in loader:
                xb, yb, wb = xb.to(self.device), yb.to(self.device), wb.to(self.device)
                opt.zero_grad()
                logits = model(xb)
                loss   = (ce(logits, yb) * wb).mean()
                loss.backward()
                opt.step()
        model.eval()

    @staticmethod
    def _update_weights(weights, misclassified, accuracy):
        # error rate -> classifier confidence -> bump bad samples
        error = max(1.0 - accuracy, 1e-10)
        alpha = 0.5 * np.log((1 - error) / error)
        adjusted = weights * np.exp(alpha * misclassified)
        return adjusted / adjusted.sum()

    # ---- step 3 ----------------------------------------------------------

    @torch.inference_mode()
    def evaluate_model(self, model, X, y):
        model.eval()
        xt = torch.tensor(np.asarray(X, dtype=np.float32)).to(self.device)
        preds = model(xt).argmax(-1).cpu().numpy()
        misclassified = (preds != np.asarray(y)).astype(np.float32)
        accuracy = 1.0 - misclassified.mean()
        return float(accuracy), misclassified

    # ---- step 4 ----------------------------------------------------------

    @torch.inference_mode()
    def combine_predictions(self, X) -> np.ndarray:
        if not self.learners:
            raise RuntimeError("No weak classifiers cleared the threshold. "
                               "Lower val_threshold or train longer.")
        X = np.asarray(X, dtype=np.float32)
        xt = torch.tensor(X).to(self.device)

        # collect per-model votes and accumulate weighted scores
        scores = np.zeros((X.shape[0], len(self.classes_)))
        for learner in self.learners:
            probs = learner.model(xt).softmax(-1).cpu().numpy()
            scores += learner.weight * probs
        return scores

    def predict_proba(self, X) -> np.ndarray:
        scores = self.combine_predictions(X)
        s = scores.sum(axis=1, keepdims=True)
        s[s == 0] = 1
        return scores / s

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    # ---- inspection ------------------------------------------------------

    def majority_vote(self, X) -> np.ndarray:
        """Hard voting variant - useful for sanity-checking against the
        weighted soft vote."""
        votes_per_sample = []
        for x in X:
            tally: Counter = Counter()
            for learner in self.learners:
                with torch.no_grad():
                    pred = learner.model(
                        torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
                    ).argmax(-1).item()
                tally[pred] += learner.weight
            votes_per_sample.append(tally.most_common(1)[0][0])
        return np.array(votes_per_sample)
