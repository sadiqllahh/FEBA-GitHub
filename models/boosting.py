"""
Boosting classifier (the BoostingClassifier described in the report).

Trains a sequence of BiLSTM weak learners. After every iteration we look
at which samples were misclassified by the *combined* prediction so far,
recompute the classifier weight from the error rate, and bump the weights
on misclassified samples for the next round.

This file is pretty much what the report spells out as "BoostingClassifier
class"; F-EBA (next file) extends the idea by holding ten weak learners
trained against shifting weight distributions.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .. import config
from .bilstm import BiLSTMClassifier


def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.tensor(x, dtype=torch.float32)


class BoostingClassifier:
    """
    Sequential boosting with BiLSTM weak learners.

    `n_iterations` controls the depth of the ensemble. Each call to fit()
    produces `n_iterations` lists of trained weak classifiers, one list per
    class label (binary case has two single-element lists per round).

    Predictions are made by averaging the weighted votes of every weak
    classifier across iterations.
    """

    def __init__(self,
                 input_dim: int,
                 n_iterations: int = 5,
                 norm: str         = "layer",
                 epochs: int       = config.FEBA_EPOCHS,
                 batch_size: int   = config.FEBA_BATCH_SIZE,
                 lr: float         = config.FEBA_LR,
                 device: str | None = None):
        self.input_dim    = input_dim
        self.n_iterations = n_iterations
        self.norm         = norm
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.device       = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Populated during fit()
        self.weak_classifiers: list[list[BiLSTMClassifier]] = []
        self.classifier_weights: list[float]                = []
        self.classes_: np.ndarray | None                    = None

    # ----------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weights: np.ndarray | None = None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        n_samples = X.shape[0]
        if sample_weights is None:
            sample_weights = np.full(n_samples, 1.0 / n_samples)

        for it in range(self.n_iterations):
            round_classifiers: list[BiLSTMClassifier] = []
            for cls in self.classes_:
                binary = (y == cls).astype(np.int64)
                model  = self._train_weak_classifier(X, binary, sample_weights)
                round_classifiers.append(model)

            self.weak_classifiers.append(round_classifiers)
            combined = self._get_combined_predictions(X, only_latest=True)
            preds    = self.classes_[np.argmax(combined, axis=1)]

            wrong   = preds != y
            error   = (sample_weights * wrong).sum() / sample_weights.sum()
            error   = max(error, 1e-10)            # avoid div/0
            alpha   = 0.5 * np.log((1 - error) / error)
            self.classifier_weights.append(float(alpha))

            # bump misclassified samples, then re-normalise
            sample_weights = sample_weights * np.exp(alpha * wrong)
            sample_weights = sample_weights / sample_weights.sum()

        return self

    # ----------------------------------------------------------------------

    def _train_weak_classifier(self, X, y, sample_weights) -> BiLSTMClassifier:
        model = BiLSTMClassifier(
            input_dim=self.input_dim,
            num_classes=2,
            norm=self.norm,
        ).to(self.device)
        opt   = optim.Adam(model.parameters(), lr=self.lr)
        # weighted CE - reduction="none" so we can multiply per-sample weights
        ce    = nn.CrossEntropyLoss(reduction="none")

        ds = TensorDataset(
            _to_tensor(X),
            torch.tensor(y, dtype=torch.long),
            _to_tensor(sample_weights),
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
        return model

    # ----------------------------------------------------------------------

    def _get_combined_predictions(self, X, only_latest: bool = False) -> np.ndarray:
        """
        Returns a (N, n_classes) score matrix. When only_latest is True we
        score with just the last round's weak classifiers - that's the
        snapshot we need to update sample weights mid-fit.
        """
        rounds = (self.weak_classifiers[-1:] if only_latest
                  else self.weak_classifiers)
        weights = ([1.0] if only_latest
                   else self.classifier_weights[:len(rounds)])

        scores = np.zeros((X.shape[0], len(self.classes_)))
        with torch.no_grad():
            xt = _to_tensor(X).to(self.device)
            for w, round_classifiers in zip(weights, rounds):
                for ci, model in enumerate(round_classifiers):
                    logits = model(xt).softmax(-1).cpu().numpy()
                    # logits[:, 1] is "this is class ci" probability
                    scores[:, ci] += w * logits[:, 1]
        return scores

    # ----------------------------------------------------------------------

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        scores = self._get_combined_predictions(X)
        # normalise rows so they look like probabilities
        scores = scores - scores.min(axis=1, keepdims=True)
        s = scores.sum(axis=1, keepdims=True)
        s[s == 0] = 1
        return scores / s

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
