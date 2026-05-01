"""
Hyperparameter tuning - normalisation sweep.

The report compares batch / instance / layer normalisation across epochs
and reports that layer norm wins (~98% accuracy). This module trains a
fresh F-EBA under each normalisation and records per-epoch accuracy and
loss curves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import config
from models.bilstm import BiLSTMClassifier


@dataclass
class TuningRun:
    """Per-epoch metrics for one normalisation setting."""
    norm: str
    accuracy_per_epoch: list[float] = field(default_factory=list)
    loss_per_epoch:     list[float] = field(default_factory=list)


def _train_one(model, X_tr, y_tr, X_val, y_val, epochs, lr, batch_size, device):
    opt = optim.Adam(model.parameters(), lr=lr)
    ce  = nn.CrossEntropyLoss()
    ds  = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    accs, losses = [], []
    for _ in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = ce(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)

        # validation snapshot
        model.eval()
        with torch.no_grad():
            xt = torch.tensor(X_val, dtype=torch.float32).to(device)
            preds = model(xt).argmax(-1).cpu().numpy()
        accs.append(float((preds == y_val).mean()))
        losses.append(running / len(loader.dataset))
    return accs, losses


def sweep_normalisations(X_train, y_train, X_val, y_val,
                         norms=("batch", "instance", "layer"),
                         epochs: int = config.FEBA_EPOCHS,
                         lr: float   = config.FEBA_LR,
                         batch_size: int = config.FEBA_BATCH_SIZE
                         ) -> dict[str, TuningRun]:
    """
    Trains a separate BiLSTM under each normalisation strategy and returns
    a dict keyed by norm name. Plot the resulting curves to reproduce the
    figures in the hyperparameter-tuning section of the report.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs: dict[str, TuningRun] = {}
    input_dim = X_train.shape[1]

    for norm in norms:
        torch.manual_seed(config.RANDOM_SEED)
        model = BiLSTMClassifier(input_dim=input_dim, norm=norm).to(device)
        accs, losses = _train_one(model, X_train, y_train, X_val, y_val,
                                  epochs, lr, batch_size, device)
        runs[norm] = TuningRun(norm=norm,
                               accuracy_per_epoch=accs,
                               loss_per_epoch=losses)
    return runs
