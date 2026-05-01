"""
Adversarial layer for F-EBA.

Implements the Fast Gradient Sign Method (FGSM) attack and the
adversarial-augmented training loop described in the report.

Maths:

    G    = ∇_X L(f(X), y)
    X'   = X + ε · sign(G)

where ε (epsilon) is the perturbation magnitude. We compute the gradient
w.r.t. the input embeddings, take its sign, scale by ε, and add it back
to the original input. The model is then trained on a mix of the original
and the perturbed batches.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .config import FGSM_EPSILON, ADV_LOSS_WEIGHT, FEBA_BATCH_SIZE, FEBA_LR


# ---------------------------------------------------------------------------

def fgsm_attack(data: torch.Tensor,
                labels: torch.Tensor,
                model: nn.Module,
                loss_fn: nn.Module,
                epsilon: float = FGSM_EPSILON,
                clip_range: tuple[float, float] | None = None) -> torch.Tensor:
    """
    One FGSM step.

    Parameters
    ----------
    data        : (B, F) feature tensor. We treat its grad as the input
                  gradient direction.
    labels      : (B,) class labels.
    model       : model whose loss we differentiate.
    loss_fn     : loss function (cross-entropy in our case).
    epsilon     : perturbation magnitude.
    clip_range  : optional (low, high) range to clamp the perturbed data to.
                  Pass None when working on un-bounded embeddings (default
                  for BERT-style features).

    Returns
    -------
    Perturbed data tensor of the same shape, detached from the graph.
    """

    data = data.clone().detach().requires_grad_(True)
    model.eval()                                       # report-aligned step

    outputs = model(data)
    loss    = loss_fn(outputs, labels)

    model.zero_grad()
    loss.backward()

    grad_sign  = data.grad.detach().sign()
    perturbed  = data.detach() + epsilon * grad_sign
    if clip_range is not None:
        perturbed = perturbed.clamp(*clip_range)
    return perturbed.detach()


# ---------------------------------------------------------------------------

class AdversarialTrainer:
    """
    Trains any nn.Module with a mix of clean and FGSM-perturbed batches.

    The total loss for each batch is

        L_total = L_clean + λ * L_adv

    where λ (`adv_weight`) defaults to 0.5. Lower it to make the model
    rely less on adversarial examples; raise it if you want more
    robustness at the expense of clean-set accuracy.
    """

    def __init__(self,
                 model: nn.Module,
                 epsilon: float    = FGSM_EPSILON,
                 adv_weight: float = ADV_LOSS_WEIGHT,
                 lr: float         = FEBA_LR,
                 device: str | None = None):
        self.model       = model
        self.epsilon     = epsilon
        self.adv_weight  = adv_weight
        self.device      = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer   = optim.Adam(model.parameters(), lr=lr)
        self.loss_fn     = nn.CrossEntropyLoss()
        self.model.to(self.device)

    # ----------------------------------------------------------------------

    def train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)

            # 1. clean forward / backward
            self.optimizer.zero_grad()
            logits_clean = self.model(xb)
            loss_clean   = self.loss_fn(logits_clean, yb)

            # 2. generate adversarial copies
            x_adv = fgsm_attack(xb, yb, self.model, self.loss_fn,
                                epsilon=self.epsilon)

            # 3. adversarial forward / backward
            self.model.train()                         # back to train mode
            logits_adv = self.model(x_adv)
            loss_adv   = self.loss_fn(logits_adv, yb)

            total = loss_clean + self.adv_weight * loss_adv
            total.backward()
            self.optimizer.step()
            running += total.item() * xb.size(0)

        return running / len(loader.dataset)

    # ----------------------------------------------------------------------

    def fit(self, X, y, epochs: int = 10, batch_size: int = FEBA_BATCH_SIZE):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        history = []
        for epoch in range(epochs):
            avg = self.train_one_epoch(loader)
            history.append(avg)
        return history
