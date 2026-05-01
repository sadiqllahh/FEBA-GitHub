"""
BiLSTM used as a weak learner inside the F-EBA ensemble.

The model takes a fixed-size feature vector per tweet (concatenation of
sentiment + Word2Vec + self-attention + BERT after RFE selection),
re-shapes it as a unit-length sequence and runs a bidirectional LSTM over
it. A normalisation layer follows the LSTM - the kind of normalisation
used here is the knob we sweep during hyperparameter tuning.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .. import config


# ---------------------------------------------------------------------------

class _InstanceNorm1dWrap(nn.Module):
    """Instance norm expects (N, C, L). Our tensor is (N, C). Reshape and
    forward."""
    def __init__(self, num_features: int):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=True)

    def forward(self, x):
        return self.norm(x.unsqueeze(-1)).squeeze(-1)


def make_norm(kind: str, num_features: int) -> nn.Module:
    """Pick a normalisation layer by string. The hyperparameter-tuning
    section of the report sweeps over these three."""
    kind = kind.lower()
    if kind == "batch":
        return nn.BatchNorm1d(num_features)
    if kind == "instance":
        return _InstanceNorm1dWrap(num_features)
    if kind == "layer":
        return nn.LayerNorm(num_features)
    raise ValueError(f"Unknown norm: {kind!r}")


# ---------------------------------------------------------------------------

class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM with a configurable normalisation stage and a binary
    output head. Designed to act as a weak classifier inside boosting, so
    it's intentionally small.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = config.BILSTM_HIDDEN,
                 dropout: float  = config.BILSTM_DROPOUT,
                 num_classes: int = 2,
                 norm: str       = "layer"):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size  = input_dim,
            hidden_size = hidden_dim,
            num_layers  = 1,
            batch_first = True,
            bidirectional = True,
        )
        self.norm    = make_norm(norm, hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F) -> (B, 1, F) so the LSTM sees a one-step sequence
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out    = out.squeeze(1)              # (B, 2*hidden)
        out    = self.norm(out)
        out    = self.dropout(out)
        return self.head(out)
