"""
GloVe embedding implementation - abandoned.

Per the report: "as it had poor performance in most cases we decided to
remove this model." We keep the code so the abandoned-experiment narrative
in the dissertation is reproducible.

Pipeline:

  1. Build a vocabulary from the cleaned tweets.
  2. Build a windowed co-occurrence matrix.
  3. Train Gensim's wrappers / a simple PyTorch GloVe layer.

We use the PyTorch implementation here because Gensim no longer ships a
canonical GloVe trainer in recent versions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .. import config


class GloveEmbedder:
    """
    Compact GloVe trainer.

    `build_vocab` -> `build_cooccurrence` -> `train` -> `embed`. Stored
    embeddings live on `self.W + self.W_tilde` (the original paper's
    word-plus-context summation).
    """

    def __init__(self,
                 dim: int     = config.GLOVE_DIM,
                 window: int  = config.GLOVE_WINDOW,
                 x_max: int   = 100,
                 alpha: float = 0.75,
                 min_count: int = 2):
        self.dim       = dim
        self.window    = window
        self.x_max     = x_max
        self.alpha     = alpha
        self.min_count = min_count

        self.word2idx: dict[str, int] = {}
        self.W:       np.ndarray | None = None
        self.W_tilde: np.ndarray | None = None

    # ---- vocabulary -------------------------------------------------------

    def build_vocab(self, sentences: list[list[str]]):
        counter = Counter()
        for s in sentences:
            counter.update(s)
        self.word2idx = {w: i for i, (w, c) in enumerate(counter.most_common())
                         if c >= self.min_count}

    # ---- co-occurrence ----------------------------------------------------

    def build_cooccurrence(self, sentences: list[list[str]]) -> dict:
        co: dict = defaultdict(float)
        for sent in sentences:
            ids = [self.word2idx[w] for w in sent if w in self.word2idx]
            for i, w in enumerate(ids):
                lo = max(0, i - self.window)
                for j in range(lo, i):
                    distance = i - j
                    co[(w, ids[j])] += 1.0 / distance
                    co[(ids[j], w)] += 1.0 / distance
        return co

    # ---- training ---------------------------------------------------------

    def train(self, sentences: list[list[str]],
              epochs: int = 30, lr: float = 0.05) -> "GloveEmbedder":
        if not self.word2idx:
            self.build_vocab(sentences)
        co = self.build_cooccurrence(sentences)

        V = len(self.word2idx)
        # init: small random vectors + zero biases
        W       = nn.Embedding(V, self.dim).weight
        W_tilde = nn.Embedding(V, self.dim).weight
        b       = nn.Embedding(V, 1).weight
        b_tilde = nn.Embedding(V, 1).weight
        nn.init.uniform_(W,       -0.5, 0.5)
        nn.init.uniform_(W_tilde, -0.5, 0.5)

        opt = optim.Adam([W, W_tilde, b, b_tilde], lr=lr)

        pairs = list(co.items())
        for _ in range(epochs):
            opt.zero_grad()
            i_idx = torch.tensor([p[0][0] for p in pairs])
            j_idx = torch.tensor([p[0][1] for p in pairs])
            x_ij  = torch.tensor([p[1] for p in pairs], dtype=torch.float32)
            wi, wj = W[i_idx], W_tilde[j_idx]
            bi, bj = b[i_idx].squeeze(), b_tilde[j_idx].squeeze()
            f = torch.where(x_ij < self.x_max,
                            (x_ij / self.x_max) ** self.alpha,
                            torch.ones_like(x_ij))
            loss = (f * ((wi * wj).sum(-1) + bi + bj - x_ij.log()).pow(2)).mean()
            loss.backward()
            opt.step()

        self.W       = W.detach().numpy()
        self.W_tilde = W_tilde.detach().numpy()
        return self

    # ---- inference --------------------------------------------------------

    def embed(self, tokens: list[str]) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("Train first.")
        E = self.W + self.W_tilde
        vecs = [E[self.word2idx[t]] for t in tokens if t in self.word2idx]
        if not vecs:
            return np.zeros(self.dim, dtype=np.float32)
        return np.mean(vecs, axis=0).astype(np.float32)
