"""
NumPy implementation of the self-attention mechanism described in section
(d) of the report.

Two hyperparameters: input_dim (size of each token vector) and hidden_dim
(projection size for Q/K/V). For a single tweet we treat the rows of the
input as tokens and compute the attention-weighted output.
"""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax. Subtract the max before exp to avoid
    overflow on large activations."""
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class SelfAttention:
    """
    Single-head dot-product self-attention.

    Q, K, V projection matrices are initialised with a small random scale
    (Xavier-ish). Once `fit_random()` is called the layer is deterministic
    given a fixed seed - which is what we want when comparing weighted
    embeddings against unweighted baselines.
    """

    def __init__(self, input_dim: int, hidden_dim: int, seed: int = 42):
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(input_dim)
        self.W_q = rng.normal(0, scale, (input_dim, hidden_dim))
        self.W_k = rng.normal(0, scale, (input_dim, hidden_dim))
        self.W_v = rng.normal(0, scale, (input_dim, hidden_dim))

    # ----------------------------------------------------------------------

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : (T, D) array - T tokens, D = input_dim

        Returns
        -------
        (T, hidden_dim) attended representation.
        """
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        scores  = (Q @ K.T) / np.sqrt(self.hidden_dim)
        weights = softmax(scores, axis=-1)
        return weights @ V

    # ----------------------------------------------------------------------

    def attention_weights(self, x: np.ndarray) -> np.ndarray:
        """
        Same maths as __call__ but returns the (T, T) attention matrix
        instead of the projected values - useful for plotting feature
        relevance like Figure 14 in the report.
        """
        Q = x @ self.W_q
        K = x @ self.W_k
        return softmax((Q @ K.T) / np.sqrt(self.hidden_dim), axis=-1)


def attend_sequence(vec: np.ndarray, hidden_dim: int, seed: int = 42) -> np.ndarray:
    """
    Convenience wrapper for a single tweet that's already represented as a
    1D vector (e.g. averaged Word2Vec). We treat the vector as a sequence of
    1-D tokens by reshaping, run attention, and average the result back to
    1D so it slots into a feature matrix.
    """
    if vec.ndim == 1:
        vec = vec.reshape(-1, 1)
    sa = SelfAttention(input_dim=vec.shape[1], hidden_dim=hidden_dim, seed=seed)
    return sa(vec).mean(axis=0)
