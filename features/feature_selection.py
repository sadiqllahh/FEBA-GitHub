"""
Custom decision tree + Recursive Feature Elimination (RFE).

Section (f) of the report describes a tree built from scratch (gini-based
splits) with an RFE wrapper that drops the least-important feature each
round until n_features_to_select remain.

Why not sklearn? Because the report explicitly says "Pythons' classes were
used to implement a custom decision tree." We respect that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .. import config


# --------------------------------------------------------------------------
# Tree internals
# --------------------------------------------------------------------------

@dataclass
class _Node:
    """A binary tree node. Either internal (feature/threshold + children) or
    leaf (value)."""
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None
    value: Optional[int] = None

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


def _gini(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / y.size
    return 1.0 - np.sum(p ** 2)


# --------------------------------------------------------------------------
# Decision tree
# --------------------------------------------------------------------------

class DecisionTree:
    """
    Plain CART classifier.

    Parameters mirror what the report calls out: min_samples_split,
    max_depth, n_features. `n_features` is the number of features
    considered at each split (None = all of them).
    """

    def __init__(self,
                 min_samples_split: int = config.TREE_MIN_SPLIT,
                 max_depth: int         = config.TREE_MAX_DEPTH,
                 n_features: int | None = None):
        self.min_samples_split = min_samples_split
        self.max_depth         = max_depth
        self.n_features        = n_features
        self.root: _Node | None = None
        self._n_features_total: int = 0

    # ---- training --------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._n_features_total = X.shape[1]
        if self.n_features is None or self.n_features > X.shape[1]:
            self.n_features = X.shape[1]
        self.root = self._grow(X, y, depth=0)
        return self

    def _grow(self, X, y, depth) -> _Node:
        n_samples, _ = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or
            n_labels == 1 or
            n_samples < self.min_samples_split):
            return _Node(value=self._majority(y))

        feat_idx = np.random.choice(self._n_features_total,
                                    self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_idx)
        if best_feat is None:
            return _Node(value=self._majority(y))

        left_mask  = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask
        # avoid empty splits
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return _Node(value=self._majority(y))

        return _Node(
            feature=best_feat,
            threshold=best_thresh,
            left=self._grow(X[left_mask],  y[left_mask],  depth + 1),
            right=self._grow(X[right_mask], y[right_mask], depth + 1),
        )

    def _best_split(self, X, y, feat_idx):
        best_gain, best_feat, best_thresh = -1.0, None, None
        parent = _gini(y)

        for f in feat_idx:
            thresholds = np.unique(X[:, f])
            for t in thresholds:
                left = y[X[:, f] <= t]
                right = y[X[:, f] > t]
                if left.size == 0 or right.size == 0:
                    continue
                w_l, w_r = left.size / y.size, right.size / y.size
                gain = parent - (w_l * _gini(left) + w_r * _gini(right))
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, f, t
        return best_feat, best_thresh

    @staticmethod
    def _majority(y: np.ndarray) -> int:
        vals, counts = np.unique(y, return_counts=True)
        return int(vals[np.argmax(counts)])

    # ---- prediction ------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node: _Node):
        if node.is_leaf:
            return node.value
        branch = node.left if x[node.feature] <= node.threshold else node.right
        return self._traverse(x, branch)

    # ---- importance helpers ---------------------------------------------

    def feature_importance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Permutation-style importance: for each feature, shuffle its column
        and measure the accuracy drop. Sklearn-free, slow but transparent.
        """
        baseline = (self.predict(X) == y).mean()
        importances = np.zeros(X.shape[1])
        for f in range(X.shape[1]):
            saved = X[:, f].copy()
            np.random.shuffle(X[:, f])
            importances[f] = baseline - (self.predict(X) == y).mean()
            X[:, f] = saved
        return importances


# --------------------------------------------------------------------------
# RFE
# --------------------------------------------------------------------------

class RecursiveFeatureEliminator:
    """
    Wraps a DecisionTree and performs RFE.

    At each iteration we re-fit the tree on the remaining features,
    measure permutation importance, and drop the lowest-scoring one. The
    process continues until `n_features_to_select` remain.

    `selected_indices_` holds the surviving feature columns from the
    *original* X after `select()` returns. `iteration_ranks_` keeps a per-
    iteration record like Figure 18 in the report.
    """

    def __init__(self,
                 tree: DecisionTree | None = None,
                 n_features_to_select: int = config.RFE_N_FEATURES):
        self.tree = tree or DecisionTree()
        self.n_features_to_select = n_features_to_select
        self.selected_indices_: np.ndarray | None = None
        self.iteration_ranks_: list[np.ndarray]   = []

    def select(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        remaining = np.arange(X.shape[1])
        X_work = X.copy()

        while remaining.size > self.n_features_to_select:
            self.tree.fit(X_work, y)
            importances = self.tree.feature_importance(X_work, y)
            self.iteration_ranks_.append(importances.copy())

            worst_local = int(np.argmin(importances))
            X_work    = np.delete(X_work, worst_local, axis=1)
            remaining = np.delete(remaining, worst_local)

        self.selected_indices_ = remaining
        return X[:, remaining]


def select_features(X: np.ndarray, y: np.ndarray,
                    n_features: int = config.RFE_N_FEATURES):
    """One-line helper. Returns (X_reduced, selected_indices)."""
    rfe = RecursiveFeatureEliminator(n_features_to_select=n_features)
    X_red = rfe.select(X, y)
    return X_red, rfe.selected_indices_
