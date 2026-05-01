"""
Explainability via SHAP.

Three things the report describes:

  1. Mean(|SHAP value|) ranking per emotion word (the bar chart).
  2. Per-sample SHAP scatter showing positive / negative impact on the
     model's prediction.
  3. SHAP dependence plot - how feature X varies with feature Y (the
     'love' vs 'excellent' example).

We wrap shap.Explainer so the rest of the codebase can call a single
`ShapAnalyzer` regardless of whether the underlying model is the F-EBA
ensemble, the boosting baseline, or a sklearn-style estimator.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import shap
import torch
import matplotlib.pyplot as plt


class ShapAnalyzer:
    """
    Computes and plots SHAP values for any predict-style callable.

    The model passed in only needs to expose `predict_proba(X) -> (N, C)`.
    Background data is sub-sampled to speed up KernelExplainer; pass a
    larger `background_size` for tighter estimates at the cost of runtime.
    """

    def __init__(self, model, background: np.ndarray,
                 feature_names: list[str] | None = None,
                 background_size: int = 100):
        self.model = model
        # KernelExplainer is the most general - works on anything callable
        bg = shap.sample(background, background_size, random_state=0)
        self.explainer = shap.KernelExplainer(self._predict_fn, bg)
        self.feature_names = feature_names

    # ----------------------------------------------------------------------

    def _predict_fn(self, X):
        # SHAP feeds in numpy 2-D; route into the wrapped model
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        if isinstance(self.model, torch.nn.Module):
            with torch.no_grad():
                t = torch.tensor(X, dtype=torch.float32)
                return self.model(t).softmax(-1).cpu().numpy()
        return self.model(X)

    # ----------------------------------------------------------------------

    def explain(self, X: np.ndarray, n_samples: int = 200) -> np.ndarray:
        """Returns a (N, F) array of SHAP values for the positive class."""
        sample = X[:n_samples]
        values = self.explainer.shap_values(sample)
        # binary classification - SHAP returns a list of two arrays
        if isinstance(values, list):
            values = values[1]
        return np.array(values)

    # ---- plots -----------------------------------------------------------

    def plot_mean_importance(self, shap_values: np.ndarray, top_k: int = 20,
                             ax=None):
        """Bar chart of mean(|SHAP|). Matches the report's description of
        'words that have the highest mean SHAP'."""
        ax = ax or plt.gca()
        importance = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(importance)[-top_k:]
        names = (np.array(self.feature_names)[top_idx]
                 if self.feature_names else
                 [f"f{i}" for i in top_idx])

        ax.barh(range(top_k), importance[top_idx])
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(names)
        ax.set_xlabel("Mean(|SHAP|)")
        ax.set_title("Top features by SHAP importance")

    def plot_summary(self, shap_values: np.ndarray, X: np.ndarray):
        """Beeswarm summary plot - dot colour shows feature value, x-axis
        shows SHAP impact."""
        shap.summary_plot(shap_values, X, feature_names=self.feature_names,
                          show=False)

    def plot_dependence(self, shap_values: np.ndarray, X: np.ndarray,
                        feature: int | str, interaction: int | str | None = None):
        """e.g. plot_dependence(values, X, 'love', interaction='excellent')"""
        shap.dependence_plot(feature, shap_values, X,
                             feature_names=self.feature_names,
                             interaction_index=interaction, show=False)
