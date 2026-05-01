"""
Evaluation utilities.

Centralises confusion matrix, ROC, PRC, accuracy, and k-fold CV so the
hyperparameter-tuning section, the main training pipeline, and the
generalisation script all use the same metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
)
from sklearn.model_selection import StratifiedKFold

import config


# ---------------------------------------------------------------------------

@dataclass
class EvalReport:
    """All-in-one bundle of evaluation outputs for a single model run."""
    accuracy:    float
    confusion:   np.ndarray
    roc_auc:     float
    pr_auc:      float
    fpr:         np.ndarray = field(default_factory=lambda: np.array([]))
    tpr:         np.ndarray = field(default_factory=lambda: np.array([]))
    precision:   np.ndarray = field(default_factory=lambda: np.array([]))
    recall:      np.ndarray = field(default_factory=lambda: np.array([]))


# ---------------------------------------------------------------------------

def evaluate(y_true, y_pred, y_proba=None) -> EvalReport:
    """
    `y_proba` is the positive-class probability vector. If None we skip the
    curve metrics and leave them at NaN.
    """
    cm = confusion_matrix(y_true, y_pred)

    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        roc_a = auc(fpr, tpr)
        pr_a  = average_precision_score(y_true, y_proba)
    else:
        fpr = tpr = precision = recall = np.array([])
        roc_a = pr_a = float("nan")

    return EvalReport(
        accuracy = accuracy_score(y_true, y_pred),
        confusion = cm,
        roc_auc   = float(roc_a),
        pr_auc    = float(pr_a),
        fpr=fpr, tpr=tpr,
        precision=precision, recall=recall,
    )


# ---------------------------------------------------------------------------

def plot_confusion(report: EvalReport, title: str = "Confusion matrix",
                   class_names=("non-depressed", "depressed"), ax=None):
    ax = ax or plt.gca()
    cm = report.confusion
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")


def plot_roc(report: EvalReport, label: str = "model", ax=None):
    ax = ax or plt.gca()
    ax.plot(report.fpr, report.tpr, label=f"{label} (AUC={report.roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")


def plot_prc(report: EvalReport, label: str = "model", ax=None):
    ax = ax or plt.gca()
    ax.plot(report.recall, report.precision,
            label=f"{label} (AP={report.pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    ax.legend(loc="lower left")


# ---------------------------------------------------------------------------

def cross_validate(model_factory, X, y,
                   k: int = config.CV_FOLDS) -> list[EvalReport]:
    """
    `model_factory` is a no-arg callable that returns a *fresh* fitted model.
    Inside this function we re-call it for every fold and let it fit on the
    fold's training partition.

    Each fitted model must expose `.predict(X)` and `.predict_proba(X)`.
    """
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=config.RANDOM_SEED)
    reports = []
    for fold, (tr, te) in enumerate(kf.split(X, y), 1):
        model = model_factory()
        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[te])
        # binary case: positive class is column 1
        report = evaluate(y[te], model.predict(X[te]),
                          y_proba=proba[:, 1] if proba.ndim == 2 else None)
        reports.append(report)
    return reports
