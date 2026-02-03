"""Accuracy, macro-F1, confusion matrix, per-class precision/recall."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix as sk_confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Classification accuracy."""
    return float(accuracy_score(y_true, y_pred))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Macro-averaged F1 (unweighted mean over classes)."""
    return float(
        f1_score(y_true, y_pred, average="macro", zero_division=0, labels=range(num_classes))
    )


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[int] | None = None
) -> np.ndarray:
    """Confusion matrix (rows=true, cols=pred)."""
    return sk_confusion_matrix(y_true, y_pred, labels=labels)


def per_class_precision_recall(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[int] | None = None
) -> dict[str, Any]:
    """Per-class precision, recall, f1, support. Keys: precision, recall, f1, support (lists by class)."""
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=labels
    )
    return {
        "precision": p.tolist(),
        "recall": r.tolist(),
        "f1": f.tolist(),
        "support": s.tolist(),
    }
