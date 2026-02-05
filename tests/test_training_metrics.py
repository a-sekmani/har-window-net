"""Tests for training metrics: accuracy, macro_f1, confusion_matrix, per_class_precision_recall."""

import numpy as np
import pytest

from har_windownet.training.metrics import (
    accuracy,
    confusion_matrix,
    macro_f1,
    per_class_precision_recall,
)


def test_accuracy_perfect() -> None:
    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = np.array([0, 1, 2, 1, 0])
    assert accuracy(y_true, y_pred) == 1.0


def test_accuracy_half() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 0, 1])
    assert accuracy(y_true, y_pred) == 0.75


def test_accuracy_zero() -> None:
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 1, 1])
    assert accuracy(y_true, y_pred) == 0.0


def test_macro_f1_perfect() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    assert macro_f1(y_true, y_pred, num_classes=3) == pytest.approx(1.0)


def test_macro_f1_all_wrong() -> None:
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 1, 1])
    assert macro_f1(y_true, y_pred, num_classes=2) == 0.0


def test_macro_f1_mixed() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    f1 = macro_f1(y_true, y_pred, num_classes=2)
    assert 0.0 < f1 < 1.0


def test_confusion_matrix_shape() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    labels = [0, 1, 2]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    assert cm.shape == (3, 3)


def test_confusion_matrix_diagonal_perfect() -> None:
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    assert np.diag(cm).sum() == 3
    assert cm[0, 0] == 1 and cm[1, 1] == 1 and cm[2, 2] == 1


def test_per_class_precision_recall_keys() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    out = per_class_precision_recall(y_true, y_pred, labels=[0, 1])
    assert "precision" in out
    assert "recall" in out
    assert "f1" in out
    assert "support" in out
    assert len(out["precision"]) == 2
    assert len(out["recall"]) == 2
    assert len(out["f1"]) == 2
    assert len(out["support"]) == 2


def test_per_class_precision_recall_support() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    out = per_class_precision_recall(y_true, y_pred, labels=[0, 1])
    assert out["support"] == [2, 2]
