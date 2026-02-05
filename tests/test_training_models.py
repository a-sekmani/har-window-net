"""Tests for training models: get_model, TCN/GRU forward shape."""

import pytest
import torch

from har_windownet.contracts.window import WINDOW_SIZE
from har_windownet.training.models import get_model


def test_get_model_tcn() -> None:
    model = get_model("tcn", num_classes=10, input_features=51)
    assert model.__class__.__name__ == "TCN"
    assert model.num_classes == 10
    assert model.input_features == 51


def test_get_model_gru() -> None:
    model = get_model("gru", num_classes=5, input_features=51)
    assert model.__class__.__name__ == "GRUModel"
    assert model.num_classes == 5
    assert model.input_features == 51


def test_get_model_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        get_model("unknown", num_classes=10)


def test_tcn_forward_shape() -> None:
    model = get_model("tcn", num_classes=10, input_features=51)
    x = torch.zeros(4, WINDOW_SIZE, 51)
    logits = model(x)
    assert logits.shape == (4, 10)


def test_tcn_forward_custom_F() -> None:
    model = get_model("tcn", num_classes=3, input_features=95)  # e.g. combo features
    x = torch.zeros(2, WINDOW_SIZE, 95)
    logits = model(x)
    assert logits.shape == (2, 3)


def test_gru_forward_shape() -> None:
    model = get_model("gru", num_classes=10, input_features=51)
    x = torch.zeros(4, WINDOW_SIZE, 51)
    logits = model(x)
    assert logits.shape == (4, 10)


def test_gru_forward_custom_F() -> None:
    model = get_model("gru", num_classes=5, input_features=85)
    x = torch.zeros(2, WINDOW_SIZE, 85)
    logits = model(x)
    assert logits.shape == (2, 5)
