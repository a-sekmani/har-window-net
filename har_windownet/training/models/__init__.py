"""Baseline models: TCN, GRU."""

from har_windownet.training.models.tcn import TCN
from har_windownet.training.models.gru import GRUModel

MODELS = {"tcn": TCN, "gru": GRUModel}


def get_model(name: str, num_classes: int, input_features: int = 51, **kwargs):
    """Get model by name."""
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}; choose from {list(MODELS)}")
    return MODELS[name](num_classes=num_classes, input_features=input_features, **kwargs)
