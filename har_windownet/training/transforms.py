"""Optional augmentations for skeleton windows (noise, jitter). Start minimal."""

from __future__ import annotations

import torch

# Placeholder: no augmentations by default; add Gaussian noise / jitter later if needed.


def no_augment(x: torch.Tensor) -> torch.Tensor:
    """Identity transform."""
    return x
