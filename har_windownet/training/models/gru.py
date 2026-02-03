"""GRU baseline: (B, T, F) -> logits. Input T=30, F=51."""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUModel(nn.Module):
    """
    GRU for window classification. Input (B, T, F); last hidden -> FC -> logits.
    """

    def __init__(
        self,
        num_classes: int,
        input_features: int = 51,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_features,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        _, h = self.gru(x)
        h = h[-1]
        return self.fc(h)
