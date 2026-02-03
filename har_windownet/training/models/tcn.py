"""TCN baseline: (B, T, F) -> logits. Input T=30, F=51."""

from __future__ import annotations

import torch
import torch.nn as nn


class TCN(nn.Module):
    """
    Temporal Convolutional Network for window classification.

    Input: (B, T, F) with T=30, F=51. Output: (B, num_classes) logits.
    """

    def __init__(
        self,
        num_classes: int,
        input_features: int = 51,
        hidden: int = 128,
        num_layers: int = 3,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_features = input_features
        layers = []
        in_ch = input_features
        for _ in range(num_layers):
            layers.append(
                nn.Conv1d(in_ch, hidden, kernel_size, padding=kernel_size // 2)
            )
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.2))
            in_ch = hidden
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> (B, F, T) for Conv1d
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        return self.fc(x)
