"""WindowDataset: load Phase A Parquet splits, return (x, y) tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from har_windownet.contracts.labels import load_label_map
from har_windownet.contracts.window import KEYPOINT_DIM, NUM_KEYPOINTS, WINDOW_SIZE

# F = K * C = 17 * 3 = 51
INPUT_FEATURES = NUM_KEYPOINTS * KEYPOINT_DIM


class WindowDataset(torch.utils.data.Dataset):
    """
    Load windows from Phase A Parquet; return x (T, F) float32, y (int) label_id.

    Flatten keypoints to (T, F): frame-major, each frame 17 keypoints x (x,y,conf).
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        label_map_path: str | Path | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.label_map_path = Path(label_map_path or self.data_root / "label_map.json")
        if not self.label_map_path.exists():
            raise FileNotFoundError(
                f"label_map.json not found at {self.label_map_path}. "
                "Ensure --data points to a Phase A dataset (run build_dataset with --out <dir> first)."
            )
        self.label_map = load_label_map(self.label_map_path)
        self.num_classes = self.label_map["num_classes"]
        self.label_to_id = self.label_map["label_to_id"]

        parquet_path = self.data_root / "splits" / f"{split}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Split file not found: {parquet_path}")
        self.table = pq.read_table(parquet_path)
        self._length = len(self.table)

    def __len__(self) -> int:
        return self._length

    def _keypoints_to_tensor(self, keypoints: Any) -> torch.Tensor:
        """Convert keypoints (list or array) to (T, F) float32."""
        arr = np.array(keypoints, dtype=np.float32)
        if arr.shape != (WINDOW_SIZE, NUM_KEYPOINTS, KEYPOINT_DIM):
            raise ValueError(f"Expected keypoints (30, 17, 3), got {arr.shape}")
        # Flatten: (T, K, C) -> (T, K*C)
        arr = arr.reshape(WINDOW_SIZE, INPUT_FEATURES)
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        kp = self.table.column("keypoints")[idx]
        label_str = self.table.column("label")[idx]
        if hasattr(kp, "as_py"):
            kp = kp.as_py()
        if hasattr(label_str, "as_py"):
            label_str = label_str.as_py()
        label_id = self.label_to_id[str(label_str)]
        x = self._keypoints_to_tensor(kp)
        return x, label_id
