"""WindowDataset: load Phase A Parquet splits, return (x, y) tensors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from har_windownet.contracts.labels import load_label_map
from har_windownet.contracts.window import KEYPOINT_DIM, NUM_KEYPOINTS, WINDOW_SIZE

# F = K * C = 17 * 3 = 51
INPUT_FEATURES = NUM_KEYPOINTS * KEYPOINT_DIM


def _load_dataset_window_size(data_root: Path) -> int:
    """Read window_size from dataset_meta.json; default to WINDOW_SIZE (30) if missing."""
    meta_path = data_root / "dataset_meta.json"
    if not meta_path.exists():
        return WINDOW_SIZE
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        return int(meta.get("window_size", WINDOW_SIZE))
    except (json.JSONDecodeError, TypeError, ValueError):
        return WINDOW_SIZE


class WindowDataset(torch.utils.data.Dataset):
    """
    Load windows from Phase A Parquet; return x (T, F) float32, y (int) label_id.

    Flatten keypoints to (T, F): frame-major, each frame 17 keypoints x (x,y,conf).
    T (window_size) is read from dataset_meta.json so datasets built with --window-size 60 work.
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
        self.window_size = _load_dataset_window_size(self.data_root)

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
        expected = (self.window_size, NUM_KEYPOINTS, KEYPOINT_DIM)
        if arr.shape != expected:
            raise ValueError(f"Expected keypoints {expected}, got {arr.shape}")
        # Flatten: (T, K, C) -> (T, K*C)
        arr = arr.reshape(self.window_size, INPUT_FEATURES)
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
