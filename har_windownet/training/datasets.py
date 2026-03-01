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
from har_windownet.features.transforms import build_feature_pipeline, get_input_features

# F = K * C = 17 * 3 = 51 (baseline raw)
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

    Flatten keypoints to (T, F): frame-major. When feature_config is None (baseline),
    F=51 (17*3). When feature_config is set, F comes from get_input_features(feature_config).
    T (window_size) is read from dataset_meta.json.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        label_map_path: str | Path | None = None,
        feature_config: dict[str, Any] | None = None,
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
        self.feature_config = feature_config
        if feature_config is not None:
            self._feature_transform, self._input_features = build_feature_pipeline(feature_config)
        else:
            self._feature_transform = None
            self._input_features = INPUT_FEATURES

        parquet_path = self.data_root / "splits" / f"{split}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Split file not found: {parquet_path}")
        self.table = pq.read_table(parquet_path)
        self._length = len(self.table)

    @property
    def input_features(self) -> int:
        """Number of input features F (for model construction)."""
        return self._input_features

    def __len__(self) -> int:
        return self._length

    def get_class_counts(self) -> list[int]:
        """Return list of sample counts per class (ordered by class_id 0..num_classes-1)."""
        labels = self.table.column("label").to_pylist()
        counts = [0] * self.num_classes
        for label_str in labels:
            label_id = self.label_to_id[str(label_str)]
            counts[label_id] += 1
        return counts

    def _keypoints_to_tensor(self, keypoints: Any) -> torch.Tensor:
        """Convert keypoints (list or array) to (T, F) float32 (raw baseline)."""
        arr = np.array(keypoints, dtype=np.float32)
        expected = (self.window_size, NUM_KEYPOINTS, KEYPOINT_DIM)
        if arr.shape != expected:
            raise ValueError(f"Expected keypoints {expected}, got {arr.shape}")
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

        if self._feature_transform is not None:
            arr = np.array(kp, dtype=np.float32)
            if arr.shape != (self.window_size, NUM_KEYPOINTS, KEYPOINT_DIM):
                raise ValueError(f"Expected keypoints ({self.window_size}, 17, 3), got {arr.shape}")
            features = self._feature_transform(arr)
            return torch.from_numpy(features.astype(np.float32)), label_id

        x = self._keypoints_to_tensor(kp)
        return x, label_id
