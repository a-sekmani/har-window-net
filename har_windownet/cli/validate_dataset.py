"""CLI: validate Phase A dataset (splits, contract, label_map)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from har_windownet.contracts.labels import load_label_map
from har_windownet.contracts.window import WINDOW_SIZE, validate_window_dict


def _load_dataset_window_size(data_root: Path) -> int | None:
    """Read window_size from dataset_meta.json; None if file missing or invalid."""
    meta_path = data_root / "dataset_meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        return int(meta.get("window_size"))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def main() -> None:
    p = argparse.ArgumentParser(
        description="Validate Phase A dataset (Window contract, splits, label_map)"
    )
    p.add_argument(
        "--data",
        required=True,
        help="Path to dataset root (contains splits/, label_map.json)",
    )
    args = p.parse_args()
    data_root = Path(args.data)

    errors: list[str] = []
    if not data_root.is_dir():
        errors.append(f"Not a directory: {data_root}")
        for e in errors:
            print(e, file=sys.stderr)
        sys.exit(1)

    # window_size from dataset_meta (for datasets built with --window-size 60 etc.)
    dataset_window_size = _load_dataset_window_size(data_root)
    window_size = dataset_window_size if dataset_window_size is not None else WINDOW_SIZE

    # Required files
    label_map_path = data_root / "label_map.json"
    if not label_map_path.exists():
        errors.append(f"Missing label_map.json at {label_map_path}")
    else:
        try:
            label_map = load_label_map(label_map_path)
            label_to_id = label_map.get("label_to_id") or {}
        except Exception as e:
            errors.append(f"Failed to load label_map.json: {e}")
            label_to_id = {}

    splits_dir = data_root / "splits"
    if not splits_dir.is_dir():
        errors.append(f"Missing splits/ directory at {splits_dir}")

    for split in ("train", "val", "test"):
        parquet_path = splits_dir / f"{split}.parquet"
        if not parquet_path.exists():
            continue
        try:
            table = pq.read_table(parquet_path)
        except Exception as e:
            errors.append(f"Failed to read {split}.parquet: {e}")
            continue
        if "label" not in table.column_names or "keypoints" not in table.column_names:
            errors.append(f"{split}.parquet missing 'label' or 'keypoints' column")
        n = len(table)
        col_names = table.column_names
        for i in range(n):
            row = {}
            for name in col_names:
                val = table.column(name)[i]
                row[name] = val.as_py() if hasattr(val, "as_py") else val
            row_errors = validate_window_dict(row, window_size=window_size)
            for err in row_errors:
                errors.append(f"{split}.parquet row {i}: {err}")
            if label_to_id and "label" in row and row["label"] not in label_to_id:
                errors.append(f"{split}.parquet row {i}: label {row['label']!r} not in label_map")

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        sys.exit(1)
    print("Validation passed.")


if __name__ == "__main__":
    main()
