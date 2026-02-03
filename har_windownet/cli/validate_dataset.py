"""CLI: validate Phase A dataset (splits, contract, label_map)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq

from har_windownet.contracts.labels import load_label_map
from har_windownet.contracts.window import validate_window_dict


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
            row_errors = validate_window_dict(row)
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
