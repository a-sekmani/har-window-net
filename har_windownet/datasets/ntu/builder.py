"""
NTU → Window dataset builder: list samples, read/preprocess/window, split, write Parquet + metadata.
"""

from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from har_windownet.contracts.labels import build_default_label_map, save_label_map
from har_windownet.contracts.window import (
    WINDOW_SIZE,
    make_created_at,
    make_window_id,
    ts_end_ms_from_window,
)
from har_windownet.datasets.ntu.preprocess import body_to_coco17_normalized
from har_windownet.datasets.ntu.reader import (
    list_ntu_samples,
    read_ntu_npy_full,
    read_ntu_skeleton_txt_full,
    sample_id_from_path,
    select_dominant_body,
)
from har_windownet.datasets.ntu.windowing import slice_windows

# NTU sample ID ends with A + 3 digits (e.g. S001C002P003R002A013)
LABEL_PATTERN = re.compile(r"A\d{3}$")


def action_label_from_sample_id(sample_id: str) -> str | None:
    """Extract NTU action label (e.g. A013) from sample ID (e.g. S001C002P003R002A013)."""
    m = LABEL_PATTERN.search(sample_id)
    return m.group(0) if m else None


def _read_sample_bodies(path: Path) -> list[dict[str, Any]]:
    """Read all bodies from one NTU sample; auto-detect .npy vs .skeleton."""
    if path.suffix.lower() == ".npy" or path.name.endswith(".skeleton.npy"):
        return read_ntu_npy_full(path)
    return read_ntu_skeleton_txt_full(path)


def build_windows_from_sample(
    path: Path,
    *,
    projection: Literal["rgb", "depth", "3d"] = "rgb",
    window_size: int = 30,
    stride: int | None = None,
    device_id: str = "ntu-offline",
    camera_id: str = "ntu-cam",
    fps: float = 30.0,
) -> list[dict[str, Any]]:
    """
    Read one NTU sample, preprocess to COCO-17, slice into windows, return list of window dicts
    (storage-ready: keypoints as list, no Pydantic).
    """
    sample_id = sample_id_from_path(path)
    label = action_label_from_sample_id(sample_id)
    if label is None:
        return []

    bodies = _read_sample_bodies(path)
    if not bodies:
        return []

    body_idx, body = select_dominant_body(bodies, policy="most_tracked")
    seq = body_to_coco17_normalized(body, projection=projection)
    if seq.size == 0:
        return []

    windows = slice_windows(
        seq,
        window_size=window_size,
        stride=stride,
        pad_short=True,
    )
    session_id = str(uuid.uuid4())
    created_at = make_created_at()
    out: list[dict[str, Any]] = []

    for i, w in enumerate(windows):
        ts_start_ms = int(i * (stride or window_size) * (1000.0 / fps))
        ts_end_ms = ts_end_ms_from_window(window_size, fps, ts_start_ms)
        mean_conf = float(np.clip(np.mean(w[:, :, 2]), 0.0, 1.0))
        out.append({
            "id": make_window_id(),
            "device_id": device_id,
            "camera_id": camera_id,
            "session_id": session_id,
            "track_id": 1,
            "ts_start_ms": ts_start_ms,
            "ts_end_ms": ts_end_ms,
            "fps": fps,
            "window_size": window_size,
            "mean_pose_conf": mean_conf,
            "label": label,
            "label_source": "dataset",
            "created_at": created_at,
            "keypoints": w.tolist(),
            "source_body_id": body_idx,
        })
    return out


def _split_sample_ids(
    sample_ids: list[str],
    *,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[list[str], list[str], list[str]]:
    """Split unique sample IDs into train / val / test (rest = test)."""
    rng = np.random.default_rng(seed)
    unique = sorted(set(sample_ids))
    rng.shuffle(unique)
    n = len(unique)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    train = unique[:n_train]
    val = unique[n_train : n_train + n_val]
    test = unique[n_train + n_val :]
    return train, val, test


def build_dataset(
    source_dir: str | Path,
    out_dir: str | Path,
    *,
    projection: Literal["rgb", "depth", "3d"] = "rgb",
    window_size: int = 30,
    stride: int | None = None,
    seed: int = 42,
    device_id: str = "ntu-offline",
    camera_id: str = "ntu-cam",
    fps: float = 30.0,
    skip_missing: bool = True,
    export_samples_count: int = 0,
) -> dict[str, Any]:
    """
    Full pipeline: list NTU samples, build windows, 80/10/10 split by sample_id, write Parquet
    and metadata. Returns dataset_meta dict.
    """
    source_dir = Path(source_dir)
    out_dir = Path(out_dir)
    if stride is None:
        stride = window_size

    paths = list_ntu_samples(source_dir, skip_missing=skip_missing)
    if not paths:
        raise FileNotFoundError(f"No NTU samples found under {source_dir}")

    # sample_id -> list of window dicts (all windows from that sample)
    sample_to_windows: dict[str, list[dict[str, Any]]] = {}
    for p in paths:
        sid = sample_id_from_path(p)
        wins = build_windows_from_sample(
            p,
            projection=projection,
            window_size=window_size,
            stride=stride,
            device_id=device_id,
            camera_id=camera_id,
            fps=fps,
        )
        if wins:
            sample_to_windows.setdefault(sid, []).extend(wins)

    sample_ids = list(sample_to_windows.keys())
    train_ids, val_ids, test_ids = _split_sample_ids(sample_ids, seed=seed)

    def collect_windows(ids: list[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for sid in ids:
            rows.extend(sample_to_windows[sid])
        return rows

    train_rows = collect_windows(train_ids)
    val_rows = collect_windows(val_ids)
    test_rows = collect_windows(test_ids)

    # Write output
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    def write_parquet(rows: list[dict[str, Any]], name: str) -> None:
        if not rows:
            return
        table = pa.table({
            "id": [r["id"] for r in rows],
            "device_id": [r["device_id"] for r in rows],
            "camera_id": [r["camera_id"] for r in rows],
            "session_id": [r["session_id"] for r in rows],
            "track_id": [r["track_id"] for r in rows],
            "ts_start_ms": [r["ts_start_ms"] for r in rows],
            "ts_end_ms": [r["ts_end_ms"] for r in rows],
            "fps": [r["fps"] for r in rows],
            "window_size": [r["window_size"] for r in rows],
            "mean_pose_conf": [r["mean_pose_conf"] for r in rows],
            "label": [r["label"] for r in rows],
            "label_source": [r["label_source"] for r in rows],
            "created_at": [r["created_at"] for r in rows],
            "keypoints": [r["keypoints"] for r in rows],
            "source_body_id": [r.get("source_body_id") for r in rows],
        })
        pq.write_table(table, splits_dir / f"{name}.parquet")

    write_parquet(train_rows, "train")
    write_parquet(val_rows, "val")
    write_parquet(test_rows, "test")

    label_map = build_default_label_map()
    save_label_map(label_map, out_dir / "label_map.json")

    dataset_meta = {
        "source_dir": str(source_dir),
        "projection": projection,
        "window_size": window_size,
        "stride": stride,
        "fps": fps,
        "seed": seed,
        "num_samples": len(sample_ids),
        "num_train_windows": len(train_rows),
        "num_val_windows": len(val_rows),
        "num_test_windows": len(test_rows),
        "num_classes": label_map["num_classes"],
    }
    with open(out_dir / "dataset_meta.json", "w", encoding="utf-8") as f:
        json.dump(dataset_meta, f, indent=2)

    # Statistics for inspection and README "statistics" deliverable
    stats_dir = out_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    all_rows = train_rows + val_rows + test_rows
    class_counts = dict(Counter(r["label"] for r in all_rows))
    with open(stats_dir / "class_counts.json", "w", encoding="utf-8") as f:
        json.dump(class_counts, f, indent=2)
    conf_values = [r["mean_pose_conf"] for r in all_rows]
    hist, bin_edges = np.histogram(conf_values, bins=10, range=(0.0, 1.0))
    pose_conf_hist = {
        "bin_edges": bin_edges.tolist(),
        "counts": hist.tolist(),
        "min": float(min(conf_values)),
        "max": float(max(conf_values)),
        "mean": float(np.mean(conf_values)),
    }
    with open(stats_dir / "pose_conf_hist.json", "w", encoding="utf-8") as f:
        json.dump(pose_conf_hist, f, indent=2)

    if export_samples_count > 0:
        samples_dir = out_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        for i, row in enumerate(all_rows[:export_samples_count]):
            with open(samples_dir / f"window_{i:05d}.json", "w", encoding="utf-8") as f:
                json.dump(row, f, indent=2)

    return dataset_meta
