"""
Edge17 -> Window dataset builder.

Reads .skeleton.jsonl files, builds windows, splits by clip_id (80/10/10),
writes Parquet + metadata.
"""

from __future__ import annotations

import json
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from har_windownet.contracts.window import (
    make_created_at,
    make_window_id,
    ts_end_ms_from_window,
)
from har_windownet.datasets.edge17.labels import get_label
from har_windownet.datasets.edge17.reader import list_edge17_files, read_clip
from har_windownet.datasets.edge17.windowing import slice_windows


def build_windows_from_clip(
    clip: dict[str, Any],
    *,
    window_size: int = 30,
    stride: int | None = None,
    device_id: str = "edge17",
    camera_id: str = "default",
) -> list[dict[str, Any]]:
    """
    Build window dicts from a clip.

    Parameters
    ----------
    clip : dict
        Output from read_clip() with keypoints, label, fps, clip_id, meta
    window_size : int
    stride : int, optional
    device_id : str
    camera_id : str

    Returns
    -------
    list of window dicts ready for Parquet
    """
    keypoints = clip["keypoints"]
    label = clip["label"]
    fps = clip["fps"]
    clip_id = clip["clip_id"]
    meta = clip["meta"]

    if keypoints.shape[0] == 0:
        return []

    windows = slice_windows(keypoints, window_size=window_size, stride=stride, pad_short=True)
    session_id = str(uuid.uuid4())
    created_at = make_created_at()
    effective_stride = stride if stride is not None else window_size

    out: list[dict[str, Any]] = []
    for i, w in enumerate(windows):
        ts_start_ms = int(i * effective_stride * (1000.0 / fps))
        ts_end_ms = ts_end_ms_from_window(window_size, fps, ts_start_ms)
        mean_conf = float(np.clip(np.mean(w[:, :, 2]), 0.0, 1.0))
        out.append({
            "id": make_window_id(),
            "device_id": meta.get("device_id", device_id),
            "camera_id": meta.get("camera_id", camera_id),
            "session_id": meta.get("session_id", session_id),
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
            "source_clip_id": clip_id,
        })
    return out


def _split_clip_ids(
    clip_ids: list[str],
    *,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[list[str], list[str], list[str]]:
    """Split unique clip IDs into train / val / test."""
    rng = np.random.default_rng(seed)
    unique = sorted(set(clip_ids))
    rng.shuffle(unique)
    n = len(unique)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = unique[:n_train]
    val = unique[n_train : n_train + n_val]
    test = unique[n_train + n_val :]
    return train, val, test


def build_label_map(labels: list[str]) -> dict[str, Any]:
    """
    Build label_map from observed labels.

    Returns dict with label_to_id, id_to_name, label_names, num_classes.
    """
    unique_labels = sorted(set(labels))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    id_to_name = {str(i): label for i, label in enumerate(unique_labels)}
    label_names = {label: label for label in unique_labels}
    return {
        "label_to_id": label_to_id,
        "id_to_name": id_to_name,
        "label_names": label_names,
        "num_classes": len(unique_labels),
    }


def build_dataset_edge17(
    source_dir: str | Path,
    out_dir: str | Path,
    *,
    window_size: int = 30,
    stride: int | None = None,
    seed: int = 42,
    device_id: str = "edge17",
    camera_id: str = "default",
    export_samples_count: int = 0,
) -> dict[str, Any]:
    """
    Full pipeline: list Edge17 files, build windows, 80/10/10 split by clip_id,
    write Parquet and metadata.

    Parameters
    ----------
    source_dir : path
        Directory containing .skeleton.jsonl files
    out_dir : path
        Output directory
    window_size : int
        Frames per window (default 30)
    stride : int, optional
        Stride between windows (default = window_size)
    seed : int
        Random seed for split
    device_id : str
        Default device_id if not in meta
    camera_id : str
        Default camera_id if not in meta
    export_samples_count : int
        Number of sample windows to export as JSON

    Returns
    -------
    dict
        dataset_meta
    """
    source_dir = Path(source_dir)
    out_dir = Path(out_dir)
    if stride is None:
        stride = window_size

    files = list_edge17_files(source_dir)
    if not files:
        raise FileNotFoundError(f"No .skeleton.jsonl files found under {source_dir}")

    clip_to_windows: dict[str, list[dict[str, Any]]] = {}
    all_labels: list[str] = []
    fps_values: list[float] = []

    for fpath in files:
        try:
            clip = read_clip(fpath)
        except Exception:
            continue
        label = get_label(clip["meta"], fpath)
        if label == "UNKNOWN":
            continue
        clip["label"] = label
        all_labels.append(label)
        fps_values.append(clip["fps"])

        wins = build_windows_from_clip(
            clip,
            window_size=window_size,
            stride=stride,
            device_id=device_id,
            camera_id=camera_id,
        )
        if wins:
            clip_id = clip["clip_id"]
            clip_to_windows.setdefault(clip_id, []).extend(wins)

    if not clip_to_windows:
        raise ValueError(f"No valid clips with windows found in {source_dir}")

    clip_ids = list(clip_to_windows.keys())
    train_ids, val_ids, test_ids = _split_clip_ids(clip_ids, seed=seed)

    def collect_windows(ids: list[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for cid in ids:
            rows.extend(clip_to_windows.get(cid, []))
        return rows

    train_rows = collect_windows(train_ids)
    val_rows = collect_windows(val_ids)
    test_rows = collect_windows(test_ids)

    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    def write_parquet(rows: list[dict[str, Any]], name: str) -> None:
        if not rows:
            table = pa.table({
                "id": pa.array([], type=pa.string()),
                "device_id": pa.array([], type=pa.string()),
                "camera_id": pa.array([], type=pa.string()),
                "session_id": pa.array([], type=pa.string()),
                "track_id": pa.array([], type=pa.int64()),
                "ts_start_ms": pa.array([], type=pa.int64()),
                "ts_end_ms": pa.array([], type=pa.int64()),
                "fps": pa.array([], type=pa.float64()),
                "window_size": pa.array([], type=pa.int64()),
                "mean_pose_conf": pa.array([], type=pa.float64()),
                "label": pa.array([], type=pa.string()),
                "label_source": pa.array([], type=pa.string()),
                "created_at": pa.array([], type=pa.string()),
                "keypoints": pa.array([], type=pa.list_(pa.list_(pa.list_(pa.float64())))),
                "source_clip_id": pa.array([], type=pa.string()),
            })
        else:
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
                "source_clip_id": [r.get("source_clip_id", "") for r in rows],
            })
        pq.write_table(table, splits_dir / f"{name}.parquet")

    write_parquet(train_rows, "train")
    write_parquet(val_rows, "val")
    write_parquet(test_rows, "test")

    all_rows = train_rows + val_rows + test_rows
    observed_labels = [r["label"] for r in all_rows]
    label_map = build_label_map(observed_labels)
    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    avg_fps = float(np.mean(fps_values)) if fps_values else 30.0
    dataset_meta = {
        "source_dir": str(source_dir),
        "dataset_type": "edge17",
        "window_size": window_size,
        "stride": stride,
        "fps": avg_fps,
        "seed": seed,
        "keypoint_order": "coco17",
        "coords": "normalized",
        "num_clips": len(clip_ids),
        "num_train_windows": len(train_rows),
        "num_val_windows": len(val_rows),
        "num_test_windows": len(test_rows),
        "num_classes": label_map["num_classes"],
    }
    with open(out_dir / "dataset_meta.json", "w", encoding="utf-8") as f:
        json.dump(dataset_meta, f, indent=2)

    stats_dir = out_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    class_counts = dict(Counter(r["label"] for r in all_rows))
    with open(stats_dir / "class_counts.json", "w", encoding="utf-8") as f:
        json.dump(class_counts, f, indent=2)

    conf_values = [r["mean_pose_conf"] for r in all_rows]
    if conf_values:
        hist, bin_edges = np.histogram(conf_values, bins=10, range=(0.0, 1.0))
        pose_conf_hist = {
            "bin_edges": bin_edges.tolist(),
            "counts": hist.tolist(),
            "min": float(min(conf_values)),
            "max": float(max(conf_values)),
            "mean": float(np.mean(conf_values)),
        }
    else:
        pose_conf_hist = {
            "bin_edges": [],
            "counts": [],
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
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
