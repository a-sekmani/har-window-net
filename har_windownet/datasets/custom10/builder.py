"""Custom10 dataset builder: clips -> windows, 80/10/10 split by clip_id, write Parquet + metadata."""

from __future__ import annotations

import json
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from har_windownet.contracts.window import (
    make_created_at,
    make_window_id,
    ts_end_ms_from_window,
)
from har_windownet.datasets.custom10.config import CAMERA_ID, DEVICE_ID, FPS
from har_windownet.datasets.custom10.labels import (
    build_label_map_from_refs,
    save_label_map,
)
from har_windownet.datasets.custom10.preprocess import normalize_keypoints
from har_windownet.datasets.custom10.reader import list_custom10_clips, read_clip
from har_windownet.datasets.custom10.skeleton_reader import (
    read_skeleton_txt_full,
    select_dominant_body,
)
from har_windownet.datasets.custom10.windowing import slice_windows
from har_windownet.datasets.ntu.preprocess import body_to_coco17_normalized

# UUID5 namespace for stable session_id per clip_id
CUSTOM10_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://har-windownet/custom10")


def _sanitize_keypoints(w: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with 0 and clip to [0, 1] so validate_dataset and training see valid values."""
    out = np.asarray(w, dtype=np.float64)
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(out, 0.0, 1.0)


def _session_id_for_clip(clip_id: str) -> str:
    """Stable session_id per clip (UUID5 from clip_id)."""
    return str(uuid.uuid5(CUSTOM10_NAMESPACE, clip_id))


def build_windows_for_clip(
    clip_ref: Any,
    *,
    window_size: int = 30,
    stride: int | None = None,
    fps: float = 30.0,
    img_w: int = 1920,
    img_h: int = 1080,
    projection: Literal["rgb", "depth", "3d"] = "rgb",
) -> list[dict[str, Any]]:
    """
    Read one clip (JSON/NPY or .skeleton), normalize to COCO-17 [0,1], slice into windows,
    return list of window dicts (storage-ready).
    For .skeleton uses NTU reader + body_to_coco17_normalized(projection).
    """
    if stride is None:
        stride = window_size

    path = Path(clip_ref.path)
    source_body_id: int | None = None

    if path.suffix.lower() == ".skeleton":
        bodies = read_skeleton_txt_full(path)
        if not bodies:
            return []
        body_idx, body = select_dominant_body(bodies, policy="most_tracked")
        source_body_id = body_idx
        seq = body_to_coco17_normalized(body, projection=projection)
        # fps from NTU default
    else:
        kp, meta = read_clip(path)
        seq = normalize_keypoints(kp, meta, img_w, img_h)
        clip_fps = meta.get("fps")
        if clip_fps is not None:
            fps = float(clip_fps)

    windows = slice_windows(
        seq,
        window_size=window_size,
        stride=stride,
        pad_short=True,
    )
    session_id = _session_id_for_clip(clip_ref.clip_id)
    created_at = make_created_at()
    out: list[dict[str, Any]] = []

    for i, w in enumerate(windows):
        w_clean = _sanitize_keypoints(w)
        ts_start_ms = int(i * stride * (1000.0 / fps))
        ts_end_ms = ts_end_ms_from_window(window_size, fps, ts_start_ms)
        mean_conf = float(np.clip(np.mean(w_clean[:, :, 2]), 0.0, 1.0))
        out.append({
            "id": make_window_id(),
            "device_id": DEVICE_ID,
            "camera_id": CAMERA_ID,
            "session_id": session_id,
            "track_id": 1,
            "ts_start_ms": ts_start_ms,
            "ts_end_ms": ts_end_ms,
            "fps": fps,
            "window_size": window_size,
            "mean_pose_conf": mean_conf,
            "label": clip_ref.label_id,
            "label_source": "dataset",
            "created_at": created_at,
            "keypoints": w_clean.tolist(),
            "source_body_id": source_body_id,
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
    n_test = n - n_train - n_val
    train = unique[:n_train]
    val = unique[n_train : n_train + n_val]
    test = unique[n_train + n_val :]
    return train, val, test


def build_dataset_custom10(
    source_dir: str | Path,
    out_dir: str | Path,
    *,
    window_size: int = 30,
    stride: int | None = None,
    fps: float = FPS,
    img_w: int = 1920,
    img_h: int = 1080,
    projection: Literal["rgb", "depth", "3d"] = "rgb",
    seed: int = 42,
    export_samples_count: int = 0,
) -> dict[str, Any]:
    """
    Full pipeline: list Custom10 clips, build windows, 80/10/10 split by clip_id,
    write Parquet, label_map, dataset_meta, stats, optional samples. Returns dataset_meta.
    For .skeleton clips uses --projection (rgb|depth|3d) for 2D normalization.
    """
    source_dir = Path(source_dir)
    out_dir = Path(out_dir)
    if stride is None:
        stride = window_size

    refs = list_custom10_clips(source_dir)
    clip_to_windows: dict[str, list[dict[str, Any]]] = {}
    for ref in refs:
        wins = build_windows_for_clip(
            ref,
            window_size=window_size,
            stride=stride,
            fps=fps,
            img_w=img_w,
            img_h=img_h,
            projection=projection,
        )
        if wins:
            clip_to_windows[ref.clip_id] = wins

    clip_ids = list(clip_to_windows.keys())
    train_ids, val_ids, test_ids = _split_clip_ids(clip_ids, seed=seed)

    def collect(ids: list[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for cid in ids:
            rows.extend(clip_to_windows[cid])
        return rows

    train_rows = collect(train_ids)
    val_rows = collect(val_ids)
    test_rows = collect(test_ids)

    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    def write_parquet(rows: list[dict[str, Any]], name: str) -> None:
        if not rows:
            # Empty table with same schema so val/test files always exist
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
                "source_body_id": pa.array([], type=pa.int64()),
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
            "source_body_id": [r.get("source_body_id") for r in rows],
        })
        pq.write_table(table, splits_dir / f"{name}.parquet")

    write_parquet(train_rows, "train")
    write_parquet(val_rows, "val")
    write_parquet(test_rows, "test")

    label_map = build_label_map_from_refs(refs)
    save_label_map(label_map, out_dir / "label_map.json")

    created_at = make_created_at()
    dataset_meta = {
        "source_dir": str(source_dir),
        "adapter": "custom10",
        "window_size": window_size,
        "stride": stride,
        "fps": fps,
        "img_w": img_w,
        "img_h": img_h,
        "projection": projection,
        "seed": seed,
        "num_clips": len(clip_ids),
        "num_train_windows": len(train_rows),
        "num_val_windows": len(val_rows),
        "num_test_windows": len(test_rows),
        "num_classes": label_map["num_classes"],
        "created_at": created_at,
    }
    with open(out_dir / "dataset_meta.json", "w", encoding="utf-8") as f:
        json.dump(dataset_meta, f, indent=2)

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
