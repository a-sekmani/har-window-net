"""
Edge17 skeleton JSONL reader.

Reads .skeleton.jsonl files from Edge pose estimation pipeline.
Each file contains:
- Line 1: meta dict with action_id, fps, frame_count, image_w, image_h, skeleton_format, coords
- Lines 2+: frame dicts with frame_index, ts_unix_ms, persons list

Keypoints are already COCO-17 format and normalized (0..1).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

NUM_KEYPOINTS = 17
NUM_COORDS = 3  # x, y, conf


def read_jsonl_file(path: str | Path) -> dict[str, Any]:
    """
    Read a .skeleton.jsonl file and return meta + frames.

    Returns
    -------
    dict with:
        meta: dict with action_id, fps, frame_count, image_w, image_h, etc.
        frames: list of frame dicts (only those with type="frame")
    """
    path = Path(path)
    meta: dict[str, Any] | None = None
    frames: list[dict[str, Any]] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "meta":
                meta = obj
            elif obj.get("type") == "frame":
                frames.append(obj)

    if meta is None:
        raise ValueError(f"No meta line found in {path}")

    return {"meta": meta, "frames": frames}


def extract_keypoints_sequence(
    frames: list[dict[str, Any]],
    track_id: int = 1,
) -> np.ndarray:
    """
    Extract keypoints sequence for a specific track_id from frames.

    Parameters
    ----------
    frames : list of frame dicts
    track_id : int, default 1
        Which person track to extract

    Returns
    -------
    ndarray shape (T, 17, 3) float64
        T = number of frames with valid keypoints for this track_id.
        Each keypoint is (x, y, conf), all in [0, 1].
    """
    keypoints_list: list[np.ndarray] = []

    for frame in frames:
        persons = frame.get("persons", [])
        kp = None
        for person in persons:
            if person.get("track_id") == track_id:
                kp = person.get("keypoints")
                break
        if kp is None:
            continue
        kp_arr = np.array(kp, dtype=np.float64)
        if kp_arr.shape != (NUM_KEYPOINTS, NUM_COORDS):
            continue
        keypoints_list.append(kp_arr)

    if not keypoints_list:
        return np.zeros((0, NUM_KEYPOINTS, NUM_COORDS), dtype=np.float64)

    seq = np.stack(keypoints_list, axis=0)
    return seq


def sanitize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    Sanitize keypoints array: replace NaN/Inf with 0, clip x/y/conf to [0, 1].

    Parameters
    ----------
    keypoints : ndarray shape (..., 3)
        Last dimension is (x, y, conf)

    Returns
    -------
    ndarray same shape, sanitized
    """
    kp = keypoints.copy()
    kp = np.nan_to_num(kp, nan=0.0, posinf=1.0, neginf=0.0)
    kp = np.clip(kp, 0.0, 1.0)
    return kp


def read_clip(path: str | Path, track_id: int = 1) -> dict[str, Any]:
    """
    Read a .skeleton.jsonl file and return processed clip data.

    Returns
    -------
    dict with:
        meta: original meta dict
        keypoints: ndarray (T, 17, 3) sanitized
        label: str (action_id from meta, e.g. "A001")
        fps: float
        frame_count: int (original frame count from meta)
        clip_id: str (filename stem)
    """
    path = Path(path)
    data = read_jsonl_file(path)
    meta = data["meta"]
    frames = data["frames"]

    keypoints = extract_keypoints_sequence(frames, track_id=track_id)
    keypoints = sanitize_keypoints(keypoints)

    return {
        "meta": meta,
        "keypoints": keypoints,
        "label": meta.get("action_id", ""),
        "fps": float(meta.get("fps", 30.0)),
        "frame_count": int(meta.get("frame_count", len(frames))),
        "clip_id": path.stem,
    }


def list_edge17_files(source_dir: str | Path) -> list[Path]:
    """
    List all .skeleton.jsonl files under source_dir.

    Returns
    -------
    list of Path, sorted by name
    """
    source_dir = Path(source_dir)
    files = list(source_dir.rglob("*.skeleton.jsonl"))
    return sorted(files)
