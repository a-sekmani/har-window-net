"""
NTU RGB+D 120 skeleton reader.

Supports:
- .npy format from FesianXu/NTU_RGBD120_Parser_python: dict with skel_body0, rgb_body0, depth_body0.
- .skeleton format: text files from official NTU (25 joints: x,y,z, depthX/Y, colorX/Y, trackingState).

Multi-person: use select_dominant_body(bodies, policy="most_tracked"|"closest_z") to pick one body.
Store source_body_id in window/dataset_meta for traceability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

from .mapping import NUM_NTU_JOINTS

# Kinect V2: trackingState 0=NotTracked, 1=Inferred, 2=Tracked
TRACKED = 2
INFERRED = 1
NOT_TRACKED = 0

# Path to bundled missing-skeletons list (NTU RGB+D 120)
_DATA_DIR = Path(__file__).resolve().parent / "data"
_MISSING_FILE = _DATA_DIR / "ntu120_missing_skeletons.txt"

_missing_set: set[str] | None = None


def load_missing_skeletons_set() -> set[str]:
    """Load set of sample IDs (e.g. S001C002P003R002A013) with missing/incomplete skeleton."""
    global _missing_set
    if _missing_set is not None:
        return _missing_set
    if not _MISSING_FILE.exists():
        _missing_set = set()
        return _missing_set
    with open(_MISSING_FILE, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    _missing_set = set(lines)
    return _missing_set


def is_missing_sample(sample_id: str) -> bool:
    """Return True if this sample is in the official NTU 120 missing-skeletons list."""
    return sample_id in load_missing_skeletons_set()


def sample_id_from_path(path: str | Path) -> str:
    """Extract NTU sample ID from file path (e.g. S001C002P003R002A013)."""
    path = Path(path)
    name = path.stem
    # .skeleton.npy -> stem is S001C002P003R002A013.skeleton
    if name.endswith(".skeleton"):
        name = name.replace(".skeleton", "")
    return name


def read_ntu_npy_full(path: str | Path) -> list[dict[str, Any]]:
    """
    Read .npy sample and return all bodies with skel, rgb_xy, depth_xy (when present).

    FesianXu parser does not store tracking_state; use None (caller will use default conf).
    Each body dict: skel (nframe, 25, 3), rgb_xy (nframe, 25, 2) or None, depth_xy (nframe, 25, 2) or None,
    tracking_state None.
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray):
        if data.size != 1:
            raise ValueError(f"Expected dict in {path}, got array of size {data.size}")
        item = data.item()
    else:
        item = data
    if not isinstance(item, dict):
        raise ValueError(f"Expected dict in {path}, got {type(item)}")
    out: list[dict[str, Any]] = []
    b = 0
    while True:
        skel_key = f"skel_body{b}"
        if skel_key not in item:
            break
        skel = np.asarray(item[skel_key], dtype=np.float64)
        if skel.ndim != 3 or skel.shape[1] != NUM_NTU_JOINTS:
            break
        rgb_xy = None
        depth_xy = None
        if f"rgb_body{b}" in item:
            rgb_xy = np.asarray(item[f"rgb_body{b}"], dtype=np.float64)
        if f"depth_body{b}" in item:
            depth_xy = np.asarray(item[f"depth_body{b}"], dtype=np.float64)
        out.append({
            "skel": skel,
            "rgb_xy": rgb_xy,
            "depth_xy": depth_xy,
            "color_xy": rgb_xy,
            "tracking_state": None,
        })
        b += 1
    return out


def read_ntu_npy(path: str | Path, body_index: int = 0) -> np.ndarray:
    """
    Read one sample from .npy (FesianXu parser output).

    Expects dict with key skel_body0 (or skel_body1, ...): array (nframe, 25, 3) x,y,z.
    Returns (nframe, 25, 3) float64 for the chosen body.
    """
    bodies = read_ntu_npy_full(path)
    if not bodies:
        raise KeyError(f"No skel_body{body_index} in {path}")
    if body_index >= len(bodies):
        body_index = 0
    return bodies[body_index]["skel"]


def read_ntu_skeleton_txt_full(path: str | Path) -> list[dict[str, Any]]:
    """
    Read .skeleton file and return all bodies with skel, depth_xy, color_xy, tracking_state.

    Each body dict: skel (nframe, 25, 3), depth_xy (nframe, 25, 2), color_xy (nframe, 25, 2),
    tracking_state (nframe, 25) int: 0=NotTracked, 1=Inferred, 2=Tracked.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        tokens = f.read().split()
    it = iter(tokens)
    framecount = int(next(it))
    # Collect per-body per-frame: list of list of (skel, depth_xy, color_xy, tracking)
    bodies_frames: list[list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = []
    for _ in range(framecount):
        bodycount = int(next(it))
        for b in range(bodycount):
            next(it)  # bodyID
            for _ in range(6):
                next(it)
            next(it)
            next(it)  # leanX, leanY
            next(it)  # body trackingState
            joint_count = int(next(it))
            skel_frame = []
            depth_frame = []
            color_frame = []
            track_frame = []
            for _ in range(joint_count):
                x, y, z = float(next(it)), float(next(it)), float(next(it))
                dx, dy = float(next(it)), float(next(it))
                cx, cy = float(next(it)), float(next(it))
                for _ in range(4):
                    next(it)  # quat
                ts = int(next(it))
                skel_frame.append([x, y, z])
                depth_frame.append([dx, dy])
                color_frame.append([cx, cy])
                track_frame.append(ts)
            arr_skel = np.array(skel_frame, dtype=np.float64)
            arr_depth = np.array(depth_frame, dtype=np.float64)
            arr_color = np.array(color_frame, dtype=np.float64)
            arr_track = np.array(track_frame, dtype=np.int32)
            while len(bodies_frames) <= b:
                bodies_frames.append([])
            bodies_frames[b].append((arr_skel, arr_depth, arr_color, arr_track))
    out: list[dict[str, Any]] = []
    for b, frames in enumerate(bodies_frames):
        if not frames:
            continue
        skel = np.stack([f[0] for f in frames], axis=0)
        depth_xy = np.stack([f[1] for f in frames], axis=0)
        color_xy = np.stack([f[2] for f in frames], axis=0)
        tracking_state = np.stack([f[3] for f in frames], axis=0)
        out.append({
            "skel": skel,
            "depth_xy": depth_xy,
            "color_xy": color_xy,
            "tracking_state": tracking_state,
        })
    return out


def read_ntu_skeleton_txt(path: str | Path, body_index: int = 0) -> np.ndarray:
    """
    Read one sample from .skeleton text file (official NTU format); returns skel only (nframe, 25, 3).
    For full data (depth, color, tracking_state) use read_ntu_skeleton_txt_full + select_dominant_body.
    """
    bodies = read_ntu_skeleton_txt_full(path)
    if not bodies:
        return np.zeros((0, NUM_NTU_JOINTS, 3), dtype=np.float64)
    if body_index >= len(bodies):
        body_index = 0
    return bodies[body_index]["skel"]


def select_dominant_body(
    bodies: list[dict[str, Any]],
    policy: Literal["most_tracked", "closest_z"] = "most_tracked",
) -> tuple[int, dict[str, Any]]:
    """
    Choose one body from multi-person sample for consistent pipeline.

    - most_tracked: body with highest total count of joints with tracking_state==2 (Tracked).
      If no tracking_state (e.g. .npy), falls back to body 0.
    - closest_z: body with smallest mean spine-base z (joint 0) over frames (closest to camera).
    """
    if not bodies:
        raise ValueError("bodies list is empty")
    if len(bodies) == 1:
        return 0, bodies[0]
    if policy == "most_tracked":
        ts = bodies[0].get("tracking_state")
        if ts is None:
            return 0, bodies[0]
        best_idx = 0
        best_count = -1
        for i, bod in enumerate(bodies):
            t = bod.get("tracking_state")
            if t is None:
                count = 0
            else:
                count = int(np.sum(t == TRACKED))
            if count > best_count:
                best_count = count
                best_idx = i
        return best_idx, bodies[best_idx]
    # closest_z: smallest mean z of joint 0 (SpineBase)
    best_idx = 0
    best_z = float("inf")
    for i, bod in enumerate(bodies):
        skel = bod["skel"]
        z_mean = np.mean(skel[:, 0, 2])
        if z_mean < best_z:
            best_z = z_mean
            best_idx = i
    return best_idx, bodies[best_idx]


def read_ntu_sample(
    path: str | Path,
    *,
    body_index: int = 0,
    prefer_npy: bool = True,
) -> np.ndarray:
    """
    Read one NTU sample; auto-detect .npy vs .skeleton.

    Returns
    -------
    ndarray
        (nframe, 25, 3) float64, x,y,z per joint (mm or raw; normalization in preprocess).
    """
    path = Path(path)
    if prefer_npy:
        npy_path = path
        if npy_path.suffix.lower() != ".npy":
            npy_path = path.parent / f"{path.stem}.skeleton.npy"
        if npy_path.exists():
            return read_ntu_npy(npy_path, body_index=body_index)
    if path.suffix.lower() == ".skeleton" or ".skeleton" in path.name:
        return read_ntu_skeleton_txt(path, body_index=body_index)
    if path.suffix.lower() == ".npy":
        return read_ntu_npy(path, body_index=body_index)
    raise ValueError(f"Unknown format for {path}; use .npy or .skeleton")


def list_ntu_samples(source_dir: str | Path, *, skip_missing: bool = True) -> list[Path]:
    """
    List all NTU sample paths under source_dir.

    Looks for *.skeleton.npy first, then *.skeleton. Skips samples in missing-skeletons list
    when skip_missing is True.
    """
    source_dir = Path(source_dir)
    missing = load_missing_skeletons_set() if skip_missing else set()
    paths: list[Path] = []
    seen_ids: set[str] = set()
    # Prefer .npy
    for p in source_dir.rglob("*.skeleton.npy"):
        sid = sample_id_from_path(p)
        if sid in missing:
            continue
        if sid not in seen_ids:
            seen_ids.add(sid)
            paths.append(p)
    for p in source_dir.rglob("*.skeleton"):
        if p.suffix.lower() != ".skeleton":
            continue
        sid = sample_id_from_path(p)
        if sid in missing:
            continue
        if sid not in seen_ids:
            seen_ids.add(sid)
            paths.append(p)
    return sorted(paths)
