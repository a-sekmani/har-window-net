"""Custom10 clip discovery and reading (JSON/NPY/.skeleton)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .labels import parse_label_from_folder_name


@dataclass(frozen=True)
class ClipRef:
    """Reference to a single clip file: path, clip_id (for session), label_id, label_name."""

    path: Path
    clip_id: str
    label_id: str
    label_name: str


# Extensions accepted as clip files: JSON/NPY (keypoints) or NTU .skeleton text
CLIP_EXTENSIONS = (".json", ".npy", ".skeleton")


def list_custom10_clips(source_dir: str | Path) -> list[ClipRef]:
    """
    Scan source_dir for label subfolders (A001_* or 001_*) and collect all .json, .npy, and .skeleton clip files.
    Returns list of ClipRef. Raises if no valid label folders found.
    """
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"Custom10 source is not a directory: {source_dir}"
        )
    refs: list[ClipRef] = []
    for child in sorted(source_dir.iterdir()):
        if not child.is_dir():
            continue
        parsed = parse_label_from_folder_name(child.name)
        if parsed is None:
            continue
        label_id, label_name = parsed
        for f in sorted(child.iterdir()):
            if not f.is_file():
                continue
            if f.suffix.lower() not in CLIP_EXTENSIONS:
                continue
            clip_id = f"{child.name}/{f.stem}"
            refs.append(
                ClipRef(
                    path=f,
                    clip_id=clip_id,
                    label_id=label_id,
                    label_name=label_name,
                )
            )
    if not refs:
        _raise_no_clips_found(source_dir)
    return refs


def _raise_no_clips_found(source_dir: Path) -> None:
    """Build a clear diagnostic when no Custom10 clips are found."""
    resolved = source_dir.resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Custom10 source path does not exist: {resolved}. "
            "Use an absolute path or a path relative to the current working directory."
        )
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"Custom10 source is not a directory: {resolved}"
        )
    subdirs = sorted(c for c in resolved.iterdir() if c.is_dir())
    # Check which subdirs match the label pattern and how many clips each has
    details: list[str] = []
    for d in subdirs:
        parsed = parse_label_from_folder_name(d.name)
        files = [f for f in d.iterdir() if f.is_file() and f.suffix.lower() in CLIP_EXTENSIONS]
        if parsed is None:
            details.append(f"  - {d.name}/ (name not matched; needs A + 1–3 digits + _ + name; has {len(files)} .json/.npy/.skeleton)")
        elif not files:
            details.append(f"  - {d.name}/ (matched but no .json, .npy or .skeleton files)")
        else:
            details.append(f"  - {d.name}/ (matched, {len(files)} clip(s))")
    sublist = "\n".join(details) if details else "  (no subfolders)"
    raise FileNotFoundError(
        f"No Custom10 clips found under {resolved}\n"
        f"Subfolders found:\n{sublist}\n"
        "Required: subfolders named like A001_WALKING, A1_drink_water, or A43_falling (A + 1–3 digits + _ + name) "
        "each containing at least one .json, .npy or .skeleton clip file."
    )


def read_clip(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Read a single clip file (JSON or NPY). Returns (keypoints (T,K,3), meta).
    meta contains: fps, format (coco17_norm or coco17_pixel), img_w, img_h when available.
    Raises with clear message on unsupported shape or missing fields.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Clip file not found: {path}")

    meta: dict[str, Any] = {}

    if path.suffix.lower() == ".json":
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "keypoints" not in data:
            raise ValueError(
                f"JSON clip must contain 'keypoints'; keys present: {list(data.keys())}"
            )
        kp = np.asarray(data["keypoints"], dtype=np.float64)
        meta["fps"] = data.get("fps")
        meta["format"] = data.get("format", "coco17_norm")
        meta["img_w"] = data.get("img_w")
        meta["img_h"] = data.get("img_h")
    else:
        if path.suffix.lower() != ".npy":
            raise ValueError(
                f"Unsupported clip extension: {path.suffix}; use .json or .npy"
            )
        raw = np.load(path, allow_pickle=True)
        if isinstance(raw, np.ndarray):
            if raw.ndim == 3:
                kp = np.asarray(raw, dtype=np.float64)
                meta["format"] = "coco17_norm"
            elif raw.size == 1 and isinstance(raw.item(), dict):
                data = raw.item()
                if "keypoints" not in data:
                    raise ValueError(
                        "NPY dict clip must contain 'keypoints'; "
                        f"keys present: {list(data.keys())}"
                    )
                kp = np.asarray(data["keypoints"], dtype=np.float64)
                meta["fps"] = data.get("fps")
                meta["format"] = data.get("format", "coco17_norm")
                meta["img_w"] = data.get("img_w")
                meta["img_h"] = data.get("img_h")
            else:
                raise ValueError(
                    "NPY clip must be (T,K,3) array or dict with 'keypoints'; "
                    f"got array shape {getattr(raw, 'shape', None)}"
                )
        else:
            raise ValueError(
                f"Unsupported NPY content type: {type(raw)}"
            )

    if kp.ndim != 3 or kp.shape[2] != 3:
        raise ValueError(
            f"Unsupported keypoints shape: expected (T,17,3) or (T,25,3), got {kp.shape}"
        )
    k = kp.shape[1]
    if k not in (17, 25):
        raise ValueError(
            f"Unsupported keypoint count K={k}; only 17 (COCO-17) or 25 (NTU, will be mapped to 17) are supported."
        )
    return kp, meta
