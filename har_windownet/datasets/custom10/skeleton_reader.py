"""
Read NTU-format .skeleton text files for Custom10.

Reuses NTU reader: each body has skel (T,25,3), depth_xy (T,25,2), color_xy (T,25,2),
tracking_state (T,25) with 0=NotTracked, 1=Inferred, 2=Tracked.
Conversion to COCO-17 and normalization (rgb/depth/3d) is done in the builder via
ntu.preprocess.body_to_coco17_normalized.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Re-export NTU skeleton text reader and body selection for Custom10 .skeleton clips
from har_windownet.datasets.ntu.reader import (
    read_ntu_skeleton_txt_full,
    select_dominant_body,
)

__all__ = ["read_skeleton_txt_full", "select_dominant_body"]


def read_skeleton_txt_full(path: str | Path) -> list[dict[str, Any]]:
    """
    Read .skeleton file (NTU text format) and return all bodies.

    Returns
    -------
    list[dict]
        Each body: skel (T, 25, 3), depth_xy (T, 25, 2), color_xy (T, 25, 2),
        tracking_state (T, 25) int: 0=NotTracked, 1=Inferred, 2=Tracked.
    """
    return read_ntu_skeleton_txt_full(path)
