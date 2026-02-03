"""
Normalize NTU skeleton to 2D [0, 1] for Window contract.

Prefer RGB or depth projection for stable scale; 3D by scene range is unstable
across samples (camera distance varies). Use --projection rgb|depth|3d in CLI.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from .config import (
    DEPTH_HEIGHT,
    DEPTH_WIDTH,
    PROJECTION_3D,
    PROJECTION_DEPTH,
    PROJECTION_RGB,
    RGB_HEIGHT,
    RGB_WIDTH,
)
from .mapping import map_ntu_sequence_to_coco17


def normalize_xy_to_01(
    xy: np.ndarray,
    *,
    projection: Literal["rgb", "depth", "3d"],
    bounds: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """
    Normalize x,y coordinates to [0, 1].

    - rgb: divide by (RGB_WIDTH, RGB_HEIGHT). Use for color_xy / rgb_xy.
    - depth: divide by (DEPTH_WIDTH, DEPTH_HEIGHT). Use for depth_xy.
    - 3d: use bounds (x_min, y_min, x_max, y_max) and normalize to [0,1].
      Unstable across samples if camera distance varies; prefer rgb/depth.
    """
    xy = np.asarray(xy, dtype=np.float64)
    if projection == PROJECTION_RGB:
        xy = xy.copy()
        xy[..., 0] /= RGB_WIDTH
        xy[..., 1] /= RGB_HEIGHT
    elif projection == PROJECTION_DEPTH:
        xy = xy.copy()
        xy[..., 0] /= DEPTH_WIDTH
        xy[..., 1] /= DEPTH_HEIGHT
    elif projection == PROJECTION_3D:
        xy = xy.copy()
        if bounds is None:
            x_min, x_max = xy[..., 0].min(), xy[..., 0].max()
            y_min, y_max = xy[..., 1].min(), xy[..., 1].max()
        else:
            x_min, y_min, x_max, y_max = bounds
        span_x = max(x_max - x_min, 1e-9)
        span_y = max(y_max - y_min, 1e-9)
        xy[..., 0] = (xy[..., 0] - x_min) / span_x
        xy[..., 1] = (xy[..., 1] - y_min) / span_y
    else:
        raise ValueError(f"projection must be rgb|depth|3d, got {projection!r}")
    return np.clip(xy, 0.0, 1.0).astype(np.float64)


def body_to_coco17_normalized(
    body: dict[str, Any],
    *,
    projection: Literal["rgb", "depth", "3d"] = "rgb",
) -> np.ndarray:
    """
    Convert one body (from read_ntu_skeleton_txt_full or read_ntu_npy_full) to (N, 17, 3)
    with x,y in [0,1] and conf from tracking_state or default 1.0.

    Chooses xy source by projection: rgb → color_xy/rgb_xy, depth → depth_xy, 3d → skel x,y.
    When projection is rgb but color_xy/rgb_xy missing, falls back to depth then 3d.
    """
    from .mapping import NUM_NTU_JOINTS, map_ntu_frame_to_coco17

    skel = body["skel"]
    nframe = skel.shape[0]
    tracking_state = body.get("tracking_state")
    color_xy = body.get("color_xy")
    if color_xy is None:
        color_xy = body.get("rgb_xy")
    depth_xy = body.get("depth_xy")

    if projection == PROJECTION_RGB and color_xy is not None:
        xy_source = color_xy
        norm_proj = PROJECTION_RGB
    elif projection == PROJECTION_DEPTH and depth_xy is not None:
        xy_source = depth_xy
        norm_proj = PROJECTION_DEPTH
    elif projection == PROJECTION_3D:
        xy_source = skel[..., :2]
        norm_proj = PROJECTION_3D
    elif color_xy is not None:
        xy_source = color_xy
        norm_proj = PROJECTION_RGB
    elif depth_xy is not None:
        xy_source = depth_xy
        norm_proj = PROJECTION_DEPTH
    else:
        xy_source = skel[..., :2]
        norm_proj = PROJECTION_3D

    coco_xy = np.zeros((nframe, 17, 2), dtype=np.float64)
    coco_conf = np.zeros((nframe, 17), dtype=np.float64)
    for t in range(nframe):
        frame_xy = xy_source[t]
        ntu_frame = np.zeros((NUM_NTU_JOINTS, 3), dtype=np.float64)
        ntu_frame[:, 0] = frame_xy[:, 0]
        ntu_frame[:, 1] = frame_xy[:, 1]
        ts = tracking_state[t] if tracking_state is not None else None
        out = map_ntu_frame_to_coco17(ntu_frame, ntu_tracking_state=ts)
        coco_xy[t] = out[:, :2]
        coco_conf[t] = out[:, 2]

    xy_norm = normalize_xy_to_01(coco_xy, projection=norm_proj)
    result = np.zeros((nframe, 17, 3), dtype=np.float64)
    result[:, :, :2] = xy_norm
    result[:, :, 2] = np.clip(coco_conf, 0.0, 1.0)
    return result
