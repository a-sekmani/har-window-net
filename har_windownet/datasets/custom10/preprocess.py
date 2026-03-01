"""Normalize Custom10 keypoints to (T, 17, 3) in [0, 1]; map 25 -> 17 when needed."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import FORMAT_COCO17_NORM, FORMAT_COCO17_PIXEL

NUM_COCO17 = 17
NUM_NTU25 = 25


def normalize_keypoints(
    kps: np.ndarray,
    meta: dict[str, Any],
    img_w: int | float,
    img_h: int | float,
) -> np.ndarray:
    """
    Ensure keypoints (T, K, 3) with K in (17, 25) become (T, 17, 3) in [0, 1].
    - If format is *_norm: clip and validate range; if K=25 map to COCO-17.
    - If format is *_pixel: divide x,y by img_w/img_h (from meta or args), then clip; map 25->17 if needed.
    Raises on unsupported K or missing img_w/img_h for pixel format.
    """
    kps = np.asarray(kps, dtype=np.float64)
    if kps.ndim != 3 or kps.shape[2] != 3:
        raise ValueError(
            f"Unsupported keypoints shape: expected (T,17,3) or (T,25,3), got {kps.shape}"
        )
    t, k, _ = kps.shape
    if k not in (NUM_COCO17, NUM_NTU25):
        raise ValueError(
            f"Unsupported keypoint count K={k}; only 17 (COCO-17) or 25 (NTU, will be mapped to 17) are supported."
        )

    fmt = (meta.get("format") or "coco17_norm").lower()
    if "pixel" in fmt or fmt == FORMAT_COCO17_PIXEL:
        w = meta.get("img_w") or img_w
        h = meta.get("img_h") or img_h
        if w is None or h is None:
            raise ValueError(
                "Input keypoints not normalized and missing img_w/img_h metadata; pass --img-w/--img-h"
            )
        kps = kps.copy()
        kps[..., 0] /= float(w)
        kps[..., 1] /= float(h)
        kps = np.clip(kps, 0.0, 1.0)
    else:
        if not np.isfinite(kps).all():
            raise ValueError("keypoints contain NaN or Inf")
        if kps.min() < 0.0 or kps.max() > 1.0:
            raise ValueError(
                "keypoints x, y, conf must be in [0, 1] for norm format; got range [{}, {}]".format(
                    float(kps.min()), float(kps.max())
                )
            )
        kps = np.clip(kps, 0.0, 1.0).astype(np.float64)

    if k == NUM_NTU25:
        from har_windownet.datasets.ntu.mapping import map_ntu_sequence_to_coco17

        kps = map_ntu_sequence_to_coco17(kps, default_conf=1.0)
    return kps
