"""
Slice preprocessed keypoint sequences into fixed-size windows for Window contract.

Handles sequences shorter than window_size by padding (repeat last frame) to produce
at least one window per sample.
"""

from __future__ import annotations

import numpy as np


def slice_windows(
    keypoints: np.ndarray,
    *,
    window_size: int = 30,
    stride: int | None = None,
    pad_short: bool = True,
) -> list[np.ndarray]:
    """
    Slice a keypoint sequence (N, 17, 3) into windows of shape (window_size, 17, 3).

    Parameters
    ----------
    keypoints : ndarray
        Shape (N, 17, 3), dtype float64, normalized [0, 1].
    window_size : int
        Number of frames per window (default 30).
    stride : int, optional
        Step between window starts. If None, stride = window_size (no overlap).
    pad_short : bool
        If True, sequences with N < window_size yield one window by repeating
        the last frame. If False, short sequences yield no windows.

    Returns
    -------
    list of ndarray
        Each element shape (window_size, 17, 3).
    """
    keypoints = np.asarray(keypoints, dtype=np.float64)
    if keypoints.ndim != 3 or keypoints.shape[1] != 17 or keypoints.shape[2] != 3:
        raise ValueError(
            f"keypoints must be (N, 17, 3), got {keypoints.shape}"
        )
    n = keypoints.shape[0]
    if stride is None:
        stride = window_size
    if stride < 1:
        raise ValueError("stride must be >= 1")
    windows: list[np.ndarray] = []

    if n < window_size:
        if not pad_short:
            return []
        # Pad by repeating last frame (or first if n==0)
        if n == 0:
            pad_frame = np.zeros((17, 3), dtype=np.float64)
        else:
            pad_frame = keypoints[-1]
        padded = np.concatenate([
            keypoints,
            np.tile(pad_frame, (window_size - n, 1, 1)),
        ], axis=0)
        windows.append(padded)
        return windows

    for start in range(0, n - window_size + 1, stride):
        window = keypoints[start : start + window_size].copy()
        windows.append(window)
    return windows
