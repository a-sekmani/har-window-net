"""Slice keypoint sequences into fixed-size windows; reuse NTU logic."""

from __future__ import annotations

from har_windownet.datasets.ntu.windowing import slice_windows

__all__ = ["slice_windows"]
