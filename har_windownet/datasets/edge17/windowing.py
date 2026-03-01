"""
Edge17 windowing - reuses NTU windowing logic.

Slice keypoint sequences into fixed-size windows for Window contract.
"""

from __future__ import annotations

from har_windownet.datasets.ntu.windowing import slice_windows

__all__ = ["slice_windows"]
