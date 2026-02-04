"""Tests for Custom10 windowing (padding when T < 30)."""

import numpy as np
import pytest

from har_windownet.datasets.custom10.windowing import slice_windows


def test_windowing_padding() -> None:
    """Sequence with T < 30 yields one window with last-frame padding."""
    seq = np.random.rand(15, 17, 3).astype(np.float64)
    np.clip(seq, 0, 1, out=seq)
    out = slice_windows(seq, window_size=30, stride=30, pad_short=True)
    assert len(out) == 1
    assert out[0].shape == (30, 17, 3)
    np.testing.assert_array_almost_equal(out[0][:15], seq)
    np.testing.assert_array_almost_equal(out[0][15:], np.broadcast_to(seq[-1], (15, 17, 3)))
