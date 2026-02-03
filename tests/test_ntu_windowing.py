"""Tests for NTU windowing (slice_windows)."""

import numpy as np
import pytest

from har_windownet.datasets.ntu.windowing import slice_windows


def test_slice_windows_exact_multiple() -> None:
    """Exactly two non-overlapping windows when n=60, size=30, stride=30."""
    kp = np.random.rand(60, 17, 3).astype(np.float64)
    np.clip(kp, 0, 1, out=kp)
    out = slice_windows(kp, window_size=30, stride=30)
    assert len(out) == 2
    assert out[0].shape == (30, 17, 3)
    assert out[1].shape == (30, 17, 3)
    np.testing.assert_array_almost_equal(out[0], kp[:30])
    np.testing.assert_array_almost_equal(out[1], kp[30:60])


def test_slice_windows_with_overlap() -> None:
    """Overlapping windows when stride < window_size."""
    kp = np.random.rand(100, 17, 3).astype(np.float64)
    np.clip(kp, 0, 1, out=kp)
    out = slice_windows(kp, window_size=30, stride=15)
    assert len(out) == 5  # 0,15,30,45,60 -> 5 windows; 75+30=105 > 100
    for w in out:
        assert w.shape == (30, 17, 3)


def test_slice_windows_short_pad() -> None:
    """Sequence shorter than window_size yields one window when pad_short=True."""
    kp = np.random.rand(10, 17, 3).astype(np.float64)
    np.clip(kp, 0, 1, out=kp)
    out = slice_windows(kp, window_size=30, pad_short=True)
    assert len(out) == 1
    assert out[0].shape == (30, 17, 3)
    np.testing.assert_array_almost_equal(out[0][:10], kp)
    np.testing.assert_array_almost_equal(out[0][10:], np.broadcast_to(kp[-1], (20, 17, 3)))


def test_slice_windows_short_no_pad() -> None:
    """Sequence shorter than window_size yields no windows when pad_short=False."""
    kp = np.random.rand(10, 17, 3).astype(np.float64)
    out = slice_windows(kp, window_size=30, pad_short=False)
    assert len(out) == 0


def test_slice_windows_empty_sequence_pad() -> None:
    """Empty sequence with pad_short=True yields one zero-padded window."""
    kp = np.zeros((0, 17, 3), dtype=np.float64)
    out = slice_windows(kp, window_size=30, pad_short=True)
    assert len(out) == 1
    assert out[0].shape == (30, 17, 3)
    assert out[0].sum() == 0


def test_slice_windows_invalid_shape() -> None:
    """Invalid keypoints shape raises."""
    kp = np.random.rand(30, 25, 3)  # wrong keypoint count
    with pytest.raises(ValueError, match="17"):
        slice_windows(kp, window_size=30)


def test_slice_windows_invalid_stride() -> None:
    """stride < 1 raises."""
    kp = np.random.rand(60, 17, 3).astype(np.float64)
    with pytest.raises(ValueError, match="stride"):
        slice_windows(kp, window_size=30, stride=0)
