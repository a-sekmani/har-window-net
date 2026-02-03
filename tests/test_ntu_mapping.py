"""Tests for NTU 25 → COCO-17 mapping (mapping.py)."""

import numpy as np
import pytest

from har_windownet.datasets.ntu.mapping import (
    NUM_COCO_KEYPOINTS,
    NUM_NTU_JOINTS,
    NTU25_TO_COCO17,
    TRACKING_INFERRED,
    TRACKING_NOT_TRACKED,
    TRACKING_TRACKED,
    map_ntu_frame_to_coco17,
    map_ntu_sequence_to_coco17,
    tracking_state_to_confidence,
)


def test_ntu25_to_coco17_length() -> None:
    """Mapping list has 17 entries."""
    assert len(NTU25_TO_COCO17) == NUM_COCO_KEYPOINTS


def test_ntu25_to_coco17_indices_in_range() -> None:
    """All mapping indices are in 0..24 or None."""
    for idx in NTU25_TO_COCO17:
        if idx is not None:
            assert 0 <= idx < NUM_NTU_JOINTS


def test_map_ntu_frame_to_coco17_shape(ntu_frame_25_3: np.ndarray) -> None:
    """Output shape is (17, 3)."""
    out = map_ntu_frame_to_coco17(ntu_frame_25_3)
    assert out.shape == (NUM_COCO_KEYPOINTS, 3)
    assert out.dtype == np.float64


def test_map_ntu_frame_to_coco17_default_conf(ntu_frame_25_3: np.ndarray) -> None:
    """When no 4th column, confidence is default 1.0."""
    out = map_ntu_frame_to_coco17(ntu_frame_25_3, default_conf=0.7)
    np.testing.assert_array_almost_equal(out[:, 2], np.full(17, 0.7))


def test_map_ntu_frame_to_coco17_with_confidence() -> None:
    """When 4th column present, it is used as confidence (clipped to [0,1])."""
    ntu = np.random.rand(25, 4).astype(np.float64)
    ntu[:, 3] = np.linspace(0, 1, 25)
    out = map_ntu_frame_to_coco17(ntu)
    assert out[:, 2].min() >= 0 and out[:, 2].max() <= 1


def test_map_ntu_frame_to_coco17_wrong_rows_raises() -> None:
    """Input with wrong number of rows raises."""
    with pytest.raises(ValueError, match="25 rows"):
        map_ntu_frame_to_coco17(np.zeros((20, 3)))


def test_map_ntu_frame_to_coco17_wrong_cols_raises() -> None:
    """Input with fewer than 3 columns raises."""
    with pytest.raises(ValueError, match="at least 3"):
        map_ntu_frame_to_coco17(np.zeros((25, 2)))


def test_map_ntu_sequence_to_coco17_shape() -> None:
    """Sequence (N, 25, 3) -> (N, 17, 3)."""
    ntu = np.random.rand(10, 25, 3).astype(np.float64)
    out = map_ntu_sequence_to_coco17(ntu)
    assert out.shape == (10, NUM_COCO_KEYPOINTS, 3)


def test_map_ntu_sequence_to_coco17_consistency_with_frame() -> None:
    """First frame of sequence matches map_ntu_frame_to_coco17 of first frame."""
    ntu = np.random.rand(5, 25, 3).astype(np.float64)
    seq_out = map_ntu_sequence_to_coco17(ntu)
    frame_out = map_ntu_frame_to_coco17(ntu[0])
    np.testing.assert_array_almost_equal(seq_out[0], frame_out)


def test_tracking_state_to_confidence() -> None:
    """tracking_state 0→0.0, 1→0.5, 2→1.0."""
    assert tracking_state_to_confidence(TRACKING_NOT_TRACKED) == 0.0
    assert tracking_state_to_confidence(TRACKING_INFERRED) == 0.5
    assert tracking_state_to_confidence(TRACKING_TRACKED) == 1.0


def test_map_ntu_frame_to_coco17_with_tracking_state(ntu_frame_25_3: np.ndarray) -> None:
    """When ntu_tracking_state is provided, conf uses tracking_state_to_confidence."""
    ts = np.full(25, TRACKING_TRACKED, dtype=np.int32)
    ts[3] = TRACKING_INFERRED
    ts[5] = TRACKING_NOT_TRACKED
    out = map_ntu_frame_to_coco17(ntu_frame_25_3, ntu_tracking_state=ts)
    assert out[0, 2] == 0.5
    assert out[7, 2] == 0.0
    assert out[5, 2] == 1.0
