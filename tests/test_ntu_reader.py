"""Tests for NTU skeleton reader (reader.py)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from har_windownet.datasets.ntu.reader import (
    is_missing_sample,
    list_ntu_samples,
    load_missing_skeletons_set,
    read_ntu_npy,
    read_ntu_npy_full,
    read_ntu_sample,
    sample_id_from_path,
    select_dominant_body,
)


def test_sample_id_from_path_skeleton_npy() -> None:
    """sample_id_from_path strips .skeleton from stem for .skeleton.npy."""
    p = Path("/some/S001C002P003R002A013.skeleton.npy")
    assert sample_id_from_path(p) == "S001C002P003R002A013"


def test_sample_id_from_path_skeleton() -> None:
    """sample_id_from_path for .skeleton file."""
    p = Path("/some/S001C002P003R002A013.skeleton")
    assert sample_id_from_path(p) == "S001C002P003R002A013"


def test_load_missing_skeletons_set() -> None:
    """Missing set is loaded and contains known missing sample."""
    missing = load_missing_skeletons_set()
    assert isinstance(missing, set)
    assert "S001C002P005R002A008" in missing


def test_is_missing_sample_true() -> None:
    """Known missing sample returns True."""
    assert is_missing_sample("S001C002P005R002A008") is True


def test_is_missing_sample_false() -> None:
    """Non-missing sample ID returns False."""
    assert is_missing_sample("S001C001P001R001A001") is False


def test_read_ntu_npy_valid() -> None:
    """read_ntu_npy reads dict with skel_body0."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.skeleton.npy"
        data = {"skel_body0": np.random.rand(20, 25, 3).astype(np.float64)}
        np.save(path, data, allow_pickle=True)
        out = read_ntu_npy(path)
        assert out.shape == (20, 25, 3)
        assert out.dtype == np.float64


def test_read_ntu_npy_wrong_key_raises() -> None:
    """read_ntu_npy raises when skel_body0 missing."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.npy"
        np.save(path, {"other": np.zeros((1, 25, 3))}, allow_pickle=True)
        with pytest.raises(KeyError, match="skel_body"):
            read_ntu_npy(path)


def test_read_ntu_npy_not_dict_raises() -> None:
    """read_ntu_npy raises when file does not contain dict (e.g. raw array)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "arr.npy"
        np.save(path, np.zeros((5, 25, 3)))
        with pytest.raises(ValueError, match="Expected dict|array of size"):
            read_ntu_npy(path)


def test_read_ntu_sample_prefers_npy() -> None:
    """read_ntu_sample uses .npy when both .npy and .skeleton could exist."""
    with tempfile.TemporaryDirectory() as tmp:
        npy_path = Path(tmp) / "S001C001P001R001A001.skeleton.npy"
        data = {"skel_body0": np.random.rand(10, 25, 3).astype(np.float64)}
        np.save(npy_path, data, allow_pickle=True)
        out = read_ntu_sample(npy_path)
        assert out.shape == (10, 25, 3)


def test_list_ntu_samples_empty_dir() -> None:
    """list_ntu_samples returns empty list for dir with no samples."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = list_ntu_samples(tmp)
        assert paths == []


def test_list_ntu_samples_finds_npy() -> None:
    """list_ntu_samples finds .skeleton.npy and skips missing."""
    with tempfile.TemporaryDirectory() as tmp:
        # Use a non-missing ID
        path = Path(tmp) / "S001C001P001R001A001.skeleton.npy"
        data = {"skel_body0": np.random.rand(5, 25, 3).astype(np.float64)}
        np.save(path, data, allow_pickle=True)
        paths = list_ntu_samples(tmp)
        assert len(paths) == 1
        assert paths[0].name == "S001C001P001R001A001.skeleton.npy"


def test_list_ntu_samples_skips_missing() -> None:
    """list_ntu_samples skips samples in missing list when skip_missing=True."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "S001C002P005R002A008.skeleton.npy"
        data = {"skel_body0": np.random.rand(5, 25, 3).astype(np.float64)}
        np.save(path, data, allow_pickle=True)
        paths = list_ntu_samples(tmp, skip_missing=True)
        assert len(paths) == 0
        paths_no_skip = list_ntu_samples(tmp, skip_missing=False)
        assert len(paths_no_skip) == 1


def test_read_ntu_npy_full_returns_bodies() -> None:
    """read_ntu_npy_full returns list of body dicts with skel, rgb_xy, depth_xy, tracking_state."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.skeleton.npy"
        data = {
            "skel_body0": np.random.rand(10, 25, 3).astype(np.float64),
            "rgb_body0": np.random.rand(10, 25, 2).astype(np.float64),
        }
        np.save(path, data, allow_pickle=True)
        bodies = read_ntu_npy_full(path)
        assert len(bodies) == 1
        assert bodies[0]["skel"].shape == (10, 25, 3)
        assert bodies[0]["rgb_xy"] is not None
        assert bodies[0]["tracking_state"] is None


def test_select_dominant_body_single() -> None:
    """select_dominant_body returns (0, body) when only one body."""
    body = {"skel": np.zeros((5, 25, 3)), "tracking_state": None}
    idx, b = select_dominant_body([body])
    assert idx == 0
    assert b is body


def test_select_dominant_body_most_tracked() -> None:
    """select_dominant_body with most_tracked picks body with more Tracked joints."""
    ts0 = np.full((3, 25), 2, dtype=np.int32)
    ts1 = np.full((3, 25), 0, dtype=np.int32)
    ts1[0, :10] = 2
    bodies = [
        {"skel": np.zeros((3, 25, 3)), "tracking_state": ts0},
        {"skel": np.zeros((3, 25, 3)), "tracking_state": ts1},
    ]
    idx, _ = select_dominant_body(bodies, policy="most_tracked")
    assert idx == 0
