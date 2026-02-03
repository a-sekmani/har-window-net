"""Tests for NTU builder (action_label_from_sample_id, build_dataset)."""

from pathlib import Path

import pytest

from har_windownet.datasets.ntu.builder import action_label_from_sample_id, build_dataset


def test_action_label_from_sample_id() -> None:
    """Extract A013 from standard NTU sample ID."""
    assert action_label_from_sample_id("S001C002P003R002A013") == "A013"
    assert action_label_from_sample_id("S120C003P001R001A001") == "A001"
    assert action_label_from_sample_id("S001C002P003R002A120") == "A120"


def test_action_label_from_sample_id_invalid() -> None:
    """Non-matching ID returns None."""
    assert action_label_from_sample_id("invalid") is None
    assert action_label_from_sample_id("S001C002P003R002") is None


def test_build_dataset_no_samples_raises() -> None:
    """build_dataset raises when source dir has no NTU samples."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "empty.txt").write_text("")
        with pytest.raises(FileNotFoundError, match="No NTU samples"):
            build_dataset(d, Path(d) / "out")
