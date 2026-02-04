"""Tests for Custom10 builder: split by clip, no leakage, outputs written."""

import tempfile
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from har_windownet.datasets.custom10.builder import build_dataset_custom10

ASSETS = Path(__file__).resolve().parent / "assets" / "custom10"


def _session_set(table):
    col = table.column("session_id")
    return {col[i].as_py() if hasattr(col[i], "as_py") else col[i] for i in range(len(col))}


def test_split_by_clip_no_leakage() -> None:
    """No clip_id (session_id) appears in more than one split."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        build_dataset_custom10(ASSETS, out, seed=42, window_size=30, stride=30)
        train = pq.read_table(out / "splits" / "train.parquet")
        val = pq.read_table(out / "splits" / "val.parquet")
        test = pq.read_table(out / "splits" / "test.parquet")
        train_s = _session_set(train)
        val_s = _session_set(val)
        test_s = _session_set(test)
        assert train_s & val_s == set()
        assert train_s & test_s == set()
        assert val_s & test_s == set()


def test_build_dataset_custom10_writes_outputs() -> None:
    """build_dataset_custom10 produces label_map.json, splits/*.parquet, dataset_meta.json, stats/*.json."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        meta = build_dataset_custom10(ASSETS, out, seed=42, window_size=30, stride=30)
        assert (out / "label_map.json").exists()
        assert (out / "splits" / "train.parquet").exists()
        assert (out / "splits" / "val.parquet").exists()
        assert (out / "splits" / "test.parquet").exists()
        assert (out / "dataset_meta.json").exists()
        assert (out / "stats" / "class_counts.json").exists()
        assert (out / "stats" / "pose_conf_hist.json").exists()
        assert meta["adapter"] == "custom10"
        assert meta["num_clips"] == 4
        assert "num_train_windows" in meta
        assert "num_val_windows" in meta
        assert "num_test_windows" in meta
        assert meta["num_classes"] == 2
