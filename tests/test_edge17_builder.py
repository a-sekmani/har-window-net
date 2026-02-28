"""Tests for Edge17 builder: split by clip, no leakage, outputs written."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from har_windownet.datasets.edge17.builder import (
    build_dataset_edge17,
    build_label_map,
    build_windows_from_clip,
)
from har_windownet.datasets.edge17.reader import read_clip

ASSETS = Path(__file__).resolve().parent / "assets" / "edge17"


def _make_sample_jsonl(path: Path, action_id: str, frame_count: int = 50) -> None:
    """Create a minimal .skeleton.jsonl file for testing."""
    with open(path, "w", encoding="utf-8") as f:
        meta = {
            "type": "meta",
            "action_id": action_id,
            "fps": 30.0,
            "frame_count": frame_count,
            "skeleton_format": "coco17",
            "coords": "normalized",
        }
        f.write(json.dumps(meta) + "\n")

        for i in range(frame_count):
            keypoints = [[0.5 + i * 0.001, 0.5 + i * 0.001, 0.9] for _ in range(17)]
            frame = {
                "type": "frame",
                "frame_index": i,
                "ts_unix_ms": 1000000 + i * 33,
                "persons": [{"track_id": 1, "keypoints": keypoints}],
            }
            f.write(json.dumps(frame) + "\n")


def _clip_id_set(table) -> set:
    """Extract unique source_clip_ids from parquet table."""
    col = table.column("source_clip_id")
    return {col[i].as_py() for i in range(len(col))}


class TestBuildWindowsFromClip:
    def test_builds_windows_from_sample(self) -> None:
        """build_windows_from_clip creates windows from clip."""
        clip = read_clip(ASSETS / "sample.skeleton.jsonl")
        windows = build_windows_from_clip(clip, window_size=30, stride=15)

        assert len(windows) > 0
        for w in windows:
            assert "id" in w
            assert "keypoints" in w
            assert "label" in w
            assert w["window_size"] == 30
            assert len(w["keypoints"]) == 30
            assert len(w["keypoints"][0]) == 17
            assert len(w["keypoints"][0][0]) == 3

    def test_respects_stride(self) -> None:
        """build_windows_from_clip uses stride for overlap."""
        clip = read_clip(ASSETS / "sample.skeleton.jsonl")
        windows_15 = build_windows_from_clip(clip, window_size=30, stride=15)
        windows_30 = build_windows_from_clip(clip, window_size=30, stride=30)

        assert len(windows_15) > len(windows_30)


class TestBuildLabelMap:
    def test_builds_correct_map(self) -> None:
        """build_label_map creates label_to_id and id_to_name."""
        labels = ["A001", "A008", "A001", "A003"]
        label_map = build_label_map(labels)

        assert label_map["num_classes"] == 3
        assert "A001" in label_map["label_to_id"]
        assert "A003" in label_map["label_to_id"]
        assert "A008" in label_map["label_to_id"]
        assert "0" in label_map["id_to_name"] or 0 in label_map["id_to_name"]


class TestBuildDatasetEdge17:
    def test_writes_all_outputs(self, tmp_path: Path) -> None:
        """build_dataset_edge17 produces all expected output files."""
        source = tmp_path / "source"
        source.mkdir()
        _make_sample_jsonl(source / "clip1.skeleton.jsonl", "A001", 50)
        _make_sample_jsonl(source / "clip2.skeleton.jsonl", "A002", 50)

        out = tmp_path / "output"
        meta = build_dataset_edge17(source, out, window_size=30, stride=30, seed=42)

        assert (out / "label_map.json").exists()
        assert (out / "splits" / "train.parquet").exists()
        assert (out / "splits" / "val.parquet").exists()
        assert (out / "splits" / "test.parquet").exists()
        assert (out / "dataset_meta.json").exists()
        assert (out / "stats" / "class_counts.json").exists()
        assert (out / "stats" / "pose_conf_hist.json").exists()

        assert meta["dataset_type"] == "edge17"
        assert meta["keypoint_order"] == "coco17"
        assert meta["coords"] == "normalized"

    def test_split_by_clip_no_leakage(self, tmp_path: Path) -> None:
        """No clip_id appears in more than one split."""
        source = tmp_path / "source"
        source.mkdir()
        for i in range(10):
            action = f"A{(i % 3) + 1:03d}"
            _make_sample_jsonl(source / f"clip{i:02d}.skeleton.jsonl", action, 60)

        out = tmp_path / "output"
        build_dataset_edge17(source, out, window_size=30, stride=30, seed=42)

        train = pq.read_table(out / "splits" / "train.parquet")
        val = pq.read_table(out / "splits" / "val.parquet")
        test = pq.read_table(out / "splits" / "test.parquet")

        train_clips = _clip_id_set(train)
        val_clips = _clip_id_set(val)
        test_clips = _clip_id_set(test)

        assert train_clips & val_clips == set()
        assert train_clips & test_clips == set()
        assert val_clips & test_clips == set()

    def test_respects_window_size_and_stride(self, tmp_path: Path) -> None:
        """Dataset meta contains correct window_size and stride."""
        source = tmp_path / "source"
        source.mkdir()
        _make_sample_jsonl(source / "clip1.skeleton.jsonl", "A001", 100)

        out = tmp_path / "output"
        meta = build_dataset_edge17(source, out, window_size=25, stride=10, seed=42)

        assert meta["window_size"] == 25
        assert meta["stride"] == 10

    def test_exports_samples(self, tmp_path: Path) -> None:
        """build_dataset_edge17 exports sample JSONs when requested."""
        source = tmp_path / "source"
        source.mkdir()
        _make_sample_jsonl(source / "clip1.skeleton.jsonl", "A001", 100)

        out = tmp_path / "output"
        build_dataset_edge17(source, out, window_size=30, stride=30, export_samples_count=2)

        samples_dir = out / "samples"
        assert samples_dir.exists()
        sample_files = list(samples_dir.glob("*.json"))
        assert len(sample_files) == 2

    def test_raises_on_empty_source(self, tmp_path: Path) -> None:
        """build_dataset_edge17 raises if no JSONL files found."""
        source = tmp_path / "empty"
        source.mkdir()
        out = tmp_path / "output"

        with pytest.raises(FileNotFoundError):
            build_dataset_edge17(source, out)

    def test_window_shape_in_parquet(self, tmp_path: Path) -> None:
        """Windows in parquet have correct shape (window_size, 17, 3)."""
        source = tmp_path / "source"
        source.mkdir()
        _make_sample_jsonl(source / "clip1.skeleton.jsonl", "A001", 50)

        out = tmp_path / "output"
        build_dataset_edge17(source, out, window_size=30, stride=30, seed=42)

        train = pq.read_table(out / "splits" / "train.parquet")
        if train.num_rows > 0:
            kp = train.column("keypoints")[0].as_py()
            assert len(kp) == 30
            assert len(kp[0]) == 17
            assert len(kp[0][0]) == 3
