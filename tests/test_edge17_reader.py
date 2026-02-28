"""Tests for Edge17 JSONL reader."""

from pathlib import Path

import numpy as np
import pytest

from har_windownet.datasets.edge17.reader import (
    extract_keypoints_sequence,
    list_edge17_files,
    read_clip,
    read_jsonl_file,
    sanitize_keypoints,
)
from har_windownet.datasets.edge17.labels import (
    extract_label_from_filename,
    extract_label_from_meta,
    get_label,
    normalize_label,
)

ASSETS = Path(__file__).resolve().parent / "assets" / "edge17"


class TestReadJsonlFile:
    def test_reads_meta_and_frames(self) -> None:
        """read_jsonl_file returns meta and frames list."""
        path = ASSETS / "sample.skeleton.jsonl"
        data = read_jsonl_file(path)

        assert "meta" in data
        assert "frames" in data
        assert data["meta"]["type"] == "meta"
        assert data["meta"]["action_id"] == "A001"
        assert data["meta"]["fps"] == 30.0
        assert data["meta"]["skeleton_format"] == "coco17"
        assert data["meta"]["coords"] == "normalized"
        assert len(data["frames"]) == 50

    def test_raises_on_missing_meta(self, tmp_path: Path) -> None:
        """read_jsonl_file raises if no meta line."""
        bad_file = tmp_path / "no_meta.jsonl"
        bad_file.write_text('{"type": "frame", "frame_index": 0, "persons": []}\n')

        with pytest.raises(ValueError, match="No meta line found"):
            read_jsonl_file(bad_file)


class TestExtractKeypointsSequence:
    def test_extracts_correct_shape(self) -> None:
        """extract_keypoints_sequence returns (T, 17, 3) array."""
        path = ASSETS / "sample.skeleton.jsonl"
        data = read_jsonl_file(path)
        frames = data["frames"]

        kp = extract_keypoints_sequence(frames, track_id=1)
        assert kp.shape[0] == 49
        assert kp.shape[1] == 17
        assert kp.shape[2] == 3

    def test_handles_empty_persons(self) -> None:
        """extract_keypoints_sequence skips frames with empty persons."""
        path = ASSETS / "sample.skeleton.jsonl"
        data = read_jsonl_file(path)
        frames = data["frames"]

        kp = extract_keypoints_sequence(frames, track_id=1)
        assert kp.shape[0] == 49

    def test_handles_missing_track_id(self) -> None:
        """extract_keypoints_sequence returns empty for non-existent track."""
        path = ASSETS / "sample.skeleton.jsonl"
        data = read_jsonl_file(path)
        frames = data["frames"]

        kp = extract_keypoints_sequence(frames, track_id=999)
        assert kp.shape == (0, 17, 3)


class TestSanitizeKeypoints:
    def test_clips_values(self) -> None:
        """sanitize_keypoints clips values to [0, 1]."""
        kp = np.array([[[1.5, -0.1, 0.5]]])
        sanitized = sanitize_keypoints(kp)

        assert sanitized[0, 0, 0] == 1.0
        assert sanitized[0, 0, 1] == 0.0
        assert sanitized[0, 0, 2] == 0.5

    def test_replaces_nan_inf(self) -> None:
        """sanitize_keypoints replaces NaN/Inf with safe values."""
        kp = np.array([[[np.nan, np.inf, -np.inf]]])
        sanitized = sanitize_keypoints(kp)

        assert not np.any(np.isnan(sanitized))
        assert not np.any(np.isinf(sanitized))


class TestReadClip:
    def test_returns_expected_dict(self) -> None:
        """read_clip returns dict with keypoints, label, fps, clip_id."""
        path = ASSETS / "sample.skeleton.jsonl"
        clip = read_clip(path)

        assert "meta" in clip
        assert "keypoints" in clip
        assert "label" in clip
        assert "fps" in clip
        assert "clip_id" in clip

        assert clip["label"] == "A001"
        assert clip["fps"] == 30.0
        assert clip["clip_id"] == "sample.skeleton"
        assert clip["keypoints"].shape == (49, 17, 3)

    def test_keypoints_are_sanitized(self) -> None:
        """read_clip returns sanitized keypoints."""
        path = ASSETS / "sample.skeleton.jsonl"
        clip = read_clip(path)
        kp = clip["keypoints"]

        assert not np.any(np.isnan(kp))
        assert not np.any(np.isinf(kp))
        assert np.all(kp >= 0.0)
        assert np.all(kp <= 1.0)


class TestListEdge17Files:
    def test_finds_jsonl_files(self, tmp_path: Path) -> None:
        """list_edge17_files finds .skeleton.jsonl files."""
        (tmp_path / "a.skeleton.jsonl").touch()
        (tmp_path / "b.skeleton.jsonl").touch()
        (tmp_path / "c.txt").touch()
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "d.skeleton.jsonl").touch()

        files = list_edge17_files(tmp_path)
        names = [f.name for f in files]

        assert len(files) == 3
        assert "a.skeleton.jsonl" in names
        assert "b.skeleton.jsonl" in names
        assert "d.skeleton.jsonl" in names


class TestLabels:
    def test_extract_label_from_meta(self) -> None:
        """extract_label_from_meta returns normalized label."""
        assert extract_label_from_meta({"action_id": "A001"}) == "A001"
        assert extract_label_from_meta({"action_id": "A1"}) == "A001"
        assert extract_label_from_meta({"action_id": "A08"}) == "A008"
        assert extract_label_from_meta({}) is None

    def test_extract_label_from_filename(self) -> None:
        """extract_label_from_filename extracts A### from filename."""
        assert extract_label_from_filename("S001C001P001R001A008.skeleton.jsonl") == "A008"
        assert extract_label_from_filename("sample_A1_test.jsonl") == "A001"
        assert extract_label_from_filename("no_label.jsonl") is None

    def test_normalize_label(self) -> None:
        """normalize_label pads to 3 digits."""
        assert normalize_label("A1") == "A001"
        assert normalize_label("A08") == "A008"
        assert normalize_label("A120") == "A120"
        assert normalize_label("INVALID") == "INVALID"

    def test_get_label_prefers_meta(self) -> None:
        """get_label uses meta over filename."""
        label = get_label({"action_id": "A005"}, "S001A008.jsonl")
        assert label == "A005"

    def test_get_label_fallback_to_filename(self) -> None:
        """get_label falls back to filename if no meta."""
        label = get_label({}, "S001C001P001R001A012.skeleton.jsonl")
        assert label == "A012"

    def test_get_label_unknown(self) -> None:
        """get_label returns UNKNOWN if no label found."""
        label = get_label({}, "no_label.jsonl")
        assert label == "UNKNOWN"
