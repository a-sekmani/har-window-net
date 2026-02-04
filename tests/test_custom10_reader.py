"""Tests for Custom10 reader (list_clips, read_clip)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from har_windownet.datasets.custom10.reader import list_custom10_clips, read_clip

ASSETS = Path(__file__).resolve().parent / "assets" / "custom10"


def _minimal_skeleton_content() -> str:
    """One frame, one body, 25 joints (NTU text format). All zeros."""
    # reader: framecount, bodycount, bodyID, 6 floats, leanX leanY, bodyState, joint_count → 13 tokens then 25*12
    header = "1 1 0 0 0 0 0 0 0 0 0 0 25"  # 12 tokens so token[11]=25=joint_count
    joint = "0 0 0 0 0 0 0 0 0 0 0 0"  # x y z dx dy cx cy 4*quat ts (12 values)
    joints = " ".join([joint] * 25)
    return f"{header} {joints}"


def test_list_custom10_clips_includes_skeleton() -> None:
    """list_custom10_clips discovers .skeleton files in label subfolders."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "A001_WALK").mkdir()
        (root / "A001_WALK" / "sample.skeleton").write_text(_minimal_skeleton_content())
        refs = list_custom10_clips(root)
        assert len(refs) == 1
        assert refs[0].path.suffix.lower() == ".skeleton"
        assert refs[0].clip_id == "A001_WALK/sample"
        assert refs[0].label_id == "A001"


def test_read_clip_json_shape() -> None:
    """read_clip on JSON returns (T, K, 3) and meta with fps, format."""
    path = ASSETS / "A001_WALKING" / "clip_short.json"
    kp, meta = read_clip(path)
    assert kp.ndim == 3
    assert kp.shape[1] == 17
    assert kp.shape[2] == 3
    assert kp.shape[0] == 15
    assert meta.get("fps") == 30
    assert meta.get("format") == "coco17_norm"


def test_list_custom10_clips() -> None:
    """list_custom10_clips discovers A001_WALKING and A002_SITTING with 2 clips each."""
    refs = list_custom10_clips(ASSETS)
    assert len(refs) == 4
    label_ids = {r.label_id for r in refs}
    assert label_ids == {"A001", "A002"}
    clip_ids = {r.clip_id for r in refs}
    assert "A001_WALKING/clip_short" in clip_ids
    assert "A001_WALKING/clip_long" in clip_ids
    assert "A002_SITTING/clip_short" in clip_ids
    assert "A002_SITTING/clip_long" in clip_ids
