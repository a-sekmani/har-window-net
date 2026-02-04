"""Tests that Custom10-built windows pass Window contract validation."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from har_windownet.contracts.window import validate_window_dict
from har_windownet.datasets.custom10.builder import build_windows_for_clip
from har_windownet.datasets.custom10.reader import ClipRef

ASSETS = Path(__file__).resolve().parent / "assets" / "custom10"


def _minimal_skeleton_content() -> str:
    """One frame, one body, 25 joints (NTU text). 13 tokens (bodyID,6,leanX,leanY,bodyState,25) then 25*12."""
    header = "1 1 0 0 0 0 0 0 0 0 0 0 25"  # 12 tokens so token[11]=25=joint_count
    joint = "0 0 0 0 0 0 0 0 0 0 0 0"
    return f"{header} {' '.join([joint] * 25)}"


def test_window_contract_valid_custom10() -> None:
    """Windows built from Custom10 clip pass validate_window_dict; keypoints (30,17,3) in [0,1]."""
    ref = ClipRef(
        path=ASSETS / "A001_WALKING" / "clip_short.json",
        clip_id="A001_WALKING/clip_short",
        label_id="A001",
        label_name="WALKING",
    )
    wins = build_windows_for_clip(ref, window_size=30, stride=30, fps=30.0)
    assert len(wins) == 1
    row = wins[0]
    errors = validate_window_dict(row)
    assert errors == []
    assert len(row["keypoints"]) == 30
    assert len(row["keypoints"][0]) == 17
    assert len(row["keypoints"][0][0]) == 3
    arr = np.array(row["keypoints"])
    assert arr.min() >= 0.0 and arr.max() <= 1.0


def test_window_contract_valid_custom10_skeleton() -> None:
    """Windows built from .skeleton (NTU text) pass validate_window_dict; projection rgb."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "A001_WALK").mkdir()
        (root / "A001_WALK" / "sample.skeleton").write_text(_minimal_skeleton_content())
        ref = ClipRef(
            path=root / "A001_WALK" / "sample.skeleton",
            clip_id="A001_WALK/sample",
            label_id="A001",
            label_name="WALK",
        )
        wins = build_windows_for_clip(ref, window_size=30, stride=30, projection="rgb")
    assert len(wins) == 1
    row = wins[0]
    errors = validate_window_dict(row)
    assert errors == []
    assert len(row["keypoints"]) == 30
    assert len(row["keypoints"][0]) == 17
    assert row.get("source_body_id") == 0
    arr = np.array(row["keypoints"])
    assert arr.min() >= 0.0 and arr.max() <= 1.0
