"""Tests for Window contract (window.py)."""

import uuid

import numpy as np
import pytest

from har_windownet.contracts.window import (
    KEYPOINT_DIM,
    NUM_KEYPOINTS,
    WINDOW_SIZE,
    WindowContract,
    make_created_at,
    make_window_id,
    validate_window_dict,
    window_json_schema,
)


def test_window_contract_valid(valid_window_dict: dict) -> None:
    """Valid window dict creates WindowContract and serializes."""
    w = WindowContract.model_validate(valid_window_dict)
    assert w.window_size == 30
    assert w.fps == 30
    assert w.label_source == "dataset"
    assert w.keypoints.shape == (WINDOW_SIZE, NUM_KEYPOINTS, KEYPOINT_DIM)
    dumped = w.model_dump_for_storage()
    assert isinstance(dumped["keypoints"], list)
    assert WindowContract.model_validate_from_storage(dumped).id == w.id


def test_window_contract_keypoints_shape_rejected() -> None:
    """Wrong keypoints shape raises."""
    d = {
        "id": str(uuid.uuid4()),
        "device_id": "x",
        "camera_id": "y",
        "session_id": str(uuid.uuid4()),
        "track_id": 1,
        "ts_start_ms": 0,
        "ts_end_ms": 1000,
        "fps": 30,
        "window_size": 30,
        "mean_pose_conf": 0.5,
        "label": "A001",
        "label_source": "dataset",
        "created_at": "2024-01-01T00:00:00+00:00",
        "keypoints": np.zeros((20, 17, 3)).tolist(),
    }
    with pytest.raises(ValueError, match="keypoints shape"):
        WindowContract.model_validate(d)


def test_window_contract_keypoints_nan_rejected(valid_window_dict: dict) -> None:
    """NaN in keypoints raises."""
    kp = np.array(valid_window_dict["keypoints"])
    kp[0, 0, 0] = np.nan
    valid_window_dict["keypoints"] = kp.tolist()
    with pytest.raises(ValueError, match="NaN|Inf"):
        WindowContract.model_validate(valid_window_dict)


def test_window_contract_keypoints_range_rejected(valid_window_dict: dict) -> None:
    """Keypoints outside [0,1] raise."""
    kp = np.array(valid_window_dict["keypoints"])
    kp[0, 0, 0] = 1.5
    valid_window_dict["keypoints"] = kp.tolist()
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        WindowContract.model_validate(valid_window_dict)


def test_window_contract_invalid_uuid(valid_window_dict: dict) -> None:
    """Invalid UUID in id or session_id raises."""
    valid_window_dict["id"] = "not-a-uuid"
    with pytest.raises(ValueError):
        WindowContract.model_validate(valid_window_dict)


def test_validate_window_dict_valid(valid_window_dict: dict) -> None:
    """validate_window_dict returns no errors for valid dict."""
    errors = validate_window_dict(valid_window_dict)
    assert errors == []


def test_validate_window_dict_missing_field(valid_window_dict: dict) -> None:
    """Missing required field is reported."""
    del valid_window_dict["label"]
    errors = validate_window_dict(valid_window_dict)
    assert any("label" in e for e in errors)


def test_validate_window_dict_label_source(valid_window_dict: dict) -> None:
    """label_source != 'dataset' is reported."""
    valid_window_dict["label_source"] = "manual"
    errors = validate_window_dict(valid_window_dict)
    assert any("dataset" in e for e in errors)


def test_validate_window_dict_bad_keypoints_shape(valid_window_dict: dict) -> None:
    """Wrong keypoints shape is reported."""
    valid_window_dict["keypoints"] = [[0.0] * 3] * 17
    errors = validate_window_dict(valid_window_dict)
    assert any("shape" in e for e in errors)


def test_window_json_schema() -> None:
    """JSON schema is a dict with expected top-level keys."""
    schema = window_json_schema()
    assert isinstance(schema, dict)
    assert "properties" in schema or "title" in schema


def test_make_window_id() -> None:
    """make_window_id returns a valid UUID string."""
    uid = make_window_id()
    uuid.UUID(uid)


def test_make_created_at() -> None:
    """make_created_at returns an ISO-like string."""
    s = make_created_at()
    assert "T" in s and "-" in s


def test_ts_end_ms_from_window() -> None:
    """ts_end_ms = ts_start_ms + (window_size-1)*(1000/fps)."""
    from har_windownet.contracts.window import ts_end_ms_from_window
    assert ts_end_ms_from_window(30, 30.0, 0) == 966
    assert ts_end_ms_from_window(30, 30.0, 100) == 1066
