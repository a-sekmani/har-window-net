"""Pytest fixtures shared across test modules."""

import uuid

import numpy as np
import pytest

from har_windownet.contracts.window import (
    KEYPOINT_DIM,
    NUM_KEYPOINTS,
    WINDOW_SIZE,
)


@pytest.fixture
def valid_keypoints() -> np.ndarray:
    """Keypoints array of shape (30, 17, 3) with values in [0, 1]."""
    np.random.seed(42)
    return np.clip(np.random.rand(WINDOW_SIZE, NUM_KEYPOINTS, KEYPOINT_DIM), 0.0, 1.0).astype(
        np.float64
    )


@pytest.fixture
def valid_window_dict(valid_keypoints: np.ndarray) -> dict:
    """Minimal valid window dict for contract validation."""
    return {
        "id": str(uuid.uuid4()),
        "device_id": "ntu-offline",
        "camera_id": "ntu-cam",
        "session_id": str(uuid.uuid4()),
        "track_id": 1,
        "ts_start_ms": 0,
        "ts_end_ms": 1000,
        "fps": 30,
        "window_size": 30,
        "mean_pose_conf": 0.9,
        "label": "A001",
        "label_source": "dataset",
        "created_at": "2024-01-01T00:00:00+00:00",
        "keypoints": valid_keypoints.tolist(),
    }


@pytest.fixture
def ntu_frame_25_3() -> np.ndarray:
    """Single NTU frame shape (25, 3) x,y,z in mm."""
    np.random.seed(123)
    return np.random.randn(25, 3).astype(np.float64) * 500 + 1000
