"""Tests for feature transforms: get_input_features, build_feature_pipeline, Normalize, Velocity, Angles."""

import numpy as np
import pytest

from har_windownet.contracts.window import KEYPOINT_DIM, NUM_KEYPOINTS, WINDOW_SIZE
from har_windownet.features.transforms import (
    RAW_FEATURES,
    NUM_ANGLES,
    AnglesTransform,
    NormalizePoseTransform,
    VelocityTransform,
    build_feature_pipeline,
    get_input_features,
)


@pytest.fixture
def keypoints_30_17_3() -> np.ndarray:
    """Keypoints (T=30, 17, 3) in [0, 1] for tests."""
    np.random.seed(42)
    return np.clip(np.random.rand(WINDOW_SIZE, NUM_KEYPOINTS, KEYPOINT_DIM), 0.0, 1.0).astype(
        np.float32
    )


# --- get_input_features ---


def test_get_input_features_raw_keep() -> None:
    assert get_input_features({"features": "raw", "conf_mode": "keep"}) == RAW_FEATURES  # 51


def test_get_input_features_raw_drop() -> None:
    assert get_input_features({"features": "raw", "conf_mode": "drop"}) == NUM_KEYPOINTS * 2  # 34


def test_get_input_features_norm() -> None:
    assert get_input_features({"features": "norm", "conf_mode": "keep"}) == 51
    assert get_input_features({"features": "norm", "conf_mode": "drop"}) == 34


def test_get_input_features_vel() -> None:
    # base + 17*2 (dx, dy)
    assert get_input_features({"features": "vel", "conf_mode": "keep"}) == 51 + 34
    assert get_input_features({"features": "vel", "conf_mode": "drop"}) == 34 + 34


def test_get_input_features_angles() -> None:
    assert get_input_features({"features": "angles", "conf_mode": "keep"}) == 51 + NUM_ANGLES
    assert get_input_features({"features": "angles", "conf_mode": "drop"}) == 34 + NUM_ANGLES


def test_get_input_features_combo() -> None:
    # base + vel (34) + angles (10)
    assert get_input_features({"features": "combo", "conf_mode": "keep"}) == 51 + 34 + NUM_ANGLES
    assert get_input_features({"features": "combo", "conf_mode": "drop"}) == 34 + 34 + NUM_ANGLES


def test_get_input_features_defaults() -> None:
    assert get_input_features({}) == RAW_FEATURES


# --- build_feature_pipeline output shape ---


def test_build_pipeline_raw_shape(keypoints_30_17_3: np.ndarray) -> None:
    fn, F = build_feature_pipeline({"features": "raw", "conf_mode": "keep"})
    out = fn(keypoints_30_17_3)
    assert out.shape == (WINDOW_SIZE, F)
    assert F == 51


def test_build_pipeline_raw_drop_shape(keypoints_30_17_3: np.ndarray) -> None:
    fn, F = build_feature_pipeline({"features": "raw", "conf_mode": "drop"})
    out = fn(keypoints_30_17_3)
    assert out.shape == (WINDOW_SIZE, F)
    assert F == 34


def test_build_pipeline_norm_shape(keypoints_30_17_3: np.ndarray) -> None:
    fn, F = build_feature_pipeline({"features": "norm", "conf_mode": "keep"})
    out = fn(keypoints_30_17_3)
    assert out.shape == (WINDOW_SIZE, F)
    assert F == 51


def test_build_pipeline_vel_shape(keypoints_30_17_3: np.ndarray) -> None:
    fn, F = build_feature_pipeline({"features": "vel", "conf_mode": "keep"})
    out = fn(keypoints_30_17_3)
    assert out.shape == (WINDOW_SIZE, F)
    assert F == 51 + 34


def test_build_pipeline_angles_shape(keypoints_30_17_3: np.ndarray) -> None:
    fn, F = build_feature_pipeline({"features": "angles", "conf_mode": "keep"})
    out = fn(keypoints_30_17_3)
    assert out.shape == (WINDOW_SIZE, F)
    assert F == 51 + NUM_ANGLES


def test_build_pipeline_combo_shape(keypoints_30_17_3: np.ndarray) -> None:
    fn, F = build_feature_pipeline({"features": "combo", "conf_mode": "keep"})
    out = fn(keypoints_30_17_3)
    assert out.shape == (WINDOW_SIZE, F)
    assert F == 51 + 34 + NUM_ANGLES


def test_build_pipeline_output_dtype(keypoints_30_17_3: np.ndarray) -> None:
    fn, _ = build_feature_pipeline({"features": "raw"})
    out = fn(keypoints_30_17_3)
    assert out.dtype == np.float32


# --- NormalizePoseTransform ---


def test_normalize_pose_output_shape(keypoints_30_17_3: np.ndarray) -> None:
    t = NormalizePoseTransform(norm_center="auto", norm_scale="auto", conf_mode="keep")
    out = t(keypoints_30_17_3)
    assert out.shape == (WINDOW_SIZE, NUM_KEYPOINTS, KEYPOINT_DIM)


def test_normalize_pose_conf_drop(keypoints_30_17_3: np.ndarray) -> None:
    t = NormalizePoseTransform(conf_mode="drop")
    out = t(keypoints_30_17_3)
    assert out.shape == (WINDOW_SIZE, NUM_KEYPOINTS, 2)


def test_normalize_pose_clamped(keypoints_30_17_3: np.ndarray) -> None:
    t = NormalizePoseTransform(clamp_range=(-2.0, 2.0))
    out = t(keypoints_30_17_3)
    assert np.all(out[:, :, :2] >= -2.0) and np.all(out[:, :, :2] <= 2.0)


# --- VelocityTransform ---


def test_velocity_output_shape(keypoints_30_17_3: np.ndarray) -> None:
    t = VelocityTransform(include_dconf=False)
    # input can be (T, 17, 2) or (T, 17, 3) after normalize
    kp = keypoints_30_17_3[:, :, :2]
    out = t(kp)
    assert out.shape == (WINDOW_SIZE, NUM_KEYPOINTS * 2)


def test_velocity_first_frame_zeros(keypoints_30_17_3: np.ndarray) -> None:
    t = VelocityTransform(include_dconf=False)
    kp = keypoints_30_17_3[:, :, :2].astype(np.float32)
    out = t(kp)
    assert np.all(out[0] == 0.0)


# --- AnglesTransform ---


def test_angles_output_shape(keypoints_30_17_3: np.ndarray) -> None:
    t = AnglesTransform()
    xy = keypoints_30_17_3[:, :, :2].astype(np.float32)
    out = t(xy)
    assert out.shape == (WINDOW_SIZE, NUM_ANGLES)


def test_angles_in_range(keypoints_30_17_3: np.ndarray) -> None:
    t = AnglesTransform()
    xy = keypoints_30_17_3[:, :, :2].astype(np.float32)
    out = t(xy)
    assert np.all(out >= 0.0) and np.all(out <= 1.0)
