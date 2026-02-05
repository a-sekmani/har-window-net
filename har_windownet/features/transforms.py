"""
Runtime feature transforms for skeleton windows [T, 17, 3] (x, y, conf in [0,1]).

Phase C: normalization, velocity, angles; applied during data loading.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

# COCO-17 indices (from har_windownet.datasets.ntu.mapping)
NUM_KEYPOINTS = 17
KEYPOINT_DIM = 3
# 0 nose, 1-4 head, 5 L_shoulder, 6 R_shoulder, 7 L_elbow, 8 R_elbow,
# 9 L_wrist, 10 R_wrist, 11 L_hip, 12 R_hip, 13 L_knee, 14 R_knee, 15 L_ankle, 16 R_ankle
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

RAW_FEATURES = NUM_KEYPOINTS * KEYPOINT_DIM  # 51
NUM_ANGLES = 10  # elbow L/R, knee L/R, shoulder-hip L/R, torso, shoulder L/R
EPS = 1e-8
DEFAULT_CLAMP = (-3.0, 3.0)


def _center_and_scale(
    kp: np.ndarray,
    norm_center: str,
    norm_scale: str,
    clamp_range: tuple[float, float],
) -> np.ndarray:
    """Center by hips/shoulders, scale by shoulder/hip distance, clamp. kp: [T, 17, 3]."""
    kp = np.asarray(kp, dtype=np.float64)
    xy = kp[:, :, :2].copy()
    conf = kp[:, :, 2:3]

    # Center: subtract midpoint of hips or shoulders
    if norm_center == "hips":
        center = (xy[:, L_HIP] + xy[:, R_HIP]) / 2
    elif norm_center == "shoulders":
        center = (xy[:, L_SHOULDER] + xy[:, R_SHOULDER]) / 2
    else:  # auto: use hips
        center = (xy[:, L_HIP] + xy[:, R_HIP]) / 2
    center = center[:, np.newaxis, :]
    xy = xy - center

    # Scale: divide by shoulder or hip distance
    if norm_scale == "shoulders":
        scale = np.linalg.norm(xy[:, L_SHOULDER] - xy[:, R_SHOULDER], axis=1, keepdims=True)
    elif norm_scale == "hips":
        scale = np.linalg.norm(xy[:, L_HIP] - xy[:, R_HIP], axis=1, keepdims=True)
    else:  # auto: shoulders
        scale = np.linalg.norm(xy[:, L_SHOULDER] - xy[:, R_SHOULDER], axis=1, keepdims=True)
    scale = np.maximum(scale, EPS)
    scale = scale[:, np.newaxis, :]
    xy = xy / scale

    lo, hi = clamp_range
    xy = np.clip(xy, lo, hi)
    out = np.concatenate([xy, conf], axis=2)
    return out.astype(np.float32)


def _apply_conf_mode(kp: np.ndarray, conf_mode: str) -> np.ndarray:
    """Apply conf_mode: keep, mask (x,y *= conf), or drop (return 17x2)."""
    if conf_mode == "keep":
        return kp
    if conf_mode == "mask":
        kp = kp.copy()
        kp[:, :, :2] *= kp[:, :, 2:3]
        return kp
    # drop: return only x, y
    return kp[:, :, :2].astype(np.float32)


class NormalizePoseTransform:
    """
    Center (hips/shoulders), scale by shoulder/hip distance, clamp, optional conf handling.
    Input/output: [T, 17, 3] or [T, 17, 2] if conf_mode=drop.
    """

    def __init__(
        self,
        norm_center: str = "auto",
        norm_scale: str = "auto",
        clamp_range: tuple[float, float] = DEFAULT_CLAMP,
        conf_mode: str = "keep",
    ) -> None:
        self.norm_center = norm_center
        self.norm_scale = norm_scale
        self.clamp_range = clamp_range
        self.conf_mode = conf_mode

    def __call__(self, kp: np.ndarray) -> np.ndarray:
        kp = _center_and_scale(kp, self.norm_center, self.norm_scale, self.clamp_range)
        return _apply_conf_mode(kp, self.conf_mode)


class VelocityTransform:
    """
    Per-frame dx, dy (and optional dconf). Frame 0: zeros.
    Input: [T, 17, C] (C=2 or 3). Output: [T, 17*2] (dx, dy only).
    """

    def __init__(self, include_dconf: bool = False) -> None:
        self.include_dconf = include_dconf

    def __call__(self, kp: np.ndarray) -> np.ndarray:
        kp = np.asarray(kp, dtype=np.float32)
        T = kp.shape[0]
        diff = np.zeros_like(kp)
        diff[1:] = kp[1:] - kp[:-1]
        # Frame 0: keep zeros (or could repeat diff[1])
        dx = diff[:, :, 0]   # [T, 17]
        dy = diff[:, :, 1]   # [T, 17]
        out = np.stack([dx, dy], axis=2)  # [T, 17, 2]
        if self.include_dconf and kp.shape[2] >= 3:
            dconf = diff[:, :, 2]
            out = np.concatenate([out, dconf[:, :, np.newaxis]], axis=2)
        return out.reshape(T, -1).astype(np.float32)


def _angle_at_b(a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Angle at b between vectors (b->a) and (b->c). a,b,c: [T, 2]. Returns [T] in [0,1] (rad/pi)."""
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba, axis=1, keepdims=True)
    nbc = np.linalg.norm(bc, axis=1, keepdims=True)
    nba = np.maximum(nba, eps)
    nbc = np.maximum(nbc, eps)
    cos_angle = np.sum((ba / nba) * (bc / nbc), axis=1)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return (angle_rad / np.pi).astype(np.float32)


class AnglesTransform:
    """
    2D joint angles from COCO-17 keypoints. Output [T, NUM_ANGLES] in [0,1].
    Angles: elbow L/R, knee L/R, shoulder-hip L/R, torso tilt, shoulder L/R.
    """

    def __call__(self, kp: np.ndarray) -> np.ndarray:
        kp = np.asarray(kp, dtype=np.float32)
        xy = kp[:, :, :2]
        T = kp.shape[0]

        def pt(j: int) -> np.ndarray:
            return xy[:, j, :]

        angles_list = [
            _angle_at_b(pt(L_SHOULDER), pt(L_ELBOW), pt(L_WRIST)),   # 0 elbow L
            _angle_at_b(pt(R_SHOULDER), pt(R_ELBOW), pt(R_WRIST)),   # 1 elbow R
            _angle_at_b(pt(L_HIP), pt(L_KNEE), pt(L_ANKLE)),         # 2 knee L
            _angle_at_b(pt(R_HIP), pt(R_KNEE), pt(R_ANKLE)),         # 3 knee R
            _angle_at_b(pt(L_SHOULDER), pt(L_HIP), pt(L_KNEE)),      # 4 shoulder-hip L
            _angle_at_b(pt(R_SHOULDER), pt(R_HIP), pt(R_KNEE)),      # 5 shoulder-hip R
            _angle_at_b(pt(L_HIP), pt(L_SHOULDER), pt(L_ELBOW)),     # 6 shoulder L
            _angle_at_b(pt(R_HIP), pt(R_SHOULDER), pt(R_ELBOW)),     # 7 shoulder R
        ]
        # 8: Torso tilt (angle of shoulder_center - hip_center with y-axis)
        shoulder_center = (pt(L_SHOULDER) + pt(R_SHOULDER)) / 2
        hip_center = (pt(L_HIP) + pt(R_HIP)) / 2
        v = shoulder_center - hip_center
        y_axis = np.zeros_like(v)
        y_axis[:, 1] = 1.0
        nv = np.linalg.norm(v, axis=1, keepdims=True)
        nv = np.maximum(nv, EPS)
        cos_tilt = np.sum((v / nv) * y_axis, axis=1)
        cos_tilt = np.clip(cos_tilt, -1.0, 1.0)
        torso_tilt = np.arccos(cos_tilt) / np.pi
        angles_list.append(torso_tilt.astype(np.float32))
        # 9: spine orientation
        spine = _angle_at_b(pt(L_HIP), pt(R_HIP), pt(R_SHOULDER))
        angles_list.append(spine)

        out = np.stack(angles_list, axis=1)
        assert out.shape == (T, NUM_ANGLES)
        return out


class ComposeTransforms:
    """Apply a list of transforms in order. Each receives/returns arrays; final output is concatenated by the pipeline."""

    def __init__(self, transforms: list[Callable[[np.ndarray], np.ndarray]]) -> None:
        self.transforms = transforms

    def __call__(self, kp: np.ndarray) -> np.ndarray:
        x = kp
        for t in self.transforms:
            x = t(x)
        return x


def build_feature_pipeline(config: dict[str, Any]) -> tuple[Callable[[np.ndarray], np.ndarray], int]:
    """
    Build a single callable that takes keypoints [T, 17, 3] and returns features [T, F].
    Returns (transform_fn, F).
    """
    features = config.get("features", "raw")
    conf_mode = config.get("conf_mode", "keep")
    norm_center = config.get("norm_center", "auto")
    norm_scale = config.get("norm_scale", "auto")
    clamp_range = tuple(config.get("clamp_range", list(DEFAULT_CLAMP)))

    if features == "raw":
        # No transform: flatten to [T, 51] (or 34 if we supported drop for raw; plan says raw = baseline so keep 51)
        def raw_fn(kp: np.ndarray) -> np.ndarray:
            kp = np.asarray(kp, dtype=np.float32)
            if conf_mode == "drop":
                kp = kp[:, :, :2]
            return kp.reshape(kp.shape[0], -1)

        F = get_input_features(config)
        return raw_fn, F

    # Normalize first when norm/vel/angles/combo
    norm_transform = NormalizePoseTransform(
        norm_center=norm_center,
        norm_scale=norm_scale,
        clamp_range=clamp_range,
        conf_mode=conf_mode,
    )

    if features == "norm":
        def fn(kp: np.ndarray) -> np.ndarray:
            out = norm_transform(kp)
            return out.reshape(out.shape[0], -1)
        return fn, get_input_features(config)

    if features == "vel":
        vel_transform = VelocityTransform(include_dconf=False)

        def fn(kp: np.ndarray) -> np.ndarray:
            normed = norm_transform(kp)
            vel = vel_transform(normed)
            norm_flat = normed.reshape(normed.shape[0], -1)
            return np.concatenate([norm_flat, vel], axis=1)
        return fn, get_input_features(config)

    if features == "angles":
        angles_transform = AnglesTransform()

        def fn(kp: np.ndarray) -> np.ndarray:
            normed = norm_transform(kp)
            ang = angles_transform(normed)
            norm_flat = normed.reshape(normed.shape[0], -1)
            return np.concatenate([norm_flat, ang], axis=1)
        return fn, get_input_features(config)

    # combo: norm + vel + angles
    vel_transform = VelocityTransform(include_dconf=False)
    angles_transform = AnglesTransform()

    def fn(kp: np.ndarray) -> np.ndarray:
        normed = norm_transform(kp)
        vel = vel_transform(normed)
        ang = angles_transform(normed)
        norm_flat = normed.reshape(normed.shape[0], -1)
        return np.concatenate([norm_flat, vel, ang], axis=1)

    return fn, get_input_features(config)


def get_input_features(config: dict[str, Any]) -> int:
    """
    Return the number of input features F for the given feature config.
    """
    features = config.get("features", "raw")
    conf_mode = config.get("conf_mode", "keep")

    if conf_mode == "drop":
        base_channels = NUM_KEYPOINTS * 2  # 34
    else:
        base_channels = RAW_FEATURES  # 51

    if features == "raw":
        return base_channels
    if features == "norm":
        return base_channels
    if features == "vel":
        return base_channels + NUM_KEYPOINTS * 2  # +34 for dx, dy
    if features == "angles":
        return base_channels + NUM_ANGLES
    if features == "combo":
        return base_channels + NUM_KEYPOINTS * 2 + NUM_ANGLES
    return RAW_FEATURES
