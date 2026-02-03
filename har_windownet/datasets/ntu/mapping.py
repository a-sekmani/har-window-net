"""
Mapping NTU RGB+D 25 joints (Kinect V2 order) → COCO-17 keypoints.

NTU/Kinect V2 order (0-based): SpineBase(0), SpineMid(1), Neck(2), Head(3),
ShoulderLeft(4), ElbowLeft(5), WristLeft(6), HandLeft(7), ShoulderRight(8),
ElbowRight(9), WristRight(10), HandRight(11), HipLeft(12), KneeLeft(13),
AnkleLeft(14), FootLeft(15), HipRight(16), KneeRight(17), AnkleRight(18),
FootRight(19), SpineShoulder(20), HandTipLeft(21), ThumbLeft(22),
HandTipRight(23), ThumbRight(24).

COCO-17 order: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow,
9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip,
13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle.

Where NTU has no direct joint (nose, eyes, ears), we use Head(3) as proxy.
Confidence for those can be 0 if Head is not tracked; some actions depend on head.
"""

from __future__ import annotations

import numpy as np

# Kinect V2 trackingState: 0=NotTracked, 1=Inferred, 2=Tracked
TRACKING_NOT_TRACKED = 0
TRACKING_INFERRED = 1
TRACKING_TRACKED = 2


def tracking_state_to_confidence(tracking_state: int) -> float:
    """
    Map Kinect V2 joint trackingState to confidence in [0, 1].

    - 2 (Tracked): 1.0
    - 1 (Inferred): 0.5
    - 0 (NotTracked): 0.0
    """
    if tracking_state == TRACKING_TRACKED:
        return 1.0
    if tracking_state == TRACKING_INFERRED:
        return 0.5
    return 0.0


# COCO-17 index -> NTU 25 joint index (0..24). None = use fallback (Head).
NTU25_TO_COCO17: list[int | None] = [
    3,   # 0: nose -> Head
    3,   # 1: left_eye -> Head
    3,   # 2: right_eye -> Head
    3,   # 3: left_ear -> Head
    3,   # 4: right_ear -> Head
    4,   # 5: left_shoulder -> ShoulderLeft
    8,   # 6: right_shoulder -> ShoulderRight
    5,   # 7: left_elbow -> ElbowLeft
    9,   # 8: right_elbow -> ElbowRight
    6,   # 9: left_wrist -> WristLeft
    10,  # 10: right_wrist -> WristRight
    12,  # 11: left_hip -> HipLeft
    16,  # 12: right_hip -> HipRight
    13,  # 13: left_knee -> KneeLeft
    17,  # 14: right_knee -> KneeRight
    14,  # 15: left_ankle -> AnkleLeft
    18,  # 16: right_ankle -> AnkleRight
]

NUM_COCO_KEYPOINTS = 17
NUM_NTU_JOINTS = 25


def map_ntu_frame_to_coco17(
    ntu_joints: np.ndarray,
    *,
    use_xy_from: str = "3d",
    default_conf: float = 1.0,
    ntu_tracking_state: np.ndarray | None = None,
) -> np.ndarray:
    """
    Map one frame from NTU (25, 3) or (25, 4+) to COCO-17 (17, 3) = (x, y, conf).

    Confidence: if ntu_tracking_state (25,) is provided, use tracking_state_to_confidence
    per joint; else if ntu_joints has 4th column use it (clipped); else default_conf.
    Nose/ears/eyes use Head(3) proxy; their conf can be 0 if Head not tracked.

    Parameters
    ----------
    ntu_joints : ndarray
        Shape (25, 3) for (x,y,z) or (25, 4+) where [:,3] can be confidence.
    use_xy_from : str
        Unused here; preprocess chooses rgb/depth/3d.
    default_conf : float
        Used when no tracking_state and no 4th column (e.g. .npy without conf).
    ntu_tracking_state : ndarray
        Optional (25,) int: 0=NotTracked, 1=Inferred, 2=Tracked. Overrides 4th column.

    Returns
    -------
    ndarray
        Shape (17, 3), dtype float64: (x, y, confidence). x,y not normalized.
    """
    ntu_joints = np.asarray(ntu_joints, dtype=np.float64)
    if ntu_joints.shape[0] != NUM_NTU_JOINTS:
        raise ValueError(f"ntu_joints must have 25 rows, got {ntu_joints.shape[0]}")
    if ntu_joints.shape[1] < 3:
        raise ValueError("ntu_joints must have at least 3 columns (x,y,z)")

    out = np.zeros((NUM_COCO_KEYPOINTS, 3), dtype=np.float64)
    for c in range(NUM_COCO_KEYPOINTS):
        idx = NTU25_TO_COCO17[c]
        if idx is not None:
            out[c, 0] = ntu_joints[idx, 0]
            out[c, 1] = ntu_joints[idx, 1]
            if ntu_tracking_state is not None:
                out[c, 2] = tracking_state_to_confidence(int(ntu_tracking_state[idx]))
            elif ntu_joints.shape[1] >= 4 and np.isfinite(ntu_joints[idx, 3]):
                out[c, 2] = float(np.clip(ntu_joints[idx, 3], 0.0, 1.0))
            else:
                out[c, 2] = default_conf
        else:
            out[c, 2] = 0.0
    return out


def map_ntu_sequence_to_coco17(
    ntu_frames: np.ndarray,
    *,
    ntu_tracking_state: np.ndarray | None = None,
    **kwargs: float | str,
) -> np.ndarray:
    """
    Map a sequence of frames (N, 25, 3) or (N, 25, 4) to (N, 17, 3).

    ntu_tracking_state: optional (N, 25) int per frame; passed per frame to map_ntu_frame_to_coco17.
    """
    ntu_frames = np.asarray(ntu_frames)
    n = ntu_frames.shape[0]
    out = np.zeros((n, NUM_COCO_KEYPOINTS, 3), dtype=np.float64)
    for i in range(n):
        ts = ntu_tracking_state[i] if ntu_tracking_state is not None else None
        out[i] = map_ntu_frame_to_coco17(ntu_frames[i], ntu_tracking_state=ts, **kwargs)
    return out
