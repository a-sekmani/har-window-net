"""Window contract: schema and validators for cloud-compatible window format."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator

# Fixed dimensions for cloud compatibility
WINDOW_SIZE = 30
NUM_KEYPOINTS = 17
KEYPOINT_DIM = 3  # x, y, conf


def _validate_keypoints_shape(v: Any) -> np.ndarray:
    """Ensure keypoints is numpy array with shape (T, K, 3)."""
    if isinstance(v, list):
        arr = np.array(v, dtype=np.float64)
    elif isinstance(v, np.ndarray):
        arr = np.asarray(v, dtype=np.float64)
    else:
        raise ValueError("keypoints must be list or ndarray")
    if arr.shape != (WINDOW_SIZE, NUM_KEYPOINTS, KEYPOINT_DIM):
        raise ValueError(
            f"keypoints shape must be ({WINDOW_SIZE}, {NUM_KEYPOINTS}, {KEYPOINT_DIM}), got {arr.shape}"
        )
    return arr


def _validate_keypoints_values(arr: np.ndarray) -> np.ndarray:
    """Check no NaN/Inf and x, y, conf in [0, 1]."""
    if not np.isfinite(arr).all():
        raise ValueError("keypoints must not contain NaN or Inf")
    if arr.min() < 0.0 or arr.max() > 1.0:
        raise ValueError("keypoints x, y, conf must be in [0, 1]")
    return arr


class WindowContract(BaseModel):
    """Window schema: one person, one camera, fixed-length frame sequence. Cloud-compatible."""

    model_config = {"arbitrary_types_allowed": True}

    id: str = Field(..., description="UUID of the window")
    device_id: str = Field(..., description="e.g. ntu-offline")
    camera_id: str = Field(..., description="e.g. ntu-cam")
    session_id: str = Field(..., description="UUID per video/sample")
    track_id: int = Field(1, description="usually 1 for NTU single person")
    ts_start_ms: int = Field(..., description="start time in ms (e.g. 0 for offline)")
    ts_end_ms: int = Field(..., description="end time in ms; use (window_size-1)*(1000/fps) for consistency")
    fps: float = Field(30.0, description="frames per second (float for cloud compatibility)")
    window_size: int = Field(30, description="number of frames T")
    mean_pose_conf: float = Field(..., ge=0.0, le=1.0, description="mean keypoint confidence")
    label: str = Field(..., description="activity label from NTU e.g. A001")
    label_source: Literal["dataset"] = Field("dataset", description="source of label")
    created_at: str = Field(..., description="ISO timestamp when built")
    keypoints: Any = Field(..., description="array [T][K][3] = (x, y, conf), normalized [0,1]")
    source_body_id: int | None = Field(None, description="NTU body index used (0=dominant); for multi-person traceability")

    @field_validator("id", "session_id")
    @classmethod
    def valid_uuid(cls, v: str) -> str:
        uuid.UUID(v)
        return v

    @field_validator("keypoints", mode="before")
    @classmethod
    def validate_keypoints(cls, v: Any) -> np.ndarray:
        arr = _validate_keypoints_shape(v)
        return _validate_keypoints_values(arr)

    def model_dump_for_storage(self) -> dict[str, Any]:
        """Serialize for Parquet/JSON: keypoints as nested list."""
        d = self.model_dump(mode="python")
        kp = d.get("keypoints")
        if isinstance(kp, np.ndarray):
            d["keypoints"] = kp.tolist()
        return d

    @classmethod
    def model_validate_from_storage(cls, obj: dict[str, Any]) -> WindowContract:
        """Deserialize from storage; keypoints may be list."""
        return cls.model_validate(obj)


def validate_window_dict(data: dict[str, Any]) -> list[str]:
    """
    Validate a window dict (e.g. from Parquet row) without building full Pydantic model.
    Returns list of error messages; empty if valid.
    fps may be int or float (cloud uses float). source_body_id is optional.
    """
    errors: list[str] = []
    required = {
        "id", "device_id", "camera_id", "session_id", "track_id",
        "ts_start_ms", "ts_end_ms", "fps", "window_size", "mean_pose_conf",
        "label", "label_source", "created_at", "keypoints",
    }
    for k in required:
        if k not in data:
            errors.append(f"missing field: {k}")
    if "label_source" in data and data["label_source"] != "dataset":
        errors.append("label_source must be 'dataset'")
    kp = data.get("keypoints")
    if kp is not None:
        try:
            arr = np.array(kp)
            if arr.shape != (WINDOW_SIZE, NUM_KEYPOINTS, KEYPOINT_DIM):
                errors.append(f"keypoints shape must be (30, 17, 3), got {arr.shape}")
            elif not np.isfinite(arr).all():
                errors.append("keypoints contain NaN or Inf")
            elif arr.min() < 0.0 or arr.max() > 1.0:
                errors.append("keypoints x, y, conf must be in [0, 1]")
        except Exception as e:
            errors.append(f"keypoints validation failed: {e}")
    return errors


def window_json_schema() -> dict[str, Any]:
    """Export JSON Schema for Window contract (e.g. for docs)."""
    return WindowContract.model_json_schema()


def make_window_id() -> str:
    return str(uuid.uuid4())


def make_created_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def ts_end_ms_from_window(window_size: int, fps: float, ts_start_ms: int = 0) -> int:
    """
    Compute ts_end_ms consistently for offline NTU: last frame end time.
    ts_end_ms = ts_start_ms + (window_size - 1) * (1000 / fps).
    """
    return ts_start_ms + int((window_size - 1) * (1000.0 / fps))
