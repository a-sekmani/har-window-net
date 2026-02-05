"""Export PyTorch checkpoint to ONNX and write model_meta.json."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

from har_windownet.contracts.window import NUM_KEYPOINTS, WINDOW_SIZE
from har_windownet.training.models import get_model

# Baseline F = 17 * 3 = 51
INPUT_FEATURES_DEFAULT = NUM_KEYPOINTS * 3


def export_to_onnx(
    checkpoint_path: str | Path,
    out_dir: str | Path,
    label_map_path: str | Path | None = None,
) -> None:
    """
    Load checkpoint, export model to ONNX, write model_meta.json and label_map.json.

    Uses input_features and feature_config from checkpoint when present (Phase C).
    If label_map_path is None, uses build_default_label_map from contracts.labels.
    """
    from har_windownet.contracts.labels import build_default_label_map, load_label_map, save_label_map

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    num_classes = ckpt["num_classes"]
    model_name = ckpt["model_name"]
    input_features = ckpt.get("input_features", INPUT_FEATURES_DEFAULT)
    feature_config = ckpt.get("feature_config")

    model = get_model(model_name, num_classes=num_classes, input_features=input_features)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.zeros(1, WINDOW_SIZE, input_features)
    onnx_path = out_dir / "model.onnx"
    # Use opset 18 to match current torch.onnx exporter; 14 triggers failed downgrade (axes adapter).
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["keypoints"],
        output_names=["logits"],
        dynamic_axes={"keypoints": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
    )

    meta = {
        "input_shape": [1, WINDOW_SIZE, input_features],
        "window_size": WINDOW_SIZE,
        "input_features": input_features,
        "fps": 30.0,
        "keypoint_order": "coco17",
        "features": ["x", "y", "conf"],
        "training_dataset": "ntu120",
        "num_classes": num_classes,
    }
    if feature_config is not None:
        meta["feature_spec"] = feature_config
    with open(out_dir / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if label_map_path and Path(label_map_path).exists():
        label_map = load_label_map(label_map_path)
    else:
        label_map = build_default_label_map()
    save_label_map(label_map, out_dir / "label_map.json")
