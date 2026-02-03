"""Offline inference: load ONNX, run on Window dict, return pred_label, pred_label_id, probs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from har_windownet.contracts.window import NUM_KEYPOINTS, WINDOW_SIZE

KEYPOINT_DIM = 3
INPUT_FEATURES = NUM_KEYPOINTS * KEYPOINT_DIM


def window_to_input(keypoints: Any) -> np.ndarray:
    """Convert keypoints (list or array) to (1, T, F) float32."""
    arr = np.array(keypoints, dtype=np.float32)
    if arr.shape != (WINDOW_SIZE, NUM_KEYPOINTS, KEYPOINT_DIM):
        raise ValueError(f"Expected keypoints (30, 17, 3), got {arr.shape}")
    arr = arr.reshape(1, WINDOW_SIZE, INPUT_FEATURES)
    return arr


class ONNXInference:
    """Load ONNX + metadata and run inference on Window dicts."""

    def __init__(self, model_path: str | Path, export_dir: str | Path | None = None) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for inference. Install with: pip install onnxruntime "
                "(or pip install -e \".[export-inference]\"). On Python 3.14 a wheel may not exist; use Python 3.11 or 3.12."
            ) from None

        self.model_path = Path(model_path)
        self.export_dir = Path(export_dir or self.model_path.parent)
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        with open(self.export_dir / "model_meta.json") as f:
            self.meta = json.load(f)
        with open(self.export_dir / "label_map.json") as f:
            self.label_map = json.load(f)
        self.id_to_name = self.label_map["id_to_name"]

    def predict(
        self, window: dict[str, Any], return_probs: bool = True
    ) -> dict[str, Any]:
        """
        Run inference on one Window dict. Returns pred_label, pred_label_id, and optionally probs.
        """
        keypoints = window["keypoints"]
        x = window_to_input(keypoints)
        input_name = self.session.get_inputs()[0].name
        logits = self.session.run(None, {input_name: x})[0][0]
        pred_id = int(np.argmax(logits))
        pred_label = self._id_to_label(pred_id)
        out = {"pred_label": pred_label, "pred_label_id": pred_id}
        if return_probs:
            exp = np.exp(logits - logits.max())
            probs = (exp / exp.sum()).tolist()
            out["probs"] = probs
        return out

    def _id_to_label(self, label_id: int) -> str:
        """Resolve label_id to NTU label string (e.g. A001)."""
        label_to_id = self.label_map.get("label_to_id", {})
        for k, v in label_to_id.items():
            if v == label_id:
                return k
        return f"A{label_id + 1:03d}"


def run_inference_cli() -> None:
    """CLI: --model path/model.onnx --window path/sample_windows.json [--export-dir]"""
    import argparse

    p = argparse.ArgumentParser(description="Run ONNX inference on Window JSON")
    p.add_argument("--model", required=True, help="Path to model.onnx")
    p.add_argument("--window", required=True, help="Path to window JSON or sample_*.json (list)")
    p.add_argument("--export-dir", default=None, help="Dir with model_meta.json and label_map.json (default: model dir)")
    p.add_argument("--no-probs", action="store_true", help="Do not output probs")
    args = p.parse_args()

    inf = ONNXInference(args.model, export_dir=args.export_dir)
    window_path = Path(args.window)
    with open(window_path) as f:
        data = json.load(f)

    if isinstance(data, list):
        windows = data
    else:
        windows = [data]

    for i, w in enumerate(windows):
        out = inf.predict(w, return_probs=not args.no_probs)
        print(f"Window {i}: pred_label={out['pred_label']} pred_label_id={out['pred_label_id']}")
        if "probs" in out and not args.no_probs:
            top3 = np.argsort(out["probs"])[-3:][::-1]
            print(f"  top3: {[(inf._id_to_label(j), out['probs'][j]) for j in top3]}")


if __name__ == "__main__":
    run_inference_cli()
