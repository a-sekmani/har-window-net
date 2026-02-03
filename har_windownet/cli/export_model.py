"""CLI: export checkpoint to ONNX + model_meta.json + label_map.json."""

from __future__ import annotations

import argparse

from har_windownet.export.onnx_export import export_to_onnx


def main() -> None:
    p = argparse.ArgumentParser(description="Export model to ONNX")
    p.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    p.add_argument("--out", required=True, help="Output dir (e.g. runs/exp01/export)")
    p.add_argument(
        "--label-map",
        default=None,
        help="Path to label_map.json (or use --data to point to Phase A dir)",
    )
    p.add_argument(
        "--data",
        default=None,
        help="Phase A data root; if set, use data/label_map.json",
    )
    args = p.parse_args()
    label_map_path = args.label_map
    if label_map_path is None and args.data:
        from pathlib import Path
        label_map_path = Path(args.data) / "label_map.json"
    export_to_onnx(
        checkpoint_path=args.checkpoint,
        out_dir=args.out,
        label_map_path=label_map_path,
    )
    print(f"Exported to {args.out}")


if __name__ == "__main__":
    main()
