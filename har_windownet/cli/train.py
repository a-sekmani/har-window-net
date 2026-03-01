"""CLI: train model on Phase A windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from har_windownet.training.trainer import run_training


def main() -> None:
    p = argparse.ArgumentParser(description="Train HAR model on Phase A windows")
    p.add_argument("--data", required=True, help="Path to Phase A output (e.g. data_out/ntu120_windows)")
    p.add_argument("--out", required=True, help="Output dir for checkpoints (e.g. runs/exp01)")
    p.add_argument("--model", default="tcn", choices=["tcn", "gru"], help="Model name")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--device", default=None, help="cuda or cpu")
    p.add_argument(
        "--features",
        default="raw",
        choices=["raw", "norm", "vel", "angles", "combo"],
        help="Input feature set",
    )
    p.add_argument(
        "--conf-mode",
        default="keep",
        choices=["keep", "drop"],
        help="Confidence: keep (x,y,conf) or drop (x,y only)",
    )
    p.add_argument("--norm-center", default="auto", help="Normalize center (e.g. auto, midhip)")
    p.add_argument("--norm-scale", default="auto", help="Normalize scale (e.g. auto, fixed)")
    p.add_argument(
        "--class-weights",
        action="store_true",
        help="Use inverse class frequency weights for imbalanced datasets",
    )
    p.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor (0.0 = no smoothing, 0.1 = recommended)",
    )
    p.add_argument(
        "--lr-scheduler",
        action="store_true",
        help="Use CosineAnnealingLR scheduler",
    )
    args = p.parse_args()

    feature_config = {
        "features": args.features,
        "conf_mode": args.conf_mode,
        "norm_center": args.norm_center,
        "norm_scale": args.norm_scale,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "config.json"
    run_config = {
        "data": args.data,
        "out": str(out_dir),
        "model": args.model,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "device": args.device,
        "feature_config": feature_config,
        "class_weights": args.class_weights,
        "label_smoothing": args.label_smoothing,
        "lr_scheduler": args.lr_scheduler,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    run_training(
        data_root=args.data,
        out_dir=args.out,
        model_name=args.model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        feature_config=feature_config,
        use_class_weights=args.class_weights,
        label_smoothing=args.label_smoothing,
        use_lr_scheduler=args.lr_scheduler,
    )


if __name__ == "__main__":
    main()
