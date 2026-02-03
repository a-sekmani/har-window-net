"""CLI: train model on Phase A windows."""

from __future__ import annotations

import argparse

from har_windownet.training.trainer import run_training


def main() -> None:
    p = argparse.ArgumentParser(description="Train HAR model on Phase A windows")
    p.add_argument("--data", required=True, help="Path to Phase A output (e.g. data_out/ntu120_windows)")
    p.add_argument("--out", required=True, help="Output dir for checkpoints (e.g. runs/exp01)")
    p.add_argument("--model", default="tcn", choices=["tcn", "gru"], help="Model name")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default=None, help="cuda or cpu")
    args = p.parse_args()
    run_training(
        data_root=args.data,
        out_dir=args.out,
        model_name=args.model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
