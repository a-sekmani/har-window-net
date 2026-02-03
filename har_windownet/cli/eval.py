"""CLI: evaluate checkpoint on a split and write reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from har_windownet.training.datasets import WindowDataset
from har_windownet.training.metrics import (
    accuracy,
    confusion_matrix,
    macro_f1,
    per_class_precision_recall,
)
from har_windownet.training.models import get_model


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate model on Phase A split")
    p.add_argument("--data", required=True, help="Path to Phase A output dir")
    p.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--out", default=None, help="Reports dir (default: same as checkpoint dir)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    if args.out is None:
        out_dir = ckpt_path.parent / "reports"
    else:
        out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_classes = ckpt["num_classes"]
    model_name = ckpt["model_name"]

    model = get_model(model_name, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    ds = WindowDataset(args.data, args.split)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(pred.tolist())
            all_labels.extend(y.numpy().tolist())

    import numpy as np
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    labels = list(range(num_classes))

    acc = accuracy(y_true, y_pred)
    f1 = macro_f1(y_true, y_pred, num_classes)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = per_class_precision_recall(y_true, y_pred, labels=labels)

    metrics = {
        "split": args.split,
        "accuracy": acc,
        "macro_f1": f1,
        "per_class": per_class,
    }
    metrics_path = out_dir / f"{args.split}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Accuracy: {acc:.4f}  Macro-F1: {f1:.4f}")
    print(f"Metrics saved to {metrics_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar()
        plt.title(f"Confusion matrix ({args.split})")
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.savefig(out_dir / "confusion_matrix.png", dpi=100, bbox_inches="tight")
        plt.close()
        print(f"Confusion matrix saved to {out_dir / 'confusion_matrix.png'}")
    except Exception as e:
        print(f"Could not save confusion matrix plot: {e}")


if __name__ == "__main__":
    main()
