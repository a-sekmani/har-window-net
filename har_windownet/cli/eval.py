"""CLI: evaluate checkpoint on a split and write reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from har_windownet.contracts.labels import load_label_map
from har_windownet.training.datasets import WindowDataset
from har_windownet.training.metrics import (
    accuracy,
    confusion_matrix,
    macro_f1,
    per_class_precision_recall,
)
from har_windownet.training.models import get_model


def _class_names_from_label_map(label_map: dict, num_classes: int) -> list[str]:
    """Build display names for axes: id_to_label (e.g. A001) or 'id_label' when id_to_name exists."""
    id_to_name = label_map.get("id_to_name") or {}
    label_to_id = label_map.get("label_to_id") or {}
    id_to_label = {str(v): k for k, v in label_to_id.items()}
    names = []
    for i in range(num_classes):
        sid = str(i)
        label_part = id_to_label.get(sid, sid)
        name_part = id_to_name.get(sid, "")
        if name_part:
            names.append(f"{label_part}_{name_part}")
        else:
            names.append(label_part)
    return names


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
    run_dir = ckpt_path.parent
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            run_config = json.load(f)
        feature_config = run_config.get("feature_config")
        data_from_config = run_config.get("data")
    else:
        run_config = None
        feature_config = None
        data_from_config = None

    data_dir = Path(args.data) if args.data else (Path(data_from_config) if data_from_config else None)
    if data_dir is None or not data_dir.exists():
        raise SystemExit(
            "Missing --data. Either pass --data <path> or use a checkpoint from a run that has config.json with 'data'."
        )

    if args.out is None:
        out_dir = run_dir / "reports"
    else:
        out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map = load_label_map(data_dir / "label_map.json")
    num_classes = label_map["num_classes"]
    class_names = _class_names_from_label_map(label_map, num_classes)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ckpt.get("num_classes") != num_classes:
        raise ValueError(
            f"Checkpoint num_classes ({ckpt.get('num_classes')}) != dataset num_classes ({num_classes})"
        )
    model_name = ckpt["model_name"]
    input_features = ckpt.get("input_features", 51)
    if feature_config is None:
        feature_config = ckpt.get("feature_config")

    model = get_model(model_name, num_classes=num_classes, input_features=input_features)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    ds = WindowDataset(data_dir, args.split, feature_config=feature_config)
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
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Accuracy: {acc:.4f}  Macro-F1: {f1:.4f}")
    print(f"Metrics saved to {metrics_path}")

    per_class_csv = out_dir / "per_class.csv"
    with open(per_class_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "label", "precision", "recall", "f1", "support"])
        for i in range(num_classes):
            w.writerow([
                i,
                class_names[i],
                per_class["precision"][i],
                per_class["recall"][i],
                per_class["f1"][i],
                per_class["support"][i],
            ])
    print(f"Per-class metrics saved to {per_class_csv}")

    # Class map: print to terminal and write CSV
    print("\nClass map:")
    for i in range(num_classes):
        print(f"  {i} -> {class_names[i]}")
    class_map_csv = out_dir / "class_map.csv"
    with open(class_map_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "label"])
        for i in range(num_classes):
            w.writerow([i, class_names[i]])
    print(f"Class map saved to {class_map_csv}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(max(8, num_classes * 0.5), max(6, num_classes * 0.4)))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar()
        plt.title(f"Confusion matrix ({args.split})")
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.xticks(range(num_classes), class_names, rotation=90)
        plt.yticks(range(num_classes), class_names)
        plt.tight_layout()
        plt.savefig(out_dir / "confusion_matrix.png", dpi=100, bbox_inches="tight")
        plt.close()
        print(f"Confusion matrix saved to {out_dir / 'confusion_matrix.png'}")
    except Exception as e:
        print(f"Could not save confusion matrix plot: {e}")


if __name__ == "__main__":
    main()
