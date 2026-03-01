"""Training loop: train/val, logging, checkpoints (best by val macro-F1, last)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from har_windownet.training.datasets import WindowDataset
from har_windownet.training.metrics import accuracy, macro_f1
from har_windownet.training.models import get_model


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n if n else 0.0


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        pred = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(pred.tolist())
        all_labels.extend(y.cpu().numpy().tolist())
    import numpy as np
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    avg_loss = total_loss / n if n else 0.0
    acc = accuracy(labels, preds)
    f1 = macro_f1(labels, preds, num_classes)
    return avg_loss, acc, f1


def run_training(
    data_root: str | Path,
    out_dir: str | Path,
    model_name: str = "tcn",
    batch_size: int = 64,
    epochs: int = 50,
    lr: float = 1e-3,
    seed: int = 42,
    device: str | None = None,
    feature_config: dict | None = None,
    use_class_weights: bool = False,
    label_smoothing: float = 0.0,
    use_lr_scheduler: bool = False,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    train_ds = WindowDataset(data_root, "train", feature_config=feature_config)
    val_ds = WindowDataset(data_root, "val", feature_config=feature_config)
    num_classes = train_ds.num_classes
    input_features = train_ds.input_features

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    model = get_model(
        model_name, num_classes=num_classes, input_features=input_features
    ).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Class weights for imbalanced datasets
    class_weights = None
    if use_class_weights:
        class_counts = train_ds.get_class_counts()
        class_counts_arr = np.array(class_counts, dtype=np.float32)
        class_counts_arr = np.maximum(class_counts_arr, 1.0)
        weights = 1.0 / class_counts_arr
        weights = weights / weights.sum() * num_classes
        class_weights = torch.tensor(weights, dtype=torch.float32).to(dev)
        print(f"Using class weights: {weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    if label_smoothing > 0:
        print(f"Using label smoothing: {label_smoothing}")

    # Learning rate scheduler
    scheduler = None
    if use_lr_scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        print("Using CosineAnnealingLR scheduler")

    best_f1 = -1.0
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, dev)
        val_loss, val_acc, val_f1 = eval_epoch(
            model, val_loader, criterion, dev, num_classes
        )
        print(
            f"Epoch {epoch}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}  val_macro_f1={val_f1:.4f}"
        )
        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_macro_f1": val_f1,
            "num_classes": num_classes,
            "model_name": model_name,
            "input_features": input_features,
            "feature_config": feature_config,
        }
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(ckpt_payload, out_dir / "best.ckpt")
        torch.save(ckpt_payload, out_dir / "last.ckpt")

        # Step the scheduler after each epoch
        if scheduler is not None:
            scheduler.step()
