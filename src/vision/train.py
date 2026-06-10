"""Transfer-learning training loop: freeze head → fine-tune.

Usage: python -m vision.train
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from vision.data import get_dataloaders, seed_everything
from vision.model import (
    CHECKPOINT_PATH,
    build_model,
    freeze_backbone,
    save_checkpoint,
    unfreeze_all,
)

ROOT = Path(__file__).parents[2]
METRICS_PATH = ROOT / "reports" / "train_metrics.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.set_grad_enabled(training):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(imgs)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, f1


def train(
    head_epochs: int = 5,
    finetune_epochs: int = 5,
    batch_size: int = 32,
    head_lr: float = 1e-3,
    finetune_lr: float = 1e-4,
    seed: int = 42,
) -> dict:
    seed_everything(seed)
    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)
    model = build_model(pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    best_f1, best_epoch = 0.0, 0

    # Phase 1: train head only
    freeze_backbone(model)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=head_lr)
    print(f"Phase 1: head-only ({head_epochs} epochs, lr={head_lr})")
    for ep in range(1, head_epochs + 1):
        tr_loss, tr_f1 = _epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_f1 = _epoch(model, val_loader, criterion, None, DEVICE)
        print(f"  ep{ep:02d}  tr_f1={tr_f1:.4f}  val_f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = ep
            save_checkpoint(model, best_f1)

    # Phase 2: fine-tune entire network
    unfreeze_all(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=finetune_lr)
    print(f"Phase 2: fine-tune ({finetune_epochs} epochs, lr={finetune_lr})")
    for ep in range(1, finetune_epochs + 1):
        tr_loss, tr_f1 = _epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_f1 = _epoch(model, val_loader, criterion, None, DEVICE)
        print(f"  ep{ep:02d}  tr_f1={tr_f1:.4f}  val_f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = head_epochs + ep
            save_checkpoint(model, best_f1)

    print(f"\nBest val macro-F1={best_f1:.4f} at epoch {best_epoch}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    METRICS_PATH.parent.mkdir(exist_ok=True)
    metrics = {"best_val_f1": round(best_f1, 4), "best_epoch": best_epoch}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train()
