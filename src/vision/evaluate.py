"""Evaluate best checkpoint on the test set.

Produces:
  reports/test_metrics.json  — top-1 acc, macro-F1, weighted-F1
  reports/confusion_matrix.png

Usage: python -m vision.evaluate
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from vision.data import get_dataloaders
from vision.model import CHECKPOINT_PATH, load_checkpoint


def top_k_accuracy(labels: list[int], logits: np.ndarray, k: int = 5) -> float:
    """Fraction of samples where the true class is in the top-k predicted classes.

    A fairer secondary metric for 102-class flower classification — knowing the right
    species is 'plausible' (in top-5 guesses) is meaningful even when top-1 fails.
    """
    top_k_preds = np.argsort(logits, axis=1)[:, -k:]  # shape (N, k), ascending sort
    correct = sum(
        int(label) in top_k_preds[i].tolist()
        for i, label in enumerate(labels)
    )
    return correct / len(labels) if labels else 0.0

ROOT = Path(__file__).parents[2]
REPORTS = ROOT / "reports"


def evaluate(checkpoint: Path = CHECKPOINT_PATH) -> dict:
    REPORTS.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_checkpoint(checkpoint, device=device)

    _, _, test_loader = get_dataloaders(batch_size=64)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    top1 = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    metrics = {
        "top1_accuracy": round(top1, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "n_test": len(all_labels),
        "n_classes": 102,
    }
    (REPORTS / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Test  top1={top1:.4f}  macro-F1={macro_f1:.4f}  weighted-F1={weighted_f1:.4f}")

    # per-class report (top 10 worst classes logged)
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    (REPORTS / "classification_report.json").write_text(json.dumps(report, indent=2))

    # confusion matrix (compact: only show if ≤20 classes, else skip viz)
    cm = confusion_matrix(all_labels, all_preds)
    np.save(REPORTS / "confusion_matrix.npy", cm)
    if cm.shape[0] <= 20:
        fig, ax = plt.subplots(figsize=(12, 10))
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        fig.tight_layout()
        fig.savefig(REPORTS / "confusion_matrix.png", dpi=100)
        plt.close(fig)
    else:
        # for 102 classes draw a heat-map without ticks
        fig, ax = plt.subplots(figsize=(14, 12))
        ax.imshow(cm, aspect="auto", cmap="Blues")
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title("Confusion matrix (Flowers102 test set)")
        fig.tight_layout()
        fig.savefig(REPORTS / "confusion_matrix.png", dpi=100)
        plt.close(fig)

    return metrics


if __name__ == "__main__":
    evaluate()
