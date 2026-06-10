"""EfficientNet-B0 backbone via timm with a new classification head."""

from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn

from vision.data import NUM_CLASSES

ROOT = Path(__file__).parents[2]
CHECKPOINT_PATH = ROOT / "models" / "best.pt"


def build_model(pretrained: bool = True, num_classes: int = NUM_CLASSES) -> nn.Module:
    model = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=num_classes)
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except the classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def save_checkpoint(model: nn.Module, val_f1: float, path: Path = CHECKPOINT_PATH) -> None:
    path.parent.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "val_f1": val_f1}, path)


def load_checkpoint(path: Path = CHECKPOINT_PATH, device: str = "cpu") -> nn.Module:
    model = build_model(pretrained=False)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
