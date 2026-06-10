"""Tests for model construction and inference shape."""

from __future__ import annotations

import torch

from vision.model import build_model, freeze_backbone, unfreeze_all


def test_model_output_shape():
    model = build_model(pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 102)


def test_freeze_backbone_reduces_params():
    model = build_model(pretrained=False)
    freeze_backbone(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    assert trainable < total
    assert trainable > 0


def test_unfreeze_all_restores_params():
    model = build_model(pretrained=False)
    freeze_backbone(model)
    unfreeze_all(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    assert trainable == total


def test_checkpoint_roundtrip(tmp_path):
    from vision.model import load_checkpoint, save_checkpoint

    model = build_model(pretrained=False)
    ckpt = tmp_path / "test.pt"
    save_checkpoint(model, val_f1=0.5, path=ckpt)

    loaded = load_checkpoint(path=ckpt, device="cpu")
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = loaded(x)
    assert out.shape == (1, 102)
