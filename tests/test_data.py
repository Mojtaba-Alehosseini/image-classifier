"""Tests for data transforms — no actual dataset download needed."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from vision.data import NUM_CLASSES, get_transforms


def _fake_img() -> Image.Image:
    arr = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_train_transform_output_shape():
    tf = get_transforms(train=True)
    out = tf(_fake_img())
    assert out.shape == (3, 224, 224)


def test_eval_transform_output_shape():
    tf = get_transforms(train=False)
    out = tf(_fake_img())
    assert out.shape == (3, 224, 224)


def test_eval_transform_is_deterministic():
    tf = get_transforms(train=False)
    img = _fake_img()
    t1 = tf(img)
    t2 = tf(img)
    assert torch.allclose(t1, t2)


def test_eval_transform_has_no_random_aug():
    """Eval transform must be deterministic (no RandomHorizontalFlip etc.)."""
    tf = get_transforms(train=False)
    img = _fake_img()
    results = [tf(img) for _ in range(10)]
    for r in results[1:]:
        assert torch.allclose(results[0], r)


def test_num_classes():
    assert NUM_CLASSES == 102
