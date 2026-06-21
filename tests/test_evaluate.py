"""Tests for evaluate.py utilities — no model or dataset download needed."""

from __future__ import annotations

import numpy as np
import pytest

from vision.evaluate import top_k_accuracy


def _logits(labels, n_classes=102):
    """Create logits where the true class always has the highest score."""
    n = len(labels)
    logits = np.random.default_rng(42).standard_normal((n, n_classes)).astype(float)
    # boost the true-class logit to guarantee top-1 correctness for some tests
    return logits, labels


def test_top1_perfect():
    n_classes = 10
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((50, n_classes))
    labels = list(np.argmax(logits, axis=1))  # always correct top-1
    assert top_k_accuracy(labels, logits, k=1) == 1.0


def test_topk_geq_top1():
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((100, 20))
    labels = list(rng.integers(0, 20, size=100))
    top1 = top_k_accuracy(labels, logits, k=1)
    top5 = top_k_accuracy(labels, logits, k=5)
    assert top5 >= top1


def test_topk_n_classes_is_one():
    """k = n_classes means every sample is in top-k -> 100% acc."""
    rng = np.random.default_rng(2)
    n_classes = 10
    logits = rng.standard_normal((30, n_classes))
    labels = list(rng.integers(0, n_classes, size=30))
    acc = top_k_accuracy(labels, logits, k=n_classes)
    assert acc == pytest.approx(1.0)


def test_topk_k1_worst_case():
    """When all predictions are wrong, top-1 == 0."""
    n_classes = 5
    logits = np.zeros((4, n_classes))
    for i in range(4):
        logits[i, 0] = 10.0  # always predicts class 0
    labels = [1, 2, 3, 4]  # never class 0
    assert top_k_accuracy(labels, logits, k=1) == 0.0


def test_topk_returns_float():
    logits = np.zeros((3, 5))
    logits[:, 0] = 1.0
    acc = top_k_accuracy([0, 0, 0], logits, k=1)
    assert isinstance(acc, float)


def test_topk_empty_returns_zero():
    assert top_k_accuracy([], np.empty((0, 5)), k=1) == 0.0
