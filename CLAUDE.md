# Project: Image classification with transfer learning

## What this is
Fine-tune a pretrained timm backbone (EfficientNet-B0) on Oxford Flowers102 (102 classes).
Evaluate with confusion matrix + per-class F1; add Grad-CAM; ship a Gradio demo.

## Stack
Python 3.11 · PyTorch · timm · torchvision · scikit-learn · pytorch-grad-cam · Gradio · pytest · ruff.

## Commands
- Train: `python -m vision.train`
- Eval: `python -m vision.evaluate`   (after `pip install -e .`)
- Demo: `python app.py`
- Tests: `pytest -q`
- Lint: `ruff check .`

## Conventions
- Seeds: torch.manual_seed(42), np.random.seed(42), random.seed(42).
- Phase 1: freeze backbone, train head 5 epochs lr=1e-3. Phase 2: unfreeze, fine-tune 5 epochs lr=1e-4.
- Save best checkpoint by val macro-F1.
- grad-cam package pip name is `grad-cam`; imports as `pytorch_grad_cam`.
- Package layout: src/ with pip install -e .; modules as `python -m vision.train`.

## Done per step
Verify command passes → commit.
