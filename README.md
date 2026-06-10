# Image Classifier — Transfer Learning + Grad-CAM

Fine-tune EfficientNet-B0 on Oxford Flowers102 (102 flower species); evaluate with
confusion matrix and per-class F1; explain every prediction with Grad-CAM;
ship a drag-and-drop Gradio demo.

## Key results

| Metric | Value |
|---|---|
| Test top-1 accuracy | **83.9 %** |
| Test macro-F1 | **0.833** |
| Test weighted-F1 | 0.837 |
| Best val macro-F1 (epoch 9) | 0.862 |
| Dataset | Flowers102 — 1,020 train / 1,020 val / 6,149 test |
| Backbone | EfficientNet-B0 (timm, pretrained ImageNet) |
| Training | 5 head-only epochs + 5 fine-tune epochs, CPU |

Numbers come from `reports/test_metrics.json` and `reports/train_metrics.json`, both committed.

## Grad-CAM example

```
python -m vision.cam --image path/to/flower.jpg
# → reports/cam_<name>.png  (original + heatmap side-by-side)
```

The heatmap confirms the model attends to petals/stamens, not background — consistent
with what Grad-CAM should surface for a fine-grained species classifier.

## Architecture

```
image
  │
  ▼
OpenCV transforms (random crop, flip, colour jitter for train;
                   resize 256 → centre-crop 224 for eval)
  │
  ▼
EfficientNet-B0 (timm, pretrained)
  │  Phase 1: freeze backbone → train head  (5 epochs, lr=1e-3)
  │  Phase 2: unfreeze all   → fine-tune    (5 epochs, lr=1e-4)
  │  Save best checkpoint by val macro-F1
  ▼
102-class softmax head
  │
  ├─► top-k prediction + confidence
  └─► Grad-CAM heatmap (target layer: conv_head)
```

## Stack

Python 3.11 · PyTorch · timm · torchvision · pytorch-grad-cam · scikit-learn · Gradio · pytest · ruff

## Quickstart

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install timm "grad-cam" scikit-learn gradio pillow matplotlib
pip install -e .

# Train (downloads Flowers102 on first run, ~345 MB)
python -m vision.train

# Evaluate on test set
python -m vision.evaluate

# Grad-CAM for a single image
python -m vision.cam --image path/to/flower.jpg

# Gradio demo (http://localhost:7860)
python app.py
```

## Training curve

| Phase | Epoch | Train macro-F1 | Val macro-F1 |
|---|---|---|---|
| Head-only | 1 | 0.053 | 0.176 |
| Head-only | 2 | 0.443 | 0.461 |
| Head-only | 3 | 0.762 | 0.632 |
| Head-only | 4 | 0.881 | 0.694 |
| Head-only | 5 | 0.938 | 0.726 |
| Fine-tune | 6 | 0.954 | 0.814 |
| Fine-tune | 7 | 0.988 | 0.845 |
| Fine-tune | 8 | 1.000 | 0.849 |
| **Fine-tune** | **9** | **0.999** | **0.862** ✓ checkpoint |
| Fine-tune | 10 | — | — |

## Error analysis (where it fails)

The confusion matrix (`reports/confusion_matrix.png`) shows most errors cluster
between visually similar species — e.g., rose vs. lenten rose, or different dahlia
varieties. Root cause: 10 training images per class is very few for fine-grained
distinctions at the petal level. Remedies: more augmentation, test-time augmentation,
or a larger training split.

## Tests

```bash
pytest -q
# 9 passed
```

- `test_data.py` — transform shapes, eval determinism, no random aug at test time
- `test_model.py` — output shape, freeze/unfreeze param counts, checkpoint round-trip

## Dataset & licence

Oxford 102 Category Flower Dataset — used for non-commercial research (Maria-Elena Nilsback
and Andrew Zisserman, 2008). Model weights via timm (Apache-2.0). Code: MIT — see [LICENSE](LICENSE).
