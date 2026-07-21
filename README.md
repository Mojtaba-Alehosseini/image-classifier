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

> **Note on top-k accuracy:** For 102 classes, top-5 accuracy (was the true species among the 5 highest-confidence guesses?) is a standard supplementary metric. Use `evaluate.top_k_accuracy(labels, logits, k=5)` — implemented and tested.

## Grad-CAM example

```
python -m vision.cam --image path/to/flower.jpg
# → reports/cam_<name>.png  (original + heatmap side-by-side)
```

The heatmap confirms the model attends to petals and stamens rather than background, which is
what you'd want to see from a fine-grained species classifier.

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
| **Fine-tune** | **9** | **0.999** | **0.862** (checkpoint) |

Epoch 9 held the best validation macro-F1 at 0.862, so that's the checkpoint that ships;
`reports/train_metrics.json` records `best_epoch: 9`. Per-epoch rows above were captured from
the training log, and only the best-epoch summary is persisted to disk.

## Error analysis (where it fails)

The confusion matrix (`reports/confusion_matrix.png`) shows most errors clustering between
visually similar species: rose against lenten rose, or one dahlia variety against another.
The root cause is simply that 10 training images per class is very little to learn
petal-level distinctions from. More augmentation, test-time augmentation, or a larger
training split would all help.

## Tests

```bash
pytest -q
# 15 passed
```

- `test_data.py` covers transform shapes, eval determinism, and no random aug at test time
- `test_model.py` covers output shape, freeze/unfreeze param counts, checkpoint round-trip
- `test_evaluate.py` covers `top_k_accuracy`, hermetic (no model or dataset download)

## Dataset & licence

Oxford 102 Category Flower Dataset, used for non-commercial research (Maria-Elena Nilsback
and Andrew Zisserman, 2008). Model weights via timm (Apache-2.0). Code is MIT, see
[LICENSE](LICENSE).

---

Built by [Mojtaba Alehosseini](https://github.com/Mojtaba-Alehosseini), data scientist.
