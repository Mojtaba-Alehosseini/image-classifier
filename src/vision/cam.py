"""Grad-CAM visualisation for a single image.

Uses pytorch_grad_cam (pip: grad-cam).

Usage:
  python -m vision.cam --image path/to/flower.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from vision.data import get_transforms
from vision.model import load_checkpoint

ROOT = Path(__file__).parents[3]
REPORTS = ROOT / "reports"


def compute_cam(image_path: Path, checkpoint: Path | None = None) -> tuple[np.ndarray, int, float]:
    """Return (heatmap_rgb uint8, predicted_class, confidence)."""
    from vision.model import CHECKPOINT_PATH

    ckpt = checkpoint or CHECKPOINT_PATH
    model = load_checkpoint(ckpt, device="cpu")

    tf = get_transforms(train=False)
    pil_img = Image.open(image_path).convert("RGB")
    tensor = tf(pil_img).unsqueeze(0)

    # EfficientNet-B0 target layer: last conv block
    target_layers = [model.conv_head]

    with GradCAM(model=model, target_layers=target_layers) as gcam:
        logits = model(tensor)
        pred_class = int(logits.argmax(1).item())
        confidence = float(torch.softmax(logits, dim=1)[0, pred_class].item())
        targets = [ClassifierOutputTarget(pred_class)]
        grayscale_cam = gcam(input_tensor=tensor, targets=targets)[0]

    # normalise input image for overlay
    img_np = np.array(pil_img.resize((224, 224))) / 255.0
    heatmap = show_cam_on_image(img_np.astype(np.float32), grayscale_cam, use_rgb=True)
    return heatmap, pred_class, confidence


def save_cam(
    image_path: Path, out_path: Path | None = None, checkpoint: Path | None = None
) -> Path:
    heatmap, pred_class, confidence = compute_cam(image_path, checkpoint)
    out = out_path or (REPORTS / f"cam_{image_path.stem}.png")
    out.parent.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(Image.open(image_path).resize((224, 224)))
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(heatmap)
    axes[1].set_title(f"Grad-CAM  class={pred_class}  conf={confidence:.2f}")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"Saved CAM to {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    save_cam(args.image, args.out)
