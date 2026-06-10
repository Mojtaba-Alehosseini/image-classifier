"""Gradio demo: upload a flower image → top-5 predictions + Grad-CAM heatmap.

Usage: python app.py
"""

from __future__ import annotations

import gradio as gr
import torch
from PIL import Image

from vision.data import get_transforms
from vision.model import CHECKPOINT_PATH, load_checkpoint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Flowers102 class names (official Oxford ordering, 1-indexed → 0-indexed offset)
# Brief readable names for the Gradio interface
FLOWER_NAMES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea",
    "english marigold", "tiger lily", "moon orchid", "bird of paradise",
    "monkshood", "globe thistle",
    "snapdragon", "colt's foot", "king protea", "spear thistle", "yellow iris",
    "globe flower", "purple coneflower", "peruvian lily", "balloon flower", "giant white arum lily",
    "fire lily", "pincushion flower", "fritillary", "red ginger", "grape hyacinth",
    "corn poppy", "prince of wales feathers", "stemless gentian", "artichoke", "sweet william",
    "carnation", "garden phlox", "love in the mist", "mexican aster", "alpine sea holly",
    "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip", "lenten rose",
    "barbeton daisy", "daffodil", "sword lily", "poinsettia", "bolero deep blue",
    "wallflower", "marigold", "buttercup", "oxeye daisy", "common dandelion",
    "petunia", "wild pansy", "primula", "sunflower", "pelargonium",
    "bishop of llandaff", "gaura", "geranium", "orange dahlia", "pink-yellow dahlia",
    "cautleya spicata", "japanese anemone", "black-eyed susan", "silverbush", "californian poppy",
    "osteospermum", "spring crocus", "bearded iris", "windflower", "tree poppy",
    "gazania", "azalea", "water lily", "rose", "thorn apple",
    "morning glory", "passion flower", "lotus", "toad lily", "anthurium",
    "frangipani", "clematis", "hibiscus", "columbine", "desert-rose",
    "tree mallow", "magnolia", "cyclamen", "watercress", "canna lily",
    "hippeastrum", "bee balm", "pink quill", "foxglove", "bougainvillea",
    "camellia", "mallow", "mexican petunia", "bromelia", "blanket flower",
    "trumpet creeper", "blackberry lily",
]


def _load_model():
    if not CHECKPOINT_PATH.exists():
        return None
    return load_checkpoint(CHECKPOINT_PATH, device=DEVICE)


_model = _load_model()


def predict(image: Image.Image) -> tuple[dict, Image.Image]:
    if _model is None:
        return {"error": "No checkpoint found — run python -m vision.train first"}, image

    tf = get_transforms(train=False)
    tensor = tf(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top5 = probs.topk(5)
    label_scores = {
        FLOWER_NAMES[idx]: float(score)
        for idx, score in zip(top5.indices.tolist(), top5.values.tolist())
    }

    # Grad-CAM heatmap
    try:
        import numpy as np
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        pred_class = int(probs.argmax().item())
        target_layers = [_model.conv_head]
        with GradCAM(model=_model, target_layers=target_layers) as gcam:
            grayscale_cam = gcam(
                input_tensor=tensor, targets=[ClassifierOutputTarget(pred_class)]
            )[0]

        img_resized = image.resize((224, 224))
        img_np = np.array(img_resized) / 255.0
        heatmap = show_cam_on_image(img_np.astype(np.float32), grayscale_cam, use_rgb=True)
        cam_image = Image.fromarray(heatmap)
    except Exception:
        cam_image = image

    return label_scores, cam_image


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a flower image"),
    outputs=[
        gr.Label(num_top_classes=5, label="Top-5 predictions"),
        gr.Image(label="Grad-CAM heatmap"),
    ],
    title="Flowers102 Classifier",
    description=(
        "EfficientNet-B0 fine-tuned on Oxford Flowers102 (102 flower species). "
        "Upload any flower photo to see the top-5 predictions "
        "and where the model is looking (Grad-CAM)."
    ),
    examples=[],
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
