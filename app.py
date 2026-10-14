"""TexTag web app: upload a texture, get the top-3 pattern tags and the inference time."""

import sys
from pathlib import Path

import gradio as gr
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from textag.infer import load_model, predict  # noqa: E402

# Load the model once at startup, then make one warm-up prediction
model, classes = load_model(ROOT / "models" / "textag.pt")
predict(model, classes, Image.new("RGB", (256, 256)))


def classify(image):
    """Gradio function: image in -> ({label: probability} for the top 3, inference time)."""
    if image is None:
        return {}, ""
    preds, seconds = predict(model, classes, image, top_k=3)
    return {label: prob for label, prob in preds}, f"{seconds:.3f} s"


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Texture image"),
    outputs=[
        gr.Label(num_top_classes=3, label="Top 3 tags"),
        gr.Textbox(label="Inference time (seconds)"),
    ],
    title="TexTag: texture auto-tagger",
    description=(
        "Upload a texture (or click an example) to get its top 3 pattern tags out of 47, "
        "e.g. cracked, woven, scaly, marbled, honeycombed. "
        "Model: ResNet18 fine-tuned on the Describable Textures Dataset (DTD). "
        "Test set: 62% exact, 83% when counting the top 3."
    ),
    examples=sorted(str(p) for p in (ROOT / "examples").glob("*.jpg")),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
