# TexTag: texture auto-tagger for game assets

**Capstone: image classification web app (PyTorch → Gradio → Hugging Face Spaces).**

## Problem

Game studios and artists collect thousands of texture files with unhelpful names
(`tex_final_v3.png`). TexTag looks at a texture image and tags it with what it looks like
(cracked, marbled, scaly, woven, ...), so asset libraries become searchable.

- **Type:** multi-class **classifier** (one label out of 47).
- **Input:** an RGB image of a texture or material surface.
- **Output:** the predicted texture class, top-3 probabilities, and the inference time.
- **Core challenge:** many classes look alike (dotted vs polka-dotted vs sprinkled; lined vs
  striped vs banded), and there are only 40 training images per class.

## Dataset

[Describable Textures Dataset (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/) via
`torchvision.datasets.DTD`: 5,640 images, 47 classes. Official partition 1:
1,880 train / 1,880 validation / 1,880 test (40 per class each).

## Plan

| Step | What |
|---|---|
| 1 | Data pipeline (`src/textag/data.py`): `DataLoader`, augmentation on train only |
| 2 | EDA notebook: class balance, sample images, image sizes |
| 3 | Baseline: small CNN from scratch (`SmallCNN`) |
| 4 | Main model: pretrained ResNet18 (transfer learning), `CrossEntropyLoss`, `AdamW`, LR scheduling, early stopping |
| 5 | Evaluation on the test set: accuracy, precision/recall/F1, confusion matrix |
| 6 | Inference with a timer (`src/textag/infer.py`) |
| 7 | Gradio app (`app.py`) and Hugging Face Space |

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
python app.py          # after training saves models/textag.pt
```
