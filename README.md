# TexTag: texture auto-tagger for game assets

Upload a texture and TexTag tags it by what it looks like (cracked, woven, scaly, marbled, honeycombed, ...),
with its top 3 guesses and the inference time. Built with PyTorch and Gradio.

**Demo:** `python app.py --share` starts the app and prints a temporary public link (`https://….gradio.live`, valid 72 hours). See [Run locally](#run-locally).

![TexTag app](docs/app.png)

## Problem

Texture libraries pile up thousands of files with unhelpful names (`tex_final_v3.png`, `T_Floor_01_D.png`).
Searching them by name only works if someone named them well. TexTag looks at the image itself and tags it,
so textures can be found by what they look like.

- **Type:** multi-class image classifier (1 of 47 texture patterns)
- **Input:** an RGB texture image
- **Output:** top 3 tags with probabilities, plus inference time in seconds
- **Challenge:** many classes look alike (dotted vs polka-dotted, pitted vs porous), and there are only 40 training images per class

## Dataset

[Describable Textures Dataset (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/), loaded with `torchvision.datasets.DTD`:
5,640 images, 47 classes, official partition 1 = 1,880 train / 1,880 validation / 1,880 test (40 per class each).
EDA: perfectly balanced, no corrupted files, all RGB, sizes mostly 300–640 px ([01_eda](notebooks/01_eda.ipynb)).

## Approach

| Step | Notebook / code | What |
|---|---|---|
| 1 | `src/textag/data.py` | `DataLoader`s; augmentation on training images only (random crop, flips, rotation, color jitter); resize 256 → crop 224; ImageNet normalization |
| 2 | [01_eda](notebooks/01_eda.ipynb) | Class balance, sample grid, look-alike classes, image sizes, color stats |
| 3 | [02_baseline](notebooks/02_baseline.ipynb) | **SmallCNN** from scratch: 4 conv blocks + 2 dense layers, 428k weights (VGG-style, an extended TinyVGG) |
| 4 | [03_resnet](notebooks/03_resnet.ipynb) | **ResNet18 transfer learning:** phase 1 trains only a new last layer (frozen backbone), phase 2 fine-tunes all layers with a 10× smaller learning rate |
| 5 | [04_evaluate](notebooks/04_evaluate.ipynb) | Test set: accuracy, top-3, precision/recall/F1, confusion matrix, mistakes |
| 6 | [05_inference](notebooks/05_inference.ipynb), `src/textag/infer.py` | Single-image prediction timed with `time.perf_counter()` |
| 7 | `app.py` | Gradio app: image in → top 3 tags + inference time |
| 8 | [06_clip](notebooks/06_clip.ipynb) | **Model 3, CLIP:** zero-shot (no training) and a linear probe (new last layer on frozen CLIP), plus text search |

**Training setup** (same for every model, so the comparison is fair): `CrossEntropyLoss` (standard for multi-class),
`AdamW` (weight decay 1e-4), batch size 32, `ReduceLROnPlateau` (halve the LR when validation loss stalls for 3 epochs),
early stopping on validation loss (keep the best epoch). Training ran on a Google Colab T4 GPU (`src/textag/engine.py`).

| Hyperparameter | SmallCNN | ResNet18 phase 1 | ResNet18 phase 2 |
|---|---|---|---|
| Learning rate | 1e-3 | 1e-3 | 1e-4 |
| Max epochs / early-stopping patience | 40 / 7 | 15 / 5 | 20 / 5 |
| Dropout | 0.3 | 0.3 | 0.3 |

These are standard defaults rather than a grid search: each run takes ~4–18 minutes on a GPU, and early stopping
plus the LR scheduler tune the number of epochs and the learning rate automatically.

## Results

| Model | Trainable weights | Val accuracy |
|---|---|---|
| Random guessing | – | 2.1% |
| SmallCNN (from scratch, 40 epochs) | 428k | 28.4% |
| ResNet18 phase 1 (feature extractor) | 24k | 56.2% |
| **ResNet18 phase 2 (fine-tuned)** | 11.2M | **62.4%** |
| CLIP ViT-B/32 zero-shot (no training) | 0 | 43.9% |
| CLIP ViT-B/32 linear probe | 24k | 74.2% |

**Final model on the test set** (never used for training or model selection):

| Metric | Test |
|---|---|
| Accuracy | **62.3%** |
| Top-3 accuracy | **82.6%** |
| Macro precision / recall / F1 | 63.4% / 62.3% / 62.0% |
| Inference time (CPU, 2 cores) | ~0.04–0.05 s per image |

**Key findings**

- **Transfer learning more than doubles the from-scratch baseline** (28.4% → 62.4%). With 40 images per class there isn't enough data to learn good filters from scratch.
- **Early stopping mattered:** fine-tuning peaked at epoch 3, then train accuracy climbed to 90% while validation loss rose (overfitting). Early stopping kept the epoch-3 weights.
- **Test ≈ validation** (62.3% vs 62.4%): picking the best epoch on validation didn't inflate the score.
- **Top-3 is 82.6%:** the right tag is among the 3 suggestions 4 times out of 5. Mistakes are usually look-alikes (dotted ↔ polka-dotted is the biggest confusion), so for search and tagging they're still useful results.
- **Best classes** are regular patterns (chequered, freckled, paisley); **worst** are vague ones (blotchy, marbled, bumpy).

**Model 3: CLIP** ([06_clip](notebooks/06_clip.ipynb)). CLIP learned from ~400M images with captions and turns both images and text into 512 numbers.

| CLIP ViT-B/32 | Test accuracy | Test top-3 |
|---|---|---|
| Zero-shot (compare each image with "a cracked texture." etc., no training) | 43.8% | 64.4% |
| Linear probe (new last layer on frozen CLIP, learning rate picked by a small grid search) | **74.1%** | **90.1%** |

- Training only a last layer on frozen CLIP beats fine-tuning all of ResNet18 (74.1% vs 62.3%): what the backbone learned beforehand mattered more than fine-tuning.
- The cost: CLIP's image encoder is ~8× bigger (~350 MB vs 45 MB) and ~3× slower per image on CPU. The app still uses ResNet18.
- CLIP can search with words outside the 47 classes ("rusty metal", "wood planks", "brick wall" find the right textures), the basis for a texture-search tool.

## Repository

```
app.py                  Gradio app (single-label classifier)
search_app.py           Gradio search app: tag, search, find similar (our multi-label ResNet18, or CLIP with --model clip)
src/textag/             data.py, model.py, engine.py (training loop), infer.py, multilabel.py, search.py
notebooks/              01_eda … 07_search (executed, with outputs)
tests/                  smoke tests (pytest)
examples/               example images for the app (from the DTD test set)
scripts/deploy_space.py upload the app to a Hugging Face Space
models/textag.pt        trained weights used by the app (45 MB, produced by 03_resnet)
models/*_multilabel.pt  multi-label weights; textag_multilabel.pt is used by the search app (from 07_search)
```

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py            # open http://127.0.0.1:7860
python app.py --share    # also prints a public link, valid 72 hours, while the app runs

python search_app.py                           # search app; library = DTD test images
python search_app.py --folder path/to/textures # search your own folder of textures
python search_app.py --model clip              # same app with CLIP: search with any words (port 7861)
```

The search app indexes its library on the first start (every image through the model once, ~2 min for 1,880 images
on a laptop CPU) and saves the index in `data/`, so later starts are instant; images added to a `--folder` later are
indexed at the next search. It also takes `--share`. `--model clip` needs `pip install open_clip_torch` (~600 MB of weights
on first use) and is ~2–3× slower to index.

`requirements.txt` pins the versions used for deployment (Linux, CPU). On an **Intel Mac**, current PyTorch isn't available;
use Python 3.12 and `pip install torch==2.2.2 torchvision==0.17.2 "numpy<2" "gradio>=5" pillow` instead.

**Train:** open `notebooks/02_baseline.ipynb` or `03_resnet.ipynb` in Google Colab (GPU runtime) and run all; the first cell clones this repo.
**Test:** `pip install -r requirements-dev.txt && pytest -q`

## Deploy

- **Code, notebooks and weights:** this GitHub repository.
- **Public demo:** `python app.py --share` (Gradio share link, 72 hours, while the app runs).
- **Hugging Face Space (optional):** `python scripts/deploy_space.py` uploads `app.py`, `requirements.txt`, `src/textag/`,
  the weights and the examples to a Space (needs `hf auth login` with a write token). Hugging Face now requires a paid plan
  to create Gradio Spaces on its free CPU hardware, so this isn't used for the public demo.

## Limitations

- **Patterns, not materials:** DTD's 47 classes describe patterns (grooved, cracked), not materials (wood, metal, stone).
- **Photos, not game art:** trained on real photos; stylized or hand-painted textures get lower confidence.
- **One label per image:** real textures often contain several patterns (e.g. a tiled floor with swirl motifs).
- **Data license:** DTD is intended for research use, so the trained model and examples are for demonstration, not commercial use.

## Future work

- **Switch to CLIP** for more accurate tags and open-vocabulary search (any words, e.g. "mossy stone"), with tags as filters
- **Multi-label tagging** (several tags per texture)
- **Game-engine plugin** (e.g. Godot): index a project's textures once, then search by content instead of filename; group PBR map sets (base color, normal, roughness, ...) into one material and tag the base color
- Retrain on game-style textures and material classes

## References

- Cimpoi et al. (2014), *Describing Textures in the Wild* (DTD). [arXiv:1311.3618](https://arxiv.org/abs/1311.3618)
- He et al. (2015), *Deep Residual Learning for Image Recognition* (ResNet). [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Yosinski et al. (2014), *How transferable are features in deep neural networks?* [arXiv:1411.1792](https://arxiv.org/abs/1411.1792)
- Simonyan & Zisserman (2014), *Very Deep Convolutional Networks* (VGG). [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)
- Radford et al. (2021), *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
- Wang et al., *CNN Explainer* (TinyVGG). [poloclub.github.io/cnn-explainer](https://poloclub.github.io/cnn-explainer/)
