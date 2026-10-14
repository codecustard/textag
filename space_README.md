---
title: TexTag
emoji: 🧱
colorFrom: yellow
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
python_version: "3.11"
app_file: app.py
pinned: false
short_description: Texture pattern auto-tagger (ResNet18 on DTD)
---

# TexTag: texture auto-tagger

Upload a texture and get its top 3 pattern tags (out of 47, e.g. cracked, woven, scaly, marbled, honeycombed) plus the inference time.

- **Model:** ResNet18 pretrained on ImageNet, fine-tuned on the Describable Textures Dataset (DTD)
- **Test set:** 62.3% accuracy, 82.6% top-3 accuracy
- **Speed:** about 0.05 s per image on CPU

Code, notebooks and results: https://github.com/codecustard/textag
