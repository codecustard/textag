"""Texture search: tag an image, search by tags or text, find similar images.

Two models ("searchers"), same index and search code:
  ResNetSearcher: our multi-label ResNet18 (47 tags, sigmoid scores)
  ClipSearcher:   CLIP ViT-B/32, frozen, no training (47 tags zero-shot, plus free-text search)

1. Index (once): run every image in a folder through the model and keep
   - its 47 tag scores
   - its embedding: 512 numbers describing the image, scaled to length 1
   The index is saved to a file, so the next start is instant.
   update_index keeps it current: only images added since last time go through the model.
2. Search: only math on the saved numbers, no model needed.
   - tags:    rank images by score(tag 1) x score(tag 2) x ...  (high only if high on every tag)
   - similar: rank images by similarity of their embedding to the query image's embedding
"""
import time
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image

from textag.data import eval_tf
from textag.multilabel import embed

THRESHOLD = 0.35       # show tags scoring at least this; picked on validation in 07_search
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tga"}


def list_images(folder) -> List[Path]:
    """Every image file in a folder and its subfolders, sorted."""
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_TYPES)


class ResNetSearcher:
    """Our multi-label ResNet18 (models/textag_multilabel.pt)."""

    def __init__(self, model, classes: List[str]):
        self.model = model.eval()
        self.classes = classes
        self.transform = eval_tf                     # same preprocessing as in training

    def describe(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """One pass through the model. Returns (tag_scores, embeddings) for a batch of images.

        The embedding is what goes into the last layer (fc), so the tag scores are just that
        last layer applied to it (same result as model(X), one pass instead of two).
        """
        features = embed(self.model, X)                       # (batch, 512)
        tag_scores = torch.sigmoid(self.model.fc(features))   # (batch, 47), each tag 0-1 on its own
        embeddings = features / features.norm(dim=1, keepdim=True)
        return tag_scores, embeddings


class ClipSearcher:
    """CLIP ViT-B/32 with OpenAI's weights, frozen (as in 06_clip). Needs `pip install open_clip_torch`."""

    def __init__(self, classes: List[str]):
        import open_clip                             # only needed for CLIP
        self.model, _, self.transform = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
        self.classes = classes
        self.tag_text = self.encode_text([f"a {c} texture." for c in classes])   # (47, 512)

    def encode_text(self, sentences: List[str]) -> torch.Tensor:
        """Sentences -> 512 numbers each, scaled to length 1."""
        with torch.inference_mode():
            text = self.model.encode_text(self.tokenizer(sentences))
        return text / text.norm(dim=1, keepdim=True)

    def describe(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (tag_scores, embeddings). Tag scores: zero-shot, 47 probabilities that add up to 1."""
        features = self.model.encode_image(X)
        embeddings = features / features.norm(dim=1, keepdim=True)            # (batch, 512)
        tag_scores = torch.softmax(100 * embeddings @ self.tag_text.T, dim=1)  # 100 = CLIP's own scale
        return tag_scores, embeddings


def build_index(searcher, paths: List[Path], batch_size: int = 32) -> dict:
    """Run every image through the model once. Returns {"paths", "tag_scores", "embeddings"}."""
    kept, all_scores, all_embeddings = [], [], []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            images, batch_paths = [], []
            for path in paths[start:start + batch_size]:
                try:
                    images.append(searcher.transform(Image.open(path).convert("RGB")))
                    batch_paths.append(path)
                except OSError:
                    print(f"skipped (can't open): {path}")
            if not images:
                continue
            tag_scores, embeddings = searcher.describe(torch.stack(images))
            kept += batch_paths
            all_scores.append(tag_scores)
            all_embeddings.append(embeddings)
            print(f"indexed {start + len(images)}/{len(paths)}", end="\r")
    print()
    if not kept:                                   # nothing could be opened
        return {"paths": [], "tag_scores": torch.zeros(0, len(searcher.classes)), "embeddings": torch.zeros(0, 512)}
    return {"paths": [str(p) for p in kept],
            "tag_scores": torch.cat(all_scores),
            "embeddings": torch.cat(all_embeddings)}


def update_index(searcher, index: dict, paths: List[Path], index_file: Path):
    """Bring a saved index up to date with the images in `paths`.

    Images no longer in `paths` are dropped; new images go through the model (only those).
    Saves the index if anything changed. Returns (index, number added, number removed).
    """
    wanted = set(str(p) for p in paths)
    known = set(index["paths"])

    # 1. Keep the rows of images that are still there
    keep = [i for i, p in enumerate(index["paths"]) if p in wanted]
    removed = len(index["paths"]) - len(keep)

    # 2. Find images that aren't in the index yet
    new_paths = [p for p in paths if str(p) not in known]
    if not new_paths and removed == 0:
        return index, 0, 0

    index = {"paths": [index["paths"][i] for i in keep],
             "tag_scores": index["tag_scores"][keep],
             "embeddings": index["embeddings"][keep]}

    # 3. Index only the new images and add them at the end
    added = 0
    if new_paths:
        new = build_index(searcher, new_paths)
        added = len(new["paths"])
        index = {"paths": index["paths"] + new["paths"],
                 "tag_scores": torch.cat([index["tag_scores"], new["tag_scores"]]),
                 "embeddings": torch.cat([index["embeddings"], new["embeddings"]])}

    torch.save(index, index_file)
    return index, added, removed


def load_or_build_index(searcher, paths: List[Path], index_file: Path) -> dict:
    """Load the saved index and update it (only new images are indexed); build it if there is none yet."""
    if index_file.exists():
        index, added, removed = update_index(searcher, torch.load(index_file), paths, index_file)
        if added or removed:
            print(f"index updated: {added} added, {removed} removed")
        return index
    start = time.perf_counter()
    index = build_index(searcher, paths)
    print(f"indexed {len(index['paths'])} images in {time.perf_counter() - start:.0f} s")
    index_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(index, index_file)
    return index


def tag_image(searcher, image: Image.Image, threshold: float = THRESHOLD, top_k: int = None):
    """Tags for one image, best first. Returns ([(tag, score), ...], seconds).

    Every tag scoring at least `threshold`, or, with top_k, the top_k tags whatever their score.
    """
    start = time.perf_counter()
    x = searcher.transform(image.convert("RGB")).unsqueeze(0)    # add a batch dimension
    with torch.inference_mode():
        tag_scores, _ = searcher.describe(x)
    tag_scores = tag_scores[0]
    seconds = time.perf_counter() - start

    tags = []
    for i in tag_scores.argsort(descending=True).tolist():
        if top_k is not None and len(tags) == top_k:
            break
        if top_k is not None or tag_scores[i] >= threshold:
            tags.append((searcher.classes[i], tag_scores[i].item()))
    return tags, seconds


def parse_query(text: str, classes: List[str]) -> Tuple[List[str], List[str]]:
    """Split a query into words. Returns (tags it knows, words it ignores)."""
    tags, ignored = [], []
    for word in text.lower().replace(",", " ").split():
        if word in classes:
            tags.append(word)
        else:
            ignored.append(word)
    return tags, ignored


def search_tags(index: dict, classes: List[str], tags: List[str], k: int = 12) -> List[Tuple[str, float]]:
    """Top k images for one or more tags: score = product of the tags' scores. Returns [(path, score), ...]."""
    scores = torch.ones(len(index["paths"]))
    for tag in tags:
        scores = scores * index["tag_scores"][:, classes.index(tag)]
    top = scores.topk(min(k, len(scores)))
    return [(index["paths"][i], s) for s, i in zip(top.values.tolist(), top.indices.tolist())]


def search_text(searcher, index: dict, text: str, k: int = 12) -> List[Tuple[str, float]]:
    """CLIP only: top k images for any text, e.g. "stained glass". Returns [(path, similarity), ...]."""
    query = searcher.encode_text([f"a photo of {text}"])[0]   # (512,)
    similarity = index["embeddings"] @ query                   # (images,)
    top = similarity.topk(min(k, len(similarity)))
    return [(index["paths"][i], s) for s, i in zip(top.values.tolist(), top.indices.tolist())]


def find_similar(searcher, index: dict, image: Image.Image, k: int = 12) -> Tuple[List[Tuple[str, float]], float]:
    """Top k images most similar to `image`. Returns ([(path, similarity), ...], seconds)."""
    start = time.perf_counter()
    x = searcher.transform(image.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        _, query = searcher.describe(x)                     # (1, 512)
    similarity = index["embeddings"] @ query[0]             # (images,)
    top = similarity.topk(min(k, len(similarity)))
    seconds = time.perf_counter() - start
    return [(index["paths"][i], s) for s, i in zip(top.values.tolist(), top.indices.tolist())], seconds
