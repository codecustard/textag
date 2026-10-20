"""TexTag search app: tag a texture, search a texture library, or find similar textures.

Two models to compare on the same library:
  --model resnet (default): our multi-label ResNet18 (models/textag_multilabel.pt, from 07_search);
                            search by the 47 tags
  --model clip:             CLIP ViT-B/32, no training (needs `pip install open_clip_torch`);
                            search with any text, e.g. "stained glass"
The first start indexes the library (every image through the model once) and saves the index in data/,
so later starts are instant. Each model has its own index.

python search_app.py                            # library = DTD test images (downloaded on first run, ~600 MB)
python search_app.py --folder path/to/textures  # library = your own folder of images (subfolders included);
                                                #   images added to the folder later are indexed at the next search
python search_app.py --model clip               # CLIP instead of our model (runs on port 7861, so both can run)
python search_app.py --share                    # also a temporary public link (https://....gradio.live, 72 hours)
"""

import argparse
import sys
from pathlib import Path

import gradio as gr
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from textag.infer import load_model  # noqa: E402
from textag import search  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--folder", help="folder of texture images to search (default: DTD test images)")
parser.add_argument("--model", choices=["resnet", "clip"], default="resnet", help="which model to search with")
parser.add_argument("--share", action="store_true", help="also create a temporary public link")
args, _ = parser.parse_known_args()
USE_CLIP = args.model == "clip"

# 1. Load the model once
MODEL_FILE = ROOT / "models" / "textag_multilabel.pt"
if USE_CLIP:
    classes = torch.load(MODEL_FILE, map_location="cpu")["classes"]   # just the 47 tag names
    searcher = search.ClipSearcher(classes)
    model_name = "CLIP ViT-B/32 (no training)"
else:
    model, classes = load_model(MODEL_FILE)
    searcher = search.ResNetSearcher(model, classes)
    model_name = "our multi-label ResNet18"

# 2. Pick the library and load (or build) its index
if args.folder:
    library = Path(args.folder).expanduser().resolve()
    library_name = library.name
    paths = search.list_images(library)
    index_name = f"index_{library.name}"
else:
    from torchvision import datasets
    from textag.data import DATA_DIR
    library = DATA_DIR
    library_name = "DTD test images"
    paths = [Path(p) for p in datasets.DTD(DATA_DIR, split="test", partition=1, download=True)._image_files]
    index_name = "index_dtd_test"
index_file = ROOT / "data" / (index_name + ("_clip" if USE_CLIP else "") + ".pt")

if not paths:
    sys.exit(f"no images found in {library}")
print(f"library: {library} ({len(paths)} images), model: {model_name}")
index = search.load_or_build_index(searcher, paths, index_file)


def refresh():
    """With --folder: pick up images added to (or removed from) the folder since the last search.

    Returns a short note for the app, e.g. "library updated: 1 new, 0 removed (5642 images)".
    """
    global index
    if not args.folder:
        return ""
    index, added, removed = search.update_index(searcher, index, search.list_images(library), index_file)
    if not added and not removed:
        return ""
    return f"library updated: {added} new, {removed} removed ({len(index['paths'])} images)"


MAX_RESULTS = 200     # more than this makes the gallery slow to load


def results_slider():
    """How many results to show (each tab that shows a gallery gets its own)."""
    return gr.Slider(minimum=4, maximum=min(MAX_RESULTS, len(index["paths"])), value=12, step=4, label="Results to show")


def caption(path, score):
    """Gallery caption: parent folder / file name, and the score."""
    path = Path(path)
    return f"{path.parent.name}/{path.name}  ({score:.2f})"


# 3. One function per tab

def tag(image):
    """Image in -> ({tag: score}, inference time).

    ResNet18: every tag above the threshold. CLIP: its top 3 of the 47 tags (zero-shot).
    """
    if image is None:
        return {}, ""
    if USE_CLIP:
        tags, seconds = search.tag_image(searcher, image, top_k=3)
    else:
        tags, seconds = search.tag_image(searcher, image)
    if not tags:
        return {"(no tag above 35%)": 0.0}, f"{seconds:.3f} s"
    return {name: score for name, score in tags}, f"{seconds:.3f} s"


def search_by_tags(selected, count=12):
    """ResNet18: chosen tags in -> (gallery of the best matches, which words were used / ignored)."""
    tags, ignored = search.parse_query(" ".join(selected or []), classes)
    if not tags:
        return [], "No known tags in the query. Try e.g. striped, honeycombed, banded striped."
    note = refresh()
    results = search.search_tags(index, classes, tags, k=int(count))
    info = "searched: " + " + ".join(tags)
    if ignored:
        info += "   |   ignored (not one of the 47 tags): " + ", ".join(ignored)
    if note:
        info += "   |   " + note
    return [(path, caption(path, score)) for path, score in results], info


def search_by_text(text, count=12):
    """CLIP: any text in -> (gallery of the best matches, the query CLIP compared against)."""
    if not text or not text.strip():
        return [], "Type something, e.g. stained glass, rusty metal, hexagon pattern."
    note = refresh()
    results = search.search_text(searcher, index, text.strip(), k=int(count))
    info = f'searched: "a photo of {text.strip()}"' + (f"   |   {note}" if note else "")
    return [(path, caption(path, score)) for path, score in results], info


def similar(image, count=12):
    """Image in -> (gallery of the most similar library images, search time)."""
    if image is None:
        return [], ""
    note = refresh()
    results, seconds = search.find_similar(searcher, index, image, k=int(count))
    info = f"{seconds:.3f} s" + (f"   |   {note}" if note else "")
    return [(path, caption(path, score)) for path, score in results], info


examples = sorted(str(p) for p in (ROOT / "examples").glob("*.jpg"))

if USE_CLIP:
    tag_label, tag_description = "Top 3 tags (zero-shot)", "CLIP's 3 best matches among the 47 tags, with no training on DTD."
else:
    tag_label = f"Tags (score ≥ {search.THRESHOLD:.0%})"
    tag_description = "Every tag the model gives at least 35% (one image can have several)."

tag_tab = gr.Interface(
    fn=tag,
    inputs=gr.Image(type="pil", label="Texture image"),
    outputs=[gr.Label(label=tag_label), gr.Textbox(label="Inference time (seconds)")],
    description=tag_description,
    examples=examples,
    flagging_mode="never",
)

if USE_CLIP:
    search_tab = gr.Interface(
        fn=search_by_text,
        inputs=[gr.Textbox(label="Search", placeholder="any words, e.g. stained glass"), results_slider()],
        outputs=[gr.Gallery(label="Best matches", columns=4, height="auto"), gr.Textbox(label="Query")],
        description="Search the library with any words: CLIP compares your text with every image.",
        examples=[["stained glass", 12], ["rusty metal", 12], ["hexagon pattern", 12], ["wood planks", 12],
                  ["honeycombed", 12]],
        flagging_mode="never",
    )
    search_title = "Search by text"
else:
    search_tab = gr.Interface(
        fn=search_by_tags,
        inputs=[gr.Dropdown(choices=classes, multiselect=True, allow_custom_value=True, label="Tags",
                            info="Start typing to see matching tags; pick one or more"),
                results_slider()],
        outputs=[gr.Gallery(label="Best matches", columns=4, height="auto"), gr.Textbox(label="Query")],
        description=("Search the library by one or more of the 47 tags. With several tags, an image needs "
                     "a high score on every one. Words that aren't tags are ignored."),
        examples=[[["striped"], 12], [["honeycombed"], 12], [["cracked"], 12], [["banded", "striped"], 12],
                  [["braided", "woven"], 12]],
        flagging_mode="never",
    )
    search_title = "Search by tags"

similar_tab = gr.Interface(
    fn=similar,
    inputs=[gr.Image(type="pil", label="Texture image"), results_slider()],
    outputs=[gr.Gallery(label="Most similar", columns=4, height="auto"), gr.Textbox(label="Search time")],
    description="Find the library images that look most like this one (compares the model's 512-number embeddings).",
    examples=[[path, 12] for path in examples],
    flagging_mode="never",
)

demo = gr.TabbedInterface(
    [tag_tab, search_tab, similar_tab],
    ["Tag", search_title, "Find similar"],
    title=f"TexTag search: {len(index['paths'])} textures ({library_name}), {model_name}",
)

if __name__ == "__main__":
    # allowed_paths lets the gallery show images from the library folder;
    # CLIP runs on port 7861 so both apps can run side by side
    demo.launch(share=args.share, allowed_paths=[str(library)], server_port=7861 if USE_CLIP else 7860)
