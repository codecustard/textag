"""TexTag search app: tag a texture, search a texture library by tags, or find similar textures.

Uses the multi-label ResNet18 (models/textag_multilabel.pt, from 07_search).
The first start indexes the library (every image through the model once) and saves the index in data/,
so later starts are instant.

python search_app.py                          # library = DTD test images (downloaded on first run, ~600 MB)
python search_app.py --folder path/to/textures  # library = your own folder of images (subfolders included);
                                                #   images added to the folder later are indexed at the next search
python search_app.py --share                  # also a temporary public link (https://....gradio.live, 72 hours)
"""

import argparse
import sys
from pathlib import Path

import gradio as gr
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from textag.infer import load_model  # noqa: E402
from textag import search  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--folder", help="folder of texture images to search (default: DTD test images)")
parser.add_argument("--share", action="store_true", help="also create a temporary public link")
args, _ = parser.parse_known_args()

# 1. Load the model once
model, classes = load_model(ROOT / "models" / "textag_multilabel.pt")

# 2. Pick the library and load (or build) its index
if args.folder:
    library = Path(args.folder).expanduser().resolve()
    library_name = library.name
    paths = search.list_images(library)
    index_file = ROOT / "data" / f"index_{library.name}.pt"
else:
    from torchvision import datasets
    from textag.data import DATA_DIR
    library = DATA_DIR
    library_name = "DTD test images"
    paths = [Path(p) for p in datasets.DTD(DATA_DIR, split="test", partition=1, download=True)._image_files]
    index_file = ROOT / "data" / "index_dtd_test.pt"

if not paths:
    sys.exit(f"no images found in {library}")
print(f"library: {library} ({len(paths)} images)")
index = search.load_or_build_index(model, paths, index_file)


def refresh():
    """With --folder: pick up images added to (or removed from) the folder since the last search.

    Returns a short note for the app, e.g. "library updated: 1 new image indexed".
    """
    global index
    if not args.folder:
        return ""
    index, added, removed = search.update_index(model, index, search.list_images(library), index_file)
    if not added and not removed:
        return ""
    return f"library updated: {added} new, {removed} removed ({len(index['paths'])} images)"


def caption(path, score):
    """Gallery caption: parent folder / file name, and the score."""
    path = Path(path)
    return f"{path.parent.name}/{path.name}  ({score:.2f})"


# 3. One function per tab

def tag(image):
    """Image in -> ({tag: score} for every tag above the threshold, inference time)."""
    if image is None:
        return {}, ""
    tags, seconds = search.tag_image(model, classes, image)
    if not tags:
        return {"(no tag above 35%)": 0.0}, f"{seconds:.3f} s"
    return {name: score for name, score in tags}, f"{seconds:.3f} s"


def search_by_tags(selected):
    """Chosen tags in -> (gallery of the best matches, which words were used / ignored)."""
    tags, ignored = search.parse_query(" ".join(selected or []), classes)
    if not tags:
        return [], "No known tags in the query. Try e.g. striped, honeycombed, banded striped."
    note = refresh()
    results = search.search_tags(index, classes, tags)
    info = "searched: " + " + ".join(tags)
    if ignored:
        info += "   |   ignored (not one of the 47 tags): " + ", ".join(ignored)
    if note:
        info += "   |   " + note
    return [(path, caption(path, score)) for path, score in results], info


def similar(image):
    """Image in -> (gallery of the most similar library images, search time)."""
    if image is None:
        return [], ""
    note = refresh()
    results, seconds = search.find_similar(model, index, image)
    info = f"{seconds:.3f} s" + (f"   |   {note}" if note else "")
    return [(path, caption(path, score)) for path, score in results], info


examples = sorted(str(p) for p in (ROOT / "examples").glob("*.jpg"))

tag_tab = gr.Interface(
    fn=tag,
    inputs=gr.Image(type="pil", label="Texture image"),
    outputs=[gr.Label(label=f"Tags (score ≥ {search.THRESHOLD:.0%})"), gr.Textbox(label="Inference time (seconds)")],
    description="Every tag the model gives at least 35% (one image can have several).",
    examples=examples,
    flagging_mode="never",
)

search_tab = gr.Interface(
    fn=search_by_tags,
    inputs=gr.Dropdown(choices=classes, multiselect=True, allow_custom_value=True, label="Tags",
                       info="Start typing to see matching tags; pick one or more"),
    outputs=[gr.Gallery(label="Best matches", columns=4, height="auto"), gr.Textbox(label="Query")],
    description=("Search the library by one or more of the 47 tags. With several tags, an image needs "
                 "a high score on every one. Words that aren't tags are ignored."),
    examples=[[["striped"]], [["honeycombed"]], [["cracked"]], [["banded", "striped"]], [["braided", "woven"]]],
    flagging_mode="never",
)

similar_tab = gr.Interface(
    fn=similar,
    inputs=gr.Image(type="pil", label="Texture image"),
    outputs=[gr.Gallery(label="Most similar", columns=4, height="auto"), gr.Textbox(label="Search time")],
    description="Find the library images that look most like this one (compares the model's 512-number embeddings).",
    examples=examples,
    flagging_mode="never",
)

demo = gr.TabbedInterface(
    [tag_tab, search_tab, similar_tab],
    ["Tag", "Search by tags", "Find similar"],
    title=f"TexTag search: {len(index['paths'])} textures ({library_name})",
)

if __name__ == "__main__":
    # allowed_paths lets the gallery show images from the library folder
    demo.launch(share=args.share, allowed_paths=[str(library)])
