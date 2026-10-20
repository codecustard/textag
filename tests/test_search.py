import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from textag import search  # noqa: E402
from textag.model import build  # noqa: E402

CLASSES = ["banded", "dotted", "striped"]


def test_parse_query_splits_known_tags_and_ignored_words():
    tags, ignored = search.parse_query("Banded, rusty striped", CLASSES)
    assert tags == ["banded", "striped"]
    assert ignored == ["rusty"]


def test_search_tags_needs_every_tag():
    index = {"paths": ["a.jpg", "b.jpg", "c.jpg"],
             "tag_scores": torch.tensor([[0.9, 0.0, 0.1],     # banded only
                                         [0.8, 0.0, 0.7],     # banded + striped
                                         [0.1, 0.9, 0.9]])}   # dotted + striped
    results = search.search_tags(index, CLASSES, ["banded", "striped"], k=3)
    assert results[0][0] == "b.jpg"


def test_index_tag_and_similar_on_a_folder(tmp_path):
    for i, color in enumerate(["red", "green", "blue"]):
        Image.new("RGB", (64, 64), color).save(tmp_path / f"{i}.png")
    searcher = search.ResNetSearcher(build("resnet18", num_classes=3, pretrained=False), CLASSES)
    paths = search.list_images(tmp_path)
    index = search.load_or_build_index(searcher, paths, tmp_path / "index.pt")
    assert index["tag_scores"].shape == (3, 3) and index["embeddings"].shape == (3, 512)

    tags, seconds = search.tag_image(searcher, Image.new("RGB", (64, 64), "red"), threshold=0.0)
    assert len(tags) == 3 and seconds > 0

    results, _ = search.find_similar(searcher, index, Image.open(paths[0]), k=1)
    assert results[0][0] == str(paths[0])          # an image is most similar to itself


def test_update_index_adds_new_and_drops_removed_images(tmp_path):
    Image.new("RGB", (64, 64), "red").save(tmp_path / "a.png")
    Image.new("RGB", (64, 64), "blue").save(tmp_path / "b.png")
    searcher = search.ResNetSearcher(build("resnet18", num_classes=3, pretrained=False), CLASSES)
    index_file = tmp_path / "index.pt"
    index = search.load_or_build_index(searcher, search.list_images(tmp_path), index_file)

    (tmp_path / "a.png").unlink()                                   # remove one image
    Image.new("RGB", (64, 64), "green").save(tmp_path / "c.png")    # add one image
    index, added, removed = search.update_index(searcher, index, search.list_images(tmp_path), index_file)

    assert (added, removed) == (1, 1)
    assert sorted(Path(p).name for p in index["paths"]) == ["b.png", "c.png"]
    assert index["embeddings"].shape == (2, 512)


def test_tag_image_top_k_returns_k_tags_whatever_the_score():
    searcher = search.ResNetSearcher(build("resnet18", num_classes=3, pretrained=False), CLASSES)
    tags, _ = search.tag_image(searcher, Image.new("RGB", (64, 64), "red"), threshold=1.1, top_k=2)
    assert len(tags) == 2
