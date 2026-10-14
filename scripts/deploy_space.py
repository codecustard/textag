"""Deploy the app to a Hugging Face Space.

Usage (after `hf auth login` with a write token):
    python scripts/deploy_space.py                 # -> <your-username>/textag
    python scripts/deploy_space.py someone/name    # custom Space id

Uploads only what the app needs: app.py, requirements.txt, src/textag/*.py,
models/textag.pt, examples/*.jpg, and space_README.md (as the Space's README.md).
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
FILES = ["app.py", "requirements.txt", "src/textag/*.py", "models/textag.pt", "examples/*.jpg"]


def main():
    weights = ROOT / "models" / "textag.pt"
    if not weights.exists():
        sys.exit(f"Missing {weights}. Put the trained weights there first.")

    api = HfApi()
    repo_id = sys.argv[1] if len(sys.argv) > 1 else f"{api.whoami()['name']}/textag"

    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=ROOT,
        allow_patterns=FILES,
        commit_message="Deploy TexTag app",
    )
    api.upload_file(
        repo_id=repo_id,
        repo_type="space",
        path_or_fileobj=ROOT / "space_README.md",
        path_in_repo="README.md",
        commit_message="Space README",
    )
    print(f"Done: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
