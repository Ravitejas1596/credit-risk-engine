from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, upload_folder


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN in your shell (do not hardcode it).")

    username = os.environ.get("HF_USERNAME", "ravitejas1596")
    space_name = os.environ.get("HF_SPACE_NAME", "credit-risk-engine")
    repo_id = f"{username}/{space_name}"

    api = HfApi(token=token)

    # Create Space (Docker) if missing
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        private=False,
        exist_ok=True,
        space_sdk="docker",
    )

    repo_root = Path(__file__).resolve().parents[1]
    upload_folder(
        repo_id=repo_id,
        folder_path=str(repo_root),
        repo_type="space",
        token=token,
        commit_message="Deploy Space",
    )

    print(f"Deployed: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()

