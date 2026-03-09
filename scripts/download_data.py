from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def _ensure_kaggle_json() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    local = repo_root / "kaggle.json"
    workspace_root = repo_root.parent / "kaggle.json"
    home = Path.home() / ".kaggle" / "kaggle.json"

    if home.exists():
        return

    if not local.exists() and not workspace_root.exists():
        raise FileNotFoundError(
            "Kaggle credentials not found. Put kaggle.json at ~/.kaggle/kaggle.json "
            "or copy it to the repo root (it is gitignored)."
        )

    src = local if local.exists() else workspace_root
    home.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, home)
    os.chmod(home, 0o600)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    _ensure_kaggle_json()

    competition = "GiveMeSomeCredit"
    zip_path = raw_dir / f"{competition}.zip"

    cmd = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        competition,
        "-p",
        str(raw_dir),
        "--force",
    ]
    subprocess.run(cmd, check=True)

    # Kaggle names the file after the competition
    if zip_path.exists():
        shutil.unpack_archive(str(zip_path), str(raw_dir))

    expected = raw_dir / "cs-training.csv"
    if not expected.exists():
        raise FileNotFoundError(
            f"Expected {expected} after download/unzip. "
            "Make sure you've accepted the Kaggle competition rules."
        )

    print(f"Downloaded to {raw_dir}")


if __name__ == "__main__":
    main()
