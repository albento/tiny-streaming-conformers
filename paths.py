"""The data and run folders: the paths set in .env (or in the environment, which wins), else the
default folders inside the repository.

PolyChirp's code only knows its own folders (polychirp/raw_dataset/, polychirp/datasets/<collection>/,
its decoded-audio cache polychirp/.cache/dataset_cache/<collection>/): when POLYCHIRP_DATA_DIR is set,
link_polychirp() turns them into links to it.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def folder(variable: str, default: Path | None) -> Path | None:
    value = os.environ.get(variable, "").strip()
    return Path(value).expanduser().resolve() if value else default


DATA = folder("DATA_DIR", ROOT / "data")
RUNS = folder("RUNS_DIR", ROOT / "runs")
POLYCHIRP_DATA = folder("POLYCHIRP_DATA_DIR", None)


def link(path: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        if path.resolve() == target.resolve():
            return
        path.unlink()
    elif path.exists():
        if any(path.iterdir()):
            raise SystemExit(f"{path} holds data: move its content to {target}, then rerun")
        path.rmdir()
    path.symlink_to(target, target_is_directory=True)


def link_polychirp(collection: str) -> None:
    if POLYCHIRP_DATA is not None:
        link(ROOT / "polychirp" / "raw_dataset", POLYCHIRP_DATA / "raw_dataset")
        link(ROOT / "polychirp" / "datasets" / collection, POLYCHIRP_DATA / "datasets" / collection)
        link(ROOT / "polychirp" / ".cache" / "dataset_cache" / collection, POLYCHIRP_DATA / "dataset_cache" / collection)
