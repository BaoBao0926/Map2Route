"""Download helpers for the Lang2LTL-2 lifted-translation checkpoint."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_LT_CHECKPOINT_URL = (
    "https://drive.google.com/drive/folders/"
    "1rZl8tblyVj-pZZW4OgbO1NJwMIT2fwx9?usp=sharing"
)

MODEL_FILES = ("pytorch_model.bin", "model.safetensors")
TOKENIZER_FILES = ("tokenizer.json", "spiece.model")


def checkpoint_looks_ready(path: Path) -> bool:
    """Return whether path looks like a HuggingFace seq2seq checkpoint."""
    if not path.is_dir():
        return False
    has_config = (path / "config.json").is_file()
    has_model = any((path / name).is_file() for name in MODEL_FILES)
    has_tokenizer = any((path / name).is_file() for name in TOKENIZER_FILES)
    return has_config and has_model and has_tokenizer


def _iter_candidate_dirs(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    yield root
    if root.is_dir():
        for path in root.rglob("*"):
            if path.is_dir():
                yield path


def find_downloaded_checkpoint(search_root: Path) -> Path | None:
    for candidate in _iter_candidate_dirs(search_root):
        if checkpoint_looks_ready(candidate):
            return candidate
    return None


def _import_or_install_gdown(*, verbose: bool) -> object:
    try:
        return importlib.import_module("gdown")
    except ImportError:
        if verbose:
            print("[lang2ltlv2] installing gdown for checkpoint download", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "gdown"],
            check=True,
        )
        return importlib.import_module("gdown")


def _copy_checkpoint_tree(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def ensure_lt_checkpoint(
    checkpoint_path: Path,
    *,
    download: bool,
    url: str = DEFAULT_LT_CHECKPOINT_URL,
    verbose: bool = False,
) -> tuple[Path, dict[str, object]]:
    """Ensure the LT checkpoint exists, optionally downloading it from Drive."""
    checkpoint_path = checkpoint_path.expanduser().resolve()
    info: dict[str, object] = {
        "checkpoint_path": str(checkpoint_path),
        "download_requested": download,
        "download_url": url,
    }
    if checkpoint_looks_ready(checkpoint_path):
        info["status"] = "ready"
        return checkpoint_path, info

    if checkpoint_path.exists() and not checkpoint_path.is_dir():
        raise FileExistsError(f"LT checkpoint path exists but is not a directory: {checkpoint_path}")

    existing_candidate = find_downloaded_checkpoint(checkpoint_path)
    if existing_candidate is not None:
        info["status"] = "ready_nested"
        info["detected_checkpoint_path"] = str(existing_candidate)
        return existing_candidate, info

    if not download:
        info["status"] = "missing"
        raise FileNotFoundError(
            "LT checkpoint is missing. Re-run with --download-lt-checkpoint "
            f"or pass --lt-model-path. Expected: {checkpoint_path}"
        )

    download_root = checkpoint_path.parent
    download_root.mkdir(parents=True, exist_ok=True)
    before = {path.resolve() for path in download_root.rglob("*")} if download_root.exists() else set()
    gdown = _import_or_install_gdown(verbose=verbose)
    if verbose:
        print(f"[lang2ltlv2] downloading LT checkpoint from {url}", flush=True)
    downloaded = gdown.download_folder(
        url=url,
        output=str(download_root),
        quiet=not verbose,
        use_cookies=False,
    )
    info["downloaded_files"] = [str(path) for path in downloaded or []]

    candidate = find_downloaded_checkpoint(checkpoint_path)
    if candidate is None:
        candidate = find_downloaded_checkpoint(download_root)
    if candidate is None:
        after = {path.resolve() for path in download_root.rglob("*")}
        new_paths = sorted(str(path) for path in after - before)
        info["status"] = "downloaded_but_checkpoint_not_found"
        info["new_paths"] = new_paths[:100]
        raise FileNotFoundError(
            "Downloaded LT checkpoint files, but could not find a HuggingFace "
            f"checkpoint under {download_root}. New paths: {new_paths[:20]}"
        )

    if candidate.resolve() != checkpoint_path:
        _copy_checkpoint_tree(candidate, checkpoint_path)
        candidate = checkpoint_path

    info["status"] = "downloaded"
    info["detected_checkpoint_path"] = str(candidate)
    return candidate, info
