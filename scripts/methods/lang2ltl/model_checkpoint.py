"""Checkpoint helpers for original Lang2LTL model weights."""

from __future__ import annotations

from pathlib import Path

from scripts.methods.lang2ltlv2.lt_checkpoint import (
    DEFAULT_LT_CHECKPOINT_URL as DEFAULT_LANG2LTL_T5_CHECKPOINT_URL,
    checkpoint_looks_ready,
    ensure_lt_checkpoint,
)


DEFAULT_LANG2LTL_T5_CHECKPOINT_PATH = (
    Path.home() / "ground" / "models" / "lang2ltl" / "t5-base" / "checkpoint-best"
)
ALL_LANG2LTL_WEIGHTS_URL = (
    "https://drive.google.com/drive/folders/"
    "1Rk_JICbHOArWZE6TRxwJZnHVd4wQ5abL?usp=sharing"
)


def ensure_lang2ltl_t5_checkpoint(
    checkpoint_path: Path,
    *,
    download: bool,
    url: str = DEFAULT_LANG2LTL_T5_CHECKPOINT_URL,
    verbose: bool = False,
) -> tuple[Path, dict[str, object]]:
    """Ensure the original Lang2LTL T5 symbolic translator checkpoint exists."""
    return ensure_lt_checkpoint(
        checkpoint_path,
        download=download,
        url=url,
        verbose=verbose,
    )
