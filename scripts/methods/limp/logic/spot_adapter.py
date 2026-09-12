"""Spot availability checks for the LIMP adapter."""

from __future__ import annotations


def spot_available() -> bool:
    try:
        import spot  # noqa: F401
    except Exception:
        return False
    return True

