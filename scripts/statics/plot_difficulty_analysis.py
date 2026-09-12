#!/usr/bin/env python3
"""Plot compact 2-panel and 3-panel benchmark difficulty figures."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


STATICS_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = STATICS_ROOT / "difficulty_analysis_summary.json"
DEFAULT_OUTPUT_DIR = STATICS_ROOT / "figures"
METHODS = ("GroundPlan", "OSG-LLM")
DISPLAY_NAMES = {
    "GroundPlan": "GroundPlan",
    "OSG-LLM": "OSG-LLM (Second Best)",
}
COLORS = {"GroundPlan": "#D95F32", "OSG-LLM": "#377EB8"}
MARKERS = {"GroundPlan": "o", "OSG-LLM": "s"}
PANELS = {
    "reference_depth": "Reference depth",
    "candidate_instances": "Candidates",
    "hard_constraints": "Hard constraints",
}
PANEL_TITLES = {
    "reference_depth": "Reference depth",
    "candidate_instances": "Candidate instances",
    "hard_constraints": "Hard constraints",
}
COMPACT_PANEL_TITLES = {
    "reference_depth": "Ref. depth",
    "candidate_instances": "Candidates",
    "hard_constraints": "Hard constr.",
}
COMPACT_AXIS_LABELS = {
    "reference_depth": "Ref. depth",
    "candidate_instances": "Candidates",
    "hard_constraints": "Hard constr.",
}


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("factors"), dict):
        raise ValueError(f"Invalid difficulty summary: {path}")
    return payload


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.35,
            "lines.markersize": 4.0,
            "figure.dpi": 150,
            "savefig.dpi": 400,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def factor_bins(summary: Mapping[str, Any], factor: str) -> Sequence[Mapping[str, Any]]:
    factor_summary = summary["factors"].get(factor)
    if not isinstance(factor_summary, Mapping):
        raise ValueError(f"Missing factor {factor!r}")
    bins = factor_summary.get("bins")
    if not isinstance(bins, Sequence) or isinstance(bins, (str, bytes)):
        raise ValueError(f"Missing bins for factor {factor!r}")
    return bins


def draw_panel(
    axis: Axes,
    summary: Mapping[str, Any],
    factor: str,
    panel_label: str,
    *,
    show_ylabel: bool,
    compact: bool,
) -> None:
    bins = factor_bins(summary, factor)
    positions = list(range(len(bins)))
    for method in METHODS:
        values = [float(item["mean_hcs"][method]) for item in bins]
        axis.plot(
            positions,
            values,
            color=COLORS[method],
            marker=MARKERS[method],
            markerfacecolor="white" if method == "OSG-LLM" else COLORS[method],
            markeredgewidth=0.9,
            label=DISPLAY_NAMES[method],
            zorder=3,
        )

    title = COMPACT_PANEL_TITLES[factor] if compact else PANEL_TITLES[factor]
    x_label = COMPACT_AXIS_LABELS[factor] if compact else PANELS[factor]
    axis.set_title(f"({panel_label}) {title}", loc="left", pad=3.0)
    axis.set_xticks(positions, [str(item["label"]) for item in bins])
    axis.set_xlabel(x_label, labelpad=2.0)
    if show_ylabel:
        axis.set_ylabel("Mean HCS ↑", labelpad=2.0)
    axis.set_ylim(0.0, 0.82)
    axis.set_yticks((0.0, 0.2, 0.4, 0.6, 0.8))
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, linestyle="--", alpha=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", length=2.2, width=0.6, pad=1.5)
    axis.spines["left"].set_color("#555555")
    axis.spines["bottom"].set_color("#555555")


def make_figure(summary: Mapping[str, Any], factors: Sequence[str]) -> Figure:
    # Both variants fit an ICRA-style single column.  The three-panel version
    # is intentionally denser so it can be compared at the actual paper size.
    width = 3.45
    figure, axes = plt.subplots(
        1,
        len(factors),
        figsize=(width, 1.72),
        sharey=True,
        facecolor="white",
    )
    axes_sequence = [axes] if len(factors) == 1 else list(axes)
    for index, (axis, factor) in enumerate(zip(axes_sequence, factors, strict=True)):
        draw_panel(
            axis,
            summary,
            factor,
            chr(ord("a") + index),
            show_ylabel=index == 0,
            compact=len(factors) == 3,
        )

    handles, labels = axes_sequence[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.6,
        handletextpad=0.5,
    )
    figure.subplots_adjust(
        left=0.115 if len(factors) == 2 else 0.075,
        right=0.99,
        bottom=0.25,
        top=0.79,
        wspace=0.20 if len(factors) == 2 else 0.24,
    )
    return figure


def save_figure(figure: Figure, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    figure.savefig(
        base_path.with_suffix(".png"),
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    summary = load_summary(args.input.resolve())
    output_dir = args.output_dir.resolve()
    variants = {
        "difficulty_analysis_2panel": ("reference_depth", "candidate_instances"),
        "difficulty_analysis_3panel": (
            "reference_depth",
            "candidate_instances",
            "hard_constraints",
        ),
    }
    for name, factors in variants.items():
        save_figure(make_figure(summary, factors), output_dir / name)
        print(f"saved {output_dir / (name + '.pdf')}")
        print(f"saved {output_dir / (name + '.png')}")


if __name__ == "__main__":
    main()
