#!/usr/bin/env python3
"""Plot the Easy/Hard HCS gap between the best baseline, GroundPlan, and humans."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "figures" / "hcs_current_gap"
SERIES = ("Best Baseline", "GroundPlan", "Human")
COLORS = ("#087CA5", "#E95727", "#6552B5")


def rounded_top_path(
    center: float,
    height: float,
    width: float,
    *,
    radius_x: float = 0.06,
    radius_y: float = 0.035,
) -> MplPath:
    """Return a bar outline with square lower and rounded upper corners."""
    left = center - width / 2
    right = center + width / 2
    radius_x = min(radius_x, width / 2)
    radius_y = min(radius_y, height)
    vertices = [
        (left, 0.0),
        (left, height - radius_y),
        (left, height),
        (left + radius_x, height),
        (right - radius_x, height),
        (right, height),
        (right, height - radius_y),
        (right, 0.0),
        (left, 0.0),
        (left, 0.0),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.CLOSEPOLY,
    ]
    return MplPath(vertices, codes)


def mix_color(color: str, target: str, amount: float) -> tuple[float, float, float]:
    source_rgb = np.asarray(to_rgb(color))
    target_rgb = np.asarray(to_rgb(target))
    return tuple(source_rgb * (1.0 - amount) + target_rgb * amount)


def draw_gradient_bar(
    axis: plt.Axes,
    center: float,
    height: float,
    width: float,
    color: str,
) -> None:
    """Draw a vertical gradient clipped to a rounded-top bar."""
    patch = PathPatch(
        rounded_top_path(center, height, width),
        facecolor="none",
        edgecolor="none",
        transform=axis.transData,
    )
    bottom_rgb = np.asarray(to_rgb(color))
    top_rgb = np.asarray(mix_color(color, "#FFFFFF", 0.12))
    step_count = 256
    blend = np.linspace(0.0, 1.0, step_count)[:, None]
    rgb = bottom_rgb[None, :] * (1.0 - blend) + top_rgb[None, :] * blend
    edges = np.linspace(0.0, height, step_count + 1)
    left = center - width / 2
    right = center + width / 2
    overlap = height / step_count * 0.05
    polygons = [
        [(left, lower), (right, lower), (right, upper + overlap), (left, upper + overlap)]
        for lower, upper in zip(edges[:-1], edges[1:], strict=True)
    ]
    gradient = PolyCollection(
        polygons,
        facecolors=rgb,
        edgecolors="none",
        linewidths=0,
        antialiaseds=False,
        zorder=3,
    )
    gradient.set_clip_path(patch)
    axis.add_collection(gradient)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best-baseline-easy", type=float, default=0.368)
    parser.add_argument("--best-baseline-hard", type=float, default=0.124)
    parser.add_argument("--groundplan-easy", type=float, default=0.667)
    parser.add_argument("--groundplan-hard", type=float, default=0.375)
    parser.add_argument("--human-easy", type=float, default=1.0)
    parser.add_argument("--human-hard", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path without an extension; both PNG and PDF are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = np.asarray(
        [
            [args.best_baseline_easy, args.groundplan_easy, args.human_easy],
            [args.best_baseline_hard, args.groundplan_hard, args.human_hard],
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("Every HCS value must be finite and within [0, 1].")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axis = plt.subplots(figsize=(8.2, 3.8), facecolor="white")
    axis.set_facecolor("white")

    group_centers = np.asarray([0.0, 2.05])
    offsets = np.asarray([-0.42, 0.0, 0.42])
    bar_width = 0.34
    for group_index, group_center in enumerate(group_centers):
        for series_index, offset in enumerate(offsets):
            value = float(values[group_index, series_index])
            center = float(group_center + offset)
            draw_gradient_bar(axis, center, value, bar_width, COLORS[series_index])
            axis.text(
                center,
                value + 0.028,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
                color="#111111",
                zorder=4,
            )

    legend_handles = [
        Patch(facecolor=color, edgecolor="none", label=name)
        for name, color in zip(SERIES, COLORS, strict=True)
    ]
    axis.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=3,
        frameon=False,
        fontsize=15,
        handlelength=1.25,
        handleheight=1.0,
        columnspacing=2.1,
        handletextpad=0.55,
    )

    axis.set_xlim(-0.90, 2.95)
    axis.set_ylim(0.0, 1.12)
    axis.set_xticks(group_centers, ("Easy", "Hard"))
    axis.set_yticks(np.linspace(0.0, 1.0, 6))
    axis.set_yticklabels([f"{value:.1f}" for value in np.linspace(0.0, 1.0, 6)])
    axis.grid(axis="y", color="#D8DDE6", linestyle="--", linewidth=0.8, zorder=1)
    axis.set_axisbelow(True)
    axis.tick_params(axis="x", labelsize=22, length=0, pad=10)
    axis.tick_params(axis="y", labelsize=13, colors="#222222")
    axis.spines["left"].set_color("#4B4B4B")
    axis.spines["left"].set_linewidth(1.0)
    axis.spines["bottom"].set_color("#4B4B4B")
    axis.spines["bottom"].set_linewidth(1.0)

    figure.subplots_adjust(left=0.09, right=0.99, bottom=0.18, top=0.83)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
