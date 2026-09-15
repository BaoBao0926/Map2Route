#!/usr/bin/env python3
"""Plot the overall HCS of Grounding2Route and the seven comparison methods."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath


ROOT = Path(__file__).resolve().parents[2]
METHODS = [
    "Grounding2Route",
    "OSGLLM",
    "Lang2LTL",
    "LTLCodeGen",
    "Lang2LTL2",
    "SayPlan",
    "ILN",
    "LIMP",
]
DISPLAY_NAMES = [
    "Grounding2Route",
    "OSG-LLM",
    "Lang2LTL",
    "LTLCodeGen",
    "Lang2LTL-2",
    "SayPlan",
    "ILN",
    "LIMP",
]
BASELINE_COLORS = [
    "#239B91",
    "#3788B5",
    "#526FBB",
    "#6D61B3",
    "#865EAA",
    "#9F6CA6",
    "#B47F9F",
]


def load_overall_hcs(method: str) -> float:
    summary_path = ROOT / "resources" / "methods" / method / "summary.json"
    with summary_path.open(encoding="utf-8") as summary_file:
        summary = json.load(summary_file)
    return float(summary["overall"]["metric"]["HCS"])


def mix_color(color: str, target: str, amount: float) -> tuple[float, float, float]:
    """Blend a color toward a target by an amount in [0, 1]."""
    source_rgb = np.asarray(to_rgb(color))
    target_rgb = np.asarray(to_rgb(target))
    return tuple(source_rgb * (1.0 - amount) + target_rgb * amount)


def rounded_top_path(
    center: float,
    height: float,
    width: float,
    radius_x: float = 0.045,
    radius_y: float = 0.012,
) -> MplPath:
    """Return a bar path with flat bottom corners and rounded top corners."""
    left = center - width / 2
    right = center + width / 2
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


def draw_gradient_bar(
    axis: plt.Axes,
    center: float,
    height: float,
    width: float,
    bottom_color: str | tuple[float, float, float],
    top_color: str | tuple[float, float, float],
) -> None:
    """Draw a vector gradient clipped to a rounded-top bar."""
    patch = PathPatch(
        rounded_top_path(center, height, width),
        facecolor="none",
        edgecolor="none",
        transform=axis.transData,
    )

    bottom_rgb = np.asarray(to_rgb(bottom_color))
    top_rgb = np.asarray(to_rgb(top_color))
    step_count = 256
    blend = np.linspace(0.0, 1.0, step_count)[:, None]
    rgb = bottom_rgb[None, :] * (1.0 - blend) + top_rgb[None, :] * blend
    left = center - width / 2
    right = center + width / 2
    edges = np.linspace(0.0, height, step_count + 1)
    overlap = height / step_count * 0.03
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


def main() -> None:
    values = [load_overall_hcs(method) for method in METHODS]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15.5,
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
        }
    )
    fig, axis = plt.subplots(figsize=(9.8, 3.6), facecolor="white")
    axis.set_facecolor("white")

    # Keep the original method order, with a small visual gap after our method.
    positions = [0.0] + [index * 0.70 + 0.16 for index in range(1, len(METHODS))]
    bar_width = 0.35

    draw_gradient_bar(
        axis,
        positions[0],
        values[0],
        bar_width,
        bottom_color="#FF9A3D",
        top_color="#F04E5E",
    )
    for position, value, color in zip(
        positions[1:], values[1:], BASELINE_COLORS, strict=True
    ):
        draw_gradient_bar(
            axis,
            position,
            value,
            bar_width,
            bottom_color=color,
            top_color=mix_color(color, "#FFFFFF", 0.14),
        )

    axis.set_title(
        "Overall HCS ↑",
        pad=5,
        fontsize=19,
        fontweight="semibold",
        color="#15233C",
    )
    axis.set_ylim(0, 0.65)
    axis.set_yticks([0.2, 0.4, 0.6])
    axis.set_xticks([])
    axis.grid(axis="y", color="#DDE5EF", linewidth=0.7, zorder=1)
    axis.set_axisbelow(True)
    axis.tick_params(axis="x", length=0)
    axis.tick_params(axis="y", labelsize=15.5, length=0, pad=3, colors="#58677C")
    axis.spines["bottom"].set_color("#E6EBF2")
    axis.spines["bottom"].set_linewidth(0.6)
    # Draw method labels manually so the horizontal offset is not reset by
    # Matplotlib's tick layout during the final render.
    label_shifts = {
        "Grounding2Route": 0.40,
        "ILN": 0.10,
        "LIMP": 0.15,
    }
    for index, (position, name) in enumerate(
        zip(positions, DISPLAY_NAMES, strict=True)
    ):
        label_shift = label_shifts.get(name, 0.30)
        axis.text(
            position + label_shift,
            -0.055,
            name,
            transform=axis.get_xaxis_transform(),
            ha="right",
            va="top",
            rotation=15,
            rotation_mode="anchor",
            fontsize=15.5,
            fontstyle="normal",
            fontweight="semibold" if index == 0 else "normal",
            color="#263244",
            clip_on=False,
        )

    value_colors = ["#F04E5E"] + [
        mix_color(color, "#15233C", 0.28) for color in BASELINE_COLORS
    ]
    for index, (position, value) in enumerate(
        zip(positions, values, strict=True)
    ):
        axis.text(
            position,
            value + 0.009,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=17,
            fontweight="bold" if index == 0 else "normal",
            color=value_colors[index],
            zorder=4,
        )

    axis.margins(x=0.025)
    fig.tight_layout(pad=0.28)
    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "hcs_overview.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "hcs_overview.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
