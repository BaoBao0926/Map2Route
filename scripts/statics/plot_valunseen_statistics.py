#!/usr/bin/env python3
"""Create publication-ready valunseen dataset-statistics figures.

The figures compare easy and hard episodes only. Episode-level constraint counts
are normalized by the number of episodes in each split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch
from matplotlib.text import Text
from matplotlib.transforms import ScaledTranslation


STATICS_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = STATICS_ROOT / "valunseen_dataset_statistics.json"
DEFAULT_OUTPUT_DIR = STATICS_ROOT / "figures"
EASY_COLOR = "#2A9D8F"
HARD_COLOR = "#E76F51"
PASS_COLOR = "#4C78A8"
AVOID_COLOR = "#E15759"
SOFT_COLORS = {
    "near_preference": "#72B7B2",
    "far_preference": "#4C78A8",
    "relative_preference": "#B279A2",
    "path_shape_preference": "#F2CF5B",
}
SOFT_LABELS = {
    "near_preference": "Near",
    "far_preference": "Far",
    "relative_preference": "Relative",
    "path_shape_preference": "Path shape",
}
SOFT_ORDER = tuple(SOFT_LABELS)
SPLIT_BAR_POSITIONS = (0.0, 0.58)
SPLIT_BAR_WIDTH = 0.22


def load_report(report_path: Path) -> dict[str, Any]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {report_path}")
    return payload


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def section(report: dict[str, Any], difficulty: str) -> dict[str, Any]:
    return report["episode_statistics"][difficulty]


def per_episode(value: float, episode_count: int) -> float:
    return float(value) / episode_count


def add_panel_label(axis: Axes, label: str) -> None:
    title = axis.get_title(loc="left")
    axis.set_title("", loc="left")
    axis.text(
        0.0,
        1.02,
        label,
        transform=axis.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )
    axis.text(
        0.18,
        1.02,
        title,
        transform=axis.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def annotate_bars(
    axis: Axes,
    bars: Any,
    format_string: str,
    *,
    x_offset_points: float = 0.0,
) -> None:
    for bar in bars:
        height = bar.get_height()
        axis.annotate(
            format_string.format(height),
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(x_offset_points, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="medium",
        )


def apply_split_labels(axis: Axes, easy: dict[str, Any], hard: dict[str, Any]) -> None:
    axis.set_xticks(SPLIT_BAR_POSITIONS, ["Easy", "Hard"])


def draw_scene_inventory(axis: Axes, report: dict[str, Any]) -> None:
    scene = report["scene_statistics"]
    object_data = scene["object_instances"]
    room_data = scene["room_instances"]
    cards = (
        (0.05, 1.05, "{:.0f}".format(object_data["unique_category_count"]), "Object kinds", PASS_COLOR),
        (1.05, 1.05, "{:.0f}".format(room_data["unique_category_count"]), "Room kinds", PASS_COLOR),
        (0.05, 0.12, "{:.1f}".format(object_data["average_per_scene"]), "Objects / scene", EASY_COLOR),
        (1.05, 0.12, "{:.2f}".format(room_data["average_per_scene"]), "Rooms / scene", EASY_COLOR),
    )
    axis.set_xlim(0, 2)
    axis.set_ylim(0, 2)
    axis.set_axis_off()
    axis.set_title("Scene inventory ({} scenes)".format(scene["scene_count"]), loc="left", fontweight="bold")
    for left, bottom, value, label, color in cards:
        axis.add_patch(
            FancyBboxPatch(
                (left, bottom),
                0.82,
                0.68,
                boxstyle="round,pad=0.04,rounding_size=0.06",
                facecolor="#F7F8FA",
                edgecolor=color,
                linewidth=1.4,
            )
        )
        axis.text(left + 0.41, bottom + 0.42, value, ha="center", va="center", fontsize=16, fontweight="bold", color=color)
        axis.text(left + 0.41, bottom + 0.17, label, ha="center", va="center", fontsize=10)


def draw_instruction_length(
    axis: Axes,
    easy: dict[str, Any],
    hard: dict[str, Any],
    bar_width: float = SPLIT_BAR_WIDTH,
) -> None:
    values = [
        easy["instruction_length"]["average_words"],
        hard["instruction_length"]["average_words"],
    ]
    bars = axis.bar(SPLIT_BAR_POSITIONS, values, color=[EASY_COLOR, HARD_COLOR], width=bar_width)
    apply_split_labels(axis, easy, hard)
    axis.set_ylabel("Words / instruction")
    axis.set_title("Instruction", loc="left", fontweight="bold")
    annotate_bars(axis, bars, "{:.1f}")
    axis.set_ylim(0, max(values) * 1.22)


def draw_room_traversal(axis: Axes, easy: dict[str, Any], hard: dict[str, Any]) -> None:
    values = [
        easy["room_traversal"]["average_room_transitions"],
        hard["room_traversal"]["average_room_transitions"],
    ]
    bars = axis.bar(SPLIT_BAR_POSITIONS, values, color=[EASY_COLOR, HARD_COLOR], width=SPLIT_BAR_WIDTH)
    apply_split_labels(axis, easy, hard)
    axis.set_ylabel("Transitions / episode")
    axis.set_title("Traversal", loc="left", fontweight="bold")
    annotate_bars(axis, bars, "{:.2f}")
    axis.set_ylim(0, max(values) * 1.22)


def draw_candidate_instances(
    axis: Axes,
    easy: dict[str, Any],
    hard: dict[str, Any],
    grounding: dict[str, Any],
) -> None:
    values = [
        grounding["easy"]["candidate_instances_before_relations"]["all_references"]["mean"],
        grounding["hard"]["candidate_instances_before_relations"]["all_references"]["mean"],
    ]
    bars = axis.bar(SPLIT_BAR_POSITIONS, values, color=[EASY_COLOR, HARD_COLOR], width=SPLIT_BAR_WIDTH)
    apply_split_labels(axis, easy, hard)
    axis.set_ylabel("Instances / reference")
    axis.set_title("Candidates", loc="left", fontweight="bold")
    annotate_bars(axis, bars, "{:.2f}")
    axis.set_ylim(0, max(values) * 1.22)


def draw_reference_depth(
    axis: Axes,
    easy: dict[str, Any],
    hard: dict[str, Any],
    grounding: dict[str, Any],
) -> None:
    values = [
        grounding["easy"]["referential_relational_depth"]["episode_total"]["mean"],
        grounding["hard"]["referential_relational_depth"]["episode_total"]["mean"],
    ]
    bars = axis.bar(SPLIT_BAR_POSITIONS, values, color=[EASY_COLOR, HARD_COLOR], width=SPLIT_BAR_WIDTH)
    apply_split_labels(axis, easy, hard)
    axis.set_ylabel("Total depth / episode")
    axis.set_title("Depth", loc="left", fontweight="bold")
    annotate_bars(axis, bars, "{:.2f}")
    axis.set_ylim(0, max(values) * 1.22)


def draw_hard_constraints(axis: Axes, easy: dict[str, Any], hard: dict[str, Any]) -> None:
    splits = (easy, hard)
    pass_values = [
        per_episode(item["hard_constraints"]["must_pass"], item["episode_count"])
        for item in splits
    ]
    avoid_values = [
        per_episode(item["hard_constraints"]["must_avoid"], item["episode_count"])
        for item in splits
    ]
    axis.bar([0, 1], pass_values, color=PASS_COLOR, width=0.62, label="Must pass")
    axis.bar([0, 1], avoid_values, bottom=pass_values, color=AVOID_COLOR, width=0.62, label="Must avoid")
    apply_split_labels(axis, easy, hard)
    axis.set_ylabel("Average constraints per episode")
    axis.set_title("Hard constraints", loc="left", fontweight="bold")
    for position, total in enumerate(a + b for a, b in zip(pass_values, avoid_values)):
        axis.annotate(f"{total:.2f}", (position, total), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10, fontweight="medium")
    axis.legend(frameon=False, loc="upper left")
    axis.set_ylim(0, max(a + b for a, b in zip(pass_values, avoid_values)) * 1.26)


def draw_soft_constraints(axis: Axes, easy: dict[str, Any], hard: dict[str, Any]) -> None:
    splits = (easy, hard)
    bottom = [0.0, 0.0]
    for preference_type in SOFT_ORDER:
        values = [
            per_episode(
                item["soft_constraints_excluding_clearance"]["by_preference_type"].get(preference_type, 0),
                item["episode_count"],
            )
            for item in splits
        ]
        axis.bar(
            [0, 1],
            values,
            bottom=bottom,
            color=SOFT_COLORS[preference_type],
            width=0.62,
            label=SOFT_LABELS[preference_type],
        )
        bottom = [existing + value for existing, value in zip(bottom, values)]
    apply_split_labels(axis, easy, hard)
    axis.set_ylabel("Average non-clearance SCS constraints / episode")
    axis.set_title("Soft constraints (clearance excluded)", loc="left", fontweight="bold")
    for position, total in enumerate(bottom):
        axis.annotate(f"{total:.2f}", (position, total), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10, fontweight="medium")
    axis.legend(frameon=False, loc="upper left", ncol=2, columnspacing=1.1, handlelength=1.1)
    axis.set_ylim(0, max(bottom) * 1.28)


def save_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
    *,
    tight: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_options = {"bbox_inches": "tight"} if tight else {}
    figure.savefig(output_dir / f"{stem}.pdf", **save_options)
    figure.savefig(output_dir / f"{stem}.png", **save_options)
    plt.close(figure)


def adjust_figure_font_sizes(figure: plt.Figure, delta: float) -> None:
    """Shift every rendered text size while preserving relative hierarchy."""

    for text_artist in figure.findobj(match=Text):
        text_artist.set_fontsize(max(1.0, text_artist.get_fontsize() + delta))


def build_figures(report: dict[str, Any], output_dir: Path) -> None:
    create_main_statistics_figure(report, output_dir)
    create_two_row_statistics_figure(report, output_dir)


def soft_per_episode(item: dict[str, Any], preference_type: str | None = None) -> float:
    soft = item["soft_constraints_excluding_clearance"]
    value = soft["total"] if preference_type is None else soft["by_preference_type"].get(preference_type, 0)
    return per_episode(value, item["episode_count"])


def draw_grouped_constraint_bars(
    axis: Axes,
    metrics: tuple[tuple[str, float, float], ...],
    *,
    title: str,
    show_legend: bool = False,
) -> None:
    """Draw one non-stacked Easy/Hard constraint group."""

    positions = [index * 1.10 for index in range(len(metrics))]
    width = 0.20
    pair_offset = 0.115
    easy_values = [metric[1] for metric in metrics]
    hard_values = [metric[2] for metric in metrics]
    easy_bars = axis.bar(
        [position - pair_offset for position in positions],
        easy_values,
        width=width,
        color=EASY_COLOR,
        label="Easy",
    )
    hard_bars = axis.bar(
        [position + pair_offset for position in positions],
        hard_values,
        width=width,
        color=HARD_COLOR,
        label="Hard",
    )
    axis.set_xticks(positions, [metric[0] for metric in metrics])
    axis.set_ylabel("Constraints / episode")
    axis.set_title(title, loc="left", fontweight="bold")
    headroom = 1.65 if show_legend else 1.25
    axis.set_ylim(0, max(easy_values + hard_values) * headroom)
    axis.yaxis.grid(True, color="#D9DDE3", linewidth=0.7)
    axis.set_axisbelow(True)
    if show_legend:
        axis.legend(
            handles=(easy_bars[0], hard_bars[0]),
            labels=("Easy", "Hard"),
            loc="upper right",
            ncol=2,
            frameon=False,
            handlelength=1.4,
            columnspacing=2.0,
        )
    annotate_bars(axis, easy_bars, "{:.2f}", x_offset_points=-14.0)
    annotate_bars(axis, hard_bars, "{:.2f}")


def draw_hard_constraint_comparison(
    axis: Axes,
    easy: dict[str, Any],
    hard: dict[str, Any],
) -> None:
    metrics = (
        (
            "Must pass",
            per_episode(easy["hard_constraints"]["must_pass"], easy["episode_count"]),
            per_episode(hard["hard_constraints"]["must_pass"], hard["episode_count"]),
        ),
        (
            "Must avoid",
            per_episode(easy["hard_constraints"]["must_avoid"], easy["episode_count"]),
            per_episode(hard["hard_constraints"]["must_avoid"], hard["episode_count"]),
        ),
    )
    draw_grouped_constraint_bars(
        axis,
        metrics,
        title="Hard constraints",
    )
    must_pass_label = axis.get_xticklabels()[0]
    must_pass_label.set_rotation(10)
    must_pass_label.set_rotation_mode("anchor")
    must_pass_label.set_ha("right")


def draw_soft_constraint_comparison(
    axis: Axes,
    easy: dict[str, Any],
    hard: dict[str, Any],
) -> None:
    metrics = (
        ("Near", soft_per_episode(easy, "near_preference"), soft_per_episode(hard, "near_preference")),
        ("Far", soft_per_episode(easy, "far_preference"), soft_per_episode(hard, "far_preference")),
        (
            "Relative",
            soft_per_episode(easy, "relative_preference"),
            soft_per_episode(hard, "relative_preference"),
        ),
        (
            "Path shape",
            soft_per_episode(easy, "path_shape_preference"),
            soft_per_episode(hard, "path_shape_preference"),
        ),
    )
    draw_grouped_constraint_bars(
        axis,
        metrics,
        title="Soft constraints",
        show_legend=True,
    )


def draw_unified_episode_statistics(
    axis: Axes,
    easy: dict[str, Any],
    hard: dict[str, Any],
    grounding: dict[str, Any],
) -> None:
    # Annotation offsets use points; 4.32 pt renders as 18 px at 300 DPI.
    easy_value_offset_points = -(18.0 * 72.0 / 300.0)
    # 2.4 pt renders as 10 px at 300 DPI.
    hard_value_offset_points = 10.0 * 72.0 / 300.0
    composition_metrics = (
        ("Must pass", per_episode(easy["hard_constraints"]["must_pass"], easy["episode_count"]), per_episode(hard["hard_constraints"]["must_pass"], hard["episode_count"])),
        ("Must avoid", per_episode(easy["hard_constraints"]["must_avoid"], easy["episode_count"]), per_episode(hard["hard_constraints"]["must_avoid"], hard["episode_count"])),
        ("Near", soft_per_episode(easy, "near_preference"), soft_per_episode(hard, "near_preference")),
        ("Far", soft_per_episode(easy, "far_preference"), soft_per_episode(hard, "far_preference")),
        ("Relative", soft_per_episode(easy, "relative_preference"), soft_per_episode(hard, "relative_preference")),
        ("Path-Shape", soft_per_episode(easy, "path_shape_preference"), soft_per_episode(hard, "path_shape_preference")),
    )
    width = 0.31
    instruction_position = 0
    room_position = 2
    candidate_position = 4
    depth_position = 6
    trajectory_position = 8
    composition_positions = list(range(10, 10 + len(composition_metrics)))

    word_values = [easy["instruction_length"]["average_words"], hard["instruction_length"]["average_words"]]
    easy_word_bars = axis.bar([instruction_position - width / 2], [word_values[0]], width=width, color=EASY_COLOR, label="Easy")
    hard_word_bars = axis.bar([instruction_position + width / 2], [word_values[1]], width=width, color=HARD_COLOR, label="Hard")
    axis.set_ylabel("Words")
    axis.set_ylim(0, 120)
    axis.yaxis.grid(True, color="#D9DDE3", linewidth=0.7)
    axis.set_axisbelow(True)
    annotate_bars(
        axis,
        easy_word_bars,
        "{:.1f}",
        x_offset_points=easy_value_offset_points,
    )
    annotate_bars(
        axis,
        hard_word_bars,
        "{:.1f}",
        x_offset_points=hard_value_offset_points,
    )

    room_axis = axis.twinx()
    room_values = [easy["room_traversal"]["average_room_transitions"], hard["room_traversal"]["average_room_transitions"]]
    easy_room_bars = room_axis.bar([room_position - width / 2], [room_values[0]], width=width, color=EASY_COLOR)
    hard_room_bars = room_axis.bar([room_position + width / 2], [room_values[1]], width=width, color=HARD_COLOR)
    room_axis.set_ylabel("Rooms crossed / episode")
    room_axis.set_ylim(0, max(room_values) * 1.28)
    room_axis.spines["top"].set_visible(False)
    room_axis.spines["right"].set_visible(False)
    room_axis.spines["left"].set_position(("data", 1.0))
    room_axis.spines["left"].set_linestyle("--")
    room_axis.spines["left"].set_color("#6B7280")
    room_axis.yaxis.set_label_position("left")
    room_axis.yaxis.tick_left()
    room_axis.tick_params(axis="y", labelsize=7, pad=2)
    annotate_bars(
        room_axis,
        easy_room_bars,
        "{:.2f}",
        x_offset_points=easy_value_offset_points,
    )
    annotate_bars(
        room_axis,
        hard_room_bars,
        "{:.2f}",
        x_offset_points=hard_value_offset_points,
    )

    composition_axis = axis.twinx()
    easy_composition = [metric[1] for metric in composition_metrics]
    hard_composition = [metric[2] for metric in composition_metrics]
    easy_composition_bars = composition_axis.bar([position - width / 2 for position in composition_positions], easy_composition, width=width, color=EASY_COLOR)
    hard_composition_bars = composition_axis.bar([position + width / 2 for position in composition_positions], hard_composition, width=width, color=HARD_COLOR)
    composition_axis.set_ylabel("Average constraints / episode")
    composition_axis.set_ylim(0, 5.35)
    composition_axis.spines["top"].set_visible(False)
    composition_axis.spines["right"].set_visible(False)
    composition_axis.spines["left"].set_position(("data", composition_positions[0] - 1.0))
    composition_axis.spines["left"].set_linestyle("--")
    composition_axis.spines["left"].set_color("#6B7280")
    composition_axis.yaxis.set_label_position("left")
    composition_axis.yaxis.tick_left()
    composition_axis.tick_params(axis="y", labelsize=7, pad=2)
    annotate_bars(
        composition_axis,
        easy_composition_bars,
        "{:.2f}",
        x_offset_points=easy_value_offset_points,
    )
    for index, bar in enumerate(hard_composition_bars):
        # Give the Relative Hard value (0.33) an additional 10 px right shift.
        hard_offset = (
            20.0 * 72.0 / 300.0
            if index == 4
            else hard_value_offset_points
        )
        annotate_bars(
            composition_axis,
            (bar,),
            "{:.2f}",
            x_offset_points=hard_offset,
        )
    composition_axis.axvline(composition_positions[1] + 0.5, color="#A9B0B8", linewidth=0.8, linestyle="--")


    easy_grounding = grounding["easy"]
    hard_grounding = grounding["hard"]
    candidate_values = [
        easy_grounding["candidate_instances_before_relations"]["all_references"]["mean"],
        hard_grounding["candidate_instances_before_relations"]["all_references"]["mean"],
    ]
    candidate_axis = axis.twinx()
    candidate_axis.spines["right"].set_visible(False)
    candidate_axis.spines["top"].set_visible(False)
    candidate_axis.spines["left"].set_position(("data", candidate_position - 1.0))
    candidate_axis.spines["left"].set_linestyle("--")
    candidate_axis.spines["left"].set_color("#6B7280")
    candidate_axis.yaxis.set_label_position("left")
    candidate_axis.yaxis.tick_left()
    candidate_axis.tick_params(axis="y", labelsize=7, pad=2)
    easy_candidate_bars = candidate_axis.bar([candidate_position - width / 2], [candidate_values[0]], width=width, color=EASY_COLOR)
    hard_candidate_bars = candidate_axis.bar([candidate_position + width / 2], [candidate_values[1]], width=width, color=HARD_COLOR)
    candidate_axis.set_ylabel("Instances / reference")
    candidate_axis.set_ylim(0, max(candidate_values) * 1.28)
    annotate_bars(
        candidate_axis,
        easy_candidate_bars,
        "{:.2f}",
        x_offset_points=easy_value_offset_points,
    )
    annotate_bars(
        candidate_axis,
        hard_candidate_bars,
        "{:.2f}",
        x_offset_points=hard_value_offset_points,
    )

    depth_values = [
        easy_grounding["referential_relational_depth"]["episode_total"]["mean"],
        hard_grounding["referential_relational_depth"]["episode_total"]["mean"],
    ]
    depth_axis = axis.twinx()
    depth_axis.spines["right"].set_visible(False)
    depth_axis.spines["top"].set_visible(False)
    depth_axis.spines["left"].set_position(("data", depth_position - 1.0))
    depth_axis.spines["left"].set_linestyle("--")
    depth_axis.spines["left"].set_color("#6B7280")
    depth_axis.yaxis.set_label_position("left")
    depth_axis.yaxis.tick_left()
    depth_axis.tick_params(axis="y", labelsize=7, pad=2)
    easy_depth_bars = depth_axis.bar([depth_position - width / 2], [depth_values[0]], width=width, color=EASY_COLOR)
    hard_depth_bars = depth_axis.bar([depth_position + width / 2], [depth_values[1]], width=width, color=HARD_COLOR)
    depth_axis.set_ylabel("Depth / episode")
    depth_axis.set_ylim(0, max(depth_values) * 1.35)
    annotate_bars(
        depth_axis,
        easy_depth_bars,
        "{:.2f}",
        x_offset_points=easy_value_offset_points,
    )
    annotate_bars(
        depth_axis,
        hard_depth_bars,
        "{:.2f}",
        x_offset_points=hard_value_offset_points,
    )

    trajectory_values = [28.0, 66.2]
    trajectory_axis = axis.twinx()
    trajectory_axis.spines["right"].set_visible(False)
    trajectory_axis.spines["top"].set_visible(False)
    trajectory_axis.spines["left"].set_position(("data", trajectory_position - 1.0))
    trajectory_axis.spines["left"].set_linestyle("--")
    trajectory_axis.spines["left"].set_color("#6B7280")
    trajectory_axis.yaxis.set_label_position("left")
    trajectory_axis.yaxis.tick_left()
    trajectory_axis.tick_params(axis="y", labelsize=7, pad=2)
    easy_trajectory_bars = trajectory_axis.bar(
        [trajectory_position - width / 2],
        [trajectory_values[0]],
        width=width,
        color=EASY_COLOR,
    )
    hard_trajectory_bars = trajectory_axis.bar(
        [trajectory_position + width / 2],
        [trajectory_values[1]],
        width=width,
        color=HARD_COLOR,
    )
    trajectory_axis.set_ylabel("Meters / episode")
    trajectory_axis.set_ylim(0, max(trajectory_values) * 1.28)
    annotate_bars(
        trajectory_axis,
        easy_trajectory_bars,
        "{:.1f}",
        x_offset_points=easy_value_offset_points,
    )
    annotate_bars(
        trajectory_axis,
        hard_trajectory_bars,
        "{:.1f}",
        x_offset_points=hard_value_offset_points,
    )

    labels = [
        "Instruction length",
        "Crossed room",
        "Initial candidates",
        "Reference depth",
        "Trajectory length",
        *[metric[0] for metric in composition_metrics],
    ]
    positions = [
        instruction_position,
        room_position,
        candidate_position,
        depth_position,
        trajectory_position,
        *composition_positions,
    ]
    axis.set_xticks(positions, labels)
    axis.set_xlim(-0.8, composition_positions[-1] + 0.8)
    tick_labels = axis.get_xticklabels()
    must_pass_label = tick_labels[5]
    must_pass_label.set_transform(
        must_pass_label.get_transform()
        + ScaledTranslation(-9.0 / 72.0, 0.0, axis.figure.dpi_scale_trans)
    )
    relative_label = tick_labels[9]
    relative_label.set_transform(
        relative_label.get_transform()
        + ScaledTranslation(-4.0 / 72.0, 0.0, axis.figure.dpi_scale_trans)
    )
    path_shape_label = tick_labels[10]
    path_shape_label.set_transform(
        path_shape_label.get_transform()
        + ScaledTranslation(4.0 / 72.0, 0.0, axis.figure.dpi_scale_trans)
    )
    axis.legend(frameon=False, loc="upper right", ncol=2)
    header_fontsize = 13
    axis.text(
        instruction_position, 1.02, "Instruction",
        transform=axis.get_xaxis_transform(), ha="center",
        fontsize=header_fontsize, fontweight="bold",
    )
    axis.text(
        room_position, 1.02, "Room traversal",
        transform=axis.get_xaxis_transform(), ha="center",
        fontsize=header_fontsize, fontweight="bold",
    )
    axis.text(
        candidate_position, 1.02, "Candidate instances",
        transform=axis.get_xaxis_transform(), ha="center",
        fontsize=header_fontsize, fontweight="bold",
    )
    axis.text(
        depth_position + 0.12, 1.02, "Reference depth",
        transform=axis.get_xaxis_transform(), ha="center",
        fontsize=header_fontsize, fontweight="bold",
    )
    axis.text(
        trajectory_position + 0.12, 1.02, "Trajectory length",
        transform=axis.get_xaxis_transform(), ha="center",
        fontsize=header_fontsize, fontweight="bold",
    )
    axis.text(
        sum(composition_positions[:2]) / 2, 1.02, "Hard constraints",
        transform=axis.get_xaxis_transform(), ha="center",
        fontsize=header_fontsize, fontweight="bold",
    )
    axis.text(
        sum(composition_positions[2:]) / len(composition_positions[2:]),
        1.02,
        "Soft constraints",
        transform=axis.get_xaxis_transform(),
        ha="center",
        fontsize=header_fontsize,
        fontweight="bold",
    )

def create_main_statistics_figure(report: dict[str, Any], output_dir: Path) -> None:
    easy = section(report, "easy")
    hard = section(report, "hard")
    figure, statistics = plt.subplots(figsize=(15.5, 3.5), layout="constrained")
    draw_unified_episode_statistics(statistics, easy, hard, report["grounding_statistics"])
    adjust_figure_font_sizes(figure, -1.0)
    save_figure(figure, output_dir, "valunseen_main_statistics_figure")


def create_two_row_statistics_figure(report: dict[str, Any], output_dir: Path) -> None:
    """Save the same statistics in a roomier two-row paper layout."""

    easy = section(report, "easy")
    hard = section(report, "hard")
    grounding = report["grounding_statistics"]

    figure = plt.figure(figsize=(7.1, 5.2))
    grid = figure.add_gridspec(
        2,
        1,
        height_ratios=(1.08, 1.0),
        hspace=0.62,
    )
    top_grid = grid[0].subgridspec(1, 2, width_ratios=(0.62, 1.38), wspace=0.34)
    bottom_grid = grid[1].subgridspec(1, 4, wspace=1.00)
    figure.subplots_adjust(left=0.12, right=0.975, bottom=0.12, top=0.90)
    hard_constraint_axis = figure.add_subplot(top_grid[0, 0])
    soft_constraint_axis = figure.add_subplot(top_grid[0, 1])
    instruction_axis = figure.add_subplot(bottom_grid[0, 0])
    room_axis = figure.add_subplot(bottom_grid[0, 1])
    candidate_axis = figure.add_subplot(bottom_grid[0, 2])
    depth_axis = figure.add_subplot(bottom_grid[0, 3])

    draw_hard_constraint_comparison(hard_constraint_axis, easy, hard)
    draw_soft_constraint_comparison(soft_constraint_axis, easy, hard)
    draw_instruction_length(instruction_axis, easy, hard)
    draw_room_traversal(room_axis, easy, hard)
    draw_candidate_instances(candidate_axis, easy, hard, grounding)
    draw_reference_depth(depth_axis, easy, hard, grounding)

    for axis in (instruction_axis, room_axis, candidate_axis, depth_axis):
        axis.yaxis.grid(True, color="#D9DDE3", linewidth=0.7)
        axis.set_axisbelow(True)

    for axis, label in zip(
        (
            hard_constraint_axis,
            soft_constraint_axis,
            instruction_axis,
            room_axis,
            candidate_axis,
            depth_axis,
        ),
        ("a", "b", "c", "d", "e", "f"),
    ):
        add_panel_label(axis, label)

    save_figure(
        figure,
        output_dir,
        "valunseen_two_row_statistics_figure",
        tight=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create valunseen statistics figures for a paper.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    build_figures(load_report(args.input), args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()
