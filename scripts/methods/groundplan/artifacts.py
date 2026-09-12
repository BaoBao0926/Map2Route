"""Debug artifact writers for GroundPlan runs."""

from __future__ import annotations

import json
from pathlib import Path
from pprint import pformat
from typing import Mapping, Sequence

import numpy as np

from scripts.methods.groundplan.ir import GPExpr, GPProgramSpec
from scripts.methods.groundplan.pipeline import GroundPlanDebugData
from scripts.methods.groundplan.planner.semantic_astar import (
    _compile_hard,
    _compile_soft,
    _goal_mask,
    _traversable_mask,
)
from scripts.methods.util.methods import (
    constraint_overlays_from_instruction,
    display_path,
    write_trajectory_image,
)


def artifact_dir_for_prediction(output_path: Path) -> Path:
    return output_path.with_suffix(".artifacts")


def save_groundplan_debug_artifacts(
    *,
    output_path: Path,
    map_state: Mapping[str, object],
    instruction: Mapping[str, object],
    trajectory: Sequence[Sequence[int]],
    steps: Mapping[str, object],
    debug_data: GroundPlanDebugData | None,
) -> dict[str, object]:
    artifact_dir = artifact_dir_for_prediction(output_path)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_known_artifacts(artifact_dir)

    written: dict[str, object] = {"directory": display_path(artifact_dir)}
    parser_path = artifact_dir / "parser.json"
    parse_payload = _mapping_or_status(steps.get("parse"))
    _write_json(parser_path, parse_payload)
    written["parser_json"] = display_path(parser_path)
    parser_intent = _parser_intent_source(parse_payload)
    if parser_intent:
        parser_intent_path = artifact_dir / "parser_intent.json"
        parser_intent_path.write_text(parser_intent.rstrip() + "\n", encoding="utf-8")
        written["parser_intent_json"] = display_path(parser_intent_path)
    parser_api_source = _parser_api_source(parse_payload)
    if parser_api_source:
        parser_api_path = artifact_dir / "parser_api.py"
        parser_api_path.write_text(parser_api_source.rstrip() + "\n", encoding="utf-8")
        written["parser_api_py"] = display_path(parser_api_path)
    parser_program_source = _parser_program_source(parse_payload)
    if parser_program_source:
        parser_program_path = artifact_dir / "parser_program.txt"
        parser_program_path.write_text(parser_program_source.rstrip() + "\n", encoding="utf-8")
        written["parser_program_txt"] = display_path(parser_program_path)

    grounding_path = artifact_dir / "grounding.json"
    grounding_payload = _mapping_or_status(steps.get("grounding"))
    _write_json(grounding_path, grounding_payload)
    written["grounding_json"] = display_path(grounding_path)

    grounded_program_path = artifact_dir / "grounded_program.py"
    grounded_program_path.write_text(
        "# Generated GroundPlan grounded IR for inspection only.\n"
        f"GROUNDED_PROGRAM = {pformat(grounding_payload, width=100, sort_dicts=False)}\n",
        encoding="utf-8",
    )
    written["grounded_program_py"] = display_path(grounded_program_path)
    if debug_data is not None and debug_data.program is not None:
        grounding_trace_path = artifact_dir / "grounding_trace.py"
        grounding_trace_path.write_text(
            _grounding_trace_source(debug_data.program, grounding_payload),
            encoding="utf-8",
        )
        written["grounding_trace_py"] = display_path(grounding_trace_path)

    planner_path = artifact_dir / "planner.json"
    _write_json(planner_path, _mapping_or_status(steps.get("planner")))
    written["planner_json"] = display_path(planner_path)

    semantic_map_path = artifact_dir / "semantic_map.png"
    gt_trajectory = _gt_trajectory(instruction)
    write_trajectory_image(
        map_state,
        trajectory,
        semantic_map_path,
        constraint_overlays=constraint_overlays_from_instruction(instruction),
        gt_trajectory=gt_trajectory,
    )
    written["semantic_map_png"] = display_path(semantic_map_path)

    if debug_data is not None and debug_data.scene is not None and debug_data.grounded is not None:
        cost_map_path = artifact_dir / "cost_map.png"
        if _write_cost_map(cost_map_path, debug_data):
            written["cost_map_png"] = display_path(cost_map_path)

    manifest_path = artifact_dir / "manifest.json"
    _write_json(manifest_path, written)
    written["manifest_json"] = display_path(manifest_path)
    return written


def _mapping_or_status(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {"status": "missing"}


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _cleanup_known_artifacts(artifact_dir: Path) -> None:
    for name in (
        "parser.json",
        "parser_api.py",
        "parser_intent.json",
        "parser_program.txt",
        "parser.dsl",
        "grounding.json",
        "grounded_program.py",
        "grounding_trace.py",
        "planner.json",
        "cost_map.png",
        "semantic_map.png",
        "manifest.json",
    ):
        path = artifact_dir / name
        if path.exists() and path.is_file():
            path.unlink()


def _parser_api_source(parse_payload: Mapping[str, object]) -> str | None:
    metadata = parse_payload.get("metadata")
    if isinstance(metadata, Mapping):
        raw_api_program = metadata.get("raw_api_program")
        if isinstance(raw_api_program, str) and raw_api_program.strip():
            return raw_api_program
    program = parse_payload.get("program")
    if isinstance(program, Mapping) and program.get("parser_mode") == "api":
        source = program.get("source")
        if isinstance(source, str) and source.strip():
            return source
    return None


def _parser_intent_source(parse_payload: Mapping[str, object]) -> str | None:
    metadata = parse_payload.get("metadata")
    if isinstance(metadata, Mapping):
        raw_intent = metadata.get("raw_intent")
        if isinstance(raw_intent, str) and raw_intent.strip():
            return raw_intent
    return None


def _parser_program_source(parse_payload: Mapping[str, object]) -> str | None:
    program = parse_payload.get("program")
    if isinstance(program, Mapping):
        source = program.get("source")
        if isinstance(source, str) and source.strip():
            return source
    return None


def _grounding_trace_source(
    program: GPProgramSpec,
    grounding_payload: Mapping[str, object],
) -> str:
    bindings = grounding_payload.get("bindings")
    binding_results = bindings if isinstance(bindings, Mapping) else {}
    identifiers = {
        "task_start",
        "start_position",
        "segment_start",
        *(binding.name for binding in program.bindings),
        *(segment.id for segment in program.segments),
    }
    lines = [
        "# Generated GroundPlan grounding trace for inspection only.",
        "# The real method does not execute this file. It runs",
        "# scripts/methods/groundplan/grounding/grounder.py as a deterministic interpreter.",
        "",
        "env = {}",
        "env['task_start'] = '<scene.start>'",
        "env['start_position'] = env['task_start']",
        "",
        "# Binding evaluation, mirroring ground_program(...):",
    ]
    for binding in program.bindings:
        expr_source = _expr_source(binding.expr, identifiers)
        result = _binding_result_summary(binding_results.get(binding.name))
        lines.append(f"{binding.name} = {expr_source}")
        lines.append(f"env[{binding.name!r}] = {binding.name}  # -> {result}")
    lines.extend(["", "# Segment grounding:"])
    for segment in program.segments:
        lines.append(f"# segment {segment.id}")
        lines.append(f"start_ref = {_expr_source(segment.start, identifiers)}")
        lines.append(f"target_ref = {_expr_source(segment.target, identifiers)}")
        for constraint in segment.constraints:
            args = ", ".join(_expr_source(expr, identifiers) for expr in constraint.exprs)
            if constraint.spatial_scope is not None:
                args = f"{args}, spatial_scope={_expr_source(constraint.spatial_scope, identifiers)}"
            lines.append(f"{constraint.kind}({args})")
    lines.extend(
        [
            "",
            "# Final grounded IR:",
            f"GROUNDED_PROGRAM = {pformat(dict(grounding_payload), width=100, sort_dicts=False)}",
            "",
        ]
    )
    return "\n".join(lines)


def _expr_source(expr: object, identifiers: set[str]) -> str:
    if isinstance(expr, GPExpr):
        args = [_expr_source(arg, identifiers) for arg in expr.args]
        args.extend(f"{key}={_expr_source(value, identifiers)}" for key, value in expr.kwargs.items())
        return f"{expr.op}({', '.join(args)})"
    if isinstance(expr, str):
        if expr in identifiers:
            return expr
        return json.dumps(expr)
    if isinstance(expr, bool):
        return "True" if expr else "False"
    if expr is None:
        return "None"
    if isinstance(expr, list):
        return "[" + ", ".join(_expr_source(item, identifiers) for item in expr) + "]"
    if isinstance(expr, tuple):
        return "(" + ", ".join(_expr_source(item, identifiers) for item in expr) + ")"
    return repr(expr)


def _binding_result_summary(value: object) -> str:
    if isinstance(value, Mapping):
        return _ref_summary(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = [_binding_result_summary(item) for item in value[:6]]
        suffix = "" if len(value) <= 6 else f", ... total={len(value)}"
        return "[" + ", ".join(items) + suffix + "]"
    return repr(value)


def _ref_summary(value: Mapping[str, object]) -> str:
    kind = value.get("kind")
    item_id = value.get("id")
    category = value.get("category")
    room_id = value.get("room_id")
    center = value.get("center")
    parts = [str(item_id or "<unknown>")]
    details = [str(kind)] if kind else []
    if category:
        details.append(str(category))
    if room_id:
        details.append(f"room={room_id}")
    if center:
        details.append(f"center={center}")
    if details:
        parts.append("(" + ", ".join(details) + ")")
    return " ".join(parts)


def _gt_trajectory(instruction: Mapping[str, object]) -> Sequence[Sequence[int]] | None:
    raw = instruction.get("human_expert_trajectory")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return raw  # type: ignore[return-value]
    return None


def _write_cost_map(path: Path, debug_data: GroundPlanDebugData, *, scale: int = 3) -> bool:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required to write GroundPlan cost maps.") from exc

    scene = debug_data.scene
    grounded = debug_data.grounded
    if scene is None or grounded is None or not grounded.segments:
        return False

    planned_by_id = {
        segment.id: segment
        for segment in (debug_data.planned.segments if debug_data.planned is not None else ())
    }
    panels = []
    label_height = 20
    for index, segment in enumerate(grounded.segments, start=1):
        goal_mask, _goal_details = _goal_mask(scene, segment.target)
        hard = _compile_hard(scene, segment.constraints)
        traversable = _traversable_mask(scene) & ~hard["forbidden"]
        goal_mask = goal_mask & traversable
        soft_field, _soft_details = _compile_soft(scene, segment.constraints)
        planned_segment = planned_by_id.get(segment.id)
        path_cells = planned_segment.path if planned_segment is not None else []
        panel = _cost_panel(
            soft_field=soft_field,
            traversable=traversable,
            forbidden=hard["forbidden"],
            required=hard["required"],
            goal=goal_mask,
            path_cells=path_cells,
            title=f"{index}: {segment.id}",
            scale=scale,
            label_height=label_height,
        )
        panels.append(panel)

    if not panels:
        return False
    width = max(panel.width for panel in panels)
    height = sum(panel.height for panel in panels)
    image = Image.new("RGB", (width, height), (245, 245, 245))
    top = 0
    for panel in panels:
        image.paste(panel, (0, top))
        top += panel.height
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return True


def _cost_panel(
    *,
    soft_field: np.ndarray,
    traversable: np.ndarray,
    forbidden: np.ndarray,
    required: Sequence[np.ndarray],
    goal: np.ndarray,
    path_cells: Sequence[Sequence[int]],
    title: str,
    scale: int,
    label_height: int,
):
    from PIL import Image, ImageDraw

    height, width = soft_field.shape
    image = Image.new("RGB", (width * scale, height * scale + label_height), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, label_height - 1), fill=(245, 245, 245))
    draw.text((4, 4), title, fill=(20, 20, 20))

    values = soft_field[traversable]
    vmax = float(np.percentile(values, 95)) if values.size else 0.0
    if vmax <= 0.0:
        vmax = float(np.max(values)) if values.size else 1.0
    if vmax <= 0.0:
        vmax = 1.0

    required_mask = np.zeros_like(goal, dtype=bool)
    for mask in required:
        required_mask |= mask

    for row in range(height):
        for col in range(width):
            if not traversable[row, col]:
                color = (28, 28, 28)
            else:
                ratio = max(0.0, min(1.0, float(soft_field[row, col]) / vmax))
                color = _heat_color(ratio)
            if forbidden[row, col]:
                color = _blend(color, (190, 36, 42), 0.70)
            if required_mask[row, col]:
                color = _blend(color, (111, 72, 180), 0.65)
            if goal[row, col]:
                color = _blend(color, (36, 150, 74), 0.70)
            draw.rectangle(
                (
                    col * scale,
                    label_height + row * scale,
                    (col + 1) * scale - 1,
                    label_height + (row + 1) * scale - 1,
                ),
                fill=color,
            )

    points = []
    for cell in path_cells:
        if len(cell) != 2:
            continue
        row, col = int(cell[0]), int(cell[1])
        if 0 <= row < height and 0 <= col < width:
            points.append((col * scale + scale // 2, label_height + row * scale + scale // 2))
    if len(points) >= 2:
        draw.line(points, fill=(0, 210, 230), width=max(1, scale))
    if points:
        radius = max(2, scale * 2)
        for point, color in ((points[0], (45, 108, 223)), (points[-1], (14, 145, 76))):
            draw.ellipse(
                (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
                fill=color,
            )
    return image


def _heat_color(ratio: float) -> tuple[int, int, int]:
    low = (252, 252, 236)
    mid = (247, 176, 73)
    high = (129, 25, 52)
    if ratio < 0.5:
        return _blend(low, mid, ratio * 2.0)
    return _blend(mid, high, (ratio - 0.5) * 2.0)


def _blend(
    base: tuple[int, int, int],
    overlay: tuple[int, int, int],
    alpha: float,
) -> tuple[int, int, int]:
    return tuple(
        max(0, min(255, round(base[index] * (1.0 - alpha) + overlay[index] * alpha)))
        for index in range(3)
    )
