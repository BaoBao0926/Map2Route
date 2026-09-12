"""Shared schema helpers for path-shape annotations embedded in instructions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


PATH_SHAPE_ANNOTATION_FIELD = "path_shape_annotation"
PATH_SHAPE_ANNOTATION_TYPE = "canonical_path_shape_reference"
PATH_SHAPE_PREFERENCE_TYPES = {
    "path_shape_preference",
    "geometric_path_preference",
}


def _string(value: object, default: str = "") -> str:
    return value.strip() if isinstance(value, str) and value.strip() else default


def _points(
    value: object,
    *,
    label: str,
    grid_size: int,
    minimum: int,
) -> list[list[int]]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list of [row, col] points.")

    result: list[list[int]] = []
    for index, point in enumerate(value):
        if (
            not isinstance(point, Sequence)
            or isinstance(point, (str, bytes))
            or len(point) != 2
        ):
            raise ValueError(f"{label}[{index}] must be [row, col].")
        row, col = point
        if (
            not isinstance(row, int)
            or isinstance(row, bool)
            or not isinstance(col, int)
            or isinstance(col, bool)
        ):
            raise ValueError(f"{label}[{index}] must contain integers.")
        if not (0 <= row < grid_size and 0 <= col < grid_size):
            raise ValueError(f"{label}[{index}] is outside the map.")
        if not result or result[-1] != [row, col]:
            result.append([row, col])

    if len(result) < minimum:
        raise ValueError(f"{label} needs at least {minimum} points.")
    return result


def normalize_path_shape_annotation(
    value: object,
    *,
    grid_size: int,
    include_history: bool = True,
    label: str = PATH_SHAPE_ANNOTATION_FIELD,
) -> dict[str, object]:
    """Validate and normalize one embedded canonical path-shape annotation."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object.")

    version = value.get("version", 1)
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError(f"{label}.version must be a positive integer.")

    annotation_type = value.get("annotation_type", PATH_SHAPE_ANNOTATION_TYPE)
    if annotation_type != PATH_SHAPE_ANNOTATION_TYPE:
        raise ValueError(
            f"{label}.annotation_type must be {PATH_SHAPE_ANNOTATION_TYPE!r}."
        )

    raw_shape_spec = value.get("shape_spec", {})
    if not isinstance(raw_shape_spec, Mapping):
        raise ValueError(f"{label}.shape_spec must be an object.")
    shape_type = _string(raw_shape_spec.get("type"), "expert_polyline")
    direction = _string(raw_shape_spec.get("direction"), "unspecified")
    controls = _points(
        raw_shape_spec.get("ordered_control_points", []),
        label=f"{label}.shape_spec.ordered_control_points",
        grid_size=grid_size,
        minimum=0,
    )
    trajectory = _points(
        value.get("shape_reference_trajectory"),
        label=f"{label}.shape_reference_trajectory",
        grid_size=grid_size,
        minimum=2,
    )

    normalized: dict[str, object] = {
        "version": version,
        "annotation_type": PATH_SHAPE_ANNOTATION_TYPE,
        "shape_spec": {
            "type": shape_type,
            "direction": direction,
            "ordered_control_points": controls,
        },
        "shape_reference_trajectory": trajectory,
        "shape_reference_source": _string(
            value.get("shape_reference_source"),
            "human_expert_canonical_v1",
        ),
        "annotator_id": _string(value.get("annotator_id")),
        "notes": _string(value.get("notes")),
        "created_at": _string(value.get("created_at")),
        "updated_at": _string(value.get("updated_at")),
    }

    if include_history:
        raw_history = value.get("history", [])
        if not isinstance(raw_history, list):
            raise ValueError(f"{label}.history must be a list.")
        normalized["history"] = [
            normalize_path_shape_annotation(
                item,
                grid_size=grid_size,
                include_history=False,
                label=f"{label}.history[{index}]",
            )
            for index, item in enumerate(raw_history)
        ]

    return normalized


def path_shape_reference_trajectory(
    constraint: Mapping[str, object],
) -> object:
    """Return the embedded canonical trajectory, or the legacy field as fallback."""

    annotation = constraint.get(PATH_SHAPE_ANNOTATION_FIELD)
    if isinstance(annotation, Mapping):
        return annotation.get("shape_reference_trajectory")
    return constraint.get("reference_trajectory")


def has_path_shape_annotation(constraint: Mapping[str, object]) -> bool:
    """Return whether a constraint contains an embedded canonical annotation."""

    annotation = constraint.get(PATH_SHAPE_ANNOTATION_FIELD)
    return (
        isinstance(annotation, Mapping)
        and annotation.get("annotation_type") == PATH_SHAPE_ANNOTATION_TYPE
        and isinstance(annotation.get("shape_reference_trajectory"), list)
        and len(annotation["shape_reference_trajectory"]) >= 2
    )
