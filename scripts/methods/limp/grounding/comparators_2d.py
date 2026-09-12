"""2D spatial comparators for LIMP CRD grounding."""

from __future__ import annotations

from scripts.methods.limp.grounding.scene_adapter import Candidate
from scripts.methods.limp.utils.geometry import distance, min_cell_distance, point_segment_distance


def is_next_to(candidate: Candidate, reference: Candidate, threshold: float = 45.0) -> bool:
    if reference.kind == "room" and candidate.kind == "object":
        return candidate.room_id == reference.instance_id or min_cell_distance(candidate.cells, reference.cells) <= threshold
    return min_cell_distance(candidate.cells, reference.cells) <= threshold


def is_left_of(candidate: Candidate, reference: Candidate, margin: float = 0.0) -> bool:
    return candidate.centroid[1] < reference.centroid[1] - margin


def is_right_of(candidate: Candidate, reference: Candidate, margin: float = 0.0) -> bool:
    return candidate.centroid[1] > reference.centroid[1] + margin


def is_above(candidate: Candidate, reference: Candidate, margin: float = 0.0) -> bool:
    return candidate.centroid[0] < reference.centroid[0] - margin


def is_below(candidate: Candidate, reference: Candidate, margin: float = 0.0) -> bool:
    return candidate.centroid[0] > reference.centroid[0] + margin


def is_between(candidate: Candidate, first: Candidate, second: Candidate, tolerance: float = 30.0) -> bool:
    first_point = first.centroid
    second_point = second.centroid
    cand_point = candidate.centroid
    if distance(first_point, second_point) <= 1e-6:
        return False
    row_min = min(first_point[0], second_point[0]) - tolerance
    row_max = max(first_point[0], second_point[0]) + tolerance
    col_min = min(first_point[1], second_point[1]) - tolerance
    col_max = max(first_point[1], second_point[1]) + tolerance
    if not (row_min <= cand_point[0] <= row_max and col_min <= cand_point[1] <= col_max):
        return False
    return point_segment_distance(cand_point, first_point, second_point) <= tolerance


def is_inside(candidate: Candidate, room: Candidate) -> bool:
    if room.kind != "room":
        return False
    if candidate.kind == "object":
        return candidate.room_id == room.instance_id
    cand_cells = set(candidate.cells)
    room_cells = set(room.cells)
    return bool(cand_cells) and cand_cells.issubset(room_cells)


def contains(room: Candidate, candidate: Candidate) -> bool:
    return is_inside(candidate, room)


def evaluate_comparator(candidate: Candidate, name: str, references: tuple[Candidate, ...]) -> bool | None:
    if name in {"isnextto", "nextto"} and len(references) == 1:
        return is_next_to(candidate, references[0])
    if name in {"isleftof", "leftof"} and len(references) == 1:
        return is_left_of(candidate, references[0])
    if name in {"isrightof", "rightof"} and len(references) == 1:
        return is_right_of(candidate, references[0])
    if name in {"isabove", "above"} and len(references) == 1:
        return is_above(candidate, references[0])
    if name in {"isbelow", "below"} and len(references) == 1:
        return is_below(candidate, references[0])
    if name in {"isbetween", "between"} and len(references) == 2:
        return is_between(candidate, references[0], references[1])
    if name in {"isinside", "inside"} and len(references) == 1:
        return is_inside(candidate, references[0])
    if name == "contains" and len(references) == 1:
        return contains(candidate, references[0])
    if name in {"isinfrontof", "infrontof", "isbehind", "behind"}:
        return None
    return None
