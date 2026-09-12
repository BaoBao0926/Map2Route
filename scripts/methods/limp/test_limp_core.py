from __future__ import annotations

from scripts.methods.limp.config import LimpConfig
from scripts.methods.limp.grounding.candidates import lookup_candidates
from scripts.methods.limp.grounding.crd_parser import parse_crd, parse_predicate
from scripts.methods.limp.grounding.referent_map import GroundingResult, ground_crd
from scripts.methods.limp.grounding.scene_adapter import (
    Candidate,
    CandidateRegistry,
    add_start_context_candidates,
)
from scripts.methods.limp.logic.ltlf_progression import (
    LTLFProgression,
    negative_atomic_propositions,
    parse_encoded_ltl,
)
from scripts.methods.limp.pipeline import _apply_extended_grounding_tiebreaks
from scripts.methods.limp.planning.progressive_planner import (
    build_symbol_regions,
    plan_progressively,
)


def _config(tmp_path, **overrides) -> LimpConfig:
    values = {
        "model": "test",
        "llm_cache_root": tmp_path,
        "near_radius": 1,
        "max_progress_steps": 8,
    }
    values.update(overrides)
    return LimpConfig(**values)


def _registry(candidates: list[Candidate] | None = None) -> CandidateRegistry:
    candidates = candidates or []
    by_category: dict[str, tuple[str, ...]] = {}
    by_kind: dict[str, tuple[str, ...]] = {}
    for candidate in candidates:
        by_category.setdefault(candidate.category, ())
        by_category[candidate.category] += (candidate.candidate_id,)
        by_kind.setdefault(candidate.kind, ())
        by_kind[candidate.kind] += (candidate.candidate_id,)
    return CandidateRegistry(
        candidates={candidate.candidate_id: candidate for candidate in candidates},
        by_kind=by_kind,
        by_category=by_category,
        object_room={},
    )


def _candidate(
    candidate_id: str,
    category: str,
    cell: tuple[int, int],
    *,
    kind: str = "object",
    instance_id: int = 1,
) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        kind=kind,
        instance_id=instance_id,
        category=category,
        name=candidate_id,
        cells=(cell,),
        centroid=(float(cell[0]), float(cell[1])),
        bbox=(cell[0], cell[1], cell[0], cell[1]),
    )


def test_complete_transition_planner_satisfies_conjunction(tmp_path) -> None:
    grid = [[True] * 7 for _ in range(7)]
    progression = LTLFProgression(parse_encoded_ltl("F (A & B)"), ["A", "B"])

    result = plan_progressively(
        {"start_pose": {"row": 0, "col": 0}},
        grid,
        progression,
        {"A": {(2, 2), (2, 3)}, "B": {(2, 3), (2, 4)}},
        {"A": "chair", "B": "table"},
        {},
        _registry(),
        [],
        _config(tmp_path),
    )

    assert result.status == "SUCCESS_ACCEPTING_STATE"
    assert result.trajectory[-1] == [2, 3]
    assert result.planner_segments[0]["true_symbols_at_end"] == ["A", "B"]


def test_transition_planner_routes_around_rejecting_cells(tmp_path) -> None:
    grid = [[True] * 7 for _ in range(7)]
    forbidden = {(row, 3) for row in range(1, 6)}
    progression = LTLFProgression(parse_encoded_ltl("G (!C) & F A"), ["C", "A"])

    result = plan_progressively(
        {"start_pose": {"row": 3, "col": 0}},
        grid,
        progression,
        {"A": {(3, 6)}, "C": forbidden},
        {"A": "goal", "C": "obstacle"},
        {},
        _registry(),
        [],
        _config(tmp_path),
    )

    assert result.status == "SUCCESS_ACCEPTING_STATE"
    assert not ({tuple(cell) for cell in result.trajectory} & forbidden)
    assert result.planner_segments[0]["forbidden_cell_count"] == len(forbidden)


def test_rejecting_start_is_reported_as_logic_violation(tmp_path) -> None:
    progression = LTLFProgression(parse_encoded_ltl("A U B"), ["A", "B"])
    result = plan_progressively(
        {"start_pose": {"row": 0, "col": 0}},
        [[True] * 3 for _ in range(3)],
        progression,
        {"A": {(1, 1)}, "B": {(2, 2)}},
        {"A": "guide", "B": "goal"},
        {},
        _registry(),
        [],
        _config(tmp_path),
    )
    assert result.status == "LOGIC_VIOLATION_AT_START"
    assert result.task_state_sequence[-1]["failed"] is True


def test_start_context_and_selector_aliases_are_groundable() -> None:
    room = _candidate("room_7", "bedroom", (1, 1), kind="room", instance_id=7)
    plant = _candidate("object_8", "house_plant", (2, 2), instance_id=8)
    registry = _registry([room, plant])
    add_start_context_candidates(registry, (1, 1))

    assert lookup_candidates(registry, "robot") == ("context_start",)
    assert lookup_candidates(registry, "starting_room") == ("room_7",)
    assert lookup_candidates(registry, "houseplant") == ("object_8",)
    assert lookup_candidates(registry, "houseplant", extended=False) == ()


def test_isnextto_prefers_closest_object_mask() -> None:
    near_bed = _candidate("object_1", "bed", (2, 1), instance_id=1)
    far_bed = _candidate("object_2", "bed", (8, 8), instance_id=2)
    dresser = _candidate("object_3", "dresser", (1, 1), instance_id=3)
    result = ground_crd(parse_crd("bed::isnextto(dresser)"), _registry([near_bed, far_bed, dresser]))
    faithful_result = ground_crd(parse_crd("bed::isnextto(dresser)"), _registry([near_bed, far_bed, dresser]), extended=False)

    assert result.status == "GROUNDING_SUCCESS"
    assert result.selected_candidate_id == "object_1"
    assert faithful_result.status == "AMBIGUOUS_GROUNDING"


def test_farthest_selector_uses_straight_line_distance(tmp_path) -> None:
    near = _candidate("object_1", "chair", (2, 2), instance_id=1)
    far = _candidate("object_2", "chair", (8, 8), instance_id=2)
    registry = _registry([near, far])
    grounding = GroundingResult(
        referent="farthest_chair",
        status="AMBIGUOUS_GROUNDING",
        candidate_ids_before_filtering=("object_1", "object_2"),
        candidate_ids_after_filtering=("object_1", "object_2"),
    )

    _apply_extended_grounding_tiebreaks(
        {"farthest_chair": grounding},
        registry,
        [[True] * 10 for _ in range(10)],
        (0, 0),
        _config(tmp_path),
    )

    assert grounding.status == "GROUNDING_SUCCESS"
    assert grounding.selected_candidate_id == "object_2"
    assert grounding.selected_candidate_ids == ("object_2",)
    assert grounding.disambiguation["policy"] == "farthest_straight_line_candidate_from_start"


def test_negative_proposition_uses_avoid_radius(tmp_path) -> None:
    sofa = _candidate("object_1", "sofa", (4, 4), instance_id=1)
    registry = _registry([sofa])
    grounding = GroundingResult(
        referent="sofa",
        status="GROUNDING_SUCCESS",
        selected_candidate_id="object_1",
        selected_candidate_ids=("object_1",),
    )
    formula = parse_encoded_ltl("G (!C) & F A")

    regions, _referents, _visit_regions, records = build_symbol_regions(
        [parse_predicate("C", "near[sofa]")],
        {"sofa": grounding},
        registry,
        [[True] * 9 for _ in range(9)],
        _config(tmp_path, near_radius=1, avoid_radius=3),
        negative_atomic_propositions(formula),
    )

    assert negative_atomic_propositions(formula) == {"C"}
    assert (4, 1) in regions["C"]
    assert records[0]["region_role"] == "avoidance"
    assert records[0]["region_radius"] == 3
