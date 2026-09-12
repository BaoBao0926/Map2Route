from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from scripts.methods.groundplan.constraint_evaluation import (
    apply_scope_ablation,
    canonicalize_constraint,
    evaluate_constraint_grounding,
)
from scripts.methods.groundplan.grounding.code import execute_grounding_code, verify_grounded_program
from scripts.methods.groundplan.ir import GPRef, GroundedConstraint, GroundedProgram, GroundedSegment, PlannedSegment
from scripts.methods.groundplan.planner.semantic_astar import active_constraints_for_segment, plan_grounded_program


class _Scene:
    def __init__(self) -> None:
        room = GPRef("room", "room_1", category="living_room", cells=frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}))
        self.rooms = {room.id: room}
        self.entities = {
            "object_1": GPRef("entity", "object_1", category="sofa", room_id="room_1", cells=frozenset({(0, 0)})),
            "object_2": GPRef("entity", "object_2", category="table", room_id="room_1", cells=frozenset({(0, 1)})),
            "object_tv": GPRef("entity", "object_tv", category="tv_stand", room_id="room_1", cells=frozenset({(1, 0)})),
        }
        self.start = GPRef("position", "task_start", room_id="room_1", cells=frozenset({(1, 1)}))

    def room_of(self, ref: GPRef) -> GPRef:
        return self.rooms[ref.room_id or "room_1"]

    def refs_by_category(self, kind: str, category: str | None = None) -> list[GPRef]:
        source = self.entities if kind == "entity" else self.rooms
        return [ref for ref in source.values() if category is None or ref.category == category]


class _RelativeWaypointScene:
    def __init__(self) -> None:
        self.grid_size = 24
        self.resolution = 0.1
        self.traversable = np.ones((self.grid_size, self.grid_size), dtype=bool)
        room_cells = frozenset((row, col) for row in range(self.grid_size) for col in range(self.grid_size))
        room = GPRef("room", "room_1", category="living_room", cells=room_cells, center=(11.5, 11.5))
        self.rooms = {room.id: room}
        self.entities = {
            "object_a": GPRef("entity", "object_a", category="sofa", room_id=room.id, center=(6.0, 6.0), cells=frozenset({(6, 6)})),
            "object_b": GPRef("entity", "object_b", category="table", room_id=room.id, center=(17.0, 17.0), cells=frozenset({(17, 17)})),
        }
        self.start = GPRef("position", "task_start", room_id=room.id, center=(2.0, 2.0), cells=frozenset({(2, 2)}))

    def refs_by_category(self, kind: str, category: str | None = None) -> list[GPRef]:
        source = self.entities if kind == "entity" else self.rooms
        return [ref for ref in source.values() if category is None or ref.category == category]

    def room_of(self, ref: GPRef) -> GPRef:
        return self.rooms[ref.room_id or "room_1"]

    def region_from_cells(self, identifier: str, cells: set[tuple[int, int]], construction: dict[str, object]) -> GPRef:
        frozen = frozenset(cells)
        center = (
            sum(row for row, _col in frozen) / len(frozen),
            sum(col for _row, col in frozen) / len(frozen),
        )
        return GPRef("region", identifier, room_id=str(construction["room_id"]), center=center, cells=frozen, construction=construction)


class DirectCapConstraintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = _Scene()
        self.first = self.scene.entities["object_1"]
        self.second = self.scene.entities["object_2"]
        self.tv = self.scene.entities["object_tv"]

    def _program(self, constraint: GroundedConstraint) -> GroundedProgram:
        return GroundedProgram((GroundedSegment("s1", self.scene.start, self.first, (constraint,)),), {})

    def test_direct_cap_preserves_serialized_provenance(self) -> None:
        source = """def ground():
    first = entities(\"sofa\")[0]
    second = entities(\"table\")[0]
    set_segment_context(\"s1\", task_start)
    c = constraint(\"prefer_relative\", first, \"closer_to\", second, source_text=\"stay closer to the sofa than the table\")
    s1 = segment(\"s1\", task_start, first, c)
    return task([s1], {\"first\": first, \"second\": second})
"""
        grounded = execute_grounding_code(source, self.scene, expected_program=None, allow_helpers=False).grounded
        constraint = grounded.segments[0].constraints[0].to_json()
        self.assertEqual(constraint["constraint_id"], "s1_c0")
        self.assertEqual(constraint["source_text"], "stay closer to the sofa than the table")
        self.assertEqual(constraint["argument_roles"], ["closer_ref", "farther_ref"])
        self.assertEqual(constraint["hardness"], "soft")

    def test_direct_cap_scope_api_preserves_default_global_and_multi_segment(self) -> None:
        source = """def ground():
    sofa = entities(\"sofa\")[0]
    tv = entities(\"tv_stand\")[0]
    set_segment_context(\"s2\", task_start)
    default_scope = constraint(\"prefer_far\", tv)
    global_scope = constraint(\"prefer_far\", tv, segment_scope=\"global\")
    multi_scope = constraint(\"prefer_far\", tv, segment_scope=[\"s1\", \"s2\"])
    s1 = segment(\"s1\", task_start, sofa)
    s2 = segment(\"s2\", sofa, tv, [default_scope, global_scope, multi_scope])
    return task([s1, s2])
"""
        grounded = execute_grounding_code(source, self.scene, expected_program=None, allow_helpers=False).grounded
        scopes = [constraint.segment_scope for constraint in grounded.segments[1].constraints]
        self.assertEqual(scopes, [("s2",), (), ("s1", "s2")])

    def test_arity_scope_and_duplicate_segment_validation(self) -> None:
        invalid = GroundedConstraint("prefer_relative", (self.first,), ("s1",), constraint_id="s1_c0", argument_roles=("closer_ref",), hardness="soft", relation="closer_to")
        errors = verify_grounded_program(self._program(invalid), self.scene, expected_program=None)["errors"]
        self.assertTrue(any(item["kind"] == "CONSTRAINT_ARITY" for item in errors))
        valid = GroundedConstraint("prefer_near", (self.first,), ("missing",), constraint_id="s1_c0", argument_roles=("reference",), hardness="soft")
        errors = verify_grounded_program(self._program(valid), self.scene, expected_program=None)["errors"]
        self.assertTrue(any(item["kind"] == "INVALID_SEGMENT_SCOPE" for item in errors))
        duplicated = GroundedProgram((self._program(valid).segments[0], self._program(valid).segments[0]), {})
        errors = verify_grounded_program(duplicated, self.scene, expected_program=None)["errors"]
        self.assertTrue(any(item["kind"] == "DUPLICATE_SEGMENT_ID" for item in errors))

    def test_between_construction_and_scope_ablation(self) -> None:
        region = GPRef("region", "between_object_1_object_2", cells=frozenset({(0, 0)}), construction={"type": "between", "references": ["object_1", "object_2"]})
        constraint = GroundedConstraint("forbid", (region,), ("s1",), constraint_id="s1_c0", argument_roles=("region",), hardness="hard")
        canonical = canonicalize_constraint(constraint.to_json())
        self.assertEqual(canonical["construction"][0][0], "between")
        ablated = apply_scope_ablation(self._program(constraint), "no_scope")
        self.assertEqual(ablated.segments[0].constraints[0].segment_scope, ())
        self.assertIsNone(ablated.segments[0].constraints[0].spatial_scope)

    def test_no_soft_constraints_preserves_hard_constraints(self) -> None:
        hard = GroundedConstraint("forbid", (self.tv,), ("s1",), constraint_id="hard", argument_roles=("region",), hardness="hard")
        soft = GroundedConstraint("prefer_near", (self.first,), ("s1",), constraint_id="soft", argument_roles=("reference",), hardness="soft")
        program = GroundedProgram(
            (GroundedSegment("s1", self.scene.start, self.first, (hard, soft)),),
            {},
        )

        ablated = apply_scope_ablation(program, "no_soft_constraints")

        self.assertEqual(ablated.segments[0].constraints, (hard,))

    def test_no_soft_constraints_collapses_generated_path_shape_sequence(self) -> None:
        waypoints = tuple(
            GPRef(
                "region",
                f"circle_{index}",
                room_id="room_1",
                center=(float(index), float(index)),
                cells=frozenset({(index, index)}),
                construction={"type": "circle_waypoint", "index": index},
            )
            for index in range(3)
        )
        shape = GroundedConstraint(
            "require_visit_in_order",
            waypoints,
            ("s1",),
            constraint_id="shape",
            argument_roles=("region", "region", "region"),
            hardness="hard",
        )

        ablated = apply_scope_ablation(self._program(shape), "no_soft_constraints")

        self.assertEqual(len(ablated.segments[0].constraints), 1)
        anchor = ablated.segments[0].constraints[0]
        self.assertEqual(anchor.kind, "require_visit")
        self.assertEqual(len(anchor.refs), 1)
        self.assertEqual(
            anchor.refs[0].cells,
            frozenset({(0, 0), (1, 1), (2, 2)}),
        )
        self.assertEqual(
            anchor.refs[0].construction["type"],
            "no_soft_path_anchor",
        )

    def test_no_soft_constraints_preserves_real_ordered_hard_visits(self) -> None:
        ordered = GroundedConstraint(
            "require_visit_in_order",
            (self.first, self.second),
            ("s1",),
            constraint_id="ordered",
            argument_roles=("region", "region"),
            hardness="hard",
        )

        ablated = apply_scope_ablation(self._program(ordered), "no_soft_constraints")

        self.assertEqual(ablated.segments[0].constraints, (ordered,))

    def test_constraint_evaluator_reports_exact_missing_and_extra(self) -> None:
        constraint = GroundedConstraint("prefer_near", (self.first,), ("s1",), constraint_id="s1_c0", argument_roles=("reference",), hardness="soft")
        instruction = {"soft_constraints": [{"constraint_id": "gt1", "preference_type": "near_preference", "reference_region": {"object_id": 1}, "scope": {"type": "between_hard_constraints", "to_order": 2}}], "hard_constraints": []}
        report = evaluate_constraint_grounding(self._program(constraint), instruction, self.scene)
        self.assertEqual(report["constraint_f1"], 1.0)
        extra = GroundedConstraint("prefer_far", (self.second,), ("s1",), constraint_id="s1_c1", argument_roles=("reference",), hardness="soft")
        program = GroundedProgram((GroundedSegment("s1", self.scene.start, self.first, (constraint, extra)),), {})
        report = evaluate_constraint_grounding(program, instruction, self.scene)
        self.assertEqual(report["constraint_recall"], 1.0)
        self.assertLess(report["constraint_precision"], 1.0)


    def test_segment_scope_controls_planner_activation_not_host_container(self) -> None:
        """The far-from-TV constraint is physically declared in s2 in every case."""
        def program_for(scope: tuple[str, ...]) -> GroundedProgram:
            far_from_tv = GroundedConstraint(
                "prefer_far",
                (self.tv,),
                scope,
                constraint_id="far_from_tv",
                argument_roles=("reference",),
                hardness="soft",
            )
            return GroundedProgram(
                (
                    GroundedSegment("s1", self.scene.start, self.first, ()),
                    GroundedSegment("s2", self.first, self.second, (far_from_tv,)),
                ),
                {},
            )

        cases = {
            "s2_only": (("s2",), ([], ["far_from_tv"])),
            "global": ((), (["far_from_tv"], ["far_from_tv"])),
            # Stored in s2, but executes in s1: scope—not the container—wins.
            "s1_only": (("s1",), (["far_from_tv"], [])),
        }
        for _name, (scope, expected) in cases.items():
            program = program_for(scope)
            actual = tuple(
                [constraint.constraint_id for constraint in active_constraints_for_segment(program, segment_id)]
                for segment_id in ("s1", "s2")
            )
            self.assertEqual(actual, expected)

            # Exercise the actual planner entry point. The search is mocked only
            # to isolate scope activation from map geometry.
            received: list[list[str]] = []

            def fake_plan_segment(_scene, segment, start, *, constraints, max_expansions, **_kwargs):
                received.append([constraint.constraint_id for constraint in constraints])
                return PlannedSegment(segment.id, [[start[0], start[1]]], start, "success", 0.0, 0)

            with patch(
                "scripts.methods.groundplan.planner.semantic_astar.plan_segment",
                side_effect=fake_plan_segment,
            ):
                plan_grounded_program(self.scene, program, planner_mode="sequential_greedy_astar")
            self.assertEqual(tuple(received), expected)

    def test_cross_segment_scope_is_valid_even_when_host_differs(self) -> None:
        constraint = GroundedConstraint(
            "prefer_far",
            (self.tv,),
            ("s1",),
            constraint_id="far_from_tv",
            argument_roles=("reference",),
            hardness="soft",
        )
        program = GroundedProgram(
            (
                GroundedSegment("s1", self.scene.start, self.first, ()),
                GroundedSegment("s2", self.first, self.second, (constraint,)),
            ),
            {},
        )
        errors = verify_grounded_program(program, self.scene, expected_program=None)["errors"]
        self.assertFalse(any(item["kind"] == "SEGMENT_SCOPE_MISMATCH" for item in errors))

    def test_anchor_relative_helper_produces_ordered_hard_waypoints(self) -> None:
        scene = _RelativeWaypointScene()
        source = """def s_shape(first, second):
    return [
        relative_waypoint(first, second, along=0.15, lateral=0.40),
        relative_waypoint(first, second, along=0.50, lateral=0.00),
        relative_waypoint(first, second, along=0.85, lateral=-0.40),
    ]

def ground():
    first = entities("sofa")[0]
    second = entities("table")[0]
    set_segment_context("s1", task_start)
    shape = s_shape(first, second)
    s1 = segment("s1", task_start, second, [constraint("require_visit_in_order", shape)])
    return task([s1])
"""
        grounded = execute_grounding_code(source, scene, expected_program=None, allow_helpers=True).grounded
        constraint = grounded.segments[0].constraints[0]
        self.assertEqual(constraint.kind, "require_visit_in_order")
        self.assertEqual(len(constraint.refs), 3)
        self.assertEqual([ref.construction["along"] for ref in constraint.refs], [0.15, 0.5, 0.85])
        self.assertEqual([ref.construction["lateral_meters"] for ref in constraint.refs], [0.4, 0.0, -0.4])
        self.assertTrue(all(ref.construction["type"] == "relative_waypoint" for ref in constraint.refs))


if __name__ == "__main__":
    unittest.main()
