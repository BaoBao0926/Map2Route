from __future__ import annotations

import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.methods.groundplan.grounding.grounder import ground_program
from scripts.methods.groundplan.ir import GPRef
from scripts.methods.groundplan.parse.direct_id import (
    DirectIDParseError,
    direct_id_to_program,
)
from scripts.methods.groundplan.parse.ltl import LTLParseError, ltl_to_program
from scripts.methods.groundplan.parse.parser import parse_instruction
from scripts.methods.groundplan.parse.scene_catalog import compact_scene_catalog
from scripts.methods.groundplan.pipeline import (
    GroundPlanRunConfig,
    resolve_grounding_architecture,
)
from scripts.methods.groundplan.run import (
    REPRESENTATION_VARIANTS,
    _resolved_parse_mode,
)


class _Scene:
    def __init__(self) -> None:
        self.grid_size = 4
        self.resolution = 0.05
        room = GPRef(
            "room",
            "room_1",
            category="living_room",
            center=(1.5, 1.5),
            cells=frozenset((row, col) for row in range(4) for col in range(4)),
        )
        self.rooms = {room.id: room}
        self.entities = {
            "object_1": GPRef(
                "entity",
                "object_1",
                category="sofa",
                room_id=room.id,
                center=(1.0, 1.0),
                cells=frozenset({(1, 1)}),
            ),
            "object_2": GPRef(
                "entity",
                "object_2",
                category="table",
                room_id=room.id,
                center=(2.0, 2.0),
                cells=frozenset({(2, 2)}),
            ),
            "object_3": GPRef(
                "entity",
                "object_3",
                category="chair",
                room_id=room.id,
                center=(1.0, 2.0),
                cells=frozenset({(1, 2)}),
            ),
        }
        self.start = GPRef(
            "position",
            "task_start",
            room_id=room.id,
            center=(0.0, 0.0),
            cells=frozenset({(0, 0)}),
        )
        self.passages: dict[tuple[str, str], list[GPRef]] = {}

    def room_of(self, ref: GPRef) -> GPRef:
        return self.rooms[ref.room_id or "room_1"]

    def region_from_cells(
        self,
        region_id: str,
        cells: set[tuple[int, int]],
        construction: dict[str, object],
    ) -> GPRef:
        return GPRef(
            "region",
            region_id,
            center=(
                sum(row for row, _col in cells) / len(cells),
                sum(col for _row, col in cells) / len(cells),
            ),
            cells=frozenset(cells),
            construction=construction,
        )


class GroundingRepresentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = _Scene()

    def test_compact_catalog_contains_no_dense_map_payload(self) -> None:
        catalog = compact_scene_catalog({}, {}, scene=self.scene)  # type: ignore[arg-type]
        serialized = json.dumps(catalog)
        self.assertNotIn("layers", catalog)
        self.assertNotIn("object_footprints", catalog)
        self.assertEqual(len(catalog["objects"]), 3)  # type: ignore[arg-type]
        self.assertLess(len(serialized), 5000)

    def test_direct_id_compiles_ids_scopes_and_preferences(self) -> None:
        payload = {
            "segments": [
                {"id": "s1", "target_id": "object_1"},
                {"id": "s2", "target_id": "object_2"},
            ],
            "constraints": [
                {
                    "kind": "forbid",
                    "ref_ids": ["object_3"],
                    "segment_ids": ["s2"],
                },
                {
                    "kind": "prefer_near",
                    "ref_ids": ["object_1"],
                    "segment_ids": ["s2"],
                    "spatial_scope_ids": ["room_1"],
                },
            ],
        }
        program = direct_id_to_program(payload, scene=self.scene)  # type: ignore[arg-type]
        grounded = ground_program(program, self.scene)  # type: ignore[arg-type]
        self.assertEqual(grounded.status, "success")
        self.assertEqual([segment.target.id for segment in grounded.segments], ["object_1", "object_2"])
        self.assertEqual([item.kind for item in grounded.segments[1].constraints], ["forbid", "prefer_near"])
        self.assertEqual(grounded.segments[1].constraints[0].segment_scope, ("s2",))

    def test_direct_id_rejects_hallucinated_ids(self) -> None:
        with self.assertRaisesRegex(DirectIDParseError, "Unknown compact-catalog"):
            direct_id_to_program(
                {"segments": [{"id": "s1", "target_id": "object_999"}]},
                scene=self.scene,  # type: ignore[arg-type]
            )

    def test_ltl_compiles_nested_eventuality_order_and_global_avoid(self) -> None:
        payload = {
            "formula": "F(v1 & F(v2)) & G(!a1)",
            "propositions": [
                {"name": "v1", "kind": "visit", "ref_ids": ["object_1"]},
                {"name": "v2", "kind": "visit", "ref_ids": ["object_2"]},
                {"name": "a1", "kind": "avoid", "ref_ids": ["object_3"]},
                {
                    "name": "p1",
                    "kind": "prefer_far",
                    "ref_ids": ["object_3"],
                    "during": ["v2"],
                    "spatial_scope_ids": ["room_1"],
                },
            ],
        }
        program, formula = ltl_to_program(payload, scene=self.scene)  # type: ignore[arg-type]
        grounded = ground_program(program, self.scene)  # type: ignore[arg-type]
        self.assertEqual(formula, payload["formula"])
        self.assertEqual([segment.id for segment in grounded.segments], ["v1", "v2"])
        self.assertEqual(
            [[item.kind for item in segment.constraints] for segment in grounded.segments],
            [["forbid"], ["forbid", "prefer_far"]],
        )

    def test_ltl_rejects_unordered_parallel_eventualities(self) -> None:
        payload = {
            "formula": "F(v1) & F(v2)",
            "propositions": [
                {"name": "v1", "kind": "visit", "ref_ids": ["object_1"]},
                {"name": "v2", "kind": "visit", "ref_ids": ["object_2"]},
            ],
        }
        with self.assertRaisesRegex(LTLParseError, "ordered eventuality chain"):
            ltl_to_program(payload, scene=self.scene)  # type: ignore[arg-type]

    def test_cli_representation_selects_matching_parser(self) -> None:
        self.assertEqual(REPRESENTATION_VARIANTS["json"], "ir_to_code")
        self.assertEqual(REPRESENTATION_VARIANTS["direct_id"], "direct_id")
        self.assertEqual(REPRESENTATION_VARIANTS["ltl"], "ltl")
        args = SimpleNamespace(parse_mode="intent")
        self.assertEqual(_resolved_parse_mode(args, "direct_id"), "direct_id")
        self.assertEqual(_resolved_parse_mode(args, "ltl"), "ltl")
        self.assertEqual(_resolved_parse_mode(args, "full"), "intent")

    def test_standard_representation_ablations_are_one_shot(self) -> None:
        for representation in ("json", "direct_id", "ltl"):
            with self.subTest(representation=representation):
                architecture = resolve_grounding_architecture(
                    GroundPlanRunConfig(
                        grounding_variant=REPRESENTATION_VARIANTS[representation],
                    )
                )
                self.assertEqual(architecture.repair_policy, "off")
                self.assertFalse(architecture.code_refinement)
                self.assertFalse(architecture.allow_helpers)

    def test_representation_parsers_receive_zero_repair_budget_by_default(self) -> None:
        parsers = {
            "intent": "scripts.methods.groundplan.parse.parser.llm_parse_intent_instruction",
            "api": "scripts.methods.groundplan.parse.parser.llm_parse_api_program_instruction",
            "direct_id": "scripts.methods.groundplan.parse.parser.llm_parse_direct_id_instruction",
            "ltl": "scripts.methods.groundplan.parse.parser.llm_parse_ltl_instruction",
        }
        program = SimpleNamespace()
        for mode, target in parsers.items():
            with self.subTest(mode=mode), patch(target, return_value=(program, {})) as parser:
                parsed, _metadata = parse_instruction(
                    {"instruction": "Go to the sofa."},
                    {},
                    mode=mode,
                    llm_client=SimpleNamespace(),  # type: ignore[arg-type]
                )
                self.assertIs(parsed, program)
                self.assertEqual(parser.call_args.kwargs["max_parse_repairs"], 0)


if __name__ == "__main__":
    unittest.main()
