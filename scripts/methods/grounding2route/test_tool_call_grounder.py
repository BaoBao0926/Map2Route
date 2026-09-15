from __future__ import annotations

from types import SimpleNamespace
import unittest

from scripts.methods.grounding2route.grounding.code import GroundingRuntime
from scripts.methods.grounding2route.grounding.semantic_api import (
    SEMANTIC_API_NAMES,
    gemini_function_declarations,
)
from scripts.methods.grounding2route.grounding.tool_call import ToolCallGrounder, _ToolSession
from scripts.methods.grounding2route.test_grounding_representations import _Scene
from scripts.methods.grounding2route.pipeline import Grounding2RouteRunConfig, resolve_grounding_architecture
from scripts.methods.grounding2route.run import REPRESENTATION_VARIANTS


class _ToolScene(_Scene):
    def refs_by_category(self, kind: str, category: str | None):
        values = self.entities.values() if kind == "entity" else self.rooms.values()
        return [ref for ref in values if category is None or ref.category == category]


class _FakeToolClient:
    def __init__(self, calls: list[dict[str, object]]) -> None:
        self.calls = list(calls)
        self.model = "fake-gemini"

    def tool_turn(self, **_kwargs: object) -> tuple[dict[str, object], dict[str, object]]:
        if not self.calls:
            return ({"role": "model", "parts": [{"text": "unable"}]}, {"usage_metadata": {}})
        call = self.calls.pop(0)
        return (
            {"role": "model", "parts": [{"functionCall": call}]},
            {"usage_metadata": {"promptTokenCount": 10, "candidatesTokenCount": 2}},
        )


class ToolCallGrounderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = _ToolScene()

    def test_registry_exactly_matches_cag_runtime(self) -> None:
        runtime_names = set(GroundingRuntime(self.scene).namespace()) - {"task_start", "start_position"}
        self.assertEqual(runtime_names, set(SEMANTIC_API_NAMES))
        self.assertEqual({item["name"] for item in gemini_function_declarations()}, runtime_names)

    def test_native_calls_build_existing_grounded_program_without_code(self) -> None:
        client = _FakeToolClient([
            {"name": "entities", "args": {"category": "chair"}},
            {"name": "unique", "args": {"candidates": "value_0001"}},
            {"name": "set_segment_context", "args": {"segment_id": "s1", "start": "start_position"}},
            {"name": "segment", "args": {"segment_id": "s1", "start": "start_position", "target": "ref_0002", "constraints": []}},
            {"name": "task", "args": {"segments": ["value_0003"], "bindings": [{"name": "target", "value": "ref_0002"}]}},
        ])
        result = ToolCallGrounder(client, max_tool_calls=8, max_llm_turns=8).ground(  # type: ignore[arg-type]
            "Go to the chair.", self.scene, None,
            entity_categories=["chair"], room_categories=["living_room"],
            scene_summary={"object_counts": {"chair": 1}, "room_counts": {"living_room": 1}},
            max_repairs=0,
        )
        self.assertIsNone(result.failure_reason)
        self.assertIsNotNone(result.grounded)
        assert result.grounded is not None
        self.assertEqual(result.grounded.segments[0].target.id, "object_3")
        self.assertEqual(result.statistics["semantic_api_calls"], 5)
        self.assertEqual(result.statistics["llm_invocations"], 5)
        self.assertEqual(result.statistics["llm_input_tokens"], 50)

    def test_hallucinated_handle_is_rejected_by_tool_adapter(self) -> None:
        client = _FakeToolClient([
            {"name": "kth_nearest", "args": {"candidates": "chair_17", "reference": "start_position"}},
            {"name": "task", "args": {"segments": ["fabricated_segment"]}},
        ])
        result = ToolCallGrounder(client, max_tool_calls=4, max_llm_turns=4).ground(  # type: ignore[arg-type]
            "Go to the chair.", self.scene, None,
            entity_categories=["chair"], room_categories=["living_room"], scene_summary={},
        )
        self.assertEqual(result.tool_trace[0]["status"], "failed")
        self.assertEqual(result.tool_trace[1]["status"], "failed")
        self.assertIsNone(result.grounded)

    def test_exact_failed_query_executes_only_three_times_then_other_queries_continue(self) -> None:
        session = _ToolSession(self.scene, max_failed_query_attempts=99)  # type: ignore[arg-type]
        self.assertEqual(session.max_failed_query_attempts, 3)
        attempts = [session.call("rooms", {"category": "garage"}) for _ in range(4)]
        self.assertIn("UNKNOWN_ROOM_CATEGORY", attempts[0][1]["error"])
        self.assertEqual(attempts[2][1]["remaining_failed_attempts"], 0)
        self.assertEqual(attempts[3][1]["error"], "QUERY_RETRY_LIMIT_REACHED")
        self.assertEqual([item["status"] for item in session.trace], ["failed", "failed", "failed", "blocked_retry_limit"])
        self.assertEqual(session.executed_calls, 3)
        result, payload = session.call("entities", {"category": "chair"})
        self.assertIsNotNone(result)
        self.assertIn("handle", payload)
        self.assertEqual(session.executed_calls, 4)
        candidates = str(payload["handle"])
        empty_attempts = [
            session.call("exclude", {"candidates": candidates, "excluded": candidates})
            for _ in range(4)
        ]
        self.assertTrue(empty_attempts[0][1]["empty_result"])
        self.assertEqual(empty_attempts[2][1]["remaining_failed_attempts"], 0)
        self.assertEqual(empty_attempts[3][1]["error"], "QUERY_RETRY_LIMIT_REACHED")
        self.assertEqual(session.executed_calls, 7)

    def test_cli_aliases_resolve_to_tool_architecture(self) -> None:
        self.assertEqual(REPRESENTATION_VARIANTS["tool_call"], "tool_call")
        self.assertEqual(REPRESENTATION_VARIANTS["function_call"], "tool_call")
        architecture = resolve_grounding_architecture(Grounding2RouteRunConfig(grounding_variant="tool_call"))
        self.assertTrue(architecture.uses_tool_calls)
        self.assertFalse(architecture.uses_code)
        self.assertFalse(architecture.uses_ir)
        self.assertEqual(architecture.repair_policy, "tool")
        no_repair = resolve_grounding_architecture(Grounding2RouteRunConfig(grounding_variant="tool_call", execution_repair="off"))
        self.assertEqual(no_repair.repair_policy, "off")


if __name__ == "__main__":
    unittest.main()
