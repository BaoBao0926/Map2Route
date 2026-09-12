"""Optional LLM scene-graph heuristic guidance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import re
from pathlib import Path

from scripts.methods.lang2ltl.translator import configured_api_key, configured_model, gemini_response_text
from scripts.methods.osgllm.scene_graph.graph_types import SceneGraph


HEURISTIC_SYSTEM_PROMPT = """You provide high-level scene-graph search guidance.

Return JSON only:
{"preferred_nodes": ["room_2", "room_8", "object_4"], "reason": "short reason"}

Rules:
- Use only node ids present in the provided scene graph.
- Do not output coordinates or a dense path.
- Do not change the required LTL goals.
- This is only a heuristic preference for search ordering.
"""


def _cache_key(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(json.dumps(part, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:24]


def _json_from_text(text: str) -> dict[str, object]:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("LLM heuristic response must be a JSON object.")
    return payload


def _read_cache(root: Path | None, key: str) -> dict[str, object] | None:
    if root is None:
        return None
    path = root / "heuristic" / f"{key}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_cache(root: Path | None, key: str, payload: Mapping[str, object]) -> None:
    if root is None:
        return
    path = root / "heuristic" / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _compact_graph(graph: SceneGraph) -> dict[str, object]:
    return {
        "rooms": [
            {
                "id": room.node_id,
                "category": room.category,
                "connected_rooms": list(graph.room_adjacency.get(room.node_id, ())),
            }
            for room in graph.by_kind("room")
        ],
        "objects": [
            {
                "id": obj.node_id,
                "category": obj.category,
                "parent_id": obj.parent_id,
            }
            for obj in graph.by_kind("object")
        ],
    }


def llm_preferred_nodes(
    graph: SceneGraph,
    instruction: Mapping[str, object],
    goals: Sequence[str],
    *,
    model: str | None,
    cache_root: Path | None,
    overwrite_cache: bool,
    cache_only: bool = False,
) -> dict[str, object]:
    instruction_text = str(instruction.get("instruction", ""))
    prompt_payload = {
        "instruction": instruction_text,
        "required_goal_aps": list(goals),
        "scene_graph": _compact_graph(graph),
    }
    key = _cache_key("heuristic_v1", prompt_payload)
    if not overwrite_cache:
        cached = _read_cache(cache_root, key)
        if cached is not None:
            cached["cache_status"] = "hit"
            return cached
    if cache_only:
        raise RuntimeError(
            "Required cached LLM heuristic was not found "
            f"for cache key {key}."
        )
    if not configured_api_key():
        raise RuntimeError("Gemini API key is not configured.")
    chosen_model = model or configured_model()
    raw = gemini_response_text(
        model=chosen_model,
        system_prompt=HEURISTIC_SYSTEM_PROMPT,
        user_prompt=json.dumps(prompt_payload, ensure_ascii=False),
    )
    payload = _json_from_text(raw)
    raw_nodes = payload.get("preferred_nodes", [])
    valid_nodes: list[str] = []
    rejected_nodes: list[str] = []
    if isinstance(raw_nodes, Sequence) and not isinstance(raw_nodes, (str, bytes)):
        for item in raw_nodes:
            node_id = str(item)
            if node_id in graph.regions:
                valid_nodes.append(node_id)
            else:
                rejected_nodes.append(node_id)
    result = {
        "cache_status": "miss",
        "model": chosen_model,
        "raw_response": raw,
        "preferred_nodes": list(dict.fromkeys(valid_nodes)),
        "rejected_nodes": rejected_nodes,
        "reason": str(payload.get("reason", "")),
        "validation": {
            "valid": bool(valid_nodes),
            "rejected_count": len(rejected_nodes),
        },
    }
    _write_cache(cache_root, key, result)
    return result

