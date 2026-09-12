"""Prompt templates for the ILN SemPathBench adapter."""

from __future__ import annotations

import json
from typing import Mapping


GROUNDING_SYSTEM_PROMPT = """You are the SemPathBench grounding adapter for an ILN-style navigation method.

Convert the instruction into one or more area-level stages. This is adapter logic, not the original ILN PassageCostEvaluator.

Return JSON only:
{
  "start_area": "room_1",
  "stages": [
    {"stage_id": "stage_1", "target_type": "area", "area_id": "room_2", "source": "ordered_room_mention"},
    {"stage_id": "stage_2", "target_type": "object", "area_id": "room_3", "object_id": "object_7", "source": "final_object_goal"}
  ],
  "adapter_graph_constraints": {
    "forbidden_areas": [],
    "soft_avoid_areas": [],
    "forbidden_passages": [],
    "unsupported_preferences": []
  },
  "adapter_notes": []
}

Rules:
- Use only rooms, objects, and passages listed in the graph/inventory.
- Infer object targets from the instruction text and the object inventory. Do not assume hidden annotations.
- Use ordered stages only when the instruction explicitly implies order.
- Put only coarse room/passage avoid constraints in adapter_graph_constraints.
- Do not produce dense trajectory cells.
"""


STRICT_DESTINATION_SYSTEM_PROMPT = """You are the destination-area selector for an ILN navigation system.

Select exactly one destination area for the complete navigation command. ILN is an area-level navigator: do not create ordered stages, select object instances, interpret route constraints, or output dense cells. If the command mentions several places, select only the final destination. Object categories may be listed as semantic landmarks of an area; use them only to identify the destination area.

Return JSON only:
{
  "destination_area": "room_2",
  "reason": "brief area-level justification"
}

Rules:
- Use exactly one area listed in the area-passage graph.
- Do not return an object ID, waypoint list, intermediate destination, or route constraint.
- Do not invent rooms or semantic landmarks.
"""


PASSAGE_COST_SYSTEM_PROMPT = """You are the PassageCostEvaluator from an ILN-style robot navigation system.

Your job is narrow: given a current area, destination area, area-passage graph, and passage experience, output door/passage traversal costs.

Return JSON only:
{
  "door_costs": {
    "passage_1": {"cost": 10}
  },
  "navigation_task": ["room_1", "room_2"],
  "notes": []
}

Rules:
- Keep navigation_task equal to the provided current and destination areas.
- Use 10 as the base cost for passages with no evidence.
- Use lower costs for passages likely easy/open and higher costs for passages likely inaccessible.
- Do not decompose the instruction, select object instances, parse instruction avoid clauses, or output dense cells.
"""


REPAIR_JSON_SYSTEM_PROMPT = """Repair malformed JSON.

Return only one valid JSON object. Do not add markdown or explanation.
"""


def grounding_user_prompt(
    *,
    instruction_text: str,
    graph_payload: Mapping[str, object],
) -> str:
    return json.dumps(
        {
            "instruction": instruction_text,
            "current_area": graph_payload.get("current_area"),
            "areas": graph_payload.get("areas"),
            "passages": graph_payload.get("passages"),
            "inventory": graph_payload.get("inventory"),
        },
        ensure_ascii=False,
        indent=2,
    )


def strict_destination_user_prompt(
    *,
    instruction_text: str,
    graph_payload: Mapping[str, object],
    area_landmarks: Mapping[str, object],
) -> str:
    return json.dumps(
        {
            "human_command": instruction_text,
            "current_area": graph_payload.get("current_area"),
            "area_passage_graph": {
                "areas": graph_payload.get("areas"),
                "passages": graph_payload.get("passages"),
            },
            "semantic_landmarks_by_area": area_landmarks,
        },
        ensure_ascii=False,
        indent=2,
    )


def passage_cost_user_prompt(
    *,
    instruction_text: str,
    graph_payload: Mapping[str, object],
    current_area: str,
    destination_area: str,
    passage_history: object,
) -> str:
    return json.dumps(
        {
            "human_command": instruction_text,
            "current_area": current_area,
            "destination_area": destination_area,
            "area_passage_graph": {
                "areas": graph_payload.get("areas"),
                "passages": graph_payload.get("passages"),
            },
            "experience_file": passage_history,
        },
        ensure_ascii=False,
        indent=2,
    )

