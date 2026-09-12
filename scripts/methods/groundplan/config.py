"""Configuration defaults for the GroundPlan method."""

from __future__ import annotations

from pathlib import Path

from scripts.make_instruction.make_instruction import REPO_ROOT


METHOD_NAME = "GroundPlan"
METHOD_ROOT = REPO_ROOT / "resources" / "methods" / "groundplan" / "main_result"
PROMPT_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL = "gemini-2.5-flash"
PARSE_MODES = ("intent", "api", "direct_id", "ltl")
PLANNER_MODES = (
    "sequential_greedy_astar",
    "global_layered_dp",
    "global_progress_astar",
)
DEFAULT_PLANNER_MODE = "sequential_greedy_astar"
RELATIVE_COST_MODES = ("difference", "ratio")
DEFAULT_GROUNDING_VARIANT = "direct_cap_repair"
GROUNDING_VARIANTS = (
    "direct_dsl",
    "direct_cap",
    "direct_cap_repair",
    "ir_to_code",
    "ir_code_refine",
    "ir_code_refine_code_repair",
    "full",
    "direct_id",
    "ltl",
    "tool_call",
)

OBJECT_GOAL_RADIUS_METERS = 0.30
NEAR_RADIUS_METERS = 1.50
FAR_SIGMA_METERS = 1.50
RELATIVE_RADIUS_METERS = 1.50
CLEARANCE_RADIUS_METERS = 0.75
PATH_SHAPE_RADIUS_METERS = 0.90

WEIGHT_NEAR = 8.00
WEIGHT_FAR = 96.00
# Metric-aligned relative cost selected by an oracle-annotation sweep over the
# first 20 val-unseen episodes containing a relative preference.
DEFAULT_RELATIVE_COST_MODE = "ratio"
WEIGHT_RELATIVE = 64.00
WEIGHT_PATH_SHAPE = 18.00
WEIGHT_CLEARANCE = 36.00
# Penalizes heading changes while planning circle/path-shape waypoint stages.
# Small values remove local zig-zags without overwhelming semantic costs.
WEIGHT_SMOOTHNESS = 0.02

# Circle waypoint snapping prefers cells with clearance from walls, obstacles,
# and room boundaries. Object loops can sit farther into open space, while room
# loops should stay closer to the intended room-scale loop.
CIRCLE_OBJECT_WAYPOINT_CLEARANCE_CELLS = 16.0
CIRCLE_OBJECT_WAYPOINT_CLEARANCE_WEIGHT = 8.0
CIRCLE_ROOM_WAYPOINT_CLEARANCE_CELLS = 10.0
CIRCLE_ROOM_WAYPOINT_CLEARANCE_WEIGHT = 16.0
CIRCLE_WAYPOINT_SEARCH_RADIUS_CELLS = 32

MAX_EXPANSIONS = 1_000_000
MAX_GOAL_HEURISTIC_CELLS = 512
# The canonical method is the largest repair-budget setting in the ablation
# suite. Lower budgets are obtained by truncating the saved attempt history,
# so all rows share exactly the same initial generation and repairs.
MAX_GROUNDING_REPAIRS = 3
MAX_TOOL_CALLS = 32
MAX_TOOL_LLM_TURNS = 40
MAX_TOOL_QUERY_RETRIES = 3

ROOM_ALIASES = {
    "livingroom": "living_room",
    "living room": "living_room",
    "bath room": "bathroom",
}

ENTITY_ALIASES = {
    "armchair": "arm_chair",
    "arm chair": "arm_chair",
    "countertop": "counter_top",
    "counter top": "counter_top",
    "plant": "house_plant",
    "houseplant": "house_plant",
    "tv": "tv_stand",
    "television": "tv_stand",
    "trash can": "garbage_can",
    "garbage bag": "garbage_bag",
    "drawer": "dresser",
    "door": "doorway",
    "door frame": "doorframe",
    "shelving unit": "shelf",
}
