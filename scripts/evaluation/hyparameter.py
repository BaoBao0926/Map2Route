"""Central evaluator hyperparameters.

Edit this file to calibrate SemPathBench evaluation without changing the
metric implementation code.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real


EVALUATION_EPSILON = 1e-6

# PLR treats a non-empty trajectory as at least one occupied grid cell, even
# when the route has no movement beyond the start point.
PLR_MIN_EFFECTIVE_PATH_LENGTH_GRID = 1.0

# SCS normalization: 1.2 means 20% worse than the human reference scores 0.
SCS_BAD_RATIO_THRESHOLD = 1.5
SCS_RATIO_EPSILON = 1e-8

# Path-shape preference uses tolerant nDTW directly, independent of SCS
# bad-ratio normalization. In grid coordinates, 1.0 means one grid cell.
# Deviations inside the tolerance band are free; beyond the band, an average
# excess deviation of success_distance cells gives exp(-1) ~= 0.37.
PATH_SHAPE_TOLERANCE_GRID = 10.0
PATH_SHAPE_NDTW_SUCCESS_DISTANCE = 4.0

# Relative preference scores compare the predicted continuous margin against
# the human reference margin. Values are in grid-cell units.
RELATIVE_PREFERENCE_HUMAN_BETA = 0.5
RELATIVE_PREFERENCE_MIN_MARGIN = 0.2
RELATIVE_PREFERENCE_MIN_VALID_LENGTH_GRID = 0.05

# Set to False to exclude move_smoothness from SCS evaluation.
EVALUATE_MOVE_SMOOTHNESS = False

TRAVERSABLE_OBJECT_CATEGORIES = {"doorframe", "doorway"}

CLEARANCE_DISTANCE_CACHE_SUFFIX = "_clearance_distance.npz"
CLEARANCE_DISTANCE_CACHE_VERSION = 1
CLEARANCE_DISTANCE_FIELD_KEY = "_clearance_distance_field"
CLEARANCE_EXCLUDED_DISTANCE_FIELDS_KEY = "_clearance_excluded_distance_fields"


@dataclass(frozen=True)
class SCSConfig:
    bad_ratio_threshold: float = SCS_BAD_RATIO_THRESHOLD
    eps: float = SCS_RATIO_EPSILON
    path_shape_tolerance_grid: float = PATH_SHAPE_TOLERANCE_GRID
    path_shape_ndtw_success_distance: float = PATH_SHAPE_NDTW_SUCCESS_DISTANCE
    relative_preference_human_beta: float = RELATIVE_PREFERENCE_HUMAN_BETA
    relative_preference_min_margin: float = RELATIVE_PREFERENCE_MIN_MARGIN
    relative_preference_min_valid_length_grid: float = (
        RELATIVE_PREFERENCE_MIN_VALID_LENGTH_GRID
    )


def active_scs_config() -> SCSConfig:
    """Return the current SCS config from module-level hyperparameters."""

    return SCSConfig(
        bad_ratio_threshold=float(SCS_BAD_RATIO_THRESHOLD),
        eps=float(SCS_RATIO_EPSILON),
        path_shape_tolerance_grid=float(PATH_SHAPE_TOLERANCE_GRID),
        path_shape_ndtw_success_distance=float(PATH_SHAPE_NDTW_SUCCESS_DISTANCE),
        relative_preference_human_beta=float(RELATIVE_PREFERENCE_HUMAN_BETA),
        relative_preference_min_margin=float(RELATIVE_PREFERENCE_MIN_MARGIN),
        relative_preference_min_valid_length_grid=float(
            RELATIVE_PREFERENCE_MIN_VALID_LENGTH_GRID
        ),
    )


def validate_scs_config(config: SCSConfig) -> None:
    if (
        not isinstance(config.bad_ratio_threshold, Real)
        or isinstance(config.bad_ratio_threshold, bool)
        or float(config.bad_ratio_threshold) <= 1.0
    ):
        raise ValueError("SCS_BAD_RATIO_THRESHOLD must be greater than 1.0.")
    if (
        not isinstance(config.eps, Real)
        or isinstance(config.eps, bool)
        or float(config.eps) <= 0.0
    ):
        raise ValueError("SCS_RATIO_EPSILON must be positive.")
    if (
        not isinstance(config.path_shape_tolerance_grid, Real)
        or isinstance(config.path_shape_tolerance_grid, bool)
        or float(config.path_shape_tolerance_grid) < 0.0
    ):
        raise ValueError("PATH_SHAPE_TOLERANCE_GRID must be non-negative.")
    if (
        not isinstance(config.path_shape_ndtw_success_distance, Real)
        or isinstance(config.path_shape_ndtw_success_distance, bool)
        or float(config.path_shape_ndtw_success_distance) <= 0.0
    ):
        raise ValueError("PATH_SHAPE_NDTW_SUCCESS_DISTANCE must be positive.")
    if (
        not isinstance(config.relative_preference_human_beta, Real)
        or isinstance(config.relative_preference_human_beta, bool)
        or float(config.relative_preference_human_beta) < 0.0
    ):
        raise ValueError("RELATIVE_PREFERENCE_HUMAN_BETA must be non-negative.")
    if (
        not isinstance(config.relative_preference_min_margin, Real)
        or isinstance(config.relative_preference_min_margin, bool)
        or float(config.relative_preference_min_margin) <= 0.0
    ):
        raise ValueError("RELATIVE_PREFERENCE_MIN_MARGIN must be positive.")
    if (
        not isinstance(config.relative_preference_min_valid_length_grid, Real)
        or isinstance(config.relative_preference_min_valid_length_grid, bool)
        or float(config.relative_preference_min_valid_length_grid) < 0.0
    ):
        raise ValueError(
            "RELATIVE_PREFERENCE_MIN_VALID_LENGTH_GRID must be non-negative."
        )


def scs_config_details(config: SCSConfig | None = None) -> dict[str, float]:
    config = active_scs_config() if config is None else config
    validate_scs_config(config)
    return {
        "bad_ratio_threshold": float(config.bad_ratio_threshold),
        "eps": float(config.eps),
        "path_shape_tolerance_grid": float(config.path_shape_tolerance_grid),
        "path_shape_ndtw_success_distance": float(config.path_shape_ndtw_success_distance),
        "relative_preference_human_beta": float(
            config.relative_preference_human_beta
        ),
        "relative_preference_min_margin": float(config.relative_preference_min_margin),
        "relative_preference_min_valid_length_grid": float(
            config.relative_preference_min_valid_length_grid
        ),
    }


def move_smoothness_enabled() -> bool:
    return bool(EVALUATE_MOVE_SMOOTHNESS)


def _validate_non_negative(value: float, label: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise ValueError(f"{label} must be non-negative for SCS normalization.")
    return parsed


def _clip_score(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def degradation_to_score(
    degradation: float,
    config: SCSConfig | None = None,
) -> float:
    """Map relative degradation to a bounded score."""

    config = active_scs_config() if config is None else config
    validate_scs_config(config)
    max_degradation = float(config.bad_ratio_threshold) - 1.0
    return _clip_score(1.0 - float(degradation) / max_degradation)


def score_smaller_is_better(
    algorithm_value: float,
    human_value: float,
    config: SCSConfig | None = None,
) -> tuple[float, float]:
    """Score a non-negative metric where lower values are preferred.

    Returns ``(score, degradation_ratio)``. A ratio of 1 means human-level
    performance; values above 1 mean worse than human.
    """

    config = active_scs_config() if config is None else config
    validate_scs_config(config)
    algorithm = _validate_non_negative(algorithm_value, "algorithm_value")
    human = _validate_non_negative(human_value, "human_value")
    eps = float(config.eps)

    if abs(human) < eps and abs(algorithm) < eps:
        return 1.0, 1.0
    if abs(human) < eps:
        return 0.0, float("inf")

    ratio = algorithm / max(abs(human), eps)
    degradation = max(0.0, ratio - 1.0)
    return degradation_to_score(degradation, config), ratio


def score_larger_is_better(
    algorithm_value: float,
    human_value: float,
    config: SCSConfig | None = None,
) -> tuple[float, float]:
    """Score a non-negative metric where higher values are preferred.

    Returns ``(score, degradation_ratio)``. A ratio of 1 means human-level
    performance; values above 1 mean worse than human.
    """

    config = active_scs_config() if config is None else config
    validate_scs_config(config)
    algorithm = _validate_non_negative(algorithm_value, "algorithm_value")
    human = _validate_non_negative(human_value, "human_value")
    eps = float(config.eps)

    if abs(human) < eps:
        return 1.0, 1.0 if abs(algorithm) < eps else 0.0
    if abs(algorithm) < eps:
        return 0.0, float("inf")

    ratio = human / max(abs(algorithm), eps)
    degradation = max(0.0, ratio - 1.0)
    return degradation_to_score(degradation, config), ratio
