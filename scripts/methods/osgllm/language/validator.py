"""Formula validation for OSG-LLM AP inventories."""

from __future__ import annotations

from scripts.methods.lang2ltl.ltl import Formula, atomic_propositions


def validate_formula(formula: Formula, inventory: set[str]) -> dict[str, object]:
    used = atomic_propositions(formula)
    missing = sorted(used - inventory)
    return {
        "valid": not missing,
        "used_atomic_propositions": sorted(used),
        "missing_atomic_propositions": missing,
        "backend": "local_formula_progression",
    }

