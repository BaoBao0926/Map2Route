"""Automaton construction facade for LIMP."""

from __future__ import annotations

from scripts.methods.limp.logic.residual_fallback import FallbackProgression, forbidden_symbols_from_encoded_ltl
from scripts.methods.limp.logic.spot_adapter import spot_available
from scripts.methods.limp.logic.ltlf_progression import LTLFProgression, parse_encoded_ltl


class AutomatonBackendUnavailable(RuntimeError):
    """Raised when the configured formal automaton backend is unavailable."""


def build_progression(
    encoded_ltl: str,
    ordered_symbols: list[str],
    *,
    backend: str = "spot",
    allow_residual_fallback: bool = False,
) -> tuple[LTLFProgression | FallbackProgression, dict[str, object]]:
    if backend == "ltlf":
        formula = parse_encoded_ltl(encoded_ltl)
        progression = LTLFProgression(
            original_formula=formula,
            ordered_symbols=list(ordered_symbols),
        )
        summary = {
            "backend": "local_ltlf_residual_formula",
            "is_debug_backend": False,
            "encoded_ltl": encoded_ltl,
            "initial_formula": str(formula),
            "ordered_symbols": list(ordered_symbols),
            "accepting_states": "formula states satisfying is_accepting(formula)",
            "method_level_deviation": (
                "Uses the local SemPathBench LTLf Formula/progress semantics "
                "instead of the official LIMP Spot/DFA wrapper."
            ),
        }
        return progression, summary

    if backend == "spot" and not spot_available():
        if not allow_residual_fallback:
            raise AutomatonBackendUnavailable(
                "Spot/formal automaton backend is unavailable; residual_fallback is debug-only."
            )
        backend = "residual-debug"

    if backend in {"residual-debug", "residual_fallback"} and not allow_residual_fallback:
        raise AutomatonBackendUnavailable(
            "residual_fallback is debug-only and requires --allow-residual-fallback."
        )

    if backend not in {"residual-debug", "residual_fallback"}:
        raise AutomatonBackendUnavailable(
            f"Automaton backend {backend!r} is not implemented in this adapter."
        )

    progression = FallbackProgression(
        ordered_symbols=list(ordered_symbols),
        forbidden_symbols=forbidden_symbols_from_encoded_ltl(encoded_ltl, ordered_symbols),
    )
    summary = {
        "backend": "residual_fallback",
        "is_debug_backend": True,
        "reason": "explicit_debug_residual_fallback",
        "ordered_symbols": list(ordered_symbols),
        "forbidden_symbols": sorted(progression.forbidden_symbols),
        "method_level_deviation": (
            "Uses a conservative sequential-eventual fallback instead of the "
            "official LIMP Spot/DFA path."
        ),
    }
    return progression, summary
