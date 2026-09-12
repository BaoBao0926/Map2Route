"""Documented fallback task progression for LIMP.

This is intentionally conservative: it handles the sequential eventual
near[...] formulas produced by the local heuristic translator and many simple
LLM outputs. It is not a replacement for the official LIMP / Spot path.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FallbackProgression:
    ordered_symbols: list[str]
    forbidden_symbols: set[str] = field(default_factory=set)
    index: int = 0
    trace: list[dict[str, object]] = field(default_factory=list)

    @property
    def accepting(self) -> bool:
        return self.index >= len(self.ordered_symbols)

    @property
    def current_symbol(self) -> str | None:
        if self.accepting:
            return None
        return self.ordered_symbols[self.index]

    def current_formula(self) -> str:
        if self.accepting:
            return "True"
        return " -> ".join(self.ordered_symbols[self.index :])

    def progress(self, true_symbols: set[str], *, stage: str) -> None:
        start_index = self.index
        while self.index < len(self.ordered_symbols) and self.ordered_symbols[self.index] in true_symbols:
            self.index += 1
        violated = sorted(self.forbidden_symbols & true_symbols)
        self.trace.append(
            {
                "stage": stage,
                "true_symbols": sorted(true_symbols),
                "start_index": start_index,
                "end_index": self.index,
                "current_symbol": self.current_symbol,
                "accepting": self.accepting,
                "violated_forbidden_symbols": violated,
            }
        )


def forbidden_symbols_from_encoded_ltl(encoded_ltl: str, symbols: list[str]) -> set[str]:
    forbidden: set[str] = set()
    for symbol in symbols:
        if f"!{symbol}" in encoded_ltl or f"~{symbol}" in encoded_ltl:
            forbidden.add(symbol)
    return forbidden

