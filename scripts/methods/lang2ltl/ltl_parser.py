"""Parse Lang2LTL/Spot-style prefix LTL strings into local Formula objects."""

from __future__ import annotations

import re
from dataclasses import dataclass

from scripts.methods.lang2ltl.ltl import (
    FALSE,
    TRUE,
    Formula,
    ap,
    ltl_always,
    ltl_and,
    ltl_eventually,
    ltl_imply,
    ltl_next,
    ltl_not,
    ltl_or,
    ltl_until,
)


UNARY_OPERATORS = {"!", "F", "G", "X"}
BINARY_OPERATORS = {"&", "|", "U", "i", "->", "e", "<->", "M"}


@dataclass(frozen=True)
class ParseResult:
    formula: Formula
    warnings: tuple[str, ...] = ()


def _strip_wrappers(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:text|ltl)?\s*(.*?)```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    if stripped.lower().startswith("ltl:"):
        stripped = stripped.split(":", 1)[1].strip()
    return stripped


def tokenize_prefix_ltl(text: str) -> list[str]:
    """Tokenize a whitespace/punctuation-tolerant prefix LTL string."""
    stripped = _strip_wrappers(text)
    stripped = stripped.replace("(", " ").replace(")", " ").replace(",", " ")
    stripped = stripped.replace("<->", " <-> ").replace("->", " -> ")
    tokens = [token for token in stripped.split() if token]
    return tokens


def _clean_ap_token(token: str) -> str:
    return token.strip().strip("\"'`")


def parse_prefix_ltl(text: str) -> ParseResult:
    tokens = tokenize_prefix_ltl(text)
    if not tokens:
        raise ValueError("Cannot parse empty LTL formula.")
    index = 0
    warnings: list[str] = []

    def parse_one() -> Formula:
        nonlocal index
        if index >= len(tokens):
            raise ValueError("Unexpected end of LTL formula.")
        token = tokens[index]
        index += 1

        lowered = token.lower()
        if lowered == "true":
            return TRUE
        if lowered == "false":
            return FALSE
        if token in UNARY_OPERATORS:
            child = parse_one()
            if token == "!":
                return ltl_not(child)
            if token == "F":
                return ltl_eventually(child)
            if token == "G":
                return ltl_always(child)
            if token == "X":
                return ltl_next(child)
        if token in BINARY_OPERATORS:
            left = parse_one()
            right = parse_one()
            if token == "&":
                return ltl_and(left, right)
            if token == "|":
                return ltl_or(left, right)
            if token == "U":
                return ltl_until(left, right)
            if token in {"i", "->"}:
                return ltl_imply(left, right)
            if token in {"e", "<->"}:
                return ltl_and(ltl_imply(left, right), ltl_imply(right, left))
            if token == "M":
                warnings.append(
                    "Operator M is approximated as until in the local planner."
                )
                return ltl_until(left, right)
        return ap(_clean_ap_token(token))

    formula = parse_one().simplify()
    if index != len(tokens):
        extras = " ".join(tokens[index:])
        raise ValueError(f"Unexpected trailing tokens in LTL formula: {extras}")
    return ParseResult(formula=formula, warnings=tuple(warnings))


def substitute_formula_aps(formula: Formula, ap_map: dict[str, str]) -> Formula:
    """Replace AP names in a Formula tree."""
    formula = formula.simplify()
    if formula.op == "ap":
        return ap(ap_map.get(str(formula.value), str(formula.value)))
    if formula.op in {"true", "false"}:
        return formula
    return Formula(
        formula.op,
        tuple(substitute_formula_aps(child, ap_map) for child in formula.args),
        formula.value,
    ).simplify()
