"""Local LTLf residual-formula progression for encoded LIMP formulas."""

from __future__ import annotations

from dataclasses import dataclass, field

from scripts.methods.lang2ltl.ltl import (
    FALSE,
    Formula,
    atomic_propositions,
    is_accepting,
    ltl_always,
    ltl_and,
    ltl_eventually,
    ltl_imply,
    ltl_next,
    ltl_not,
    ltl_or,
    ltl_until,
    positive_eventual_aps,
    progress,
)


class LTLFParseError(ValueError):
    """Raised when encoded LTL cannot be parsed by the local LTLf parser."""


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if text.startswith("->", index):
            tokens.append("->")
            index += 2
            continue
        if char in "()!&|":
            tokens.append(char)
            index += 1
            continue
        if char.isalnum() or char == "_":
            start = index
            while index < len(text) and (text[index].isalnum() or text[index] == "_"):
                index += 1
            tokens.append(text[start:index])
            continue
        raise LTLFParseError(f"Unexpected token character in encoded LTL: {char!r}")
    return tokens


def parse_encoded_ltl(text: str) -> Formula:
    tokens = _tokenize(text)
    if not tokens:
        raise LTLFParseError("Cannot parse empty encoded LTL.")
    index = 0

    def peek() -> str | None:
        return tokens[index] if index < len(tokens) else None

    def consume(expected: str | None = None) -> str:
        nonlocal index
        if index >= len(tokens):
            raise LTLFParseError("Unexpected end of encoded LTL.")
        token = tokens[index]
        if expected is not None and token != expected:
            raise LTLFParseError(f"Expected {expected!r}, got {token!r}.")
        index += 1
        return token

    def parse_primary() -> Formula:
        token = peek()
        if token is None:
            raise LTLFParseError("Unexpected end of encoded LTL.")
        if token == "(":
            consume("(")
            node = parse_imply()
            consume(")")
            return node
        if token in {"!", "F", "G", "X"}:
            consume()
            child = parse_primary()
            if token == "!":
                return ltl_not(child)
            if token == "F":
                return ltl_eventually(child)
            if token == "G":
                return ltl_always(child)
            return ltl_next(child)
        if token.lower() == "true":
            consume()
            from scripts.methods.lang2ltl.ltl import TRUE

            return TRUE
        if token.lower() == "false":
            consume()
            return FALSE
        consume()
        from scripts.methods.lang2ltl.ltl import ap

        return ap(token)

    def parse_until() -> Formula:
        node = parse_primary()
        while peek() == "U":
            consume("U")
            node = ltl_until(node, parse_primary())
        return node

    def parse_and() -> Formula:
        node = parse_until()
        while peek() == "&":
            consume("&")
            node = ltl_and(node, parse_until())
        return node

    def parse_or() -> Formula:
        node = parse_and()
        while peek() == "|":
            consume("|")
            node = ltl_or(node, parse_and())
        return node

    def parse_imply() -> Formula:
        node = parse_or()
        while peek() == "->":
            consume("->")
            node = ltl_imply(node, parse_or())
        return node

    formula = parse_imply().simplify()
    if index != len(tokens):
        raise LTLFParseError(f"Unexpected trailing encoded LTL tokens: {' '.join(tokens[index:])}")
    return formula


def _always_negated_symbols(formula: Formula) -> set[str]:
    symbols: set[str] = set()

    def visit(node: Formula) -> None:
        node = node.simplify()
        if node.op == "always":
            child = node.args[0].simplify()
            if child.op == "not" and child.args[0].op == "ap" and child.args[0].value:
                symbols.add(str(child.args[0].value))
        for child in node.args:
            visit(child)

    visit(formula)
    return symbols


def negative_atomic_propositions(formula: Formula) -> set[str]:
    """Collect proposition occurrences that appear under odd negation."""

    symbols: set[str] = set()

    def visit(node: Formula, negated: bool = False) -> None:
        node = node.simplify()
        if node.op == "ap":
            if negated and node.value:
                symbols.add(str(node.value))
            return
        if node.op == "not":
            visit(node.args[0], not negated)
            return
        for child in node.args:
            visit(child, negated)

    visit(formula)
    return symbols


@dataclass
class LTLFProgression:
    original_formula: Formula
    ordered_symbols: list[str]
    formula: Formula | None = None
    trace: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.formula is None:
            self.formula = self.original_formula.simplify()

    @property
    def forbidden_symbols(self) -> set[str]:
        return _always_negated_symbols(self.formula or self.original_formula)

    @property
    def accepting(self) -> bool:
        return is_accepting(self.formula or self.original_formula)

    @property
    def failed(self) -> bool:
        """Return whether the residual formula is the rejecting sink."""

        return (self.formula or self.original_formula).simplify().op == "false"

    def current_formula(self) -> str:
        return str((self.formula or self.original_formula).simplify())

    def preview(self, true_symbols: set[str]) -> Formula:
        """Progress without mutating state.

        The grid TPSM planner uses this to classify every traversable cell as
        a self-loop, an enabled DFA transition, or a rejecting transition.
        """

        current = (self.formula or self.original_formula).simplify()
        return progress(current, frozenset(true_symbols)).simplify()

    @property
    def current_symbol(self) -> str | None:
        current = (self.formula or self.original_formula).simplify()
        if is_accepting(current) or current.op == "false":
            return None

        eventual = positive_eventual_aps(current)
        candidates = [symbol for symbol in self.ordered_symbols if symbol in eventual]
        if not candidates:
            all_aps = atomic_propositions(current)
            candidates = [symbol for symbol in self.ordered_symbols if symbol in all_aps]

        for symbol in candidates:
            next_formula = progress(current, frozenset({symbol})).simplify()
            if next_formula != current and next_formula.op != "false":
                return symbol
        return candidates[0] if candidates else None

    def progress(self, true_symbols: set[str], *, stage: str) -> None:
        before = (self.formula or self.original_formula).simplify()
        after = self.preview(true_symbols)
        self.formula = after
        self.trace.append(
            {
                "stage": stage,
                "true_symbols": sorted(true_symbols),
                "formula_before": str(before),
                "formula_after": str(after),
                "accepting": is_accepting(after),
            }
        )
