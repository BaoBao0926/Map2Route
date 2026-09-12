"""Small LTLf formula library used by the LTLCodeGen adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class Formula:
    op: str
    args: tuple["Formula", ...] = ()
    value: str | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "op": self.op,
            "value": self.value,
            "args": [arg.to_json() for arg in self.args],
        }

    @staticmethod
    def from_json(payload: Mapping[str, object]) -> "Formula":
        raw_args = payload.get("args", [])
        args = (
            tuple(Formula.from_json(item) for item in raw_args)
            if isinstance(raw_args, list)
            else ()
        )
        value = payload.get("value")
        return Formula(
            op=str(payload["op"]),
            args=args,
            value=value if isinstance(value, str) else None,
        ).simplify()

    def simplify(self) -> "Formula":
        if self.op in {"true", "false", "ap"}:
            return self
        args = tuple(arg.simplify() for arg in self.args)
        if self.op == "not":
            child = args[0]
            if child.op == "true":
                return FALSE
            if child.op == "false":
                return TRUE
            if child.op == "not":
                return child.args[0]
            return Formula("not", (child,))
        if self.op == "and":
            flat: list[Formula] = []
            for arg in args:
                if arg.op == "false":
                    return FALSE
                if arg.op == "true":
                    continue
                if arg.op == "and":
                    flat.extend(arg.args)
                else:
                    flat.append(arg)
            return _join("and", flat)
        if self.op == "or":
            flat = []
            for arg in args:
                if arg.op == "true":
                    return TRUE
                if arg.op == "false":
                    continue
                if arg.op == "or":
                    flat.extend(arg.args)
                else:
                    flat.append(arg)
            return _join("or", flat)
        if self.op == "imply":
            return lor(lnot(args[0]), args[1])
        return Formula(self.op, args, self.value)

    def __str__(self) -> str:
        if self.op == "true":
            return "true"
        if self.op == "false":
            return "false"
        if self.op == "ap":
            return str(self.value)
        if self.op == "not":
            return f"!({self.args[0]})"
        if self.op == "eventually":
            return f"F({self.args[0]})"
        if self.op == "always":
            return f"G({self.args[0]})"
        if self.op == "next":
            return f"X({self.args[0]})"
        if self.op == "until":
            return f"U({self.args[0]}, {self.args[1]})"
        joiner = " & " if self.op == "and" else " | "
        if self.op in {"and", "or"}:
            return "(" + joiner.join(str(arg) for arg in self.args) + ")"
        return f"{self.op}({', '.join(str(arg) for arg in self.args)})"


TRUE = Formula("true")
FALSE = Formula("false")


def _coerce(value: Formula | str) -> Formula:
    if isinstance(value, Formula):
        return value
    return ap(str(value))


def _join(op: str, args: Iterable[Formula]) -> Formula:
    values = tuple(dict.fromkeys(args))
    if not values:
        return TRUE if op == "and" else FALSE
    if len(values) == 1:
        return values[0]
    return Formula(op, values)


def ap(*args: object) -> Formula:
    if not args:
        raise ValueError("ap() requires an atomic proposition name.")
    name = str(args[-1])
    if name.startswith("reach(") and name.endswith(")"):
        name = name[len("reach(") : -1]
    return Formula("ap", value=name)


def ltl_and(first: Formula | str, second: Formula | str) -> Formula:
    return Formula("and", (_coerce(first), _coerce(second))).simplify()


def ltl_or(first: Formula | str, second: Formula | str) -> Formula:
    return Formula("or", (_coerce(first), _coerce(second))).simplify()


def ltl_not(value: Formula | str) -> Formula:
    return Formula("not", (_coerce(value),)).simplify()


def ltl_next(value: Formula | str) -> Formula:
    return Formula("next", (_coerce(value),)).simplify()


def ltl_until(first: Formula | str, second: Formula | str) -> Formula:
    return Formula("until", (_coerce(first), _coerce(second))).simplify()


def ltl_eventually(value: Formula | str) -> Formula:
    return Formula("eventually", (_coerce(value),)).simplify()


def ltl_always(value: Formula | str) -> Formula:
    return Formula("always", (_coerce(value),)).simplify()


def ltl_imply(first: Formula | str, second: Formula | str) -> Formula:
    return Formula("imply", (_coerce(first), _coerce(second))).simplify()


def lnot(value: Formula) -> Formula:
    return ltl_not(value)


def land(first: Formula, second: Formula) -> Formula:
    return ltl_and(first, second)


def lor(first: Formula, second: Formula) -> Formula:
    return ltl_or(first, second)


def progress(formula: Formula, labels: frozenset[str]) -> Formula:
    formula = formula.simplify()
    if formula.op in {"true", "false"}:
        return formula
    if formula.op == "ap":
        return TRUE if formula.value in labels else FALSE
    if formula.op == "not":
        child = formula.args[0]
        if child.op == "ap":
            return FALSE if child.value in labels else TRUE
        return ltl_not(progress(child, labels)).simplify()
    if formula.op == "and":
        result = TRUE
        for arg in formula.args:
            result = ltl_and(result, progress(arg, labels))
        return result.simplify()
    if formula.op == "or":
        result = FALSE
        for arg in formula.args:
            result = ltl_or(result, progress(arg, labels))
        return result.simplify()
    if formula.op == "eventually":
        child = formula.args[0]
        return ltl_or(progress(child, labels), formula).simplify()
    if formula.op == "always":
        child = formula.args[0]
        return ltl_and(progress(child, labels), formula).simplify()
    if formula.op == "until":
        left, right = formula.args
        return ltl_or(
            progress(right, labels),
            ltl_and(progress(left, labels), formula),
        ).simplify()
    if formula.op == "next":
        return formula.args[0].simplify()
    if formula.op == "imply":
        return progress(formula.simplify(), labels)
    raise ValueError(f"Unsupported LTL operator: {formula.op}")


def is_accepting(formula: Formula) -> bool:
    formula = formula.simplify()
    if formula.op == "true":
        return True
    if formula.op == "false":
        return False
    if formula.op == "always":
        return True
    if formula.op == "and":
        return all(is_accepting(arg) for arg in formula.args)
    if formula.op == "or":
        return any(is_accepting(arg) for arg in formula.args)
    return False


def atomic_propositions(formula: Formula) -> set[str]:
    aps: set[str] = set()

    def visit(node: Formula) -> None:
        if node.op == "ap":
            if node.value:
                aps.add(node.value)
            return
        for child in node.args:
            visit(child)

    visit(formula.simplify())
    return aps


def positive_eventual_aps(formula: Formula) -> set[str]:
    aps: set[str] = set()

    def visit(node: Formula, negated: bool = False, eventual: bool = False) -> None:
        if node.op == "ap" and eventual and not negated and node.value:
            aps.add(node.value)
        elif node.op == "not":
            visit(node.args[0], not negated, eventual)
        elif node.op == "eventually":
            visit(node.args[0], negated, True)
        elif node.op == "until":
            visit(node.args[1], negated, True)
        elif node.op in {"and", "or"}:
            for child in node.args:
                visit(child, negated, eventual)
        elif node.op in {"always", "next", "imply"}:
            for child in node.args:
                visit(child, negated, eventual)

    visit(formula)
    return aps
