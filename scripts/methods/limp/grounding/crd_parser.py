"""Parser for LIMP Composable Referent Descriptors."""

from __future__ import annotations

from dataclasses import dataclass, field

from scripts.methods.limp.utils.geometry import normalize_name


@dataclass(frozen=True)
class CRD:
    base: str
    comparators: tuple["ComparatorCall", ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ComparatorCall:
    name: str
    args: tuple[CRD, ...]


@dataclass(frozen=True)
class ParsedPredicate:
    encoded: str
    predicate: str
    referents: tuple[CRD, ...]
    raw: str


def _split_top_level(value: str, delimiter: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    index = 0
    while index < len(value):
        char = value[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif depth == 0 and value.startswith(delimiter, index):
            parts.append(value[start:index].strip())
            index += len(delimiter)
            start = index
            continue
        index += 1
    parts.append(value[start:].strip())
    return [part for part in parts if part]


def parse_crd(value: str) -> CRD:
    text = value.strip()
    parts = _split_top_level(text, "::")
    if not parts:
        return CRD(base="")
    base = normalize_name(parts[0])
    comparators: list[ComparatorCall] = []
    for item in parts[1:]:
        open_index = item.find("(")
        close_index = item.rfind(")")
        if open_index <= 0 or close_index <= open_index:
            comparators.append(ComparatorCall(name=normalize_name(item), args=()))
            continue
        name = normalize_name(item[:open_index])
        arg_text = item[open_index + 1 : close_index]
        args = tuple(parse_crd(arg) for arg in _split_top_level(arg_text, ","))
        comparators.append(ComparatorCall(name=name, args=args))
    return CRD(base=base, comparators=tuple(comparators))


def parse_predicate(encoded: str, raw: str) -> ParsedPredicate:
    text = raw.strip()
    bracket = text.find("[")
    close = text.rfind("]")
    if bracket <= 0 or close <= bracket:
        return ParsedPredicate(encoded=encoded, predicate=normalize_name(text), referents=(), raw=raw)
    predicate = normalize_name(text[:bracket])
    args_text = text[bracket + 1 : close]
    referents = tuple(parse_crd(part) for part in _split_top_level(args_text, ","))
    return ParsedPredicate(encoded=encoded, predicate=predicate, referents=referents, raw=raw)


def parsed_crd_to_dict(crd: CRD) -> dict[str, object]:
    return {
        "base": crd.base,
        "comparators": [
            {
                "name": comparator.name,
                "args": [parsed_crd_to_dict(arg) for arg in comparator.args],
            }
            for comparator in crd.comparators
        ],
    }


def parsed_predicate_to_dict(predicate: ParsedPredicate) -> dict[str, object]:
    return {
        "encoded": predicate.encoded,
        "predicate": predicate.predicate,
        "raw": predicate.raw,
        "referents": [parsed_crd_to_dict(referent) for referent in predicate.referents],
    }

