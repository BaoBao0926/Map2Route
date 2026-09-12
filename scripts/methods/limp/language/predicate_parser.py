"""LIMP predicate encoding utilities.

This follows the official LIMP convention of replacing lifted predicates such
as `near[chair]` with single-letter atomic propositions for temporal planning.
"""

from __future__ import annotations

import re
import string


RESERVED_LTL_CHARS = {"F", "G", "X", "R", "M", "W", "U"}
PREDICATE_RE = re.compile(r"(near|pick|release)\[([^\]]+?)\]")


def encode_lifted_predicates(formula: str) -> tuple[str, dict[str, str]]:
    predicates = [f"{match[0]}[{match[1]}]" for match in PREDICATE_RE.findall(formula)]
    available = [char for char in string.ascii_uppercase if char not in RESERVED_LTL_CHARS]
    if len(predicates) > len(available):
        raise ValueError("Not enough proposition symbols for lifted predicates")
    encoding: dict[str, str] = {}
    encoded = formula
    for index, predicate in enumerate(predicates):
        symbol = available[index]
        encoding[symbol] = predicate
        encoded = encoded.replace(predicate, symbol, 1)
    return encoded, encoding


def unsupported_predicates(encoding_map: dict[str, str]) -> list[str]:
    unsupported: list[str] = []
    for predicate in encoding_map.values():
        name = predicate.split("[", 1)[0].strip()
        if name != "near":
            unsupported.append(predicate)
    return unsupported


def extract_ordered_symbols(encoded_ltl: str, encoding_map: dict[str, str]) -> list[str]:
    result: list[str] = []
    for char in encoded_ltl:
        if char in encoding_map and char not in result:
            result.append(char)
    return result

