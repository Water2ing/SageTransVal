"""Lightweight source-obligation extraction for Proposal 1."""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from typing import Any

from .models import Case


def extract_python_obligations(source: str, entrypoint: str, seed_tests: list[Case]) -> dict[str, list[Case]]:
    tree = ast.parse(source)
    fn = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == entrypoint), None)
    template = list(seed_tests[0].get("args", [])) if seed_tests else [1]
    obligations: dict[str, list[Case]] = defaultdict(list)
    if fn is None:
        return {}

    constants: set[Any] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, str)):
            constants.add(node.value)
        if isinstance(node, ast.Compare):
            obligations["control flow"].extend(_numeric_cases(template, [-1, 0, 1]))
        if isinstance(node, (ast.Subscript, ast.Call)):
            obligations["api contracts"].extend(_shape_cases(template))
        if isinstance(node, (ast.Div, ast.FloorDiv, ast.Mod)):
            obligations["exceptions"].extend(_numeric_cases(template, [0]))

    int_constants = [value for value in constants if isinstance(value, int)]
    if int_constants:
        probes = sorted({value + delta for value in int_constants for delta in (-1, 0, 1)})
        obligations["boundary/type"].extend(_numeric_cases(template, probes[:6]))
    if any(isinstance(value, str) for value in constants):
        obligations["boundary/type"].extend(_string_cases(template))
    return _dedupe(obligations)


_INT_MAX_I32 = 2147483647
_INT_MIN_I32 = -2147483648


def extract_c_obligations(source: str, seed_tests: list[Case]) -> dict[str, list[Case]]:
    template = list(seed_tests[0].get("args", [])) if seed_tests else [1]
    obligations: dict[str, list[Case]] = defaultdict(list)
    ints = [int(match) for match in re.findall(r"(?<![A-Za-z_])-?\d+", source)]
    if re.search(r"\bif\s*\(", source):
        obligations["control flow"].extend(_numeric_cases(template, [-1, 0, 1]))
    if any(op in source for op in ("/", "%")):
        obligations["exceptions"].extend(_numeric_cases(template, [0]))
    if ints:
        probes = sorted({value + delta for value in ints for delta in (-1, 0, 1)})
        obligations["boundary/type"].extend(_numeric_cases(template, probes[:6]))
    if "*" in source or "NULL" in source:
        obligations["api contracts"].extend(_numeric_cases(template, [0]))

    # --- v2 C-specific probe generators ---------------------------------------
    # Integer-width probes: target signed-overflow and INT_MIN/-1 seams.
    if re.search(r"[+\-*/<>]", source) and any(isinstance(value, int) and not isinstance(value, bool) for value in template):
        width_probes = [_INT_MIN_I32, _INT_MIN_I32 + 1, -1, 0, 1, _INT_MAX_I32 - 1, _INT_MAX_I32]
        obligations["boundary/type"].extend(_numeric_cases(template, width_probes))
    # Shift-amount probes when the source uses `<<` or `>>`.
    if re.search(r"<<|>>", source):
        obligations["boundary/type"].extend(_numeric_cases(template, [0, 1, 31, 32]))
    # Sentinel/error probes when the source returns a sentinel value
    # (e.g. `return -1;`, `return INT_MIN;`, `return 0;` on guarded paths).
    if re.search(r"return\s+-1\b", source) or "INT_MIN" in source:
        obligations["exceptions"].extend(_numeric_cases(template, [_INT_MIN_I32, -1]))
    # UTF-8 / byte-vs-codepoint probes when the source has a `char*` parameter.
    if any(isinstance(value, str) for value in template):
        utf8_cases = _utf8_string_cases(template)
        obligations["boundary/type"].extend(utf8_cases)
    return _dedupe(obligations)


def _utf8_string_cases(template: list[Any]) -> list[Case]:
    """Probe strings that distinguish byte-level from codepoint-level handling."""
    positions = [idx for idx, value in enumerate(template) if isinstance(value, str)]
    if not positions:
        return []
    probes = [
        "",            # empty
        "abc",         # ASCII
        "héllo",       # mixed ASCII + multi-byte
        "日本",         # all multi-byte
        "a\xc3\xa9",   # raw byte sequence forming 'é' in UTF-8
    ]
    cases: list[Case] = []
    for probe in probes:
        args = list(template)
        for pos in positions:
            args[pos] = probe
        cases.append({"args": args})
    return cases


def generic_cases(seed_tests: list[Case], aggressive: bool = False) -> list[Case]:
    template = list(seed_tests[0].get("args", [])) if seed_tests else [1]
    cases: list[Case] = []
    pools: list[list[Any]] = []
    for value in template:
        if isinstance(value, bool):
            pools.append([False, True])
        elif isinstance(value, int):
            pools.append([-3, -1, 0, 1, 2, 7] if aggressive else [0, 1, 2])
        elif isinstance(value, str):
            pools.append(["", "a", "A", "abc"] if aggressive else ["", "abc"])
        elif isinstance(value, list):
            pools.append([[], [0], [1, -1]] if aggressive else [[], [1]])
        else:
            pools.append([value])
    for idx in range(max(len(pool) for pool in pools)):
        cases.append({"args": [pool[min(idx, len(pool) - 1)] for pool in pools]})
    return cases


def _numeric_cases(template: list[Any], probes: list[int]) -> list[Case]:
    positions = [idx for idx, value in enumerate(template) if isinstance(value, int) and not isinstance(value, bool)]
    if not positions:
        return []
    cases = []
    for probe in probes:
        args = list(template)
        args[positions[-1]] = probe
        cases.append({"args": args})
    return cases


def _string_cases(template: list[Any]) -> list[Case]:
    positions = [idx for idx, value in enumerate(template) if isinstance(value, str)]
    return [{"args": [("" if idx in positions else value) for idx, value in enumerate(template)]}] if positions else []


def _shape_cases(template: list[Any]) -> list[Case]:
    cases = []
    for idx, value in enumerate(template):
        if isinstance(value, list):
            args = list(template)
            args[idx] = []
            cases.append({"args": args})
        if isinstance(value, str):
            args = list(template)
            args[idx] = ""
            cases.append({"args": args})
    return cases


def _dedupe(obligations: dict[str, list[Case]]) -> dict[str, list[Case]]:
    out: dict[str, list[Case]] = {}
    for family, cases in obligations.items():
        seen = set()
        unique = []
        for case in cases:
            key = repr(case.get("args", []))
            if key not in seen:
                seen.add(key)
                unique.append(case)
        if unique:
            out[family] = unique
    return out
