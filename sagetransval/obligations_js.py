"""JavaScript obligation extractor (eval-plan E4).

Mirrors the Python obligation extractor in ``sagetransval.obligations`` but
operates on a JS source. Two implementation paths:

  * **Preferred:** shell out to ``scripts/js_ast_extractor.js`` which uses Node's
    built-in ``vm`` + ``acorn`` (if available) to walk the AST and return a
    JSON-formatted obligation list.
  * **Fallback:** regex-based extraction from the source. Produces a strict
    subset of obligations (branch / boundary heuristics only). Used when Node
    or acorn are not available, so this module is import-safe in any environment.

Output schema matches :func:`sagetransval.obligations.extract_python_obligations`:
``{"family": [Case, ...]}`` where each Case is ``{"args": {...}, "obligation_id": "..."}``.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parents[1] / "scripts"
JS_AST_EXTRACTOR = SCRIPT_DIR / "js_ast_extractor.js"


def extract_js_obligations(source: str, entrypoint: str, seed_tests: list[dict]) -> dict[str, list[dict]]:
    """Extract branch / boundary / API / exception obligations from a JS source."""
    if JS_AST_EXTRACTOR.exists():
        try:
            return _extract_via_node(source, entrypoint, seed_tests)
        except (subprocess.SubprocessError, json.JSONDecodeError):
            pass
    return _extract_via_regex(source, entrypoint, seed_tests)


def _extract_via_node(source: str, entrypoint: str, seed_tests: list[dict]) -> dict[str, list[dict]]:
    payload = json.dumps({"source": source, "entrypoint": entrypoint, "seed_tests": seed_tests})
    proc = subprocess.run(
        ["node", str(JS_AST_EXTRACTOR)],
        input=payload, capture_output=True, text=True, timeout=30,
    )
    if proc.returncode != 0:
        raise subprocess.SubprocessError(proc.stderr)
    return json.loads(proc.stdout)


def _extract_via_regex(source: str, entrypoint: str, seed_tests: list[dict]) -> dict[str, list[dict]]:
    """Regex-based fallback. Conservative: picks up obvious literal boundaries
    (0, 1, -1, MAX_SAFE_INTEGER, empty array, empty string, null, undefined)
    and obvious `throw` sites for exception obligations.
    """
    obligations: dict[str, list[dict]] = {
        "control flow": [],
        "boundary/type": [],
        "api": [],
        "exceptions": [],
    }
    # Boundary literals from seed tests
    boundary_args = [
        {}, {"_boundary": 0}, {"_boundary": -1}, {"_boundary": 1},
        {"_boundary_str": ""}, {"_boundary_arr": []},
        {"_boundary": "NaN"}, {"_boundary": "Infinity"},
    ]
    for i, ba in enumerate(boundary_args):
        obligations["boundary/type"].append({
            "args": ba,
            "obligation_id": f"js_boundary_{i:02d}",
        })
    # Branch obligations: one per `if` / `else if`
    if_pattern = re.compile(r"\bif\s*\(([^)]+)\)")
    for j, m in enumerate(if_pattern.finditer(source)):
        cond = m.group(1).strip()
        obligations["control flow"].append({
            "args": {"_branch_witness": cond},
            "obligation_id": f"js_branch_{j:02d}",
        })
    # Exception obligations: one per `throw`
    throw_pattern = re.compile(r"\bthrow\s+new\s+(\w+)")
    for k, m in enumerate(throw_pattern.finditer(source)):
        obligations["exceptions"].append({
            "args": {"_expected_exception": m.group(1)},
            "obligation_id": f"js_throw_{k:02d}",
        })
    # API obligations: calls to global builtins that LLMs often mistranslate
    api_pattern = re.compile(r"\b(parseInt|parseFloat|JSON\.parse|Array\.from|Math\.\w+)\(")
    for l, m in enumerate(api_pattern.finditer(source)):
        obligations["api"].append({
            "args": {"_api_call": m.group(1)},
            "obligation_id": f"js_api_{l:02d}",
        })
    return obligations
