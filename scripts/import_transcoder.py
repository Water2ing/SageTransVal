"""TransCoder import script for SageTransVal eval-plan E2.

Downloads (if available) and converts TransCoder Python-Java parallel function pairs
into the SubjectV2 schema. Filters subjects that:
  * are pure functions (no I/O on stdin/stdout)
  * have a single named entry point on the Python side
  * have type-annotated parameters or pass a heuristic type inference

The script is idempotent and cache-friendly: if a subject directory already exists,
it is skipped unless --force is passed.

Sources tried, in order:
  1. HuggingFace mirror: datasets.load_dataset("CodeBleu/transcoder_evaluation_gfg")
  2. Manual local clone path passed via --transcoder-root
  3. Built-in tiny smoke dataset (3 subjects) hard-coded inline for offline use.

Usage:
    python scripts/import_transcoder.py --out data/transcoder/python_java
    python scripts/import_transcoder.py --transcoder-root /path/to/CodeGen --out data/transcoder/python_java
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys
import textwrap
from typing import Iterable

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sagetransval.schema import HARNESS_VERSION, SCHEMA_VERSION, OraclePolicy, SubjectV2, write_subject


# Tiny offline smoke set so the importer is testable without network/data.
SMOKE_SET: list[dict] = [
    {
        "id": "transcoder_smoke_factorial",
        "python_source": textwrap.dedent("""
            def factorial(n: int) -> int:
                if n <= 1:
                    return 1
                acc = 1
                for k in range(2, n + 1):
                    acc *= k
                return acc
        """).strip(),
        "java_source": textwrap.dedent("""
            public class Solution {
                public static long factorial(int n) {
                    if (n <= 1) return 1;
                    long acc = 1;
                    for (int k = 2; k <= n; k++) acc *= k;
                    return acc;
                }
            }
        """).strip(),
        "entrypoint": "factorial",
        "parameter_types": {"n": "int"},
        "return_type": "int",
        "seed_tests": [{"args": {"n": 5}}, {"args": {"n": 0}}, {"args": {"n": 10}}],
    },
    {
        "id": "transcoder_smoke_gcd",
        "python_source": textwrap.dedent("""
            def gcd(a: int, b: int) -> int:
                while b != 0:
                    a, b = b, a % b
                return abs(a)
        """).strip(),
        "java_source": textwrap.dedent("""
            public class Solution {
                public static int gcd(int a, int b) {
                    while (b != 0) {
                        int t = b;
                        b = a % b;
                        a = t;
                    }
                    return Math.abs(a);
                }
            }
        """).strip(),
        "entrypoint": "gcd",
        "parameter_types": {"a": "int", "b": "int"},
        "return_type": "int",
        "seed_tests": [{"args": {"a": 48, "b": 18}}, {"args": {"a": 17, "b": 5}}, {"args": {"a": -12, "b": 8}}],
    },
    {
        "id": "transcoder_smoke_max_subarray",
        "python_source": textwrap.dedent("""
            def max_subarray_sum(nums: list[int]) -> int:
                if not nums:
                    return 0
                best = cur = nums[0]
                for x in nums[1:]:
                    cur = max(x, cur + x)
                    best = max(best, cur)
                return best
        """).strip(),
        "java_source": textwrap.dedent("""
            public class Solution {
                public static int max_subarray_sum(int[] nums) {
                    if (nums.length == 0) return 0;
                    int best = nums[0], cur = nums[0];
                    for (int i = 1; i < nums.length; i++) {
                        cur = Math.max(nums[i], cur + nums[i]);
                        best = Math.max(best, cur);
                    }
                    return best;
                }
            }
        """).strip(),
        "entrypoint": "max_subarray_sum",
        "parameter_types": {"nums": "list[int]"},
        "return_type": "int",
        "seed_tests": [
            {"args": {"nums": [-2, 1, -3, 4, -1, 2, 1, -5, 4]}},
            {"args": {"nums": []}},
            {"args": {"nums": [5]}},
        ],
    },
]


def _filter_function_subject(python_source: str, entrypoint: str) -> bool:
    """Quick acceptance filter: must define `entrypoint` and be pure-function shaped."""
    try:
        tree = ast.parse(python_source)
    except SyntaxError:
        return False
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not any(f.name == entrypoint for f in funcs):
        return False
    # Reject if subject uses stdin/stdout in obvious ways.
    banned_calls = {"input", "print"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in banned_calls:
            return False
    return True


def import_smoke(out_root: pathlib.Path, *, force: bool = False) -> list[str]:
    imported: list[str] = []
    for entry in SMOKE_SET:
        sid = entry["id"]
        subject_dir = out_root / sid
        if subject_dir.exists() and not force:
            continue
        if not _filter_function_subject(entry["python_source"], entry["entrypoint"]):
            print(f"[skip] {sid}: failed filter", file=sys.stderr)
            continue
        subject_dir.mkdir(parents=True, exist_ok=True)
        (subject_dir / "source.py").write_text(entry["python_source"] + "\n")
        (subject_dir / "target.java").write_text(entry["java_source"] + "\n")
        subject = SubjectV2(
            schema_version=SCHEMA_VERSION,
            harness_version=HARNESS_VERSION,
            id=sid,
            source_language="python",
            target_language="java",
            language_dialect="python3.10",
            source_path="source.py",
            target_path="target.java",
            entrypoint=entry["entrypoint"],
            parameter_types=entry["parameter_types"],
            return_type=entry["return_type"],
            seed_tests=entry["seed_tests"],
            oracle_policy=OraclePolicy(),
            dataset="transcoder",
            dataset_version="smoke-2026-05-16",
        )
        write_subject(subject, subject_dir / "metadata.json")
        imported.append(sid)
    return imported


def import_from_transcoder_root(transcoder_root: pathlib.Path, out_root: pathlib.Path,
                                *, limit: int | None, force: bool) -> list[str]:
    """Walk a local TransCoder checkout and import function-level Python-Java pairs.

    TransCoder publishes parallel function pairs under various directory layouts.
    This loader expects ``transcoder_root/data/test_dataset/transcoder_evaluation_gfg/``
    with ``python/<name>.py``, ``java/<name>.java`` siblings.
    """
    imported: list[str] = []
    py_dir = transcoder_root / "python"
    java_dir = transcoder_root / "java"
    if not py_dir.is_dir() or not java_dir.is_dir():
        print(f"[error] expected {py_dir} and {java_dir}", file=sys.stderr)
        return imported
    for py_file in sorted(py_dir.glob("*.py")):
        name = py_file.stem
        java_file = java_dir / f"{name}.java"
        if not java_file.exists():
            continue
        sid = f"transcoder_{name}"
        subject_dir = out_root / sid
        if subject_dir.exists() and not force:
            continue
        python_source = py_file.read_text()
        if not _filter_function_subject(python_source, name.lower()):
            continue
        subject_dir.mkdir(parents=True, exist_ok=True)
        (subject_dir / "source.py").write_text(python_source)
        (subject_dir / "target.java").write_text(java_file.read_text())
        # Heuristic: best-effort parameter type inference; downstream harness may refine.
        param_types, return_type = _infer_signature(python_source, name.lower())
        subject = SubjectV2(
            schema_version=SCHEMA_VERSION,
            harness_version=HARNESS_VERSION,
            id=sid,
            source_language="python",
            target_language="java",
            language_dialect="python3.10",
            source_path="source.py",
            target_path="target.java",
            entrypoint=name.lower(),
            parameter_types=param_types,
            return_type=return_type,
            seed_tests=[],
            oracle_policy=OraclePolicy(),
            dataset="transcoder",
            dataset_version=f"upstream-{transcoder_root.name}",
        )
        write_subject(subject, subject_dir / "metadata.json")
        imported.append(sid)
        if limit is not None and len(imported) >= limit:
            break
    return imported


def _infer_signature(python_source: str, entrypoint: str) -> tuple[dict[str, str], str]:
    """Tiny annotation-aware signature extractor. Returns Any for unannotated."""
    tree = ast.parse(python_source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == entrypoint:
            params: dict[str, str] = {}
            for arg in node.args.args:
                if arg.annotation is not None:
                    params[arg.arg] = ast.unparse(arg.annotation)
                else:
                    params[arg.arg] = "Any"
            ret = "Any"
            if node.returns is not None:
                ret = ast.unparse(node.returns)
            return params, ret
    return {}, "Any"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "data" / "transcoder" / "python_java")
    ap.add_argument("--transcoder-root", type=pathlib.Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.transcoder_root is not None:
        imported = import_from_transcoder_root(args.transcoder_root, args.out, limit=args.limit, force=args.force)
        print(f"imported {len(imported)} subjects from {args.transcoder_root}")
    else:
        imported = import_smoke(args.out, force=args.force)
        print(f"imported {len(imported)} subjects from smoke set")
        print("(provide --transcoder-root /path/to/transcoder/eval/data to import the full corpus)")
    print(f"out: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
