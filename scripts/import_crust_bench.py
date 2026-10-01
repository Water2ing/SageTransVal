"""CRUST-Bench import script for SageTransVal eval-plan E1.

Converts CRUST-Bench C subjects into SubjectV2-format Rust translation subjects.
CRUST-Bench upstream layout is roughly::

  CRUST-Bench/
    benchmark/
      <subject_name>/
        c/<file>.c
        rust/<file>.rs        # reference Rust, only used for cross-check
        spec.json             # entry point, parameters

If the upstream schema differs, override via flags. The importer filters subjects
whose parameter shape exceeds the supported codec (see :mod:`sagetransval.c_struct_codec`).

Like the TransCoder importer, this script can run on a built-in offline smoke set
without network or upstream access; that mode is what tests rely on.

Usage:
    python scripts/import_crust_bench.py --out data/crust_bench
    python scripts/import_crust_bench.py --crust-root /path/to/CRUST-Bench/benchmark --out data/crust_bench
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sagetransval.c_struct_codec import supported as type_supported
from sagetransval.schema import HARNESS_VERSION, SCHEMA_VERSION, OraclePolicy, SubjectV2, write_subject


SMOKE_SET = [
    {
        "id": "crust_smoke_overflow_add",
        "c_source": textwrap.dedent("""
            #include <stdint.h>
            int32_t overflow_add(int32_t a, int32_t b) {
                return (int32_t)((int64_t)a + (int64_t)b);
            }
        """).strip(),
        "entrypoint": "overflow_add",
        "parameter_types": {"a": "i32", "b": "i32"},
        "return_type": "i32",
        "seed_tests": [
            {"args": {"a": 2147483647, "b": 1}},  # overflow on naive add
            {"args": {"a": 100, "b": 200}},
            {"args": {"a": -2147483648, "b": -1}},
        ],
    },
    {
        "id": "crust_smoke_array_max",
        "c_source": textwrap.dedent("""
            #include <stdint.h>
            int64_t array_max(int64_t *arr, int32_t n) {
                if (n <= 0) return 0;
                int64_t best = arr[0];
                for (int32_t i = 1; i < n; i++) if (arr[i] > best) best = arr[i];
                return best;
            }
        """).strip(),
        "entrypoint": "array_max",
        "parameter_types": {"arr": "array<i64>", "n": "i32"},
        "return_type": "i64",
        "seed_tests": [
            {"args": {"arr": [3, 1, 4, 1, 5, 9, 2, 6], "n": 8}},
            {"args": {"arr": [], "n": 0}},
            {"args": {"arr": [-1, -5, -2], "n": 3}},
        ],
    },
    {
        "id": "crust_smoke_ownership_concat",
        "c_source": textwrap.dedent("""
            #include <stdint.h>
            #include <string.h>
            #include <stdlib.h>
            const char *concat(const char *a, const char *b) {
                size_t la = strlen(a), lb = strlen(b);
                char *out = malloc(la + lb + 1);
                memcpy(out, a, la);
                memcpy(out + la, b, lb);
                out[la + lb] = '\\0';
                return out;
            }
        """).strip(),
        "entrypoint": "concat",
        "parameter_types": {"a": "string", "b": "string"},
        "return_type": "string",
        "seed_tests": [
            {"args": {"a": "hello, ", "b": "world"}},
            {"args": {"a": "", "b": ""}},
        ],
    },
]


def import_smoke(out_root: pathlib.Path, *, force: bool = False) -> list[str]:
    imported: list[str] = []
    for entry in SMOKE_SET:
        for pname, pt in entry["parameter_types"].items():
            if not type_supported(pt):
                # In a real subject, mark as skipped rather than refuse to import.
                continue
        sid = entry["id"]
        subject_dir = out_root / sid
        if subject_dir.exists() and not force:
            continue
        subject_dir.mkdir(parents=True, exist_ok=True)
        (subject_dir / "source.c").write_text(entry["c_source"] + "\n")
        subject = SubjectV2(
            schema_version=SCHEMA_VERSION,
            harness_version=HARNESS_VERSION,
            id=sid,
            source_language="c",
            target_language="rust",
            language_dialect="c11",
            source_path="source.c",
            target_path=None,  # populated after translation
            entrypoint=entry["entrypoint"],
            parameter_types=entry["parameter_types"],
            return_type=entry["return_type"],
            seed_tests=entry["seed_tests"],
            oracle_policy=OraclePolicy(numeric_tolerance=0.0, nan_equals_nan=True),
            dataset="crust_bench",
            dataset_version="smoke-2026-05-16",
        )
        write_subject(subject, subject_dir / "metadata.json")
        imported.append(sid)
    return imported


def import_from_crust_root(crust_root: pathlib.Path, out_root: pathlib.Path,
                           *, limit: int | None, force: bool) -> list[str]:
    """Import a real CRUST-Bench checkout.

    Expects ``crust_root/<subject>/c/*.c`` and ``crust_root/<subject>/spec.json``.
    Subjects whose ``spec.json`` declares unsupported parameter types are
    written out with ``skip_reason="unsupported_struct_shape"`` instead of
    being silently dropped.
    """
    imported: list[str] = []
    for subject_dir in sorted(crust_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        spec_path = subject_dir / "spec.json"
        if not spec_path.exists():
            continue
        spec = json.loads(spec_path.read_text())
        c_files = list((subject_dir / "c").glob("*.c"))
        if not c_files:
            continue
        sid = f"crust_{subject_dir.name}"
        out_dir = out_root / sid
        if out_dir.exists() and not force:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "source.c").write_text(c_files[0].read_text())
        param_types = dict(spec.get("parameter_types", {}))
        skip = None if all(type_supported(t) for t in param_types.values()) else "unsupported_struct_shape"
        subject = SubjectV2(
            schema_version=SCHEMA_VERSION,
            harness_version=HARNESS_VERSION,
            id=sid,
            source_language="c",
            target_language="rust",
            language_dialect="c11",
            source_path="source.c",
            target_path=None,
            entrypoint=spec["entrypoint"],
            parameter_types=param_types,
            return_type=spec.get("return_type", "Any"),
            seed_tests=spec.get("seed_tests", []),
            oracle_policy=OraclePolicy(),
            dataset="crust_bench",
            dataset_version=f"upstream-{crust_root.parent.name}",
            skip_reason=skip,
        )
        write_subject(subject, out_dir / "metadata.json")
        imported.append(sid)
        if limit is not None and len(imported) >= limit:
            break
    return imported


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "data" / "crust_bench")
    ap.add_argument("--crust-root", type=pathlib.Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.crust_root is not None:
        imported = import_from_crust_root(args.crust_root, args.out, limit=args.limit, force=args.force)
        print(f"imported {len(imported)} subjects from {args.crust_root}")
    else:
        imported = import_smoke(args.out, force=args.force)
        print(f"imported {len(imported)} subjects from smoke set")
        print("(provide --crust-root /path/to/CRUST-Bench/benchmark to import upstream)")
    print(f"out: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
