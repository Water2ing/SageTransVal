"""Hybrid 100-subject C-to-Rust benchmark for Proposal 1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import TranslationSubjectRecord


def prepare_c_rust_100(out: Path, limit: int | None = None) -> tuple[list[TranslationSubjectRecord], dict[str, Any]]:
    """Create a deterministic hybrid C-to-Rust benchmark.

    The first slice is made of normalized exercise-style functions inspired by
    permissively licensed C practice material. The second slice is diagnostic
    and stresses translation hazards that are easy to miss in small curated
    corpora.
    """

    out.mkdir(parents=True, exist_ok=True)
    specs = _exercise_derived_specs()[:60] + _diagnostic_specs()[:40]
    target = min(limit or 100, 100)
    records = [_write_subject(out, spec) for spec in specs[:target]]
    report = {
        "dataset": "c_rust_100",
        "imported": len(records),
        "real_imported": sum(1 for item in records if item.metadata.get("profile") == "exercise_derived"),
        "diagnostic_generated": sum(1 for item in records if item.metadata.get("profile") == "diagnostic"),
        "skipped": [],
        "source_note": "Exercise-derived subjects are normalized function-level variants inspired by MIT-licensed Exercism C practice tasks; diagnostic subjects are generated templates.",
    }
    return records, report


def _write_subject(out: Path, spec: dict[str, Any]) -> TranslationSubjectRecord:
    subject_dir = out / spec["id"]
    subject_dir.mkdir(parents=True, exist_ok=True)
    source_path = subject_dir / "source.c"
    source_path.write_text(spec["source"], encoding="utf-8")
    record = TranslationSubjectRecord(
        id=spec["id"],
        source_language="c",
        target_language="rust",
        source_path=str(source_path.resolve()),
        target_path=None,
        entrypoint="solve",
        seed_tests=[{"args": spec["args"]}],
        metadata=spec["metadata"],
    )
    (subject_dir / "subject.json").write_text(json.dumps(record.to_json(), indent=2, sort_keys=True), encoding="utf-8")
    return record


def _metadata(name: str, profile: str, signature: dict[str, Any], source_url: str | None = None) -> dict[str, Any]:
    data = {
        "dataset": "c_rust_100",
        "profile": profile,
        "original_id": name,
        "license": "MIT" if profile == "exercise_derived" else "generated",
        "signature": signature,
    }
    if source_url:
        data["source_url"] = source_url
    return data


def _spec(
    name: str,
    source: str,
    args: list[Any],
    profile: str,
    ret: str = "int",
    params: list[str] | None = None,
    source_url: str | None = None,
) -> dict[str, Any]:
    params = params or ["int" for _ in args]
    return {
        "id": f"c_rust_100_{name}",
        "source": source.strip() + "\n",
        "args": args,
        "metadata": _metadata(name, profile, {"return": ret, "params": params}, source_url),
    }


def _exercise_derived_specs() -> list[dict[str, Any]]:
    src = "https://github.com/exercism/c"
    specs: list[dict[str, Any]] = []
    years = [1900, 1996, 2000, 2019, 2024, 2100]
    for idx, seed in enumerate(years):
        specs.append(
            _spec(
                f"exercism_leap_{idx}",
                "#include <stdbool.h>\nbool solve(int year) { return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0); }",
                [seed],
                "exercise_derived",
                ret="bool",
                source_url=src,
            )
        )
    ns = [3, 5, 7, 10, 12, 20]
    for idx, n in enumerate(ns):
        specs.append(
            _spec(
                f"exercism_difference_squares_{idx}",
                "int solve(int n) { int sum = 0; int squares = 0; for (int i = 1; i <= n; i++) { sum += i; squares += i * i; } return sum * sum - squares; }",
                [n],
                "exercise_derived",
                source_url=src,
            )
        )
    for idx, square in enumerate([1, 2, 5, 10, 16, 30]):
        specs.append(
            _spec(
                f"exercism_grains_{idx}",
                "int solve(int square) { if (square <= 0 || square > 30) return 0; return 1 << (square - 1); }",
                [square],
                "exercise_derived",
                source_url=src,
            )
        )
    for idx, n in enumerate([1, 2, 3, 6, 11, 19]):
        specs.append(
            _spec(
                f"exercism_collatz_{idx}",
                "int solve(int n) { if (n <= 0) return -1; int steps = 0; while (n != 1 && steps < 1000) { n = (n % 2 == 0) ? n / 2 : 3 * n + 1; steps++; } return steps; }",
                [n],
                "exercise_derived",
                source_url=src,
            )
        )
    for idx, n in enumerate([0, 5, 9, 153, 370, 9474]):
        specs.append(
            _spec(
                f"exercism_armstrong_{idx}",
                "#include <stdbool.h>\nint powi(int b, int e) { int out = 1; for (int i = 0; i < e; i++) out *= b; return out; }\nbool solve(int n) { if (n < 0) return false; int original = n; int tmp = n; int digits = 0; do { digits++; tmp /= 10; } while (tmp); tmp = n; int sum = 0; do { sum += powi(tmp % 10, digits); tmp /= 10; } while (tmp); return sum == original; }",
                [n],
                "exercise_derived",
                ret="bool",
                source_url=src,
            )
        )
    hamming = [("GAGCCT", "CATCGT"), ("AAAA", "AAAT"), ("", ""), ("abc", "xyz"), ("GGGG", "GGGG"), ("ACGT", "TGCA")]
    for idx, args in enumerate(hamming):
        specs.append(
            _spec(
                f"exercism_hamming_{idx}",
                "#include <string.h>\nint solve(const char *a, const char *b) { int d = 0; int i = 0; while (a[i] && b[i]) { if (a[i] != b[i]) d++; i++; } while (a[i]) { d++; i++; } while (b[i]) { d++; i++; } return d; }",
                list(args),
                "exercise_derived",
                params=["string", "string"],
                source_url=src,
            )
        )
    strings = ["quiz", "cabbage", "hello", "", "Zebra", "Rust"]
    for idx, s in enumerate(strings):
        specs.append(
            _spec(
                f"exercism_scrabble_{idx}",
                "int solve(const char *s) { int score = 0; for (int i = 0; s[i]; i++) { char c = s[i]; if (c >= 'a' && c <= 'z') c -= 32; if (c == 'Q' || c == 'Z') score += 10; else if (c == 'J' || c == 'X') score += 8; else if (c >= 'A' && c <= 'Z') score += 1; } return score; }",
                [s],
                "exercise_derived",
                params=["string"],
                source_url=src,
            )
        )
    targets = ["A", "C", "G", "T", "A", "G"]
    for idx, target in enumerate(targets):
        specs.append(
            _spec(
                f"exercism_nucleotide_{idx}",
                f"int solve(const char *s) {{ int count = 0; for (int i = 0; s[i]; i++) if (s[i] == '{target}') count++; return count; }}",
                ["ACGTAGGA"],
                "exercise_derived",
                params=["string"],
                source_url=src,
            )
        )
    colors = ["black", "brown", "red", "green", "blue", "white"]
    for idx, color in enumerate(colors):
        specs.append(
            _spec(
                f"exercism_resistor_{idx}",
                '#include <string.h>\nint solve(const char *color) { if (strcmp(color, "black") == 0) return 0; if (strcmp(color, "brown") == 0) return 1; if (strcmp(color, "red") == 0) return 2; if (strcmp(color, "green") == 0) return 5; if (strcmp(color, "blue") == 0) return 6; if (strcmp(color, "white") == 0) return 9; return -1; }',
                [color],
                "exercise_derived",
                params=["string"],
                source_url=src,
            )
        )
    arrays = [[1, 2, 3], [-1, 2, -3], [], [5], [10, -10, 7], [2, 4, 6, 8]]
    for idx, xs in enumerate(arrays):
        specs.append(
            _spec(
                f"exercism_accumulate_{idx}",
                "int solve(const int *xs, int n) { int total = 0; for (int i = 0; i < n; i++) total += xs[i] < 0 ? -xs[i] : xs[i]; return total; }",
                [xs],
                "exercise_derived",
                params=["int_array"],
                source_url=src,
            )
        )
    return specs


def _diagnostic_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for idx, k in enumerate(range(-5, 5)):
        specs.append(
            _spec(
                f"diag_boundary_clamp_{idx}",
                f"int solve(int x) {{ if (x < {k}) return {k}; if (x > {k + 10}) return {k + 10}; return x; }}",
                [k - 1],
                "diagnostic",
            )
        )
    for idx, divisor in enumerate([2, 3, 4, 5, 7, 8, 9, 10, 11, 13]):
        specs.append(
            _spec(
                f"diag_division_mod_{idx}",
                f"int solve(int x) {{ if (x == 0) return 0; return (x / {divisor}) + (x % {divisor}); }}",
                [divisor * 3 + 1],
                "diagnostic",
            )
        )
    for idx, mask in enumerate([1, 2, 3, 4, 7, 8, 15, 16, 31, 63]):
        specs.append(
            _spec(
                f"diag_bitwise_{idx}",
                f"int solve(int x) {{ int y = x ^ {mask}; return (y & {mask}) ? y : -y; }}",
                [mask + 5],
                "diagnostic",
            )
        )
    arrays = [[0], [1, 2, 3], [-3, 0, 3], [4, 4, 4], [], [9, -1], [2, 5, 8], [-5], [1, -1, 1, -1], [7, 11, 13]]
    for idx, xs in enumerate(arrays):
        specs.append(
            _spec(
                f"diag_array_threshold_{idx}",
                f"int solve(const int *xs, int n) {{ int count = 0; for (int i = 0; i < n; i++) if (xs[i] >= {idx % 5}) count++; return count; }}",
                [xs],
                "diagnostic",
                params=["int_array"],
            )
        )
    return specs
