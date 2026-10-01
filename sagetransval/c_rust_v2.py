"""V2 C-to-Rust benchmark: 30 subjects authored to expose translation divergences.

Replaces the diagnostic-half of ``c_rust_100`` for the v2 evaluation. Subjects
are grouped into three categories, all expressible inside the existing harness
(int / bool / string / int_array parameters; int / bool / string return):

* A (integer semantics, 11 subjects): exercises signed-overflow,
  shift-by-width, ``INT_MIN / -1`` and related C-UB-vs-Rust-defined seams.
* C (error handling, 10 subjects): functions whose C convention is a sentinel
  return (e.g. ``-1`` on bad input); the natural LLM Rust translation tends
  toward ``Option``/``Result`` or to ``panic!``, diverging on the error path.
* D (strings/bytes, 9 subjects): byte- vs codepoint- semantics on UTF-8
  inputs; ``strlen`` vs ``.chars().count()``; ASCII-only ``toupper`` vs
  ``to_ascii_uppercase`` on multi-byte input.

Categories B (memory/ownership), E (structs/layout) and F (floating-point) are
out-of-scope for v2 because they require harness signatures the current
adapters do not support; they remain in the design doc for a future v3 push.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import TranslationSubjectRecord


def prepare_c_rust_v2(out: Path, limit: int | None = None) -> tuple[list[TranslationSubjectRecord], dict[str, Any]]:
    out.mkdir(parents=True, exist_ok=True)
    specs = (
        _integer_specs()
        + _integer_defined_specs()
        + _error_handling_specs()
        + _error_handling_aux_specs()
        + _string_specs()
        + _string_aux_specs()
    )
    target = min(limit or len(specs), len(specs))
    records = [_write_subject(out, spec) for spec in specs[:target]]
    report = {
        "dataset": "c_rust_v2",
        "imported": len(records),
        "by_category": {
            "A_integer": sum(1 for r in records if r.metadata.get("category") == "A_integer"),
            "C_error": sum(1 for r in records if r.metadata.get("category") == "C_error"),
            "D_strings": sum(1 for r in records if r.metadata.get("category") == "D_strings"),
        },
        "skipped": [],
        "source_note": (
            "Authored 2026-05-16 / extended 2026-05-17 for the v2 evaluation. "
            "Each subject is chosen so the idiomatic LLM Rust translation either "
            "(a) panics where the C function returns a value (UB-triggering A and "
            "sentinel-error C), (b) drops a width / signedness annotation and "
            "diverges on the test input (defined-defined A), or (c) operates on "
            "Unicode codepoints where C operates on bytes (D)."
        ),
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
    file_record = TranslationSubjectRecord(
        id=record.id,
        source_language=record.source_language,
        target_language=record.target_language,
        source_path="source.c",
        target_path=None,
        entrypoint=record.entrypoint,
        seed_tests=record.seed_tests,
        metadata=record.metadata,
        oracle_policy=record.oracle_policy,
    )
    (subject_dir / "subject.json").write_text(
        json.dumps(file_record.to_json(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return record


def _spec(
    name: str,
    category: str,
    source: str,
    args: list[Any],
    *,
    ret: str = "int",
    params: list[str] | None = None,
) -> dict[str, Any]:
    params = params or ["int" for _ in args]
    metadata = {
        "dataset": "c_rust_v2",
        "category": category,
        "original_id": name,
        "license": "authored_for_this_paper",
        "signature": {"return": ret, "params": params},
    }
    return {
        "id": f"c_rust_v2_{name}",
        "source": source.strip() + "\n",
        "args": args,
        "metadata": metadata,
    }


# --- Category A: integer semantics ---------------------------------------------

def _integer_specs() -> list[dict[str, Any]]:
    return [
        _spec(
            "A_signed_add_overflow",
            "A_integer",
            # On INT_MAX+1 C is UB; idiomatic Rust panics (overflow-checks on).
            "int solve(int a, int b) { return a + b; }",
            [2147483647, 1],
        ),
        _spec(
            "A_signed_mul_overflow",
            "A_integer",
            "int solve(int a, int b) { return a * b; }",
            [46341, 46341],  # 46341^2 just exceeds INT_MAX
        ),
        _spec(
            "A_shift_at_width",
            "A_integer",
            # Shift by >= width is C UB; Rust panics.
            "int solve(int x, int k) { return x << k; }",
            [1, 32],
        ),
        _spec(
            "A_intmin_div_neg1",
            "A_integer",
            # INT_MIN / -1 is C UB (overflow); Rust panics.
            "int solve(int a, int b) { return a / b; }",
            [-2147483648, -1],
        ),
        _spec(
            "A_neg_div_truncation",
            "A_integer",
            # Both languages truncate toward zero for signed integer division
            # since C99 / Rust; this subject is the no-divergence control so we
            # can attribute v2 detections to genuine seams, not blanket panic.
            "int solve(int a, int b) { if (b == 0) return 0; return a / b; }",
            [-7, 2],
        ),
        _spec(
            "A_abs_intmin",
            "A_integer",
            # -INT_MIN is C UB; idiomatic Rust ``x.abs()`` panics on i32::MIN.
            "int solve(int x) { return x < 0 ? -x : x; }",
            [-2147483648],
        ),
    ]


# --- Category A (defined-vs-defined): integer subjects that are *defined* in
# both C and Rust but where the natural LLM translation tends to drop the
# unsigned cast or width annotation and diverge on the test input. The
# UB-safe equivalence rule does NOT suppress these because the C side is
# defined; any mismatch counts.


def _integer_defined_specs() -> list[dict[str, Any]]:
    return [
        _spec(
            "A_def_unsigned_compare",
            "A_integer",
            # ((unsigned)a > (unsigned)b) on (-1, 1): C reads as UINT_MAX > 1 = 1.
            # LLM that drops the unsigned cast computes -1 > 1 = 0.
            "int solve(int a, int b) { return ((unsigned)a > (unsigned)b) ? 1 : 0; }",
            [-1, 1],
        ),
        _spec(
            "A_def_unsigned_shr",
            "A_integer",
            # ((unsigned)x) >> 1 on -1: C reads as UINT_MAX >> 1 = INT_MAX.
            # LLM keeping i32 arithmetic shift returns -1.
            "int solve(int x) { return (int)(((unsigned)x) >> 1); }",
            [-1],
        ),
        _spec(
            "A_def_unsigned_div",
            "A_integer",
            # (unsigned)a / (unsigned)b on (-1, 3): C reads as UINT_MAX/3 = 1431655765.
            # LLM that drops the cast computes -1/3 = 0.
            "int solve(int a, int b) { return (int)(((unsigned)a) / ((unsigned)b)); }",
            [-1, 3],
        ),
        _spec(
            "A_def_unsigned_mod",
            "A_integer",
            # (unsigned)a % (unsigned)b on (-1, 7): C reads as UINT_MAX % 7 = 3.
            # LLM that drops the cast computes -1 % 7 (Rust: -1, C99: -1).
            "int solve(int a, int b) { return (int)(((unsigned)a) % ((unsigned)b)); }",
            [-1, 7],
        ),
        _spec(
            "A_def_abs_via_long",
            "A_integer",
            # Computes abs through a wider type so INT_MIN does not overflow
            # in either language. Naive LLM that ignores the long widening
            # writes ``x.abs()`` and panics with overflow-checks on i32::MIN.
            "int solve(int x) { long lx = x; long ax = lx < 0 ? -lx : lx; return (int)ax; }",
            [-2147483648],
        ),
    ]


# --- Category C: sentinel error handling ---------------------------------------

def _error_handling_specs() -> list[dict[str, Any]]:
    return [
        _spec(
            "C_sentinel_negative_input",
            "C_error",
            # Returns -1 on invalid; LLM may translate as Option::None and the
            # harness then unwraps.
            "int solve(int n) { if (n < 0) return -1; int s = 0; for (int i = 0; i <= n; i++) s += i; return s; }",
            [-5],
        ),
        _spec(
            "C_sentinel_empty_string",
            "C_error",
            "int solve(const char *s) { if (s == 0 || s[0] == 0) return -1; int c = 0; for (int i = 0; s[i]; i++) c++; return c; }",
            [""],
            params=["string"],
        ),
        _spec(
            "C_sentinel_overflow_guard",
            "C_error",
            # Returns -1 to signal "would overflow"; LLM might just translate
            # the multiplication and panic on overflow.
            "int solve(int a, int b) { if (a > 46340 || b > 46340) return -1; return a * b; }",
            [50000, 50000],
        ),
        _spec(
            "C_sentinel_lookup_miss",
            "C_error",
            '#include <string.h>\nint solve(const char *key) { if (strcmp(key, "alpha") == 0) return 1; if (strcmp(key, "beta") == 0) return 2; if (strcmp(key, "gamma") == 0) return 3; return -1; }',
            ["delta"],
            params=["string"],
        ),
        _spec(
            "C_sentinel_safe_div",
            "C_error",
            # Returns INT_MIN (a sentinel value) on divide-by-zero. LLM may
            # translate as Result::Err and the harness unwrap panics.
            "#include <limits.h>\nint solve(int a, int b) { if (b == 0) return INT_MIN; return a / b; }",
            [10, 0],
        ),
        _spec(
            "C_sentinel_index_out_of_range",
            "C_error",
            "int solve(const int *xs, int n, int idx) { if (idx < 0 || idx >= n) return -1; return xs[idx]; }",
            [[10, 20, 30], 5],
            params=["int_array", "int"],
        ),
        _spec(
            "C_sentinel_match_first",
            "C_error",
            "int solve(const int *xs, int n, int needle) { for (int i = 0; i < n; i++) if (xs[i] == needle) return i; return -1; }",
            [[1, 2, 3], 99],
            params=["int_array", "int"],
        ),
    ]


# --- Category C (auxiliary): subjects whose C surface invites idiomatic
# Option/Result Rust translations more strongly than the original C_sentinel_*
# subjects, raising the LLM's chance of recasting the sentinel as Option::None
# and panicking on the harness-side unwrap.


def _error_handling_aux_specs() -> list[dict[str, Any]]:
    return [
        _spec(
            "C_aux_parse_int",
            "C_error",
            # Parse digit string or return -1. Idiomatic Rust uses
            # ``str::parse::<i32>()`` which returns Result; the harness then
            # unwraps and panics on the non-digit input.
            "int solve(const char *s) { int n = 0; for (int i = 0; s[i]; i++) { if (s[i] < '0' || s[i] > '9') return -1; n = n*10 + (s[i]-'0'); } return n; }",
            ["12a3"],
            params=["string"],
        ),
        _spec(
            "C_aux_find_index_nonzero",
            "C_error",
            # Linear-scan-and-return-index pattern that maps naturally to
            # ``Iterator::position`` (returns Option<usize>).
            "int solve(const int *xs, int n) { for (int i = 0; i < n; i++) if (xs[i] != 0) return i; return -1; }",
            [[0, 0, 0, 0]],
            params=["int_array"],
        ),
        _spec(
            "C_aux_validate_positive",
            "C_error",
            # Validation function returning -1 on invalid input; Rust idiom
            # is Result::Err and an explicit ?-propagation.
            "int solve(int x) { if (x <= 0) return -1; return x * 2; }",
            [-3],
        ),
    ]


# --- Category D: byte vs codepoint string semantics ----------------------------

def _string_specs() -> list[dict[str, Any]]:
    return [
        _spec(
            "D_strlen_utf8",
            "D_strings",
            # strlen counts bytes; .chars().count() counts codepoints.
            "#include <string.h>\nint solve(const char *s) { return (int)strlen(s); }",
            ["héllo"],  # 6 bytes, 5 codepoints
            params=["string"],
        ),
        _spec(
            "D_index_byte",
            "D_strings",
            # Return the i-th byte. LLM may use s.chars().nth(i) which on
            # UTF-8 input returns the i-th codepoint, not byte.
            "int solve(const char *s, int i) { return (int)(unsigned char)s[i]; }",
            ["héllo", 2],
            params=["string", "int"],
        ),
        _spec(
            "D_count_char",
            "D_strings",
            # Count occurrences of an ASCII byte in s. UTF-8 multi-byte
            # codepoints may contain the byte 'l' (0x6c)? Actually 'é' = 0xC3 0xA9,
            # neither of which is 'l'; this is the control where byte and
            # codepoint scans agree.
            "int solve(const char *s, int byte) { int c = 0; for (int i = 0; s[i]; i++) if ((unsigned char)s[i] == (unsigned char)byte) c++; return c; }",
            ["héllo", ord("l")],
            params=["string", "int"],
        ),
        _spec(
            "D_count_non_ascii",
            "D_strings",
            # Count bytes >= 128 (UTF-8 continuation/leading bytes). LLM may
            # translate as .chars().filter(|c| !c.is_ascii()) which counts
            # codepoints, not bytes -- 2 vs 1 on "héllo".
            "int solve(const char *s) { int c = 0; for (int i = 0; s[i]; i++) if ((unsigned char)s[i] >= 128) c++; return c; }",
            ["héllo"],
            params=["string"],
        ),
        _spec(
            "D_reverse_bytes",
            "D_strings",
            # Sum the ASCII values when reading bytes in reverse. LLM may use
            # .chars().rev() and operate on codepoints, producing a different
            # sum on UTF-8 input.
            "int solve(const char *s) { int n = 0; while (s[n]) n++; int sum = 0; for (int i = n - 1; i >= 0; i--) sum += (unsigned char)s[i]; return sum; }",
            ["héllo"],
            params=["string"],
        ),
        _spec(
            "D_toupper_ascii",
            "D_strings",
            # C toupper is locale-dependent and typically leaves UTF-8 bytes
            # alone; the C function here only flips ASCII. LLM may use
            # s.to_uppercase() (full Unicode case folding) which differs.
            "int solve(const char *s) { int c = 0; for (int i = 0; s[i]; i++) if (s[i] >= 'A' && s[i] <= 'Z') c++; return c; }",
            ["Café"],
            params=["string"],
        ),
        _spec(
            "D_embedded_terminator",
            "D_strings",
            # C strings terminate at the first NUL; Rust strings carry an
            # explicit length. Test by counting only up to NUL byte.
            "int solve(const char *s) { int n = 0; while (s[n]) n++; return n; }",
            ["abcdef"],
            params=["string"],
        ),
    ]


# --- Category D (auxiliary): additional Unicode boundary cases ----------------


def _string_aux_specs() -> list[dict[str, Any]]:
    return [
        _spec(
            "D_aux_substring_match",
            "D_strings",
            # Counts byte-level occurrences of a single-byte pattern. On a
            # UTF-8 input where the LLM uses ``str::matches`` or ``contains``
            # over a char, codepoint vs byte semantics diverge on non-ASCII
            # leading-byte collisions.
            "int solve(const char *s, const char *pat) { int c = 0; int plen = 0; while (pat[plen]) plen++; for (int i = 0; s[i]; i++) { int ok = 1; for (int j = 0; j < plen; j++) { if (s[i+j] != pat[j]) { ok = 0; break; } } if (ok) c++; } return c; }",
            ["héllohélloA", "é"],
            params=["string", "string"],
        ),
        _spec(
            "D_aux_count_uppercase_byte",
            "D_strings",
            # Counts bytes in the ASCII uppercase range. LLM that uses
            # ``c.is_uppercase()`` on chars counts codepoints (and includes
            # non-ASCII uppercase like 'Ä'), differing on a UTF-8 input that
            # mixes ASCII uppercase with non-ASCII letters.
            "int solve(const char *s) { int c = 0; for (int i = 0; s[i]; i++) { unsigned char b = (unsigned char)s[i]; if (b >= 'A' && b <= 'Z') c++; } return c; }",
            ["ÄÖÜAB"],
            params=["string"],
        ),
    ]
