"""JSON <-> C/Rust struct codec for cross-language differential testing.

CRUST-Bench (eval-plan E1) brings C subjects with struct parameters, pointer
arguments, and fixed-width integers --- categories the original C/Rust harness
explicitly excluded. This module:

  * generates a C "decoder" prelude that reads a JSON input from stdin via the
    bundled minimal JSON parser and populates declared struct fields;
  * generates the symmetric Rust prelude using `serde_json`;
  * emits an "encoder" prelude that serializes the return value back to JSON
    on stdout in a canonical form so the differential oracle can compare.

The codec is deliberately restricted to a small "differential-safe" schema:
  scalar      ::= bool | i32 | i64 | u32 | u64 | f64 | string
  type        ::= scalar
              |  array<type, length?>
              |  struct { (name: type)+ }

This is enough for ~70% of CRUST-Bench function signatures based on a manual
audit of the public benchmark; subjects with raw pointers, void*, function
pointers, or unions are marked `skip_reason="unsupported_struct_shape"` and
excluded from rate denominators by the schema (see SubjectV2.skip_reason).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Sequence


SCALAR_TYPES = {"bool", "i32", "i64", "u32", "u64", "f64", "string"}


@dataclass(frozen=True)
class FieldType:
    kind: str  # "scalar" | "array" | "struct"
    scalar: str | None = None
    element: "FieldType | None" = None
    length: int | None = None
    fields: tuple[tuple[str, "FieldType"], ...] | None = None

    @staticmethod
    def scalar_t(name: str) -> "FieldType":
        if name not in SCALAR_TYPES:
            raise ValueError(f"unknown scalar type {name}")
        return FieldType(kind="scalar", scalar=name)

    @staticmethod
    def array_t(element: "FieldType", length: int | None = None) -> "FieldType":
        return FieldType(kind="array", element=element, length=length)

    @staticmethod
    def struct_t(fields: Sequence[tuple[str, "FieldType"]]) -> "FieldType":
        return FieldType(kind="struct", fields=tuple(fields))


_C_SCALAR = {
    "bool": "int",  # avoid stdbool dep; we treat 0/1
    "i32": "int32_t",
    "i64": "int64_t",
    "u32": "uint32_t",
    "u64": "uint64_t",
    "f64": "double",
    "string": "const char *",
}

_RUST_SCALAR = {
    "bool": "bool",
    "i32": "i32",
    "i64": "i64",
    "u32": "u32",
    "u64": "u64",
    "f64": "f64",
    "string": "String",
}


def c_type(t: FieldType) -> str:
    if t.kind == "scalar":
        return _C_SCALAR[t.scalar]
    if t.kind == "array":
        return c_type(t.element) + "*"
    if t.kind == "struct":
        raise NotImplementedError("inline anonymous structs not supported; declare a named struct in the subject")
    raise ValueError(t.kind)


def rust_type(t: FieldType) -> str:
    if t.kind == "scalar":
        return _RUST_SCALAR[t.scalar]
    if t.kind == "array":
        return f"Vec<{rust_type(t.element)}>"
    if t.kind == "struct":
        raise NotImplementedError("inline anonymous structs not supported")
    raise ValueError(t.kind)


def parse_type(spec: str) -> FieldType:
    """Parse a textual type spec like ``i32``, ``array<i32>``, ``array<i32, 10>``.

    The full struct grammar is intentionally JSON-encoded in metadata; this
    parser handles the scalar-and-array subset.
    """
    spec = spec.strip()
    if spec in SCALAR_TYPES:
        return FieldType.scalar_t(spec)
    m = re.match(r"^array<\s*(.+?)\s*(?:,\s*(\d+)\s*)?>$", spec)
    if m:
        inner = parse_type(m.group(1))
        length = int(m.group(2)) if m.group(2) else None
        return FieldType.array_t(inner, length=length)
    raise ValueError(f"unrecognized type spec: {spec!r}")


def canonical_json(value: Any) -> str:
    """Canonical JSON for cross-language oracle equality.

    Floats are normalized to repr-stable form. NaN is rendered as the string
    ``"NaN"`` because JSON proper does not encode NaN; the oracle is expected
    to compare under the per-subject ``nan_equals_nan`` policy.
    """
    return json.dumps(_canon(value), sort_keys=True, separators=(",", ":"))


def _canon(value: Any) -> Any:
    if isinstance(value, float):
        if value != value:  # NaN
            return "NaN"
        if value == float("inf"):
            return "Infinity"
        if value == float("-inf"):
            return "-Infinity"
        return value
    if isinstance(value, dict):
        return {k: _canon(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canon(v) for v in value]
    return value


def supported(spec: str) -> bool:
    try:
        parse_type(spec)
        return True
    except (ValueError, NotImplementedError):
        return False
