"""Subject-schema v2.

Versioned, validated subject metadata that survives across extension benchmarks
(TransCoder, CRUST-Bench, AlphaTrans, JS/TS). Backwards-compatible with the
legacy TranslationSubjectRecord via :func:`from_legacy`.

Design points:
  * Every record carries `schema_version` and `harness_version`. These feed into
    the translation cache key so cached translations invalidate automatically when
    the harness or subject layout changes (see also :mod:`sagetransval.cache_key`).
  * Parameter types are explicit (`parameter_types`) so JS, C, and Rust harnesses
    can deserialize JSON inputs into the right runtime types without guessing.
  * `oracle_policy` is structured: numeric tolerance, NaN handling, declared
    unordered collections, exception-class equivalence, normalization rules.
  * `language_dialect` lets a single source_language span multiple dialects
    (e.g., Python 3.10 vs 3.12) without confusing the harness.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .models import resolve_portable_path

SCHEMA_VERSION = 2
HARNESS_VERSION = 2  # Bumped when any executor adapter changes


@dataclass(frozen=True)
class OraclePolicy:
    """Structured oracle normalization rules per subject."""

    numeric_tolerance: float = 0.0
    nan_equals_nan: bool = True
    unordered_collections: tuple[str, ...] = ()  # field names whose order is unconstrained
    exception_class_equivalence: tuple[tuple[str, ...], ...] = ()  # equivalence groups
    string_case_insensitive: bool = False
    ignore_trailing_whitespace: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "OraclePolicy":
        if not data:
            return cls()
        # Coerce list-of-lists to tuple-of-tuples for the exception-equivalence field
        eq = tuple(
            tuple(g) for g in data.get("exception_class_equivalence", [])
        )
        return cls(
            numeric_tolerance=float(data.get("numeric_tolerance", 0.0)),
            nan_equals_nan=bool(data.get("nan_equals_nan", True)),
            unordered_collections=tuple(data.get("unordered_collections", [])),
            exception_class_equivalence=eq,
            string_case_insensitive=bool(data.get("string_case_insensitive", False)),
            ignore_trailing_whitespace=bool(data.get("ignore_trailing_whitespace", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "numeric_tolerance": self.numeric_tolerance,
            "nan_equals_nan": self.nan_equals_nan,
            "unordered_collections": list(self.unordered_collections),
            "exception_class_equivalence": [list(g) for g in self.exception_class_equivalence],
            "string_case_insensitive": self.string_case_insensitive,
            "ignore_trailing_whitespace": self.ignore_trailing_whitespace,
        }


@dataclass(frozen=True)
class SubjectV2:
    """File-backed translation subject in the v2 schema.

    The combination (schema_version, harness_version) is part of every cache key
    that touches this subject; bumping either invalidates downstream caches.
    """

    schema_version: int
    harness_version: int

    id: str
    source_language: str  # "python" | "c" | "java" | "rust" | "javascript" | "typescript" | "kotlin"
    target_language: str
    language_dialect: str | None  # e.g., "python3.10", "rust2021", "es2022"

    source_path: str
    target_path: str | None
    entrypoint: str
    parameter_types: dict[str, str]  # parameter_name -> language-level type string
    return_type: str  # language-level type string of the return value

    seed_tests: list[dict[str, Any]]
    oracle_policy: OraclePolicy

    # Repo-level / multi-file support (E3 / AlphaTrans).
    extra_source_files: tuple[str, ...] = ()
    extra_target_files: tuple[str, ...] = ()

    # Provenance.
    dataset: str = "unknown"
    dataset_version: str | None = None
    skip_reason: str | None = None  # set to mark a subject excluded from rate denominators

    # Free-form metadata that survives round-trip but is not used by the schema.
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "harness_version": self.harness_version,
            "id": self.id,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "language_dialect": self.language_dialect,
            "source_path": self.source_path,
            "target_path": self.target_path,
            "entrypoint": self.entrypoint,
            "parameter_types": dict(self.parameter_types),
            "return_type": self.return_type,
            "seed_tests": list(self.seed_tests),
            "oracle_policy": self.oracle_policy.to_dict(),
            "extra_source_files": list(self.extra_source_files),
            "extra_target_files": list(self.extra_target_files),
            "dataset": self.dataset,
            "dataset_version": self.dataset_version,
            "skip_reason": self.skip_reason,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any], base: Path | None = None) -> "SubjectV2":
        d = dict(data)
        d.setdefault("schema_version", SCHEMA_VERSION)
        d.setdefault("harness_version", HARNESS_VERSION)
        d.setdefault("language_dialect", None)
        d.setdefault("parameter_types", {})
        d.setdefault("return_type", "Any")
        d.setdefault("seed_tests", [])
        d.setdefault("extra_source_files", [])
        d.setdefault("extra_target_files", [])
        d.setdefault("dataset", "unknown")
        d.setdefault("dataset_version", None)
        d.setdefault("skip_reason", None)
        d.setdefault("extra", {})
        d["oracle_policy"] = OraclePolicy.from_dict(d.get("oracle_policy"))
        if base is not None:
            for key in ("source_path", "target_path"):
                val = d.get(key)
                if val:
                    d[key] = resolve_portable_path(str(val), base)
            d["extra_source_files"] = [
                resolve_portable_path(str(p), base)
                for p in d.get("extra_source_files", [])
            ]
            d["extra_target_files"] = [
                resolve_portable_path(str(p), base)
                for p in d.get("extra_target_files", [])
            ]
        return cls(
            schema_version=int(d["schema_version"]),
            harness_version=int(d["harness_version"]),
            id=d["id"],
            source_language=d["source_language"],
            target_language=d["target_language"],
            language_dialect=d.get("language_dialect"),
            source_path=d["source_path"],
            target_path=d.get("target_path"),
            entrypoint=d["entrypoint"],
            parameter_types=dict(d["parameter_types"]),
            return_type=d["return_type"],
            seed_tests=list(d["seed_tests"]),
            oracle_policy=d["oracle_policy"],
            extra_source_files=tuple(d["extra_source_files"]),
            extra_target_files=tuple(d["extra_target_files"]),
            dataset=d.get("dataset", "unknown"),
            dataset_version=d.get("dataset_version"),
            skip_reason=d.get("skip_reason"),
            extra=dict(d.get("extra", {})),
        )

    @classmethod
    def from_legacy(cls, legacy: Any) -> "SubjectV2":
        """Promote a legacy TranslationSubjectRecord to v2."""
        return cls(
            schema_version=SCHEMA_VERSION,
            harness_version=HARNESS_VERSION,
            id=getattr(legacy, "id", "unknown"),
            source_language=getattr(legacy, "source_language", "python"),
            target_language=getattr(legacy, "target_language", "java"),
            language_dialect=None,
            source_path=str(getattr(legacy, "source_path", "")),
            target_path=str(getattr(legacy, "target_path", None)) if getattr(legacy, "target_path", None) else None,
            entrypoint=getattr(legacy, "entrypoint", "solution"),
            parameter_types={},
            return_type="Any",
            seed_tests=list(getattr(legacy, "seed_tests", [])),
            oracle_policy=OraclePolicy(),
            dataset=getattr(getattr(legacy, "metadata", {}) or {}, "get", lambda *_: "legacy")("dataset", "legacy"),
        )


def load_subject(metadata_path: Path) -> SubjectV2:
    with metadata_path.open() as f:
        data = json.load(f)
    return SubjectV2.from_json(data, base=metadata_path.parent)


def write_subject(subject: SubjectV2, metadata_path: Path) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w") as f:
        json.dump(subject.to_json(), f, indent=2, sort_keys=True)
