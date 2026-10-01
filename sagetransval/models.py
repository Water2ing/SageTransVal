"""Shared schemas for cross-language translation validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


Case = dict[str, Any]


@dataclass(frozen=True)
class TranslationSubjectRecord:
    """File-backed subject used by the full Proposal 1 pipeline."""

    id: str
    source_language: str
    target_language: str
    source_path: str
    target_path: str | None
    entrypoint: str
    seed_tests: list[Case]
    metadata: dict[str, Any] = field(default_factory=dict)
    oracle_policy: str = "differential"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any], base: Path | None = None) -> "TranslationSubjectRecord":
        normalized = dict(data)
        if base is not None:
            for key in ("source_path", "target_path"):
                value = normalized.get(key)
                if value:
                    normalized[key] = resolve_portable_path(str(value), base)
        return cls(**normalized)


@dataclass(frozen=True)
class Observation:
    kind: str
    value: Any = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    timeout: bool = False
    exit_code: int | None = None

    def comparable(self) -> tuple[str, str]:
        return (self.kind, normalize_value(self.value))

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def normalize_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float, str)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(normalize_value(item) for item in value) + "]"
    return str(value)


def resolve_portable_path(value: str, base: Path) -> str:
    """Resolve relative paths and remap stale absolute artifact paths."""
    value = value.replace(chr(92), "/")
    path = Path(value)
    if path.exists():
        return str(path.resolve())

    remapped = _remap_known_artifact_path(path, base)
    if remapped is not None:
        return str(remapped)

    if not path.is_absolute():
        candidate = (base / value).resolve()
        if candidate.exists():
            return str(candidate)
        return str(candidate)
    return value


def _remap_known_artifact_path(path: Path, base: Path) -> Path | None:
    markers = ("artifact", "data", "runs", "results", "paper_tables", "configs")
    parts = path.parts
    search_roots = [base.resolve(), *base.resolve().parents]
    for marker in markers:
        indexes = [idx for idx, part in enumerate(parts) if part == marker]
        for idx in reversed(indexes):
            suffix_parts = parts[idx + 1 :] if marker == "artifact" else parts[idx:]
            if not suffix_parts:
                continue
            suffix = Path(*suffix_parts)
            for root in search_roots:
                candidate = root / suffix
                if candidate.exists():
                    return candidate.resolve()
    return None
