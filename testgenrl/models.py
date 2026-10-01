"""Shared dataclasses for TestGenRL."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ProgramSubject:
    id: str
    language: str
    path: Path
    entrypoint: str
    tests: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: dict[str, Any], base: Path) -> "ProgramSubject":
        path = Path(data["path"])
        if not path.is_absolute():
            path = base / path
        return cls(
            id=data["id"],
            language=data.get("language", "python"),
            path=path,
            entrypoint=data["entrypoint"],
            tests=list(data.get("tests", [])),
            metadata=dict(data.get("metadata", {})),
        )

    def to_json(self, base: Path | None = None) -> dict[str, Any]:
        data = asdict(self)
        path = self.path
        if base is not None:
            try:
                path = path.relative_to(base)
            except ValueError:
                pass
        data["path"] = str(path)
        return data


@dataclass(slots=True)
class Budget:
    tokens: float = 1000
    runtime_seconds: float = 30
    fuzz_iterations: float = 100
    solver_seconds: float = 0
    horizon: int = 8

    def remaining_dict(self) -> dict[str, float]:
        return {
            "tokens": self.tokens,
            "runtime_seconds": self.runtime_seconds,
            "fuzz_iterations": self.fuzz_iterations,
            "solver_seconds": self.solver_seconds,
        }

    def spend(self, cost: dict[str, float]) -> None:
        self.tokens = max(0.0, self.tokens - cost.get("tokens", 0.0))
        self.runtime_seconds = max(0.0, self.runtime_seconds - cost.get("runtime_seconds", 0.0))
        self.fuzz_iterations = max(0.0, self.fuzz_iterations - cost.get("fuzz_iterations", 0.0))
        self.solver_seconds = max(0.0, self.solver_seconds - cost.get("solver_seconds", 0.0))

    def exhausted(self) -> bool:
        return self.tokens <= 0 and self.runtime_seconds <= 0 and self.fuzz_iterations <= 0


@dataclass(slots=True)
class ExecutionResult:
    action: str
    generated_tests: list[dict[str, Any]] = field(default_factory=list)
    failures: int = 0
    invalid_tests: int = 0
    flaky_tests: int = 0
    duplicate_tests: int = 0
    coverage_delta: float = 0.0
    uncertainty_delta: float = 0.0
    residual_risk_delta: float = 0.0
    repair_overfit_detected: bool = False
    cost: dict[str, float] = field(default_factory=dict)
    observations: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Transition:
    state: list[float]
    action: int
    reward: float
    next_state: list[float]
    mask: list[bool]
    next_mask: list[bool]
    cost: dict[str, float]
    done: bool

