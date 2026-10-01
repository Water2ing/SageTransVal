"""State feature extraction."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import Budget, ExecutionResult


STATE_KEYS = [
    "coverage",
    "uncovered_obligations",
    "failures_found",
    "flaky_tests",
    "invalid_tests",
    "uncertain_oracles",
    "last_action",
    "recent_reward",
    "recent_cost",
    "remaining_tokens",
    "remaining_runtime",
    "remaining_fuzz",
    "remaining_horizon",
    "schema_valid",
    "duplicate_objectives",
    "refinement_count",
    "counterexample_exists",
    "repair_events",
    "generated_tests",
    "duplicate_tests",
    "budget_exhausted",
    "steps",
    "residual_risk",
    "stop_selected",
]


@dataclass
class EvidenceState:
    coverage: float = 0.0
    uncovered_obligations: float = 1.0
    failures_found: float = 0.0
    flaky_tests: float = 0.0
    invalid_tests: float = 0.0
    uncertain_oracles: float = 0.0
    last_action: float = -1.0
    recent_reward: float = 0.0
    recent_cost: float = 0.0
    schema_valid: float = 0.0
    duplicate_objectives: float = 0.0
    refinement_count: float = 0.0
    counterexample_exists: float = 0.0
    repair_events: float = 0.0
    generated_tests: float = 0.0
    duplicate_tests: float = 0.0
    residual_risk: float = 1.0
    stop_selected: float = 0.0
    steps: int = 0
    history: list[dict[str, float]] = field(default_factory=list)

    def features(self, budget: Budget) -> dict[str, float]:
        return {
            "coverage": self.coverage,
            "uncovered_obligations": self.uncovered_obligations,
            "failures_found": self.failures_found,
            "flaky_tests": self.flaky_tests,
            "invalid_tests": self.invalid_tests,
            "uncertain_oracles": self.uncertain_oracles,
            "last_action": self.last_action,
            "recent_reward": self.recent_reward,
            "recent_cost": self.recent_cost,
            "remaining_tokens": budget.tokens,
            "remaining_runtime": budget.runtime_seconds,
            "remaining_fuzz": budget.fuzz_iterations,
            "remaining_horizon": max(0.0, budget.horizon - self.steps),
            "schema_valid": self.schema_valid,
            "duplicate_objectives": self.duplicate_objectives,
            "refinement_count": self.refinement_count,
            "counterexample_exists": self.counterexample_exists,
            "repair_events": self.repair_events,
            "generated_tests": self.generated_tests,
            "duplicate_tests": self.duplicate_tests,
            "budget_exhausted": 1.0 if budget.exhausted() else 0.0,
            "steps": float(self.steps),
            "residual_risk": self.residual_risk,
            "stop_selected": self.stop_selected,
        }

    def vector(self, budget: Budget) -> list[float]:
        raw = self.features(budget)
        scales = {
            "remaining_tokens": 1000.0,
            "remaining_runtime": 60.0,
            "remaining_fuzz": 100.0,
            "remaining_horizon": max(float(budget.horizon), 1.0),
            "steps": max(float(budget.horizon), 1.0),
            "last_action": 10.0,
            "recent_cost": 1000.0,
        }
        out: list[float] = []
        for key in STATE_KEYS:
            value = raw[key]
            value = value / scales.get(key, 1.0)
            out.append(max(-1.0, min(1.0, float(value))))
        return out

    def apply(self, action_id: int, result: ExecutionResult, reward: float) -> None:
        self.steps += 1
        self.last_action = float(action_id)
        self.recent_reward = reward
        self.recent_cost = sum(result.cost.values())
        self.coverage = max(self.coverage, min(1.0, self.coverage + result.coverage_delta))
        self.uncovered_obligations = max(0.0, 1.0 - self.coverage)
        self.failures_found += result.failures
        self.invalid_tests += result.invalid_tests
        self.flaky_tests += result.flaky_tests
        self.duplicate_tests += result.duplicate_tests
        self.generated_tests += len(result.generated_tests)
        self.uncertain_oracles = max(0.0, self.uncertain_oracles - result.uncertainty_delta)
        self.residual_risk = max(0.0, self.residual_risk - result.residual_risk_delta)
        self.counterexample_exists = 1.0 if self.failures_found > 0 else 0.0
        if result.observations.get("schema_valid"):
            self.schema_valid = 1.0
        if "duplicate_objectives" in result.observations:
            self.duplicate_objectives += float(result.observations["duplicate_objectives"])
        if result.observations.get("refined"):
            self.refinement_count += 1.0
        if result.observations.get("repair_event"):
            self.repair_events += 1.0
        self.history.append({"action": float(action_id), "reward": reward, "cost": self.recent_cost})

