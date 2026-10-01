"""Reward function with separately logged components."""

from __future__ import annotations

from dataclasses import dataclass

from .models import ExecutionResult


@dataclass(slots=True)
class RewardConfig:
    alpha_failure: float = 10.0
    beta_coverage: float = 1.0
    gamma_uncertainty: float = 1.0
    eta_risk: float = 1.0
    lambda_cost: float = 1.0
    rho_bad_tests: float = 0.5


def compute_reward(result: ExecutionResult, config: RewardConfig | None = None) -> tuple[float, dict[str, float]]:
    config = config or RewardConfig()
    normalized_cost = (
        result.cost.get("tokens", 0.0) / 1000.0
        + result.cost.get("runtime_seconds", 0.0) / 60.0
        + result.cost.get("fuzz_iterations", 0.0) / 100.0
        + result.cost.get("solver_seconds", 0.0) / 60.0
    )
    bad_tests = result.invalid_tests + result.flaky_tests + result.duplicate_tests
    components = {
        "failure": config.alpha_failure * result.failures,
        "coverage": config.beta_coverage * result.coverage_delta,
        "uncertainty": config.gamma_uncertainty * result.uncertainty_delta,
        "risk": config.eta_risk * result.residual_risk_delta,
        "cost": -config.lambda_cost * normalized_cost,
        "bad_tests": -config.rho_bad_tests * bad_tests,
        "repair_overfit": config.alpha_failure if result.repair_overfit_detected else 0.0,
    }
    return sum(components.values()), components

