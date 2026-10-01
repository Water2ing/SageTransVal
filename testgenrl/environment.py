"""MDP episode runner."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .actions import Action, action_mask
from .dqn import ReplayBuffer
from .executors import ExecutorSuite
from .logging import JsonlLogger
from .models import Budget, ProgramSubject, Transition
from .planners import ContextualBanditPlanner, Planner
from .reward import RewardConfig, compute_reward
from .state import EvidenceState


def _hash_state(vector: list[float]) -> str:
    return hashlib.sha256(json.dumps(vector).encode("utf-8")).hexdigest()[:16]


def _log_metadata(subject: ProgramSubject) -> dict:
    return {key: value for key, value in subject.metadata.items() if key != "source_code"}


def _test_key(test: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(test.get("args", []), sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _action_family(action: Action) -> str:
    if action == Action.GENERATE_BOUNDARY_TESTS:
        return "boundary"
    if action == Action.PROPERTY_BASED_FUZZ:
        return "fuzz"
    if action == Action.LLM_SYNTHESIZE_INSTRUCTIONS:
        return "llm_instruction"
    if action in {Action.LLM_GENERATE_ASSERTIONS, Action.REFINE_UNCERTAIN_ORACLE}:
        return "llm_oracle"
    if action == Action.EXPAND_COUNTEREXAMPLE:
        return "counterexample"
    if action == Action.RETEST_REPAIR_CONTEXT:
        return "repair"
    return "stop"


class TestGenEnvironment:
    def __init__(
        self,
        cache_dir: Path,
        reward_config: RewardConfig | None = None,
        llm_provider: str = "stub",
        llm_model: str = "stub",
        cache_mode: str = "live",
        seed: int | None = None,
    ):
        self.executors = ExecutorSuite(cache_dir, llm_provider=llm_provider, llm_model=llm_model, cache_mode=cache_mode)
        self.reward_config = reward_config or RewardConfig()
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.cache_mode = cache_mode
        self.seed = seed

    def run_episode(
        self,
        subject: ProgramSubject,
        planner: Planner,
        budget: Budget,
        out_dir: Path,
    ) -> ReplayBuffer:
        out_dir.mkdir(parents=True, exist_ok=True)
        logger = JsonlLogger(out_dir / "events.jsonl")
        evidence = EvidenceState()
        replay = ReplayBuffer()
        stress_profile = subject.metadata.get("profile") in {"stress", "action_balanced"}
        seen_tests: set[str] = set()
        saturated_families: set[str] = set()

        for step in range(budget.horizon):
            state_vec = evidence.vector(budget)
            features = evidence.features(budget)
            mask = action_mask(features, budget.remaining_dict())
            action_id = planner.select(state_vec, mask)
            action = Action(action_id)
            done = action == Action.STOP

            if done:
                result = self.executors.execute(action, subject)
                reward, components = compute_reward(result, self.reward_config)
                evidence.stop_selected = 1.0
            else:
                result = self.executors.execute(action, subject)
                if stress_profile:
                    generated = result.generated_tests
                    if generated:
                        new_tests = 0
                        repeated = 0
                        for test in generated:
                            key = _test_key(test)
                            if key in seen_tests:
                                repeated += 1
                            else:
                                seen_tests.add(key)
                                new_tests += 1
                        result.duplicate_tests += repeated
                        novelty = new_tests / max(len(generated), 1)
                        result.coverage_delta *= novelty
                        result.residual_risk_delta *= novelty
                    family = _action_family(action)
                    if family in saturated_families:
                        result.coverage_delta = 0.0
                        result.residual_risk_delta = 0.0
                        result.uncertainty_delta = 0.0
                    elif family != "stop":
                        saturated_families.add(family)
                reward, components = compute_reward(result, self.reward_config)
                budget.spend(result.cost)
                evidence.apply(action_id, result, reward)
                if isinstance(planner, ContextualBanditPlanner):
                    planner.update(action_id, reward)
                done = budget.exhausted()

            next_state = evidence.vector(budget)
            next_mask = action_mask(evidence.features(budget), budget.remaining_dict())
            transition = Transition(
                state=state_vec,
                action=action_id,
                reward=reward,
                next_state=next_state,
                mask=mask,
                next_mask=next_mask,
                cost=result.cost,
                done=done,
            )
            replay.add(transition)
            logger.write(
                {
                    "subject_id": subject.id,
                    "step": step,
                    "state_hash": _hash_state(state_vec),
                    "state": state_vec,
                    "mask": mask,
                    "action": action.name,
                    "action_id": action_id,
                    "result": asdict(result),
                    "reward": reward,
                    "reward_components": components,
                    "next_state_hash": _hash_state(next_state),
                    "next_mask": next_mask,
                    "done": done,
                    "budget": asdict(budget),
                    "provider": self.llm_provider,
                    "model": self.llm_model,
                    "cache_mode": self.cache_mode,
                    "seed": self.seed,
                    "subject_metadata": _log_metadata(subject),
                }
            )
            if done:
                break
        replay.save(out_dir / "replay.json")
        return replay
