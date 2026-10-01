"""Action space and validity masking."""

from __future__ import annotations

from enum import IntEnum
from typing import Mapping


class Action(IntEnum):
    GENERATE_BOUNDARY_TESTS = 0
    PROPERTY_BASED_FUZZ = 1
    LLM_SYNTHESIZE_INSTRUCTIONS = 2
    LLM_GENERATE_ASSERTIONS = 3
    REFINE_UNCERTAIN_ORACLE = 4
    EXPAND_COUNTEREXAMPLE = 5
    RETEST_REPAIR_CONTEXT = 6
    STOP = 7

    @classmethod
    def names(cls) -> list[str]:
        return [a.name for a in cls]


MIN_COSTS: dict[Action, dict[str, float]] = {
    Action.GENERATE_BOUNDARY_TESTS: {"runtime_seconds": 0.1},
    Action.PROPERTY_BASED_FUZZ: {"runtime_seconds": 0.5, "fuzz_iterations": 5},
    Action.LLM_SYNTHESIZE_INSTRUCTIONS: {"tokens": 200},
    Action.LLM_GENERATE_ASSERTIONS: {"tokens": 150},
    Action.REFINE_UNCERTAIN_ORACLE: {"tokens": 100},
    Action.EXPAND_COUNTEREXAMPLE: {"runtime_seconds": 0.2, "fuzz_iterations": 3},
    Action.RETEST_REPAIR_CONTEXT: {"runtime_seconds": 0.2},
    Action.STOP: {},
}


def action_mask(features: Mapping[str, float], remaining: Mapping[str, float]) -> list[bool]:
    """Return a valid-action mask for the canonical action order."""

    mask: list[bool] = []
    for action in Action:
        valid = True
        for key, needed in MIN_COSTS[action].items():
            valid = valid and remaining.get(key, 0.0) >= needed
        if action in {Action.LLM_SYNTHESIZE_INSTRUCTIONS, Action.LLM_GENERATE_ASSERTIONS, Action.REFINE_UNCERTAIN_ORACLE}:
            valid = valid and remaining.get("tokens", 0.0) > 0
        if action == Action.EXPAND_COUNTEREXAMPLE:
            valid = valid and features.get("failures_found", 0.0) > 0
        if action == Action.RETEST_REPAIR_CONTEXT:
            valid = valid and features.get("repair_events", 0.0) > 0
        if action == Action.REFINE_UNCERTAIN_ORACLE:
            valid = valid and features.get("uncertain_oracles", 0.0) > 0
        if action == Action.STOP:
            valid = True
        mask.append(valid)
    return mask


def masked_argmax(values: list[float], mask: list[bool]) -> int:
    """Argmax over valid actions, defaulting to STOP when all non-stop actions are invalid."""

    if len(values) != len(Action) or len(mask) != len(Action):
        raise ValueError("values and mask must match the action space")
    best_idx = int(Action.STOP)
    best_val = float("-inf")
    for idx, (value, valid) in enumerate(zip(values, mask)):
        if valid and value > best_val:
            best_idx, best_val = idx, value
    return best_idx

