"""Baseline and learned planner adapters."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .actions import Action, MIN_COSTS, masked_argmax


class Planner:
    name = "planner"

    def select(self, state: list[float], mask: list[bool]) -> int:
        raise NotImplementedError


class ExistingTestsPlanner(Planner):
    name = "existing"

    def select(self, state: list[float], mask: list[bool]) -> int:
        return int(Action.EXPAND_COUNTEREXAMPLE) if mask[int(Action.EXPAND_COUNTEREXAMPLE)] else int(Action.STOP)


@dataclass
class FixedSchedulePlanner(Planner):
    name = "fixed"
    cursor: int = 0
    schedule: tuple[Action, ...] = (
        Action.LLM_SYNTHESIZE_INSTRUCTIONS,
        Action.GENERATE_BOUNDARY_TESTS,
        Action.PROPERTY_BASED_FUZZ,
        Action.LLM_GENERATE_ASSERTIONS,
        Action.STOP,
    )

    def select(self, state: list[float], mask: list[bool]) -> int:
        while self.cursor < len(self.schedule):
            action = self.schedule[self.cursor]
            self.cursor += 1
            if mask[int(action)]:
                return int(action)
        return int(Action.STOP)


class RandomPlanner(Planner):
    name = "random"

    def select(self, state: list[float], mask: list[bool]) -> int:
        valid = [idx for idx, ok in enumerate(mask) if ok]
        return random.choice(valid) if valid else int(Action.STOP)


class GreedyPlanner(Planner):
    name = "greedy"

    def select(self, state: list[float], mask: list[bool]) -> int:
        scores: list[float] = []
        for action in Action:
            cost = sum(MIN_COSTS[action].values()) or 0.1
            if action == Action.STOP:
                scores.append(0.0)
            elif action in {Action.GENERATE_BOUNDARY_TESTS, Action.PROPERTY_BASED_FUZZ}:
                scores.append(1.0 / cost)
            elif action.name.startswith("LLM"):
                scores.append(0.5 / cost)
            else:
                scores.append(0.25 / cost)
        return masked_argmax(scores, mask)


class ContextualBanditPlanner(Planner):
    name = "bandit"

    def __init__(self):
        self.weights = [0.0 for _ in Action]

    def select(self, state: list[float], mask: list[bool]) -> int:
        return masked_argmax(self.weights, mask)

    def update(self, action: int, reward: float) -> None:
        self.weights[action] = 0.9 * self.weights[action] + 0.1 * reward


def planner_from_name(name: str) -> Planner:
    if name == "existing":
        return ExistingTestsPlanner()
    if name == "fixed":
        return FixedSchedulePlanner()
    if name == "random":
        return RandomPlanner()
    if name == "greedy":
        return GreedyPlanner()
    if name == "bandit":
        return ContextualBanditPlanner()
    if name == "dqn":
        from .dqn import DQNPlanner

        return DQNPlanner()
    raise ValueError(f"unknown planner {name!r}")

