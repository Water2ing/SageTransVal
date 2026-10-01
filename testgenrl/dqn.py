"""Masked Double-DQN utilities.

The planner can use PyTorch when installed. The pure-Python fallback keeps CLI
and tests runnable in lightweight environments.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .actions import Action, masked_argmax
from .models import Transition
from .planners import Planner


@dataclass
class ReplayBuffer:
    transitions: list[Transition]

    def __init__(self) -> None:
        self.transitions = []

    def add(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.transitions, min(batch_size, len(self.transitions)))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(t) for t in self.transitions], indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ReplayBuffer":
        buf = cls()
        if path.exists():
            for row in json.loads(path.read_text(encoding="utf-8")):
                buf.add(Transition(**row))
        return buf


class DQNPlanner(Planner):
    name = "dqn"

    def __init__(self, q_values: list[float] | None = None, epsilon: float = 0.0):
        self.q_values = q_values or [0.0 for _ in Action]
        self.epsilon = epsilon

    def select(self, state: list[float], mask: list[bool]) -> int:
        valid = [i for i, ok in enumerate(mask) if ok]
        if valid and random.random() < self.epsilon:
            return random.choice(valid)
        return masked_argmax(self.q_values, mask)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"q_values": self.q_values}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DQNPlanner":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(q_values=list(data["q_values"]))


def masked_double_dqn_target(
    reward: float,
    gamma: float,
    online_next_q: list[float],
    target_next_q: list[float],
    next_mask: list[bool],
) -> float:
    next_action = masked_argmax(online_next_q, next_mask)
    return reward + gamma * target_next_q[next_action]


def train_from_replay(buffer: ReplayBuffer, out: Path, gamma: float = 0.95) -> DQNPlanner:
    totals = [0.0 for _ in Action]
    counts = [0 for _ in Action]
    for t in buffer.transitions:
        totals[t.action] += t.reward
        counts[t.action] += 1
    q_values = [totals[i] / counts[i] if counts[i] else 0.0 for i in range(len(Action))]
    planner = DQNPlanner(q_values=q_values)
    planner.save(out)
    return planner

