"""Command-line interface for TestGenRL."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .dqn import DQNPlanner, ReplayBuffer, train_from_replay
from .environment import TestGenEnvironment
from .metrics import write_report
from .models import Budget
from .planners import planner_from_name
from .subjects import load_subjects, prepare_dataset


def _load_budget(path: Path | None) -> Budget:
    if path is None or not path.exists():
        return Budget()
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        data = json.loads(text)
    else:
        data = {}
        for line in text.splitlines():
            if ":" in line and not line.lstrip().startswith("#"):
                key, value = line.split(":", 1)
                data[key.strip()] = float(value.strip())
    if "horizon" in data:
        data["horizon"] = int(data["horizon"])
    return Budget(**{k: v for k, v in data.items() if k in Budget.__dataclass_fields__})


def cmd_prepare(args: argparse.Namespace) -> None:
    manifest = prepare_dataset(
        Path(args.out),
        args.dataset,
        limit=args.limit,
        source_root=Path(args.source_root) if args.source_root else None,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        limit_per_source=args.limit_per_source,
        validate=args.validate,
    )
    print(f"wrote {manifest}")


def cmd_run(args: argparse.Namespace) -> None:
    if args.seed is not None:
        random.seed(args.seed)
    subjects = load_subjects(Path(args.subjects), args.split)
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(args.out) / ".llm_cache"
    env = TestGenEnvironment(
        cache_dir,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        cache_mode=args.cache_mode,
        seed=args.seed,
    )
    root = Path(args.out) / args.planner
    if args.seed is not None:
        root = root / f"seed_{args.seed}"
    for subject in subjects:
        planner = planner_from_name(args.planner)
        budget = _load_budget(Path(args.budget) if args.budget else None)
        env.run_episode(subject, planner, budget, root / subject.id)
    print(
        f"ran {len(subjects)} subjects with planner={args.planner} "
        f"provider={args.llm_provider} model={args.llm_model}; logs in {root}"
    )


def cmd_train(args: argparse.Namespace) -> None:
    buffer = ReplayBuffer()
    for path in Path(args.runs).rglob("replay.json"):
        for transition in ReplayBuffer.load(path).transitions:
            buffer.add(transition)
    planner = train_from_replay(buffer, Path(args.out))
    print(f"trained fallback DQN from {len(buffer.transitions)} transitions -> {args.out}")
    print(json.dumps({"q_values": planner.q_values}, indent=2))


def cmd_eval(args: argparse.Namespace) -> None:
    if args.seed is not None:
        random.seed(args.seed)
    subjects = load_subjects(Path(args.subjects), args.split)
    planner = DQNPlanner.load(Path(args.model))
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(args.out) / ".llm_cache"
    env = TestGenEnvironment(
        cache_dir,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        cache_mode=args.cache_mode,
        seed=args.seed,
    )
    root = Path(args.out) / "dqn"
    if args.seed is not None:
        root = root / f"seed_{args.seed}"
    for subject in subjects:
        budget = _load_budget(Path(args.budget) if args.budget else None)
        env.run_episode(subject, planner, budget, root / subject.id)
    print(f"evaluated {len(subjects)} subjects provider={args.llm_provider} model={args.llm_model}; logs in {root}")


def cmd_report(args: argparse.Namespace) -> None:
    summaries = write_report([Path(p) for p in args.runs], Path(args.out))
    print(f"wrote report for {len(summaries)} planners to {args.out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="testgenrl")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare")
    p.add_argument(
        "--dataset",
        default="humaneval",
        choices=[
            "humaneval",
            "mbpp",
            "bugsinpy",
            "toy",
            "full_pilot",
            "stress",
            "action_balanced",
            "diverse_full",
            "humaneval_real",
            "mbpp_real",
            "bugsinpy_real",
            "real_python",
            "real_python_smoke",
            "real_python_full",
            "real_python_strict",
        ],
    )
    p.add_argument("--limit", type=int)
    p.add_argument("--limit-per-source", type=int)
    p.add_argument("--source-root")
    p.add_argument("--cache-dir")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("run")
    p.add_argument("--planner", required=True, choices=["existing", "fixed", "random", "greedy", "bandit", "dqn"])
    p.add_argument("--subjects", required=True)
    p.add_argument("--budget")
    p.add_argument("--split")
    p.add_argument("--llm-provider", default="stub", choices=["stub", "openai", "gemini"])
    p.add_argument("--llm-model", default="stub")
    p.add_argument("--cache-mode", default="live", choices=["live", "replay"])
    p.add_argument("--cache-dir")
    p.add_argument("--seed", type=int)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("train")
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("eval")
    p.add_argument("--model", required=True)
    p.add_argument("--subjects", required=True)
    p.add_argument("--budget")
    p.add_argument("--split")
    p.add_argument("--llm-provider", default="stub", choices=["stub", "openai", "gemini"])
    p.add_argument("--llm-model", default="stub")
    p.add_argument("--cache-mode", default="replay", choices=["live", "replay"])
    p.add_argument("--cache-dir")
    p.add_argument("--seed", type=int)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("report")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
