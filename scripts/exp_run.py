"""Run TestGenRL experiment matrices for paper result tables."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROVIDER_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash-lite",
    "stub": "stub",
}
BUDGETS = {
    "low": {"tokens": 300, "runtime_seconds": 30, "fuzz_iterations": 100, "horizon": 8},
    "medium": {"tokens": 1000, "runtime_seconds": 30, "fuzz_iterations": 100, "horizon": 8},
    "high": {"tokens": 2000, "runtime_seconds": 30, "fuzz_iterations": 100, "horizon": 8},
}
DETERMINISTIC = ["existing", "fixed", "greedy", "bandit"]
STOCHASTIC = ["random"]
SEEDS = [0, 1, 2, 3, 4]


def run(cmd: list[str], dry_run: bool) -> None:
    print("+", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def write_budget_files(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, values in BUDGETS.items():
        path = root / f"budget_{name}.yaml"
        path.write_text("\n".join(f"{k}: {v}" for k, v in values.items()) + "\n", encoding="utf-8")
        paths[name] = path
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--providers", nargs="+", default=["openai", "gemini"], choices=sorted(PROVIDER_MODELS))
    parser.add_argument("--budgets", nargs="+", default=["low", "medium", "high"], choices=sorted(BUDGETS))
    parser.add_argument("--subjects", default="data/subjects")
    parser.add_argument("--runs", default="runs/exp")
    parser.add_argument("--results", default="results/exp")
    parser.add_argument("--models", default="models/exp")
    parser.add_argument("--budget-dir", default="configs/exp")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    py = sys.executable
    subjects = Path(args.subjects)
    budgets = write_budget_files(Path(args.budget_dir))
    run([py, "-m", "testgenrl.cli", "prepare", "--dataset", "toy", "--out", str(subjects)], args.dry_run)

    for provider in args.providers:
        model = PROVIDER_MODELS[provider]
        for budget_name in args.budgets:
            budget_path = budgets[budget_name]
            train_root = Path(args.runs) / provider / budget_name
            eval_root = Path(args.results) / provider / budget_name
            cache_dir = train_root / ".llm_cache"
            model_path = Path(args.models) / provider / f"{budget_name}_dqn.json"

            for planner in DETERMINISTIC:
                run(
                    [
                        py, "-m", "testgenrl.cli", "run",
                        "--planner", planner,
                        "--subjects", str(subjects),
                        "--budget", str(budget_path),
                        "--llm-provider", provider,
                        "--llm-model", model,
                        "--cache-mode", "live",
                        "--cache-dir", str(cache_dir),
                        "--out", str(train_root),
                    ],
                    args.dry_run,
                )

            for planner in STOCHASTIC:
                for seed in SEEDS:
                    run(
                        [
                            py, "-m", "testgenrl.cli", "run",
                            "--planner", planner,
                            "--subjects", str(subjects),
                            "--budget", str(budget_path),
                            "--llm-provider", provider,
                            "--llm-model", model,
                            "--cache-mode", "live",
                            "--cache-dir", str(cache_dir),
                            "--seed", str(seed),
                            "--out", str(train_root),
                        ],
                        args.dry_run,
                    )

            run([py, "-m", "testgenrl.cli", "train", "--runs", str(train_root), "--out", str(model_path)], args.dry_run)

            for planner in DETERMINISTIC:
                run(
                    [
                        py, "-m", "testgenrl.cli", "run",
                        "--planner", planner,
                        "--subjects", str(subjects),
                        "--budget", str(budget_path),
                        "--llm-provider", provider,
                        "--llm-model", model,
                        "--cache-mode", "replay",
                        "--cache-dir", str(cache_dir),
                        "--out", str(eval_root),
                    ],
                    args.dry_run,
                )

            for seed in SEEDS:
                run(
                    [
                        py, "-m", "testgenrl.cli", "run",
                        "--planner", "random",
                        "--subjects", str(subjects),
                        "--budget", str(budget_path),
                        "--llm-provider", provider,
                        "--llm-model", model,
                        "--cache-mode", "replay",
                        "--cache-dir", str(cache_dir),
                        "--seed", str(seed),
                        "--out", str(eval_root),
                    ],
                    args.dry_run,
                )
                run(
                    [
                        py, "-m", "testgenrl.cli", "eval",
                        "--model", str(model_path),
                        "--subjects", str(subjects),
                        "--budget", str(budget_path),
                        "--llm-provider", provider,
                        "--llm-model", model,
                        "--cache-mode", "replay",
                        "--cache-dir", str(cache_dir),
                        "--seed", str(seed),
                        "--out", str(eval_root),
                    ],
                    args.dry_run,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
