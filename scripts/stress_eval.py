"""Run the diagnostic stress-profile evaluation matrix."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.full_dataset_exp import PROVIDER_MODELS, run, run_planner, write_budget_files


DETERMINISTIC = ["existing", "fixed", "greedy", "bandit"]
STOCHASTIC = ["random"]
SEEDS = [0, 1, 2, 3, 4]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--providers", nargs="+", default=["openai", "gemini"], choices=sorted(PROVIDER_MODELS))
    parser.add_argument("--subjects", default="data/stress_subjects")
    parser.add_argument("--runs", default="runs/stress_eval")
    parser.add_argument("--results", default="results/stress_eval")
    parser.add_argument("--models", default="models/stress_eval")
    parser.add_argument("--budget-dir", default="configs/stress_eval")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    py = sys.executable
    subjects = Path(args.subjects)
    budget_paths = write_budget_files(Path(args.budget_dir), [16], ["medium"])
    run([py, "-m", "testgenrl.cli", "prepare", "--dataset", "stress", "--out", str(subjects)], args.dry_run)

    for provider in args.providers:
        model = PROVIDER_MODELS[provider]
        budget_path = budget_paths[("medium", 16)]
        train_root = Path(args.runs) / provider / "medium" / "h16"
        eval_root = Path(args.results) / provider / "medium" / "h16"
        cache_dir = train_root / ".llm_cache"
        model_path = Path(args.models) / provider / "medium_h16_dqn.json"

        for planner in DETERMINISTIC:
            run_planner(py, planner, subjects, budget_path, provider, model, "live", cache_dir, train_root, args.dry_run)
        for planner in STOCHASTIC:
            for seed in SEEDS:
                run_planner(py, planner, subjects, budget_path, provider, model, "live", cache_dir, train_root, args.dry_run, seed)

        run([py, "-m", "testgenrl.cli", "train", "--runs", str(train_root), "--out", str(model_path)], args.dry_run)

        for planner in DETERMINISTIC:
            run_planner(py, planner, subjects, budget_path, provider, model, "replay", cache_dir, eval_root, args.dry_run)
        for seed in SEEDS:
            run_planner(py, "random", subjects, budget_path, provider, model, "replay", cache_dir, eval_root, args.dry_run, seed)
            run(
                [
                    py,
                    "-m",
                    "testgenrl.cli",
                    "eval",
                    "--model",
                    str(model_path),
                    "--subjects",
                    str(subjects),
                    "--budget",
                    str(budget_path),
                    "--llm-provider",
                    provider,
                    "--llm-model",
                    model,
                    "--cache-mode",
                    "replay",
                    "--cache-dir",
                    str(cache_dir),
                    "--seed",
                    str(seed),
                    "--out",
                    str(eval_root),
                ],
                args.dry_run,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
