"""Run the diverse Python diagnostic full experiment matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.full_dataset_exp import BUDGET_TOKENS, HORIZONS, PROVIDER_MODELS, run, run_planner, write_budget_files


DETERMINISTIC = ["existing", "fixed", "greedy", "bandit"]
STOCHASTIC = ["random"]
SEEDS = [0, 1, 2, 3, 4]


def filter_manifest(subjects: Path, profiles: list[str]) -> None:
    if not profiles or "all" in profiles:
        return
    manifest_path = subjects / "manifest.json"
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed = set(profiles)
    filtered = [row for row in rows if row.get("metadata", {}).get("profile") in allowed]
    if not filtered:
        raise RuntimeError(f"profile filter removed all subjects: {sorted(allowed)}")
    manifest_path.write_text(json.dumps(filtered, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--providers", nargs="+", default=["openai", "gemini"], choices=sorted(PROVIDER_MODELS))
    parser.add_argument("--budgets", nargs="+", default=["low", "medium", "high"], choices=sorted(BUDGET_TOKENS))
    parser.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    parser.add_argument("--profile-filter", nargs="+", default=["all"], choices=["all", "full_pilot", "stress", "action_balanced"])
    parser.add_argument("--subjects", default="data/diverse_full_subjects")
    parser.add_argument("--runs", default="runs/diverse_full")
    parser.add_argument("--results", default="results/diverse_full")
    parser.add_argument("--models", default="models/diverse_full")
    parser.add_argument("--budget-dir", default="configs/diverse_full")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    py = sys.executable
    subjects = Path(args.subjects)
    budget_paths = write_budget_files(Path(args.budget_dir), args.horizons, args.budgets)
    run([py, "-m", "testgenrl.cli", "prepare", "--dataset", "diverse_full", "--out", str(subjects)], args.dry_run)
    if not args.dry_run:
        filter_manifest(subjects, args.profile_filter)

    profile_suffix = "all" if args.profile_filter == ["all"] else "_".join(args.profile_filter)
    for provider in args.providers:
        model = PROVIDER_MODELS[provider]
        for budget_name in args.budgets:
            for horizon in args.horizons:
                budget_path = budget_paths[(budget_name, horizon)]
                train_root = Path(args.runs) / profile_suffix / provider / budget_name / f"h{horizon}"
                eval_root = Path(args.results) / profile_suffix / provider / budget_name / f"h{horizon}"
                cache_dir = Path(args.runs) / profile_suffix / provider / ".llm_cache"
                model_path = Path(args.models) / profile_suffix / provider / f"{budget_name}_h{horizon}_dqn.json"

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
