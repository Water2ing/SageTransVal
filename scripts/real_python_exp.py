"""Run the real Python benchmark experiment matrix."""

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
STRICT_REAL_DATASETS = {"humaneval_real", "mbpp_real"}
DISALLOWED_STRICT_DATASETS = {
    "toy",
    "humaneval",
    "mbpp",
    "bugsinpy",
    "stress",
    "action_balanced",
    "full_pilot",
    "diverse_full",
    "bugsinpy_real",
}


def ensure_strict_real_manifest(subjects: Path) -> None:
    manifest_path = subjects / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"missing subject manifest: {manifest_path}")
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = {str(row.get("metadata", {}).get("dataset", "")) for row in rows}
    bad = sorted((datasets - STRICT_REAL_DATASETS) | (datasets & DISALLOWED_STRICT_DATASETS))
    if bad:
        raise RuntimeError(f"strict real-data run refuses non-real or proxy datasets: {', '.join(bad)}")
    missing = STRICT_REAL_DATASETS - datasets
    if missing:
        raise RuntimeError(f"strict real-data run requires both HumanEval and MBPP; missing: {', '.join(sorted(missing))}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--providers", nargs="+", default=["openai", "gemini"], choices=sorted(PROVIDER_MODELS))
    parser.add_argument("--budgets", nargs="+", default=["medium"], choices=sorted(BUDGET_TOKENS))
    parser.add_argument("--horizons", nargs="+", type=int, default=[16])
    parser.add_argument(
        "--dataset",
        default="real_python_strict",
        choices=["humaneval_real", "mbpp_real", "bugsinpy_real", "real_python", "real_python_smoke", "real_python_full", "real_python_strict"],
    )
    parser.add_argument("--subjects", default="data/real_python_strict_subjects")
    parser.add_argument("--runs", default="runs/real_python_strict")
    parser.add_argument("--results", default="results/real_python_strict")
    parser.add_argument("--models", default="models/real_python_strict")
    parser.add_argument("--budget-dir", default="configs/real_python_strict")
    parser.add_argument("--source-root")
    parser.add_argument("--cache-dir", default="data/raw")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--limit-per-source", type=int)
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    py = sys.executable
    subjects = Path(args.subjects)
    budget_paths = write_budget_files(Path(args.budget_dir), args.horizons, args.budgets)

    prepare_cmd = [
        py,
        "-m",
        "testgenrl.cli",
        "prepare",
        "--dataset",
        args.dataset,
        "--out",
        str(subjects),
        "--cache-dir",
        args.cache_dir,
    ]
    if args.source_root:
        prepare_cmd.extend(["--source-root", args.source_root])
    if args.limit:
        prepare_cmd.extend(["--limit", str(args.limit)])
    if args.limit_per_source is not None:
        prepare_cmd.extend(["--limit-per-source", str(args.limit_per_source)])
    if not args.no_validate:
        prepare_cmd.append("--validate")
    run(prepare_cmd, args.dry_run)
    if args.dataset == "real_python_strict" and not args.dry_run:
        ensure_strict_real_manifest(subjects)

    for provider in args.providers:
        model = PROVIDER_MODELS[provider]
        for budget_name in args.budgets:
            for horizon in args.horizons:
                budget_path = budget_paths[(budget_name, horizon)]
                train_root = Path(args.runs) / provider / budget_name / f"h{horizon}"
                eval_root = Path(args.results) / provider / budget_name / f"h{horizon}"
                llm_cache = Path(args.runs) / provider / ".llm_cache"
                model_path = Path(args.models) / provider / f"{budget_name}_h{horizon}_dqn.json"

                for planner in DETERMINISTIC:
                    run_planner(py, planner, subjects, budget_path, provider, model, "live", llm_cache, train_root, args.dry_run)
                for planner in STOCHASTIC:
                    for seed in SEEDS:
                        run_planner(py, planner, subjects, budget_path, provider, model, "live", llm_cache, train_root, args.dry_run, seed)

                run([py, "-m", "testgenrl.cli", "train", "--runs", str(train_root), "--out", str(model_path)], args.dry_run)

                for planner in DETERMINISTIC:
                    run_planner(py, planner, subjects, budget_path, provider, model, "replay", llm_cache, eval_root, args.dry_run)
                for seed in SEEDS:
                    run_planner(py, "random", subjects, budget_path, provider, model, "replay", llm_cache, eval_root, args.dry_run, seed)
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
                            str(llm_cache),
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
