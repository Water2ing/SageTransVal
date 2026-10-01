"""Run the 50-subject Python pilot experiment matrix."""

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
BUDGET_TOKENS = {"low": 300, "medium": 1000, "high": 2000}
HORIZONS = [8, 16, 32]
DETERMINISTIC = ["existing", "fixed", "greedy", "bandit"]
STOCHASTIC = ["random"]
SEEDS = [0, 1, 2, 3, 4]


def run(cmd: list[str], dry_run: bool) -> None:
    print("+", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def write_budget_files(root: Path, horizons: list[int], budget_names: list[str]) -> dict[tuple[str, int], Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[tuple[str, int], Path] = {}
    for budget_name in budget_names:
        for horizon in horizons:
            path = root / f"budget_{budget_name}_h{horizon}.yaml"
            path.write_text(
                "\n".join(
                    [
                        f"tokens: {BUDGET_TOKENS[budget_name]}",
                        "runtime_seconds: 30",
                        "fuzz_iterations: 100",
                        f"horizon: {horizon}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            paths[(budget_name, horizon)] = path
    return paths


def run_planner(
    py: str,
    planner: str,
    subjects: Path,
    budget_path: Path,
    provider: str,
    model: str,
    cache_mode: str,
    cache_dir: Path,
    out_root: Path,
    dry_run: bool,
    seed: int | None = None,
) -> None:
    cmd = [
        py,
        "-m",
        "testgenrl.cli",
        "run",
        "--planner",
        planner,
        "--subjects",
        str(subjects),
        "--budget",
        str(budget_path),
        "--llm-provider",
        provider,
        "--llm-model",
        model,
        "--cache-mode",
        cache_mode,
        "--cache-dir",
        str(cache_dir),
        "--out",
        str(out_root),
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    run(cmd, dry_run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--providers", nargs="+", default=["openai", "gemini"], choices=sorted(PROVIDER_MODELS))
    parser.add_argument("--budgets", nargs="+", default=["low", "medium", "high"], choices=sorted(BUDGET_TOKENS))
    parser.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    parser.add_argument("--subjects", default="data/full_pilot_subjects")
    parser.add_argument("--runs", default="runs/full_dataset")
    parser.add_argument("--results", default="results/full_dataset")
    parser.add_argument("--models", default="models/full_dataset")
    parser.add_argument("--budget-dir", default="configs/full_dataset")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    py = sys.executable
    subjects = Path(args.subjects)
    budget_paths = write_budget_files(Path(args.budget_dir), args.horizons, args.budgets)
    run([py, "-m", "testgenrl.cli", "prepare", "--dataset", "full_pilot", "--out", str(subjects)], args.dry_run)

    for provider in args.providers:
        model = PROVIDER_MODELS[provider]
        for budget_name in args.budgets:
            for horizon in args.horizons:
                budget_path = budget_paths[(budget_name, horizon)]
                train_root = Path(args.runs) / provider / budget_name / f"h{horizon}"
                eval_root = Path(args.results) / provider / budget_name / f"h{horizon}"
                cache_dir = train_root / ".llm_cache"
                model_path = Path(args.models) / provider / f"{budget_name}_h{horizon}_dqn.json"

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

