"""Run the full two-provider TestGenRL MVP experiment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROVIDERS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash-lite",
}
BASELINES = ["existing", "fixed", "greedy", "bandit"]
STOCHASTIC = {"random"}
SEEDS = [0, 1, 2, 3, 4]


def run(cmd: list[str], dry_run: bool = False) -> None:
    print("+", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="data/subjects")
    parser.add_argument("--budget", default="configs/budget.yaml")
    parser.add_argument("--runs", default="runs/full")
    parser.add_argument("--results", default="results/full")
    parser.add_argument("--models", default="models")
    parser.add_argument("--tables", default="paper_tables/full")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    py = sys.executable
    subjects = Path(args.subjects)
    run([py, "-m", "testgenrl.cli", "prepare", "--dataset", "toy", "--out", str(subjects)], args.dry_run)

    result_roots: list[str] = []
    for provider, model in PROVIDERS.items():
        provider_runs = Path(args.runs) / provider
        provider_results = Path(args.results) / provider
        cache_dir = provider_runs / ".llm_cache"
        model_path = Path(args.models) / f"{provider}_dqn.json"

        for planner in BASELINES:
            run(
                [
                    py,
                    "-m",
                    "testgenrl.cli",
                    "run",
                    "--planner",
                    planner,
                    "--subjects",
                    str(subjects),
                    "--budget",
                    args.budget,
                    "--llm-provider",
                    provider,
                    "--llm-model",
                    model,
                    "--cache-mode",
                    "live",
                    "--cache-dir",
                    str(cache_dir),
                    "--out",
                    str(provider_runs),
                ],
                args.dry_run,
            )

        for planner in STOCHASTIC:
            for seed in SEEDS:
                run(
                    [
                        py,
                        "-m",
                        "testgenrl.cli",
                        "run",
                        "--planner",
                        planner,
                        "--subjects",
                        str(subjects),
                        "--budget",
                        args.budget,
                        "--llm-provider",
                        provider,
                        "--llm-model",
                        model,
                        "--cache-mode",
                        "live",
                        "--cache-dir",
                        str(cache_dir),
                        "--seed",
                        str(seed),
                        "--out",
                        str(provider_runs),
                    ],
                    args.dry_run,
                )

        run([py, "-m", "testgenrl.cli", "train", "--runs", str(provider_runs), "--out", str(model_path)], args.dry_run)

        for planner in BASELINES:
            run(
                [
                    py,
                    "-m",
                    "testgenrl.cli",
                    "run",
                    "--planner",
                    planner,
                    "--subjects",
                    str(subjects),
                    "--budget",
                    args.budget,
                    "--llm-provider",
                    provider,
                    "--llm-model",
                    model,
                    "--cache-mode",
                    "replay",
                    "--cache-dir",
                    str(cache_dir),
                    "--out",
                    str(provider_results),
                ],
                args.dry_run,
            )

        for seed in SEEDS:
            run(
                [
                    py,
                    "-m",
                    "testgenrl.cli",
                    "run",
                    "--planner",
                    "random",
                    "--subjects",
                    str(subjects),
                    "--budget",
                    args.budget,
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
                    str(provider_results),
                ],
                args.dry_run,
            )
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
                    args.budget,
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
                    str(provider_results),
                ],
                args.dry_run,
            )

        provider_table = Path(args.tables) / provider
        run([py, "-m", "testgenrl.cli", "report", "--runs", str(provider_results), "--out", str(provider_table)], args.dry_run)
        result_roots.append(str(provider_results))

    run([py, "-m", "testgenrl.cli", "report", "--runs", *result_roots, "--out", str(Path(args.tables) / "combined")], args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
