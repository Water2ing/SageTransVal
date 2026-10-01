"""Staged driver for the Proposal 1 full cross-language upgrade."""

from __future__ import annotations

import argparse
import shutil
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _cmd(parts: list[str]) -> list[str]:
    return [sys.executable, "-m", "sagetransval.cli", *parts]


def _run(parts: list[str], dry_run: bool) -> None:
    command = _cmd(parts)
    print(shlex.join(command))
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["smoke", "python-java", "c-rust", "c-rust-100", "aggregate", "aggregate-200"],
        default="smoke",
    )
    parser.add_argument("--providers", nargs="+", default=["stub"])
    parser.add_argument("--models", nargs="+", default=["stub"])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--c-rust-limit", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if len(args.models) == 1 and len(args.providers) > 1:
        models = args.models * len(args.providers)
    else:
        models = args.models
    if len(models) != len(args.providers):
        raise SystemExit("--models must have length 1 or match --providers")

    data_root = Path("data/proposal1_full_subjects")
    runs_root = Path("runs/proposal1_translation")
    results_root = Path("results/proposal1_full")
    tables_root = Path("paper_tables/proposal1_full")
    figures_root = Path("paper-proposal-1/figures")
    runs_root_200 = Path("runs/proposal1_translation_200")
    results_root_200 = Path("results/proposal1_full_200")
    tables_root_200 = Path("paper_tables/proposal1_full_200")

    if args.stage in {"smoke", "python-java"}:
        _run(["prepare", "--dataset", "python_java", "--limit", str(args.limit), "--out", str(data_root / "python_java")], args.dry_run)
        for provider, model in zip(args.providers, models):
            translated = runs_root / provider / "python_java"
            cache_mode = "live" if provider != "stub" else "live"
            _run(
                [
                    "translate",
                    "--subjects",
                    str(data_root / "python_java"),
                    "--out",
                    str(translated),
                    "--provider",
                    provider,
                    "--model",
                    model,
                    "--cache-mode",
                    cache_mode,
                ],
                args.dry_run,
            )
            _run(["validate", "--subjects", str(translated), "--out", str(results_root / provider / "python_java")], args.dry_run)

    if args.stage in {"smoke", "c-rust"}:
        _run(["prepare", "--dataset", "c_rust", "--limit", str(args.c_rust_limit), "--out", str(data_root / "c_rust")], args.dry_run)
        for provider, model in zip(args.providers, models):
            translated = runs_root / provider / "c_rust"
            _run(
                [
                    "translate",
                    "--subjects",
                    str(data_root / "c_rust"),
                    "--out",
                    str(translated),
                    "--provider",
                    provider,
                    "--model",
                    model,
                    "--cache-mode",
                    "live",
                ],
                args.dry_run,
            )
            _run(["validate", "--subjects", str(translated), "--out", str(results_root / provider / "c_rust")], args.dry_run)

    if args.stage == "c-rust-100":
        _run(
            ["prepare", "--dataset", "c_rust_100", "--limit", str(args.c_rust_limit), "--out", str(data_root / "c_rust_100")],
            args.dry_run,
        )
        _copy_python_java_results(args.providers, results_root, results_root_200, args.dry_run)
        for provider, model in zip(args.providers, models):
            translated = runs_root_200 / provider / "c_rust_100"
            _run(
                [
                    "translate",
                    "--subjects",
                    str(data_root / "c_rust_100"),
                    "--out",
                    str(translated),
                    "--provider",
                    provider,
                    "--model",
                    model,
                    "--cache-mode",
                    "live",
                ],
                args.dry_run,
            )
            _run(["validate", "--subjects", str(translated), "--out", str(results_root_200 / provider / "c_rust_100")], args.dry_run)
        _run(["aggregate", "--results", str(results_root_200), "--out", str(tables_root_200), "--figures", str(figures_root)], args.dry_run)

    if args.stage in {"smoke", "aggregate"}:
        _run(["aggregate", "--results", str(results_root), "--out", str(tables_root), "--figures", str(figures_root)], args.dry_run)
    if args.stage == "aggregate-200":
        _copy_python_java_results(args.providers, results_root, results_root_200, args.dry_run)
        _run(["aggregate", "--results", str(results_root_200), "--out", str(tables_root_200), "--figures", str(figures_root)], args.dry_run)
    return 0


def _copy_python_java_results(providers: list[str], src_root: Path, dst_root: Path, dry_run: bool) -> None:
    for provider in providers:
        src = ROOT / src_root / provider / "python_java"
        dst = ROOT / dst_root / provider / "python_java"
        print(f"copy {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
        if dry_run:
            continue
        if not src.exists():
            print(f"warning: missing {src.relative_to(ROOT)}; aggregate will use whatever results are present")
            continue
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)


if __name__ == "__main__":
    raise SystemExit(main())
