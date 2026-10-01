"""Command-line interface for SageTransVal full Proposal 1 experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .full_pipeline import aggregate_results, prepare_subjects, translate_subjects, validate_subjects, write_repair_packets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sagetransval")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare")
    p_prepare.add_argument(
        "--dataset",
        choices=["python_java", "c_rust", "c_rust_100", "c_rust_v2", "proposal1_full", "proposal1_full_200"],
        default="python_java",
    )
    p_prepare.add_argument("--out", required=True)
    p_prepare.add_argument("--limit", type=int)
    p_prepare.add_argument("--cache-dir", default="data/raw")

    p_translate = sub.add_parser("translate")
    p_translate.add_argument("--subjects", required=True)
    p_translate.add_argument("--out", required=True)
    p_translate.add_argument("--provider", choices=["stub", "openai", "gemini"], default="stub")
    p_translate.add_argument("--model", default="stub")
    p_translate.add_argument("--cache-mode", choices=["live", "replay"], default="live")
    p_translate.add_argument("--cache-dir")

    p_validate = sub.add_parser("validate")
    p_validate.add_argument("--subjects", required=True)
    p_validate.add_argument("--out", required=True)
    p_validate.add_argument("--modes", nargs="*")

    p_repair = sub.add_parser("repair")
    p_repair.add_argument("--validation", required=True)
    p_repair.add_argument("--out", required=True)

    p_aggregate = sub.add_parser("aggregate")
    p_aggregate.add_argument("--results", required=True)
    p_aggregate.add_argument("--out", required=True)
    p_aggregate.add_argument("--figures")

    p_report = sub.add_parser("report")
    p_report.add_argument("--results", required=True)
    p_report.add_argument("--out", default="paper_tables/proposal1_full")
    p_report.add_argument("--figures", default="paper-proposal-1/figures")

    args = parser.parse_args(argv)
    if args.command == "prepare":
        report = prepare_subjects(args.dataset, Path(args.out), limit=args.limit, cache_dir=Path(args.cache_dir))
        print(f"prepared {report['imported']} subjects in {args.out}")
    elif args.command == "translate":
        translated = translate_subjects(
            Path(args.subjects),
            Path(args.out),
            provider=args.provider,
            model=args.model,
            cache_mode=args.cache_mode,
            cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        )
        print(f"translated {len(translated)} subjects to {args.out}")
    elif args.command == "validate":
        rows = validate_subjects(Path(args.subjects), Path(args.out), modes=args.modes)
        print(f"validated {len(rows)} mode rows in {args.out}")
    elif args.command == "repair":
        count = write_repair_packets(Path(args.validation), Path(args.out))
        print(f"wrote {count} repair packets to {args.out}")
    elif args.command == "aggregate":
        aggregate_results(Path(args.results), Path(args.out), Path(args.figures) if args.figures else None)
        print(f"aggregated results into {args.out}")
    elif args.command == "report":
        aggregate_results(Path(args.results), Path(args.out), Path(args.figures))
        print(f"wrote report artifacts into {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
