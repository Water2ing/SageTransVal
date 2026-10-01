"""Run the Proposal 1 SageTransVal prototype evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sagetransval import evaluate_subjects, write_outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="paper_tables/proposal1")
    parser.add_argument("--figures", default="paper-proposal-1/figures")
    args = parser.parse_args()
    rows = evaluate_subjects()
    write_outputs(rows, Path(args.out), Path(args.figures))
    print(f"wrote {len(rows)} evaluation rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
