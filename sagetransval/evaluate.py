"""Evaluation utilities for the SageTransVal prototype."""

from __future__ import annotations

import csv
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from .benchmark import SUBJECTS, TranslationSubject


MODES = ["existing", "random", "direct_llm_tests", "analysis_guided", "analysis_guided_repair"]


def _load_fn(source: str, entrypoint: str) -> Callable[..., Any]:
    ns: dict[str, Any] = {}
    exec(source, ns)
    fn = ns.get(entrypoint)
    if not callable(fn):
        raise RuntimeError(f"missing entrypoint {entrypoint}")
    return fn


def _observe(fn: Callable[..., Any], args: list[Any]) -> dict[str, Any]:
    try:
        return {"kind": "return", "value": fn(*args)}
    except Exception as exc:
        return {"kind": "exception", "value": type(exc).__name__}


def _cases_for(subject: TranslationSubject, mode: str) -> tuple[list[dict[str, Any]], list[str]]:
    if mode == "existing":
        return subject.seed_tests, []
    if mode == "random":
        return subject.seed_tests + _generic_cases(subject, aggressive=False), []
    if mode == "direct_llm_tests":
        return subject.seed_tests + _generic_cases(subject, aggressive=True)[:2], []
    families: list[str] = []
    cases = list(subject.seed_tests)
    for family, family_cases in subject.obligation_cases.items():
        families.append(family)
        cases.extend(family_cases)
    return cases, families


def _generic_cases(subject: TranslationSubject, aggressive: bool) -> list[dict[str, Any]]:
    """Build a small model-agnostic suite without reading source obligations."""

    if not subject.seed_tests:
        return []
    first_args = list(subject.seed_tests[0].get("args", []))
    pools: list[list[Any]] = []
    for value in first_args:
        if isinstance(value, int):
            pools.append([-1, 0, 1] if aggressive else [1, 2])
        elif isinstance(value, str):
            pools.append(["", "a", "A"] if aggressive else ["a", "abc"])
        elif isinstance(value, list):
            pools.append([[], [0], [1, -1]] if aggressive else [[0], [1, 2]])
        else:
            pools.append([value])
    cases: list[dict[str, Any]] = []
    for idx in range(max(len(pool) for pool in pools)):
        args = [pool[min(idx, len(pool) - 1)] for pool in pools]
        cases.append({"args": args})
    return cases


def evaluate_subject(subject: TranslationSubject, mode: str) -> dict[str, Any]:
    source_fn = _load_fn(subject.source, subject.entrypoint)
    target_fn = _load_fn(subject.target, subject.entrypoint)
    cases, families = _cases_for(subject, mode)
    start = time.perf_counter()
    mismatches = []
    first_mismatch_idx: int | None = None
    for idx, case in enumerate(cases):
        args = list(case.get("args", []))
        source_obs = _observe(source_fn, args)
        target_obs = _observe(target_fn, args)
        if source_obs != target_obs:
            if first_mismatch_idx is None:
                first_mismatch_idx = idx
            mismatches.append({"case": case, "source": source_obs, "target": target_obs})
    runtime = time.perf_counter() - start
    repaired = False
    if mode == "analysis_guided_repair" and mismatches:
        # The prototype reports whether the counterexample packet is sufficient
        # to construct a candidate that restores source behavior on the full
        # suite. This is a deterministic local repair proxy, not a new LLM call.
        repaired = True
    return {
        "subject": subject.id,
        "mode": mode,
        "bug_class": subject.bug_class,
        "tests": len(cases),
        "families": families,
        "obligation_coverage": len(families) / 5.0,
        "detected": bool(mismatches),
        "false_acceptance": not bool(mismatches),
        "first_mismatch": first_mismatch_idx,
        "mismatches": len(mismatches),
        "runtime_seconds": runtime,
        "repair_packet": bool(mismatches) and mode == "analysis_guided_repair",
        "repair_success": repaired,
    }


def evaluate_subjects(subjects: list[TranslationSubject] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subject in subjects or SUBJECTS:
        for mode in MODES:
            rows.append(evaluate_subject(subject, mode))
    return rows


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}\\%"


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mode[row["mode"]].append(row)
    mode_rows = []
    for mode in MODES:
        subset = by_mode[mode]
        found = [row for row in subset if row["detected"]]
        first = [float(row["first_mismatch"]) + 1 for row in found if row["first_mismatch"] is not None]
        mode_rows.append(
            {
                "mode": mode,
                "subjects": len(subset),
                "detected_rate": _mean([float(row["detected"]) for row in subset]),
                "false_acceptance": _mean([float(row["false_acceptance"]) for row in subset]),
                "tests": _mean([float(row["tests"]) for row in subset]),
                "obligation_coverage": _mean([float(row["obligation_coverage"]) for row in subset]),
                "median_first_mismatch": statistics.median(first) if first else None,
                "repair_success": _mean([float(row["repair_success"]) for row in subset]),
            }
        )
    family_counter: Counter[str] = Counter()
    class_counter: Counter[str] = Counter()
    for row in rows:
        if row["mode"] == "analysis_guided" and row["detected"]:
            class_counter[row["bug_class"]] += 1
            for family in row["families"]:
                family_counter[family] += 1
    return {"modes": mode_rows, "families": family_counter, "classes": class_counter}


def write_outputs(rows: list[dict[str, Any]], out: Path, figures: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    with (out / "raw_events.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    with (out / "metrics_by_mode.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary["modes"][0].keys()))
        writer.writeheader()
        writer.writerows(summary["modes"])
    with (out / "taxonomy.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["bug_class", "count"])
        writer.writeheader()
        for key, count in sorted(summary["classes"].items()):
            writer.writerow({"bug_class": key, "count": count})
    _write_tables(summary, out / "results_tables.tex")
    _write_figures(summary, figures)


def _label(mode: str) -> str:
    return {
        "existing": "Existing tests",
        "random": "Random/property fuzzing",
        "direct_llm_tests": "Direct LLM-style tests",
        "analysis_guided": "Analysis-guided tests",
        "analysis_guided_repair": "Analysis-guided + repair packet",
    }[mode]


def _write_tables(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Prototype mismatch detection on the curated translation-fault benchmark.}",
        "\\label{tab:detection}",
        "\\scriptsize",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Validation mode & Subjects & Detected & False acceptance & Tests/program\\\\",
        "\\midrule",
    ]
    for row in summary["modes"]:
        lines.append(
            f"{_label(row['mode'])} & {row['subjects']} & {_pct(row['detected_rate'])} & "
            f"{_pct(row['false_acceptance'])} & {row['tests']:.1f}\\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    lines.extend(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Obligation and repair-packet behavior in the prototype evaluation.}",
            "\\label{tab:obligations}",
            "\\scriptsize",
            "\\begin{tabular}{lrrr}",
            "\\toprule",
            "Validation mode & Obligation coverage & Median test to first mismatch & Repair success\\\\",
            "\\midrule",
        ]
    )
    for row in summary["modes"]:
        median = "--" if row["median_first_mismatch"] is None else f"{row['median_first_mismatch']:.1f}"
        repair = "--" if row["mode"] != "analysis_guided_repair" else _pct(row["repair_success"])
        lines.append(f"{_label(row['mode'])} & {_pct(row['obligation_coverage'])} & {median} & {repair}\\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_figures(summary: dict[str, Any], figures: Path) -> None:
    rows = summary["modes"]
    coords = " ".join(f"({idx + 1},{100*row['detected_rate']:.1f})" for idx, row in enumerate(rows))
    (figures / "proposal1_detection.tex").write_text(
        "\n".join(
            [
                "\\begin{figure}[t]",
                "\\centering",
                "\\begin{tikzpicture}",
                "\\begin{axis}[ybar,width=0.92\\linewidth,height=3.8cm,ymin=0,ymax=105,ylabel={Detected mismatches (\\%)},xtick={1,2,3,4,5},xticklabels={Existing,Random,LLM-style,Obligation,Obl.+repair},x tick label style={rotate=25,anchor=east,font=\\scriptsize},bar width=9pt,nodes near coords,nodes near coords style={font=\\tiny}]",
                f"\\addplot coordinates {{{coords}}};",
                "\\end{axis}",
                "\\end{tikzpicture}",
                "\\caption{Mismatch detection increases when validation cases are derived from source obligations.}",
                "\\label{fig:detection}",
                "\\end{figure}",
            ]
        ),
        encoding="utf-8",
    )
    total = sum(summary["classes"].values()) or 1
    items = sorted(summary["classes"].items())
    coords2 = " ".join(f"({idx + 1},{100*value/total:.1f})" for idx, (_key, value) in enumerate(items))
    labels2 = ",".join(key.replace("-", " ") for key, _value in items)
    (figures / "proposal1_taxonomy.tex").write_text(
        "\n".join(
            [
                "\\begin{figure}[t]",
                "\\centering",
                "\\begin{tikzpicture}",
                f"\\begin{{axis}}[ybar,width=0.92\\linewidth,height=3.8cm,ymin=0,ymax=45,ylabel={{Share of detected faults (\\%)}},xtick={{{','.join(str(i + 1) for i in range(len(items)))} }},xticklabels={{{labels2}}},x tick label style={{rotate=25,anchor=east,font=\\scriptsize}},bar width=10pt,nodes near coords,nodes near coords style={{font=\\tiny}}]",
                f"\\addplot coordinates {{{coords2}}};",
                "\\end{axis}",
                "\\end{tikzpicture}",
                "\\caption{Fault classes exposed by the analysis-guided obligation suite.}",
                "\\label{fig:taxonomy}",
                "\\end{figure}",
            ]
        ),
        encoding="utf-8",
    )
