"""Metrics and report generation."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .logging import iter_event_files, read_jsonl


PRIMARY_FIELDS = [
    "provider",
    "model",
    "planner",
    "seed",
    "subjects",
    "failures",
    "ttfm",
    "tokfm",
    "mp1k",
    "mpm",
    "coverage",
    "false_reassurance_rate",
    "budget_exhaustion_rate",
    "invalid_flaky_rate",
    "duplicate_rate",
    "return",
]


def summarize_events(event_files: list[Path]) -> dict[str, Any]:
    subjects: set[str] = set()
    failures = 0
    total_tokens = 0.0
    total_runtime = 0.0
    total_return = 0.0
    coverage = 0.0
    invalid_flaky = 0
    duplicates = 0
    tests = 0
    exhausted = 0
    first_failure_step: list[int] = []
    first_failure_tokens: list[float] = []
    actions: Counter[str] = Counter()
    provider = ""
    model = ""
    seed = ""

    for path in event_files:
        events = read_jsonl(path)
        if not events:
            continue
        provider = provider or str(events[0].get("provider", ""))
        model = model or str(events[0].get("model", ""))
        seed_value = events[0].get("seed", "")
        if seed == "" and seed_value is not None:
            seed = str(seed_value)
        subject_id = str(events[0].get("subject_id", path.parent.name))
        subjects.add(subject_id)
        running_tokens = 0.0
        saw_failure = False
        for event in events:
            actions[str(event["action"])] += 1
            result = event.get("result", {})
            cost = result.get("cost", {})
            running_tokens += float(cost.get("tokens", 0.0))
            total_tokens += float(cost.get("tokens", 0.0))
            total_runtime += float(cost.get("runtime_seconds", 0.0))
            total_return += float(event.get("reward", 0.0))
            coverage += float(result.get("coverage_delta", 0.0))
            failures += int(result.get("failures", 0))
            invalid_flaky += int(result.get("invalid_tests", 0)) + int(result.get("flaky_tests", 0))
            duplicates += int(result.get("duplicate_tests", 0))
            tests += len(result.get("generated_tests", []))
            if int(result.get("failures", 0)) > 0 and not saw_failure:
                first_failure_step.append(int(event.get("step", 0)))
                first_failure_tokens.append(running_tokens)
                saw_failure = True
        if events[-1].get("budget", {}).get("runtime_seconds", 1) <= 0:
            exhausted += 1

    n_subjects = max(1, len(subjects))
    mp1k = 1000.0 * failures / max(total_tokens, 1.0)
    mpm = failures / max(total_runtime / 60.0, 1e-6)
    return {
        "provider": provider,
        "model": model,
        "seed": seed,
        "subjects": len(subjects),
        "failures": failures,
        "ttfm": sum(first_failure_step) / len(first_failure_step) if first_failure_step else "",
        "tokfm": sum(first_failure_tokens) / len(first_failure_tokens) if first_failure_tokens else "",
        "mp1k": mp1k,
        "mpm": mpm,
        "coverage": min(1.0, coverage / n_subjects),
        "false_reassurance_rate": (n_subjects - len(first_failure_step)) / n_subjects,
        "budget_exhaustion_rate": exhausted / n_subjects,
        "invalid_flaky_rate": invalid_flaky / max(tests, 1),
        "duplicate_rate": duplicates / max(tests, 1),
        "return": total_return / n_subjects,
        "actions": dict(actions),
    }


def write_report(run_roots: list[Path], out: Path) -> dict[str, dict[str, Any]]:
    out.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, Any]] = {}
    for root in run_roots:
        grouped: defaultdict[str, list[Path]] = defaultdict(list)
        for event_file in iter_event_files(root):
            events = read_jsonl(event_file)
            if events:
                event = events[0]
                planner = event_file.parent.parent.name
                if planner.startswith("seed_"):
                    planner = event_file.parent.parent.parent.name
                provider = str(event.get("provider", ""))
                model = str(event.get("model", ""))
                seed = event.get("seed", "")
                key = "|".join([provider, model, planner, "" if seed is None else str(seed)])
            else:
                planner = event_file.parent.parent.name if event_file.parent.parent != root.parent else event_file.parent.name
                key = planner
            grouped[key].append(event_file)
        for key, files in grouped.items():
            summary = summarize_events(files)
            if not summary.get("planner"):
                summary["planner"] = key.split("|")[2] if "|" in key else key
            summaries[key] = summary

    csv_path = out / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PRIMARY_FIELDS)
        writer.writeheader()
        for summary in summaries.values():
            writer.writerow({k: summary.get(k, "") for k in PRIMARY_FIELDS})

    md_lines = ["| " + " | ".join(PRIMARY_FIELDS) + " |", "| " + " | ".join(["---"] * len(PRIMARY_FIELDS)) + " |"]
    for summary in summaries.values():
        md_lines.append("| " + " | ".join(str(round(summary.get(k, ""), 4)) if isinstance(summary.get(k, ""), float) else str(summary.get(k, "")) for k in PRIMARY_FIELDS) + " |")
    (out / "metrics.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return summaries
