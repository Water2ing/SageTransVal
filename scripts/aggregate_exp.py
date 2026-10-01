"""Aggregate TestGenRL experiment logs into paper tables and PGFPlots figures."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from testgenrl.logging import read_jsonl


METHOD_LABELS = {
    "existing": "Existing tests",
    "fixed": "Static instruction synthesis",
    "random": "Random/property fuzzing",
    "greedy": "Greedy planner",
    "bandit": "Contextual bandit",
    "dqn": "Masked replay planner",
}
PLOT_METHODS = ["fixed", "random", "bandit", "dqn"]
ACTION_ORDER = [
    "GENERATE_BOUNDARY_TESTS",
    "PROPERTY_BASED_FUZZ",
    "LLM_SYNTHESIZE_INSTRUCTIONS",
    "LLM_GENERATE_ASSERTIONS",
    "REFINE_UNCERTAIN_ORACLE",
    "EXPAND_COUNTEREXAMPLE",
    "RETEST_REPAIR_CONTEXT",
    "STOP",
]
ACTION_LABELS = {
    "GENERATE_BOUNDARY_TESTS": "Boundary",
    "PROPERTY_BASED_FUZZ": "Fuzz",
    "LLM_SYNTHESIZE_INSTRUCTIONS": "LLM instr.",
    "LLM_GENERATE_ASSERTIONS": "LLM assert.",
    "REFINE_UNCERTAIN_ORACLE": "Refine",
    "EXPAND_COUNTEREXAMPLE": "Counterex.",
    "RETEST_REPAIR_CONTEXT": "Retest",
    "STOP": "Stop",
}
BUDGET_TOKENS = {"low": 300.0, "medium": 1000.0, "high": 2000.0}
PROFILE_LABELS = {
    "full_pilot": "Full pilot",
    "stress": "Stress",
    "action_balanced": "Action-balanced",
}


def safe_float(value: Any, default: float = math.nan) -> float:
    if value in {"", None}:
        return default
    return float(value)


def infer_context(path: Path, root: Path, events: list[dict[str, Any]]) -> dict[str, str]:
    rel = path.relative_to(root)
    parts = rel.parts
    offset = 0
    if len(parts) > 1 and parts[0] in {"all", "full_pilot", "stress", "action_balanced"}:
        offset = 1
    provider = parts[offset] if len(parts) > offset else str(events[0].get("provider", "unknown"))
    budget = parts[offset + 1] if len(parts) > offset + 1 else "unknown"
    budget_state = events[0].get("budget", {})
    horizon = str(int(budget_state.get("horizon", 0))) if budget_state.get("horizon") is not None else ""
    planner_idx = offset + 2
    if len(parts) > offset + 2 and parts[offset + 2].startswith("h") and parts[offset + 2][1:].isdigit():
        horizon = parts[offset + 2][1:]
        planner_idx = offset + 3
    planner = parts[planner_idx] if len(parts) > planner_idx else "unknown"
    seed = ""
    seed_idx = planner_idx + 1
    if len(parts) > seed_idx and parts[seed_idx].startswith("seed_"):
        seed = parts[seed_idx].removeprefix("seed_")
    elif events[0].get("seed") is not None:
        seed = str(events[0].get("seed"))
    metadata = events[0].get("subject_metadata", {})
    dataset = str(metadata.get("dataset") or str(events[0].get("subject_id", "unknown")).split("_", 1)[0])
    profile = str(metadata.get("profile") or ("stress" if dataset == "stress" else "full_pilot"))
    return {
        "provider": provider,
        "budget": budget,
        "horizon": horizon,
        "planner": planner,
        "seed": seed,
        "dataset": dataset,
        "profile": profile,
        "model": str(events[0].get("model", "")),
    }


def summarize_episode(events: list[dict[str, Any]]) -> dict[str, Any]:
    total_tokens = 0.0
    total_runtime = 0.0
    total_return = 0.0
    failures = 0
    tests = 0
    duplicates = 0
    invalid_flaky = 0
    coverage = 0.0
    first_failure_step = math.nan
    first_failure_tokens = math.nan
    running_tokens = 0.0
    action_counts: Counter[str] = Counter()
    reward_by_step: dict[int, float] = {}
    for event in events:
        step = int(event.get("step", 0))
        result = event.get("result", {})
        cost = result.get("cost", {})
        tokens = float(cost.get("tokens", 0.0))
        runtime = float(cost.get("runtime_seconds", 0.0))
        running_tokens += tokens
        total_tokens += tokens
        total_runtime += runtime
        reward = float(event.get("reward", 0.0))
        total_return += reward
        reward_by_step[step] = reward_by_step.get(step, 0.0) + reward
        action_counts[str(event.get("action", ""))] += 1
        failures += int(result.get("failures", 0))
        tests += len(result.get("generated_tests", []))
        duplicates += int(result.get("duplicate_tests", 0))
        invalid_flaky += int(result.get("invalid_tests", 0)) + int(result.get("flaky_tests", 0))
        coverage += float(result.get("coverage_delta", 0.0))
        if int(result.get("failures", 0)) > 0 and math.isnan(first_failure_step):
            first_failure_step = float(step)
            first_failure_tokens = running_tokens
    return {
        "subject": str(events[0].get("subject_id", "")),
        "failures": failures,
        "failure_found": 1.0 if failures > 0 else 0.0,
        "ttfm": first_failure_step,
        "tokfm": first_failure_tokens,
        "tokens_used": total_tokens,
        "mp1k": 1000.0 * failures / max(total_tokens, 1.0),
        "mpm": failures / max(total_runtime / 60.0, 1e-6),
        "coverage": min(1.0, coverage),
        "false_reassurance": 1.0 if failures == 0 else 0.0,
        "budget_exhausted": 1.0 if events[-1].get("budget", {}).get("runtime_seconds", 1) <= 0 else 0.0,
        "invalid_flaky_rate": invalid_flaky / max(tests, 1),
        "duplicate_rate": duplicates / max(tests + duplicates, 1),
        "llm_action_share": sum(action_counts[action] for action in {"LLM_SYNTHESIZE_INSTRUCTIONS", "LLM_GENERATE_ASSERTIONS", "REFINE_UNCERTAIN_ORACLE"}) / max(sum(action_counts.values()), 1),
        "return": total_return,
        "steps": len(events),
        "actions": dict(action_counts),
        "reward_by_step": reward_by_step,
    }


def collect(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event_file in sorted(root.rglob("events.jsonl")):
        events = read_jsonl(event_file)
        if not events:
            continue
        context = infer_context(event_file, root, events)
        summary = summarize_episode(events)
        token_budget = BUDGET_TOKENS.get(context["budget"], math.nan)
        summary["token_budget_initial"] = token_budget
        summary["token_budget_used_rate"] = summary["tokens_used"] / token_budget if token_budget and not math.isnan(token_budget) else math.nan
        summary["llm_episode"] = 1.0 if summary["llm_action_share"] > 0 else 0.0
        rows.append({**context, **summary})
    if not rows:
        raise RuntimeError(f"no events.jsonl files found under {root}")
    return rows


def grouped_stats(rows: list[dict[str, Any]], keys: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[k]) for k in keys)].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        record = dict(zip(keys, key))
        record["subjects"] = len({item["subject"] for item in items})
        record["episodes"] = len(items)
        for metric in metrics:
            vals = [safe_float(item.get(metric)) for item in items]
            vals = [v for v in vals if not math.isnan(v)]
            record[f"{metric}_mean"] = mean(vals) if vals else math.nan
            record[f"{metric}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
        out.append(record)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "--"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f):
        return "--"
    return f"{f:.{digits}f}"


def pct(value: Any) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "--"
    if math.isnan(f):
        return "--"
    return f"{100*f:.1f}\\%"


def mean_present(values: list[Any]) -> float:
    vals = [safe_float(value) for value in values]
    vals = [value for value in vals if not math.isnan(value)]
    return mean(vals) if vals else math.nan


def median_present(values: list[Any]) -> float:
    vals = [safe_float(value) for value in values]
    vals = [value for value in vals if not math.isnan(value)]
    return median(vals) if vals else math.nan


def action_share(rows: list[dict[str, Any]], action: str) -> float:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row.get("actions", {}))
    total = sum(counts.values())
    return counts[action] / total if total else math.nan


def llm_share(rows: list[dict[str, Any]]) -> float:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row.get("actions", {}))
    total = sum(counts.values())
    if not total:
        return math.nan
    llm_actions = {"LLM_SYNTHESIZE_INSTRUCTIONS", "LLM_GENERATE_ASSERTIONS", "REFINE_UNCERTAIN_ORACLE"}
    return sum(counts[action] for action in llm_actions) / total


def dominant_action(rows: list[dict[str, Any]]) -> str:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row.get("actions", {}))
    if not counts:
        return "--"
    action, _ = counts.most_common(1)[0]
    return ACTION_LABELS.get(action, action.replace("_", " ").title())


def profile_label(profile: str) -> str:
    return PROFILE_LABELS.get(profile, profile.replace("_", " ").title())


def select_table_blocks(lines: list[str], labels: set[str]) -> list[str]:
    selected: list[str] = ["% Auto-generated by scripts/aggregate_exp.py"]
    block: list[str] = []
    in_table = False
    for line in lines:
        if line.startswith("\\begin{table}"):
            block = [line]
            in_table = True
            continue
        if in_table:
            block.append(line)
            if line.startswith("\\end{table}"):
                text = "\n".join(block)
                if any(f"\\label{{{label}}}" in text for label in labels):
                    selected.extend(block)
                    selected.append("")
                in_table = False
            continue
    return selected


def _main_horizon(rows: list[dict[str, Any]]) -> str:
    horizons = {str(row.get("horizon", "")) for row in rows if str(row.get("horizon", "")).isdigit()}
    if "16" in horizons:
        return "16"
    if horizons:
        return sorted(horizons, key=lambda value: int(value))[-1]
    return ""


def write_tables(out: Path, metrics_rows: list[dict[str, Any]], budget_rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    full_metrics = [row for row in metrics_rows if row.get("profile") != "stress"]
    full_budget_rows = [row for row in budget_rows if row.get("profile") != "stress"]
    full_raw_rows = [row for row in raw_rows if row.get("profile") != "stress"]
    stress_raw_rows = [row for row in raw_rows if row.get("profile") == "stress"]
    main_source = full_metrics or metrics_rows
    main_horizon = _main_horizon(main_source)
    medium = [row for row in full_metrics if row["budget"] == "medium" and (not main_horizon or row.get("horizon") == main_horizon)]
    medium_raw = [row for row in full_raw_rows if row["budget"] == "medium" and (not main_horizon or row.get("horizon") == main_horizon)]
    ordered_methods = ["existing", "random", "fixed", "greedy", "bandit", "dqn"]
    lines = ["% Auto-generated by scripts/aggregate_exp.py"]

    if medium_raw:
        lines += [
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{Full-pilot planner behavior under the medium budget at horizon {main_horizon or 'default'}.}}",
            "\\label{tab:cost-failure}",
            "\\resizebox{\\linewidth}{!}{%",
            "\\scriptsize",
            "\\begin{tabular}{p{0.24\\linewidth}rrrrlr}",
            "\\toprule",
            "Method & Failure-found rate & Median step & Coverage & Duplicate rate & Dominant action & LLM action share\\\\",
            "\\midrule",
        ]
        for method in ordered_methods:
            subset = [row for row in medium_raw if row["planner"] == method]
            if not subset:
                continue
            found = [row for row in subset if not math.isnan(safe_float(row.get("ttfm")))]
            found_rate = len(found) / len(subset)
            median_step = median_present([row["ttfm"] for row in found])
            coverage = mean_present([row.get("coverage") for row in subset])
            duplicate = mean_present([row.get("duplicate_rate") for row in subset])
            lines.append(
                f"{METHOD_LABELS[method]} & {pct(found_rate)} & {fmt(median_step)} & {pct(coverage)} & "
                f"{pct(duplicate)} & {dominant_action(subset)} & {pct(llm_share(subset))}\\\\"
            )
        lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]

    dqn_budget = [row for row in full_budget_rows if row["planner"] == "dqn" and (not main_horizon or row.get("horizon") == main_horizon)]
    if dqn_budget:
        lines += [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Budget usefulness for the masked replay-trained planner at the main horizon. Token budget used indicates whether a nominal budget is binding.}",
            "\\label{tab:learning}",
            "\\resizebox{\\linewidth}{!}{%",
            "\\scriptsize",
            "\\begin{tabular}{p{0.16\\linewidth}rrrrrr}",
            "\\toprule",
            "Budget & Horizon & Failure-found rate & Coverage & Token budget used & LLM action share & Cumulative reward\\\\",
            "\\midrule",
        ]
        for budget in ["low", "medium", "high"]:
            subset = [row for row in dqn_budget if row["budget"] == budget]
            if not subset:
                continue
            ret = mean_present([row.get("return_mean") for row in subset])
            failure_found = mean_present([row.get("failure_found_mean") for row in subset])
            coverage = mean_present([row.get("coverage_mean") for row in subset])
            token_used = mean_present([row.get("token_budget_used_rate_mean") for row in subset])
            llm = mean_present([row.get("llm_action_share_mean") for row in subset])
            lines.append(f"{budget.title()} & {main_horizon or 'default'} & {pct(failure_found)} & {pct(coverage)} & {pct(token_used)} & {pct(llm)} & {fmt(ret)}\\\\")
        lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]

    dqn_budget_horizon = [row for row in full_budget_rows if row["planner"] == "dqn"]
    if dqn_budget_horizon:
        lines += [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Token-budget and horizon sensitivity for the masked replay-trained planner.}",
            "\\label{tab:budget-horizon}",
            "\\resizebox{\\linewidth}{!}{%",
            "\\scriptsize",
            "\\begin{tabular}{p{0.12\\linewidth}rrrrr}",
            "\\toprule",
            "Setting & Failure-found rate & Coverage & Token budget used & Cumulative reward & Episodes\\\\",
            "\\midrule",
        ]
        for budget in ["low", "medium", "high"]:
            for horizon in ["8", "16", "32"]:
                subset = [row for row in dqn_budget_horizon if row["budget"] == budget and row.get("horizon") == horizon]
                if not subset:
                    continue
                episodes = sum(int(row["episodes"]) for row in subset)
                failure_found = mean_present([row.get("failure_found_mean") for row in subset])
                coverage = mean_present([row.get("coverage_mean") for row in subset])
                token_used = mean_present([row.get("token_budget_used_rate_mean") for row in subset])
                ret = mean_present([row.get("return_mean") for row in subset])
                lines.append(f"{budget}/h{horizon} & {pct(failure_found)} & {pct(coverage)} & {pct(token_used)} & {fmt(ret)} & {episodes}\\\\")
        lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]

    dqn_horizons = [row for row in full_budget_rows if row["planner"] == "dqn" and row["budget"] == "medium"]
    if dqn_horizons:
        lines += [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Horizon sensitivity for the masked replay-trained planner under the medium budget.}",
            "\\label{tab:horizon-sensitivity}",
            "\\resizebox{\\linewidth}{!}{%",
            "\\scriptsize",
            "\\begin{tabular}{p{0.29\\linewidth}rrrr}",
            "\\toprule",
            "Horizon & Episodes & Cumulative reward & Coverage & False reassurance\\\\",
            "\\midrule",
        ]
        for horizon in ["8", "16", "32"]:
            subset = [row for row in dqn_horizons if row.get("horizon") == horizon]
            if not subset:
                continue
            episodes = sum(int(row["episodes"]) for row in subset)
            ret = mean_present([row["return_mean"] for row in subset])
            coverage = mean_present([row["coverage_mean"] for row in subset])
            frr = mean_present([row["false_reassurance_mean"] for row in subset])
            lines.append(f"{horizon} & {episodes} & {fmt(ret)} & {pct(coverage)} & {pct(frr)}\\\\")
        lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]

    provider_rows = [row for row in full_budget_rows if row["budget"] == "medium" and row["planner"] == "dqn" and (not main_horizon or row.get("horizon") == main_horizon)]
    if provider_rows:
        lines += [
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{Provider comparison for the masked replay-trained planner under the medium budget at horizon {main_horizon or 'default'}.}}",
            "\\label{tab:provider-comparison}",
            "\\resizebox{\\linewidth}{!}{%",
            "\\scriptsize",
            "\\begin{tabular}{p{0.29\\linewidth}rrrr}",
            "\\toprule",
            "Provider & Episodes & Failure-found rate & Coverage & Cumulative reward\\\\",
            "\\midrule",
        ]
        for provider in sorted({row["provider"] for row in provider_rows}):
            subset = [row for row in provider_rows if row["provider"] == provider]
            episodes = sum(int(row["episodes"]) for row in subset)
            frr = mean_present([row["false_reassurance_mean"] for row in subset])
            coverage = mean_present([row["coverage_mean"] for row in subset])
            ret = mean_present([row["return_mean"] for row in subset])
            lines.append(f"{provider} & {episodes} & {pct(1.0 - frr)} & {pct(coverage)} & {fmt(ret)}\\\\")
        lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]

    if medium:
        lines += [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Planner comparison on available prototype variants under the medium budget.}",
            "\\label{tab:ablations}",
            "\\resizebox{\\linewidth}{!}{%",
            "\\scriptsize",
            "\\begin{tabular}{p{0.32\\linewidth}rrrr}",
            "\\toprule",
            "Planner variant & Coverage & Duplicate tests & Failures / min & Cumulative reward\\\\",
            "\\midrule",
        ]
        for method in ordered_methods:
            subset = [row for row in medium if row["planner"] == method]
            if not subset:
                continue
            coverage = mean_present([row["coverage_mean"] for row in subset])
            duplicate = mean_present([row["duplicate_rate_mean"] for row in subset])
            mpm = mean_present([row["mpm_mean"] for row in subset])
            ret = mean_present([row["return_mean"] for row in subset])
            lines.append(f"{METHOD_LABELS[method]} & {pct(coverage)} & {pct(duplicate)} & {fmt(mpm)} & {fmt(ret)}\\\\")
        lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]

    profiles = sorted({row.get("profile", "full_pilot") for row in raw_rows})
    profile_medium = [row for row in raw_rows if row["budget"] == "medium" and (not main_horizon or row.get("horizon") == main_horizon)]
    if len(profiles) > 1 and profile_medium:
        lines += [
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{Profile-wise planner comparison under the medium budget at horizon {main_horizon or 'default'}.}}",
            "\\label{tab:profile-planner-full}",
            "\\resizebox{\\linewidth}{!}{%",
            "\\scriptsize",
            "\\begin{tabular}{p{0.15\\linewidth}p{0.19\\linewidth}rrrrrr}",
            "\\toprule",
            "Profile & Method & Failure-found rate & Coverage & Duplicate rate & Boundary share & Fuzz share & LLM share\\\\",
            "\\midrule",
        ]
        for profile in ["full_pilot", "stress", "action_balanced"]:
            for method in ordered_methods:
                subset = [row for row in profile_medium if row.get("profile") == profile and row["planner"] == method]
                if not subset:
                    continue
                found = [row for row in subset if not math.isnan(safe_float(row.get("ttfm")))]
                found_rate = len(found) / len(subset)
                coverage = mean_present([row.get("coverage") for row in subset])
                duplicate = mean_present([row.get("duplicate_rate") for row in subset])
                lines.append(
                    f"{profile_label(profile)} & {METHOD_LABELS[method]} & {pct(found_rate)} & {pct(coverage)} & "
                    f"{pct(duplicate)} & {pct(action_share(subset, 'GENERATE_BOUNDARY_TESTS'))} & "
                    f"{pct(action_share(subset, 'PROPERTY_BASED_FUZZ'))} & {pct(llm_share(subset))}\\\\"
                )
        lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]

    stress_horizon = _main_horizon(stress_raw_rows)
    stress_medium = [row for row in stress_raw_rows if row["budget"] == "medium" and (not stress_horizon or row.get("horizon") == stress_horizon)]
    if stress_medium:
        lines += [
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{Stress-profile planner separation under the medium budget at horizon {stress_horizon or 'default'}.}}",
            "\\label{tab:stress-separation}",
            "\\resizebox{\\linewidth}{!}{%",
            "\\scriptsize",
            "\\begin{tabular}{p{0.22\\linewidth}rrrrrrr}",
            "\\toprule",
            "Method & Failure-found rate & Coverage & Duplicate rate & Boundary share & Fuzz share & LLM share & Cumulative reward\\\\",
            "\\midrule",
        ]
        for method in ordered_methods:
            subset = [row for row in stress_medium if row["planner"] == method]
            if not subset:
                continue
            found = [row for row in subset if not math.isnan(safe_float(row.get("ttfm")))]
            found_rate = len(found) / len(subset)
            coverage = mean_present([row.get("coverage") for row in subset])
            duplicate = mean_present([row.get("duplicate_rate") for row in subset])
            ret = mean_present([row["return"] for row in subset])
            lines.append(
                f"{METHOD_LABELS[method]} & {pct(found_rate)} & {pct(coverage)} & {pct(duplicate)} & "
                f"{pct(action_share(subset, 'GENERATE_BOUNDARY_TESTS'))} & "
                f"{pct(action_share(subset, 'PROPERTY_BASED_FUZZ'))} & {pct(llm_share(subset))} & {fmt(ret)}\\\\"
            )
        lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]

    main_lines = select_table_blocks(lines, {"tab:cost-failure"})
    if len(profiles) > 1 and profile_medium:
        main_lines += [
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{Compact profile-wise planner comparison under the medium budget at horizon {main_horizon or 'default'}.}}",
            "\\label{tab:profile-planner}",
            "\\resizebox{\\linewidth}{!}{%",
            "\\scriptsize",
            "\\begin{tabular}{p{0.16\\linewidth}p{0.21\\linewidth}rrrrr}",
            "\\toprule",
            "Profile & Method & Failure-found rate & Coverage & Boundary share & Fuzz share & LLM share\\\\",
            "\\midrule",
        ]
        for profile in ["full_pilot", "stress", "action_balanced"]:
            for method in ["fixed", "greedy", "bandit", "dqn"]:
                subset = [row for row in profile_medium if row.get("profile") == profile and row["planner"] == method]
                if not subset:
                    continue
                found = [row for row in subset if not math.isnan(safe_float(row.get("ttfm")))]
                found_rate = len(found) / len(subset)
                coverage = mean_present([row.get("coverage") for row in subset])
                main_lines.append(
                    f"{profile_label(profile)} & {METHOD_LABELS[method]} & {pct(found_rate)} & {pct(coverage)} & "
                    f"{pct(action_share(subset, 'GENERATE_BOUNDARY_TESTS'))} & "
                    f"{pct(action_share(subset, 'PROPERTY_BASED_FUZZ'))} & {pct(llm_share(subset))}\\\\"
                )
        main_lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]

    appendix_lines = select_table_blocks(
        lines,
        {
            "tab:learning",
            "tab:budget-horizon",
            "tab:horizon-sensitivity",
            "tab:provider-comparison",
            "tab:ablations",
            "tab:profile-planner-full",
            "tab:stress-separation",
        },
    )

    (out / "results_tables.tex").write_text("\n".join(lines), encoding="utf-8")
    (out / "results_tables_main.tex").write_text("\n".join(main_lines), encoding="utf-8")
    (out / "results_tables_appendix.tex").write_text("\n".join(appendix_lines), encoding="utf-8")


def coordinates(rows: list[dict[str, Any]], planner: str, metric: str, budget: str = "medium") -> str:
    pts = []
    for idx, provider in enumerate(sorted({row["provider"] for row in rows}), start=1):
        vals = [safe_float(row[f"{metric}_mean"]) for row in rows if row["planner"] == planner and row["budget"] == budget and row["provider"] == provider]
        if vals:
            pts.append(f"({idx},{mean(vals):.4f})")
    return " ".join(pts)


def write_figures(out: Path, metrics_rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]], fig_dir: Path | None = None) -> None:
    fig_dir = fig_dir or Path("paper-proposal-2/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows = [row for row in metrics_rows if row.get("profile") != "stress"] or metrics_rows
    raw_rows = [row for row in raw_rows if row.get("profile") != "stress"] or raw_rows
    main_horizon = _main_horizon(metrics_rows)
    providers = sorted({row["provider"] for row in metrics_rows if row["budget"] == "medium" and (not main_horizon or row.get("horizon") == main_horizon)})
    provider_labels = ",".join(providers)

    (fig_dir / "failure_efficiency.tex").write_text(
        "\\begin{tikzpicture}\n"
        "\\begin{axis}[width=0.96\\linewidth,height=44mm,ybar,bar width=5pt,"
        "symbolic x coords={%s},xtick=data,ylabel={Failure-found rate},"
        "legend style={font=\\tiny,draw=none,at={(0.5,1.06)},anchor=south,legend columns=2},"
        "ymin=0,ymax=1,ytick={0,0.25,0.5,0.75,1.0},yticklabels={0,25,50,75,100},"
        "ymajorgrids,grid style={black!10}]\n" % provider_labels
        + "\n".join(
            "\\addplot coordinates {%s};\\addlegendentry{%s}" % (
                " ".join(
                    f"({provider},{mean([1.0 - safe_float(row['false_reassurance']) for row in raw_rows if row['budget']=='medium' and row['provider']==provider and row['planner']==planner and (not main_horizon or row.get('horizon') == main_horizon)] or [0]):.4f})"
                    for provider in providers
                ),
                METHOD_LABELS[planner],
            )
            for planner in ["fixed", "random", "bandit", "dqn"]
        )
        + "\n\\end{axis}\n\\end{tikzpicture}\n",
        encoding="utf-8",
    )

    reward_group: defaultdict[tuple[str, int], list[float]] = defaultdict(list)
    for row in raw_rows:
        if row["budget"] != "medium" or row["planner"] not in PLOT_METHODS or (main_horizon and row.get("horizon") != main_horizon):
            continue
        cumulative = 0.0
        horizon = int(main_horizon or row.get("steps", 8) or 8)
        for step in range(horizon):
            cumulative += float(row["reward_by_step"].get(step, 0.0))
            reward_group[(row["planner"], step)].append(cumulative)
    horizon = int(main_horizon or 8)
    (fig_dir / "reward_learning_curve.tex").write_text(
        "\\begin{tikzpicture}\n"
        f"\\begin{{axis}}[width=0.80\\linewidth,height=40mm,font=\\scriptsize,xlabel={{Step}},ylabel={{Cumulative reward}},xmin=1,xmax={horizon},xtick={{1,4,8,12,16}},"
        "legend style={font=\\tiny,draw=none,at={(0.5,-0.28)},anchor=north,legend columns=1},"
        "ymajorgrids,grid style={black!10},mark options={solid},mark size=1.4pt]\n"
        + "\n".join(
            "\\addplot+[mark=%s,thick] coordinates {%s};\\addlegendentry{%s}" % (
                mark,
                " ".join(
                    f"({step + 1},{mean(reward_group.get((planner, step), [0.0])):.3f})"
                    for step in range(horizon)
                ),
                METHOD_LABELS[planner],
            )
            for planner, mark in [("fixed", "*"), ("random", "square*"), ("bandit", "triangle*"), ("dqn", "diamond*")]
        )
        + "\n\\end{axis}\n\\end{tikzpicture}\n",
        encoding="utf-8",
    )

    action_rows: list[dict[str, Any]] = []
    for planner in ["fixed", "random", "bandit", "dqn"]:
        counts: Counter[str] = Counter()
        for row in raw_rows:
            if row["budget"] == "medium" and row["planner"] == planner and (not main_horizon or row.get("horizon") == main_horizon):
                counts.update(row["actions"])
        total = sum(counts.values()) or 1
        action_rows.append({"planner": planner, **{action: counts[action] / total for action in ACTION_ORDER}})
    action_tex = [
        "\\begin{tikzpicture}",
        "\\begin{axis}[width=0.82\\linewidth,height=40mm,font=\\scriptsize,ybar stacked,bar width=7pt,",
        "symbolic x coords={Fixed,Random,Bandit,Replay},xtick=data,ylabel={Action share},",
        "legend style={font=\\tiny,draw=none,at={(0.5,-0.28)},anchor=north,legend columns=2},",
        "ymin=0,ymax=1,ytick={0,0.25,0.5,0.75,1.0},yticklabels={0,25,50,75,100},ymajorgrids,grid style={black!10}]",
    ]
    xlabels = {"fixed": "Fixed", "random": "Random", "bandit": "Bandit", "dqn": "Replay"}
    for action in ACTION_ORDER:
        coords = " ".join(f"({xlabels[row['planner']]},{row[action]:.4f})" for row in action_rows)
        action_tex.append(f"\\addplot coordinates {{{coords}}};\\addlegendentry{{{ACTION_LABELS[action]}}}")
    action_tex += ["\\end{axis}", "\\end{tikzpicture}", ""]
    (fig_dir / "action_distribution.tex").write_text("\n".join(action_tex), encoding="utf-8")


def write_summary(out: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# TestGenRL Experiment Summary",
        "",
        f"- Event episodes aggregated: {len(rows)}",
        f"- Providers: {', '.join(sorted({row['provider'] for row in rows}))}",
        f"- Profiles: {', '.join(sorted({row.get('profile', 'full_pilot') for row in rows}))}",
        f"- Datasets: {', '.join(sorted({row['dataset'] for row in rows}))}",
        f"- Budgets: {', '.join(sorted({row['budget'] for row in rows}))}",
        f"- Horizons: {', '.join(sorted({row['horizon'] for row in rows}))}",
        f"- Planners: {', '.join(sorted({row['planner'] for row in rows}))}",
        "",
        "All values are computed from `events.jsonl` logs; no synthetic numeric values are used.",
    ]
    (out / "results_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/exp")
    parser.add_argument("--out", default="paper_tables/exp")
    parser.add_argument("--figures", default="paper-proposal-2/figures")
    parser.add_argument("--providers", nargs="+")
    args = parser.parse_args(argv)

    root = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = collect(root)
    if args.providers:
        allowed = set(args.providers)
        rows = [row for row in rows if row["provider"] in allowed]
        if not rows:
            raise RuntimeError(f"no rows left after provider filter: {sorted(allowed)}")
    metrics = [
        "failure_found",
        "ttfm",
        "tokfm",
        "tokens_used",
        "token_budget_used_rate",
        "mp1k",
        "mpm",
        "coverage",
        "false_reassurance",
        "budget_exhausted",
        "duplicate_rate",
        "llm_action_share",
        "return",
    ]
    metrics_rows = grouped_stats(rows, ["profile", "provider", "model", "dataset", "budget", "horizon", "planner", "seed"], metrics)
    budget_rows = grouped_stats(rows, ["profile", "provider", "model", "dataset", "budget", "horizon", "planner"], metrics)
    budget_horizon_rows = grouped_stats(
        [row for row in rows if row.get("profile") != "stress" and row["planner"] == "dqn"],
        ["profile", "budget", "horizon", "planner"],
        metrics,
    )
    llm_heavy_rows = grouped_stats(
        [row for row in rows if row.get("profile") != "stress" and safe_float(row.get("llm_episode")) > 0],
        ["profile", "budget", "horizon", "planner"],
        metrics,
    )
    action_rows = []
    for row in rows:
        for action, count in row["actions"].items():
            action_rows.append({k: row[k] for k in ["profile", "provider", "model", "dataset", "budget", "horizon", "planner", "seed", "subject"]} | {"action": action, "count": count})
    profile_action_totals: defaultdict[tuple[str, str, str, str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = (row.get("profile", "full_pilot"), row["budget"], row["horizon"], row["planner"], row["provider"])
        profile_action_totals[key].update(row["actions"])
    profile_action_rows = []
    for (profile, budget, horizon, planner, provider), counts in sorted(profile_action_totals.items()):
        total = sum(counts.values()) or 1
        for action in ACTION_ORDER:
            profile_action_rows.append(
                {
                    "profile": profile,
                    "provider": provider,
                    "budget": budget,
                    "horizon": horizon,
                    "planner": planner,
                    "action": action,
                    "count": counts[action],
                    "share": counts[action] / total,
                }
            )
    write_csv(out / "metrics_by_method.csv", metrics_rows)
    write_csv(out / "budget_sensitivity.csv", budget_rows)
    write_csv(out / "budget_horizon_sensitivity.csv", budget_horizon_rows)
    write_csv(out / "llm_heavy_budget_sensitivity.csv", llm_heavy_rows)
    write_csv(out / "action_distribution.csv", action_rows)
    write_csv(out / "profile_planner_comparison.csv", budget_rows)
    write_csv(out / "profile_action_distribution.csv", profile_action_rows)
    write_tables(out, metrics_rows, budget_rows, rows)
    write_figures(out, metrics_rows, rows, Path(args.figures))
    write_summary(out, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
