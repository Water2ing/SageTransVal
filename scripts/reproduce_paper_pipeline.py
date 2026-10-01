"""Audit the staged paper evidence and run the missing C-to-Rust v2 study.

All new outputs go below ``artifact/reproduction``. This script never edits the
manuscript or the staged result tables. Existing translation snapshots may be
supplied to avoid making new provider requests.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sagetransval.full_pipeline import load_manifest, translate_subjects, validate_subjects, write_repair_packets
MODES = (
    "existing",
    "random",
    "direct_llm_tests",
    "source_obligation_guided",
    "source_obligation_guided_repair",
)
ABLATIONS = (
    "source_obligation_guided",
    "no_boundary_obligations",
    "no_branch_obligations",
    "no_exception_obligations",
)
PROVIDERS = {"openai": "gpt-4o-mini", "gemini": "gemini-2.5-flash-lite"}
PAPER_PJ = {"existing": 70.5, "random": 74.5, "direct_llm_tests": 81.0,
            "source_obligation_guided": 79.0, "source_obligation_guided_repair": 79.0}
PAPER_CR = {"existing": 5.0, "random": 8.3, "direct_llm_tests": 10.0,
            "source_obligation_guided": 11.7, "source_obligation_guided_repair": 11.7}
PAPER_POLICY_OFF = {"A_integer": 63.6, "C_error": 25.0, "D_strings": 33.3, "ALL": 41.7}
PACKET_FIELDS = (
    "source_span", "target_span", "normalized_input", "source_observation",
    "target_observation", "obligation_id", "error_class", "normalization_policy",
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def mode_summary(rows: list[dict], modes: tuple[str, ...]) -> dict[str, dict]:
    result = {}
    for mode in modes:
        subset = [r for r in rows if r.get("mode") == mode]
        result[mode] = {
            "trials": len(subset),
            "detected": sum(bool(r.get("detected")) for r in subset),
            "detected_pct": round(100 * sum(bool(r.get("detected")) for r in subset) / len(subset), 3) if subset else None,
            "tests_per_program": round(sum(r.get("tests", 0) for r in subset) / len(subset), 3) if subset else None,
            "coverage_pct": round(100 * sum(r.get("obligation_coverage", 0) for r in subset) / len(subset), 3) if subset else None,
        }
    return result


def check_grid(rows: list[dict], subjects: set[str], modes: tuple[str, ...]) -> dict:
    expected = {(s, p, m) for s in subjects for p in PROVIDERS for m in modes}
    counts = Counter((r.get("subject"), r.get("provider"), r.get("mode")) for r in rows)
    actual = set(counts)
    return {
        "expected_rows": len(expected), "actual_rows": len(rows),
        "missing": len(expected - actual), "unexpected": len(actual - expected),
        "duplicates": sum(n - 1 for n in counts.values() if n > 1),
        "complete": actual == expected and all(n == 1 for n in counts.values()),
    }


def compare(summary: dict[str, dict], paper: dict[str, float]) -> dict:
    return {mode: {"paper_pct": expected, "observed_pct": summary[mode]["detected_pct"],
                   "matches_one_decimal": summary[mode]["detected_pct"] is not None and
                   round(summary[mode]["detected_pct"], 1) == expected}
            for mode, expected in paper.items()}


def category(subject: str) -> str | None:
    for prefix, label in (("c_rust_v2_A_", "A_integer"), ("c_rust_v2_C_", "C_error"),
                          ("c_rust_v2_D_", "D_strings")):
        if subject.startswith(prefix):
            return label
    return None


def category_summary(rows: list[dict]) -> dict[str, dict]:
    return {label: mode_summary([r for r in rows if label == "ALL" or category(r.get("subject", "")) == label],
                                ("source_obligation_guided",))["source_obligation_guided"]
            for label in ("A_integer", "C_error", "D_strings", "ALL")}


def category_mode_summary(rows: list[dict]) -> dict[str, dict[str, dict]]:
    return {label: mode_summary([r for r in rows if category(r.get("subject", "")) == label], MODES)
            for label in ("A_integer", "C_error", "D_strings")}


def audit(out: Path) -> dict:
    pj_root = ROOT / "results/proposal1_full_200"
    pj = [row for provider in PROVIDERS for row in read_jsonl(pj_root / provider / "python_java/validation_events.jsonl")]
    pj_subjects = {s.id for s in load_manifest(ROOT / "runs/proposal1_translation/openai/python_java")}
    pj_summary = mode_summary(pj, MODES + ABLATIONS[1:])

    cr_root = ROOT / "data/c_rust_v2_run/subjects"
    cr_subjects = {s.id for s in load_manifest(cr_root)}
    cr_result = ROOT / "results/proposal1_c_rust_v2"
    cr = read_jsonl(cr_result / "validation_events.jsonl")
    cr_off = read_jsonl(cr_result / "validation_events_ubsan_off.jsonl")
    cr_summary = mode_summary(cr, MODES)
    off_summary = category_summary(cr_off) if cr_off else None

    repair_rows = [r for r in pj if r.get("mode") == "source_obligation_guided_repair" and r.get("repair_packet")]
    pj_packets = list((ROOT / "runs").rglob("repair_packets.jsonl"))
    packets = [p for f in pj_packets for p in read_jsonl(f)]
    replay_root = ROOT / "reproduction/pj_repair_spans"
    replay_rows = [r for provider in PROVIDERS for r in read_jsonl(replay_root / provider / "validation_events.jsonl")]
    replay_packets = [p for provider in PROVIDERS for p in read_jsonl(replay_root / f"{provider}_packets/repair_packets.jsonl")]
    original_repair = {(r["subject"], r["provider"]): bool(r["detected"]) for r in pj if r.get("mode") == "source_obligation_guided_repair"}
    replay_repair = {(r["subject"], r["provider"]): bool(r["detected"]) for r in replay_rows}
    report = {
        "python_java": {
            "grid": check_grid(pj, pj_subjects, MODES + ABLATIONS[1:]),
            "modes": pj_summary,
            "paper_comparison": compare(pj_summary, PAPER_PJ),
            "ablation_python_java_only": {m: pj_summary[m] for m in ABLATIONS},
            "repair_flags": len(repair_rows),
            "repair_rows_with_mismatch_examples": sum(bool(r.get("mismatch_examples")) for r in repair_rows),
            "packet_files": [str(f.relative_to(ROOT)) for f in pj_packets],
            "packets_with_all_fields_non_null": sum(all(p.get(k) is not None for k in PACKET_FIELDS) for p in packets),
            "repair_revalidation": {"rows": len(replay_rows), "detected": sum(replay_repair.values()),
                "detection_agrees_with_staged": replay_repair == original_repair,
                "packets": len(replay_packets),
                "packets_with_all_fields_non_null": sum(all(p.get(k) is not None for k in PACKET_FIELDS) for p in replay_packets),
                "target_file_span_fallbacks": sum(p.get("target_span", {}).get("granularity") == "file" for p in replay_packets)},
        },
        "c_rust_v2": {
            "source_subjects": len(cr_subjects),
            "source_categories": dict(Counter(category(s) for s in cr_subjects)),
            "primary_grid": check_grid(cr, cr_subjects, MODES),
            "primary_modes": cr_summary,
            "paper_comparison": compare(cr_summary, PAPER_CR),
            "policy_off_grid": check_grid(cr_off, cr_subjects, ("source_obligation_guided",)),
            "policy_off_by_category": off_summary,
            "policy_off_paper_comparison": (
                {k: {"paper_pct": v, "observed_pct": off_summary[k]["detected_pct"],
                     "matches_one_decimal": round(off_summary[k]["detected_pct"], 1) == v}
                 for k, v in PAPER_POLICY_OFF.items()} if off_summary else None),
        },
    }
    write_json(out, report)
    return report


def preflight(translations: dict[str, Path | None]) -> list[str]:
    missing = []
    if not (shutil.which("clang") or shutil.which("gcc")):
        missing.append("C compiler (clang or gcc)")
    if not shutil.which("rustc"):
        missing.append("rustc")
    for provider, path in translations.items():
        if path is None and not os.environ.get("OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY"):
            missing.append(f"{provider} translation snapshot or API key")
        if path is not None and not (path / "manifest.json").is_file():
            missing.append(f"{provider} translation manifest at {path}")
    return missing


def validated_translation_root(provider: str, path: Path, subject_ids: set[str]) -> Path:
    subjects = load_manifest(path)
    if {s.id for s in subjects} != subject_ids or len(subjects) != len(subject_ids):
        raise ValueError(f"{provider}: translation manifest must have exactly the 30 v2 subjects")
    if any(s.metadata.get("provider") != provider or s.metadata.get("model") != PROVIDERS[provider]
           or s.source_language != "c" or s.target_language != "rust"
           or not s.target_path or not Path(s.target_path).is_file()
           or not Path(s.source_path).is_file() for s in subjects):
        raise ValueError(f"{provider}: manifest model, language, or translated files are incomplete")
    return path


def run_crust(out: Path, translations: dict[str, Path | None]) -> dict:
    issues = preflight(translations)
    if issues:
        raise RuntimeError("Missing prerequisites: " + "; ".join(issues))
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing reproduction directory: {out}")
    subjects_root = ROOT / "data/c_rust_v2_run/subjects"
    subjects = load_manifest(subjects_root)
    ids = {s.id for s in subjects}
    if len(subjects) != 30 or len(ids) != 30:
        raise ValueError("Expected exactly 30 distinct C-to-Rust v2 source subjects")
    out.mkdir(parents=True)
    primary, off = [], []
    for provider, model in PROVIDERS.items():
        translated = translations[provider]
        if translated is None:
            translated = out / "translations" / provider
            translate_subjects(subjects_root, translated, provider=provider, model=model, cache_mode="live")
        translated = validated_translation_root(provider, translated, ids)
        old = os.environ.get("SAGETRANSVAL_C_UBSAN")
        try:
            os.environ["SAGETRANSVAL_C_UBSAN"] = "1"
            primary.extend(validate_subjects(translated, out / "results" / provider / "primary", modes=list(MODES)))
            os.environ["SAGETRANSVAL_C_UBSAN"] = "0"
            off.extend(validate_subjects(translated, out / "results" / provider / "ubsan_off",
                                         modes=["source_obligation_guided"]))
        finally:
            if old is None:
                os.environ.pop("SAGETRANSVAL_C_UBSAN", None)
            else:
                os.environ["SAGETRANSVAL_C_UBSAN"] = old
        write_repair_packets(out / "results" / provider / "primary", out / "packets" / provider)
    result = out / "combined"
    result.mkdir()
    for name, rows in (("validation_events.jsonl", primary), ("validation_events_ubsan_off.jsonl", off)):
        (result / name).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    if not check_grid(primary, ids, MODES)["complete"] or not check_grid(off, ids, ("source_obligation_guided",))["complete"]:
        raise RuntimeError("C/Rust v2 event grid is incomplete; see the per-provider logs")
    summary = {
        "primary_grid": check_grid(primary, ids, MODES),
        "primary_modes": mode_summary(primary, MODES),
        "primary_paper_comparison": compare(mode_summary(primary, MODES), PAPER_CR),
        "policy_off_grid": check_grid(off, ids, ("source_obligation_guided",)),
        "primary_by_category": category_mode_summary(primary),
        "policy_on_by_category": category_summary(primary),
        "policy_off_by_category": category_summary(off),
        "policy_off_paper_comparison": {label: {"paper_pct": pct,
            "observed_pct": category_summary(off)[label]["detected_pct"],
            "matches_one_decimal": round(category_summary(off)[label]["detected_pct"], 1) == pct}
            for label, pct in PAPER_POLICY_OFF.items()},
        "packet_quality": {provider: {"count": len(packets),
            "all_fields_non_null": sum(all(p.get(k) is not None for k in PACKET_FIELDS) for p in packets)}
            for provider in PROVIDERS for packets in [[*read_jsonl(out / "packets" / provider / "repair_packets.jsonl")]]},
        "published_numbers_preserved": True,
    }
    write_json(out / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit_cmd = sub.add_parser("audit", help="check staged evidence without editing paper tables")
    audit_cmd.add_argument("--out", type=Path, default=ROOT / "reproduction/audit.json")
    run_cmd = sub.add_parser("run-crust-v2", help="run missing primary and UB-policy sensitivity experiments")
    run_cmd.add_argument("--out", type=Path, default=ROOT / "reproduction/c_rust_v2")
    run_cmd.add_argument("--openai-translations", type=Path)
    run_cmd.add_argument("--gemini-translations", type=Path)
    args = parser.parse_args()
    if args.command == "audit":
        result = audit(args.out)
        print(f"Audit written to {args.out}")
        print(f"Python/Java grid complete: {result['python_java']['grid']['complete']}")
        print(f"C/Rust v2 primary grid complete: {result['c_rust_v2']['primary_grid']['complete']}")
        return 0
    translations = {"openai": args.openai_translations, "gemini": args.gemini_translations}
    try:
        result = run_crust(args.out, translations)
    except (RuntimeError, ValueError, FileExistsError) as exc:
        parser.exit(2, f"Cannot run C/Rust v2: {exc}\n")
    print(f"C/Rust v2 reproduction written to {args.out}")
    print(f"Primary grid complete: {result['primary_grid']['complete']}; policy-off grid complete: {result['policy_off_grid']['complete']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
