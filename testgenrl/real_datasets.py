"""Import real Python benchmark datasets into TestGenRL subjects."""

from __future__ import annotations

import ast
import gzip
import json
import re
import shutil
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import Any

from .dataset_catalog import BUGSINPY_PILOT_TASKS
from .models import ProgramSubject


HUMANEVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
MBPP_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json"
BUGSINPY_URL = "https://github.com/soarsmu/BugsInPy"

REAL_DATASETS = {
    "humaneval_real",
    "mbpp_real",
    "bugsinpy_real",
    "real_python",
    "real_python_smoke",
    "real_python_full",
    "real_python_strict",
}


@dataclass
class ImportReport:
    dataset: str
    imported: int = 0
    skipped: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)

    def add_skip(self, reason: str, item_id: str | int | None = None) -> None:
        self.skipped += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1
        if item_id is not None:
            self.records.append({"id": str(item_id), "status": "skipped", "reason": reason})

    def add_import(self, subject: ProgramSubject) -> None:
        self.imported += 1
        self.records.append({"id": subject.id, "status": "imported", "dataset": subject.metadata.get("dataset")})

    def to_json(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "imported": self.imported,
            "skipped": self.skipped,
            "skip_reasons": self.skip_reasons,
            "records": self.records,
        }


def _split_for_index(idx: int, total: int) -> str:
    train_cut = max(1, int(total * 0.6))
    val_cut = max(train_cut + 1, int(total * 0.8))
    if idx < train_cut:
        return "train"
    if idx < val_cut:
        return "val"
    return "test"


def _fetch(url: str, cache_dir: Path, filename: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / filename
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    return path


def _find_source_file(source_root: Path | None, names: list[str]) -> Path | None:
    if source_root is None:
        return None
    for name in names:
        candidate = source_root / name
        if candidate.exists():
            return candidate
    for name in names:
        matches = sorted(source_root.rglob(name))
        if matches:
            return matches[0]
    return None


def _safe_id(prefix: str, raw: object) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(raw)).strip("_").lower()
    if text.startswith(prefix):
        return text
    return f"{prefix}_{text}"


def _function_names(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]


def _write_subject(out: Path, subject: ProgramSubject) -> dict[str, Any]:
    subject_dir = out / subject.id
    subject_dir.mkdir(parents=True, exist_ok=True)
    source_path = subject_dir / "solution.py"
    source_path.write_text(dedent(subject.metadata.get("source_code", "")), encoding="utf-8")
    subject.path = source_path
    record = subject.to_json(out)
    (subject_dir / "subject.json").write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return record


def _validate_subject(subject: ProgramSubject) -> tuple[bool, str | None]:
    from .executors import _run_cases

    result = _run_cases(subject, subject.tests)
    if result.invalid_tests:
        return False, str(result.observations.get("oracle_error") or result.observations.get("load_error") or "invalid tests")
    if result.failures and not subject.metadata.get("known_bug"):
        return False, "reference solution fails oracle"
    return True, None


def _make_check_oracle(assertions: list[str], imports: list[str] | None, entrypoint: str) -> str:
    lines: list[str] = []
    lines.extend(imports or [])
    lines.append("")
    lines.append("def check(candidate):")
    lines.append(f"    {entrypoint} = candidate")
    for assertion in assertions:
        lines.append("    " + assertion)
    return "\n".join(lines) + "\n"


def _read_humaneval(source_root: Path | None, cache_dir: Path) -> list[dict[str, Any]]:
    path = _find_source_file(source_root, ["HumanEval.jsonl.gz", "human-eval-v2-20210705.jsonl.gz", "HumanEval.jsonl"])
    if path is None:
        path = _fetch(HUMANEVAL_URL, cache_dir, "HumanEval.jsonl.gz")
    raw = gzip.decompress(path.read_bytes()).decode("utf-8") if path.suffix == ".gz" else path.read_text(encoding="utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def _read_mbpp(source_root: Path | None, cache_dir: Path) -> list[dict[str, Any]]:
    path = _find_source_file(source_root, ["sanitized-mbpp.json", "mbpp.json"])
    if path is None:
        path = _fetch(MBPP_URL, cache_dir, "sanitized-mbpp.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _subject_from_humaneval(item: dict[str, Any], split: str) -> ProgramSubject | None:
    task_id = item.get("task_id")
    entrypoint = item.get("entry_point")
    prompt = item.get("prompt")
    solution = item.get("canonical_solution")
    oracle = item.get("test")
    if not all(isinstance(v, str) and v for v in [task_id, entrypoint, prompt, solution, oracle]):
        return None
    source = prompt + solution
    return ProgramSubject(
        id=_safe_id("humaneval_real", task_id),
        language="python",
        path=Path("solution.py"),
        entrypoint=entrypoint,
        tests=[{"oracle": "check"}],
        metadata={
            "dataset": "humaneval_real",
            "profile": "real_python",
            "split": split,
            "original_id": task_id,
            "source": "openai_human_eval",
            "source_url": HUMANEVAL_URL,
            "license": "dataset-specific; see source repository",
            "oracle_type": "callable_check",
            "oracle_code": oracle,
            "import_status": "imported",
            "source_code": source,
        },
    )


def _subject_from_mbpp(item: dict[str, Any], split: str) -> ProgramSubject | None:
    task_id = item.get("task_id")
    code = item.get("code")
    tests = item.get("test_list")
    imports = item.get("test_imports") or []
    if not isinstance(code, str) or not isinstance(tests, list) or not tests:
        return None
    names = _function_names(code)
    entrypoint = names[0] if names else None
    if not entrypoint:
        return None
    assertions = [test for test in tests if isinstance(test, str) and test.strip().startswith("assert ")]
    if not assertions:
        return None
    return ProgramSubject(
        id=_safe_id("mbpp_real", task_id),
        language="python",
        path=Path("solution.py"),
        entrypoint=entrypoint,
        tests=[{"oracle": "check"}],
        metadata={
            "dataset": "mbpp_real",
            "profile": "real_python",
            "split": split,
            "original_id": task_id,
            "source": "google_research_mbpp",
            "source_url": MBPP_URL,
            "license": "dataset-specific; see source repository",
            "oracle_type": "callable_check",
            "oracle_code": _make_check_oracle(assertions, imports, entrypoint),
            "import_status": "imported",
            "prompt": item.get("prompt", ""),
            "source_code": code,
        },
    )


def _subjects_from_bugsinpy(limit: int | None, offset: int, report: ImportReport) -> list[ProgramSubject]:
    subjects: list[ProgramSubject] = []
    selected = BUGSINPY_PILOT_TASKS[: limit or len(BUGSINPY_PILOT_TASKS)]
    total = len(selected)
    for idx, (task_id, source, entrypoint, tests) in enumerate(selected):
        subjects.append(
            ProgramSubject(
                id=_safe_id("bugsinpy_real", task_id),
                language="python",
                path=Path("solution.py"),
                entrypoint=entrypoint,
                tests=tests,
                metadata={
                    "dataset": "bugsinpy_real",
                    "profile": "real_python",
                    "split": _split_for_index(offset + idx, max(total + offset, 1)),
                    "original_id": task_id,
                    "source": "embedded_bugsinpy_compatible_subset",
                    "source_url": BUGSINPY_URL,
                    "license": "dataset-specific; see source repository",
                    "oracle_type": "args_expected",
                    "known_bug": True,
                    "import_status": "curated_function_level_proxy",
                    "source_code": source,
                },
            )
        )
    if not subjects:
        report.add_skip("no function-level BugsInPy-compatible subjects available")
    return subjects


def _import_humaneval(source_root: Path | None, cache_dir: Path, limit: int | None, offset: int, report: ImportReport) -> list[ProgramSubject]:
    rows = _read_humaneval(source_root, cache_dir)
    selected = rows[:limit] if limit is not None else rows
    subjects: list[ProgramSubject] = []
    for idx, item in enumerate(selected):
        subject = _subject_from_humaneval(item, _split_for_index(offset + idx, max(len(selected) + offset, 1)))
        if subject is None:
            report.add_skip("missing required HumanEval fields", item.get("task_id"))
        else:
            subjects.append(subject)
    return subjects


def _import_mbpp(source_root: Path | None, cache_dir: Path, limit: int | None, offset: int, report: ImportReport) -> list[ProgramSubject]:
    rows = _read_mbpp(source_root, cache_dir)
    selected = rows[:limit] if limit is not None else rows
    subjects: list[ProgramSubject] = []
    for idx, item in enumerate(selected):
        subject = _subject_from_mbpp(item, _split_for_index(offset + idx, max(len(selected) + offset, 1)))
        if subject is None:
            report.add_skip("unsupported MBPP record shape", item.get("task_id"))
        else:
            subjects.append(subject)
    return subjects


def prepare_real_dataset(
    out: Path,
    dataset: str,
    limit: int | None = None,
    limit_per_source: int | None = None,
    source_root: Path | None = None,
    cache_dir: Path | None = None,
    validate: bool = False,
    clean: bool = True,
) -> Path:
    if dataset not in REAL_DATASETS:
        raise ValueError(f"unsupported real dataset {dataset!r}")
    if clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    cache = cache_dir or (out / ".raw_cache")
    report = ImportReport(dataset=dataset)

    subjects: list[ProgramSubject] = []
    if dataset == "humaneval_real":
        subjects = _import_humaneval(source_root, cache, limit, 0, report)
    elif dataset == "mbpp_real":
        subjects = _import_mbpp(source_root, cache, limit, 0, report)
    elif dataset == "bugsinpy_real":
        subjects = _subjects_from_bugsinpy(limit, 0, report)
    else:
        per_source = limit_per_source
        if per_source is None:
            per_source = 20 if dataset in {"real_python", "real_python_smoke"} else None
        he_limit = per_source
        mbpp_limit = per_source
        subjects.extend(_import_humaneval(source_root, cache, he_limit, len(subjects), report))
        subjects.extend(_import_mbpp(source_root, cache, mbpp_limit, len(subjects), report))
        if dataset != "real_python_strict":
            bugs_limit = 10 if per_source is None else min(10, per_source)
            subjects.extend(_subjects_from_bugsinpy(bugs_limit, len(subjects), report))
        if limit is not None:
            subjects = subjects[:limit]

    manifest: list[dict[str, Any]] = []
    for subject in subjects:
        record = _write_subject(out, subject)
        concrete = ProgramSubject.from_json(record, out)
        if validate:
            ok, reason = _validate_subject(concrete)
            if not ok:
                report.add_skip(f"validation failed: {reason}", concrete.id)
                shutil.rmtree(out / concrete.id, ignore_errors=True)
                continue
        report.add_import(concrete)
        manifest.append(record)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (out / "import_report.json").write_text(json.dumps(report.to_json(), indent=2, sort_keys=True), encoding="utf-8")
    return out / "manifest.json"
