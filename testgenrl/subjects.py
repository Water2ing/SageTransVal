"""Subject loading and benchmark preparation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from textwrap import dedent

from .dataset_catalog import ACTION_BALANCED_PILOT_TASKS, BUGSINPY_PILOT_TASKS, HUMANEVAL_PILOT_TASKS, MBPP_PILOT_TASKS, STRESS_PILOT_TASKS, TaskSpec
from .models import ProgramSubject
from .real_datasets import REAL_DATASETS, prepare_real_dataset


TOY_TASKS: list[tuple[str, str, str, list[dict[str, object]]]] = [
    ("abs_val", "def solve(x: int) -> int:\n    return x if x >= 0 else -x\n", "solve", [{"args": [1], "expected": 1}, {"args": [-2], "expected": 2}]),
    ("is_even", "def solve(x: int) -> bool:\n    return x % 2 == 0\n", "solve", [{"args": [2], "expected": True}, {"args": [3], "expected": False}]),
    ("max2", "def solve(a: int, b: int) -> int:\n    return a if a >= b else b\n", "solve", [{"args": [1, 2], "expected": 2}]),
    ("min2", "def solve(a: int, b: int) -> int:\n    return a if a <= b else b\n", "solve", [{"args": [1, 2], "expected": 1}]),
    ("clamp10", "def solve(x: int) -> int:\n    return 0 if x < 0 else (10 if x > 10 else x)\n", "solve", [{"args": [-1], "expected": 0}, {"args": [11], "expected": 10}]),
    ("factorial_buggy", "def solve(n: int) -> int:\n    out = 1\n    for i in range(1, n):\n        out *= i\n    return out\n", "solve", [{"args": [3], "expected": 6}]),
    ("reverse", "def solve(s: str) -> str:\n    return s[::-1]\n", "solve", [{"args": ["ab"], "expected": "ba"}]),
    ("palindrome", "def solve(s: str) -> bool:\n    return s == s[::-1]\n", "solve", [{"args": ["aba"], "expected": True}, {"args": ["abc"], "expected": False}]),
    ("count_a", "def solve(s: str) -> int:\n    return s.count('a')\n", "solve", [{"args": ["banana"], "expected": 3}]),
    ("first_char_buggy", "def solve(s: str) -> str:\n    return s[0]\n", "solve", [{"args": [""], "expected": ""}]),
    ("safe_div", "def solve(a: int, b: int) -> int:\n    return 0 if b == 0 else a // b\n", "solve", [{"args": [4, 2], "expected": 2}, {"args": [4, 0], "expected": 0}]),
    ("sum_list", "def solve(xs: list[int]) -> int:\n    return sum(xs)\n", "solve", [{"args": [[1, 2]], "expected": 3}]),
    ("len_list", "def solve(xs: list[int]) -> int:\n    return len(xs)\n", "solve", [{"args": [[1, 2]], "expected": 2}]),
    ("contains_zero", "def solve(xs: list[int]) -> bool:\n    return 0 in xs\n", "solve", [{"args": [[1, 0]], "expected": True}]),
    ("sort_list", "def solve(xs: list[int]) -> list[int]:\n    return sorted(xs)\n", "solve", [{"args": [[2, 1]], "expected": [1, 2]}]),
    ("dedupe_buggy", "def solve(xs: list[int]) -> list[int]:\n    return list(set(xs))\n", "solve", [{"args": [[2, 1, 2]], "expected": [2, 1]}]),
    ("sign", "def solve(x: int) -> int:\n    return -1 if x < 0 else (1 if x > 0 else 0)\n", "solve", [{"args": [0], "expected": 0}]),
    ("square", "def solve(x: int) -> int:\n    return x * x\n", "solve", [{"args": [3], "expected": 9}]),
    ("starts_with_a", "def solve(s: str) -> bool:\n    return s.startswith('a')\n", "solve", [{"args": ["abc"], "expected": True}]),
    ("middle_buggy", "def solve(xs: list[int]) -> int:\n    return xs[len(xs)//2]\n", "solve", [{"args": [[]], "expected": 0}]),
]


DATASETS: dict[str, list[TaskSpec]] = {
    "toy": TOY_TASKS,
    "humaneval": HUMANEVAL_PILOT_TASKS,
    "mbpp": MBPP_PILOT_TASKS,
    "bugsinpy": BUGSINPY_PILOT_TASKS,
    "stress": STRESS_PILOT_TASKS,
    "action_balanced": ACTION_BALANCED_PILOT_TASKS,
}


STRESS_METADATA: dict[str, dict[str, object]] = {
    "boundary_zero_division": {
        "stress_category": "boundary",
        "boundary_cases": [{"args": [0]}, {"args": [1]}, {"args": [-1]}],
    },
    "boundary_empty_string": {
        "stress_category": "boundary",
        "boundary_cases": [{"args": [""]}, {"args": ["a"]}],
    },
    "boundary_empty_list": {
        "stress_category": "boundary",
        "boundary_cases": [{"args": [[]]}, {"args": [[0]]}],
    },
    "fuzz_large_square": {
        "stress_category": "fuzz",
        "boundary_cases": [{"args": [0]}, {"args": [1]}, {"args": [-1]}],
        "fuzz_cases": [{"args": [20], "expected": 400}, {"args": [-21], "expected": 441}],
    },
    "fuzz_repeated_items": {
        "stress_category": "fuzz",
        "boundary_cases": [{"args": [[]]}, {"args": [[0]]}],
        "fuzz_cases": [{"args": [[1, 1, 2]], "expected": 3}, {"args": [[0, 0, 0]], "expected": 3}],
    },
    "fuzz_long_string": {
        "stress_category": "fuzz",
        "boundary_cases": [{"args": [""]}, {"args": ["abc"]}],
        "fuzz_cases": [{"args": ["abcD"], "expected": True}, {"args": ["longlower"], "expected": True}],
    },
    "assertion_rounding": {
        "stress_category": "llm_assertion",
        "boundary_cases": [{"args": [4, 2]}, {"args": [5, 1]}],
        "llm_assertion_cases": [{"args": [5, 2], "expected": 2}],
    },
    "assertion_casefold": {
        "stress_category": "llm_assertion",
        "boundary_cases": [{"args": ["banana"]}, {"args": ["abc"]}],
        "llm_assertion_cases": [{"args": ["Aardvark"], "expected": 3}],
    },
    "counterexample_factorial": {
        "stress_category": "counterexample",
        "boundary_cases": [{"args": [0]}, {"args": [1]}],
        "counterexample_cases": [{"args": [4], "expected": 24}],
    },
    "counterexample_clamp": {
        "stress_category": "counterexample",
        "boundary_cases": [{"args": [0]}, {"args": [5]}],
        "counterexample_cases": [{"args": [-5], "expected": 0}],
    },
    "clean_sum": {
        "stress_category": "clean",
        "boundary_cases": [{"args": [[]]}, {"args": [[0]]}],
        "fuzz_cases": [{"args": [[1, -1, 2]], "expected": 2}, {"args": [[5, 5]], "expected": 10}],
    },
    "clean_palindrome": {
        "stress_category": "clean",
        "boundary_cases": [{"args": [""]}, {"args": ["a"]}],
        "fuzz_cases": [{"args": ["abba"], "expected": True}, {"args": ["abca"], "expected": False}],
    },
}


ACTION_BALANCED_METADATA: dict[str, dict[str, object]] = {
    "boundary_mod_zero": {
        "stress_category": "boundary",
        "boundary_cases": [{"args": [0]}, {"args": [1]}, {"args": [-1]}],
    },
    "boundary_first_word": {
        "stress_category": "boundary",
        "boundary_cases": [{"args": [""]}, {"args": ["hello"]}],
    },
    "fuzz_overflow_len": {
        "stress_category": "fuzz",
        "boundary_cases": [{"args": [[]]}, {"args": [[1]]}],
        "fuzz_cases": [{"args": [[1, 2, 3]], "expected": 3}, {"args": [[0, 1, 2, 3]], "expected": 4}],
    },
    "fuzz_threshold_total": {
        "stress_category": "fuzz",
        "boundary_cases": [{"args": [[]]}, {"args": [[1]]}],
        "fuzz_cases": [{"args": [[4, 4, 4]], "expected": True}, {"args": [[10]], "expected": True}],
    },
    "llm_assert_sorted_abs": {
        "stress_category": "llm_assertion",
        "boundary_cases": [{"args": [[]]}, {"args": [[1, 2]]}],
        "llm_assertion_cases": [{"args": [[-3, 2, -1]], "expected": [-1, 2, -3]}],
    },
    "llm_assert_strip_case": {
        "stress_category": "llm_assertion",
        "boundary_cases": [{"args": ["abc"]}, {"args": [""]}],
        "llm_assertion_cases": [{"args": ["  AbC  "], "expected": "AbC"}],
    },
    "counterexample_gcd_zero": {
        "stress_category": "counterexample",
        "boundary_cases": [{"args": [6, 3]}, {"args": [4, 2]}],
        "counterexample_cases": [{"args": [-6, 3], "expected": 3}, {"args": [0, 0], "expected": 0}],
    },
    "counterexample_rotate_empty": {
        "stress_category": "counterexample",
        "boundary_cases": [{"args": [[1, 2], 1]}, {"args": [[0], 0]}],
        "counterexample_cases": [{"args": [[], 3], "expected": []}],
    },
    "clean_clamp": {
        "stress_category": "clean",
        "boundary_cases": [{"args": [-1]}, {"args": [0]}, {"args": [11]}],
        "fuzz_cases": [{"args": [5], "expected": 5}, {"args": [100], "expected": 10}],
    },
    "clean_join": {
        "stress_category": "clean",
        "boundary_cases": [{"args": [[]]}, {"args": [["a"]]}],
        "fuzz_cases": [{"args": [["a", "b", "c"]], "expected": "a,b,c"}],
    },
    "stop_noop_identity": {
        "stress_category": "stop",
        "boundary_cases": [{"args": [0]}, {"args": [1]}, {"args": [-1]}],
        "fuzz_cases": [{"args": [5], "expected": 5}, {"args": [-5], "expected": -5}],
    },
    "stop_noop_reverse": {
        "stress_category": "stop",
        "boundary_cases": [{"args": [""]}, {"args": ["a"]}],
        "fuzz_cases": [{"args": ["abcd"], "expected": "dcba"}, {"args": ["race"], "expected": "ecar"}],
    },
}


def _split_for_index(idx: int, total: int) -> str:
    train_cut = max(1, int(total * 0.6))
    val_cut = max(train_cut + 1, int(total * 0.8))
    if idx < train_cut:
        return "train"
    if idx < val_cut:
        return "val"
    return "test"


def _metadata_for_task(dataset: str, task_id: str) -> dict[str, object]:
    if dataset == "stress":
        return {"profile": "stress", **STRESS_METADATA.get(task_id, {"stress_category": "mixed"})}
    if dataset == "action_balanced":
        return {"profile": "action_balanced", **ACTION_BALANCED_METADATA.get(task_id, {"stress_category": "mixed"})}
    return {"profile": "full_pilot" if dataset in {"humaneval", "mbpp", "bugsinpy"} else dataset}


def _write_tasks(out: Path, dataset: str, tasks: list[TaskSpec], limit: int | None = None, offset: int = 0) -> list[dict[str, object]]:
    selected = tasks[:limit] if limit is not None else tasks
    manifest: list[dict[str, object]] = []
    total = len(selected)
    for idx, (task_id, source, entrypoint, tests) in enumerate(selected):
        split = _split_for_index(offset + idx, total + offset)
        subject_id = task_id if dataset == "toy" or task_id.startswith(f"{dataset}_") else f"{dataset}_{task_id}"
        subject_dir = out / subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)
        source_path = subject_dir / "solution.py"
        source_path.write_text(dedent(source), encoding="utf-8")
        record = {
            "id": subject_id,
            "language": "python",
            "path": str(source_path.relative_to(out)),
            "entrypoint": entrypoint,
            "tests": tests,
            "metadata": {
                "dataset": dataset,
                "split": split,
                "original_id": task_id,
                "source": "embedded_pilot_catalog",
                "source_code": dedent(source),
                **_metadata_for_task(dataset, task_id),
            },
        }
        (subject_dir / "subject.json").write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        manifest.append(record)
    return manifest


def prepare_builtin_dataset(out: Path, dataset: str = "toy", limit: int | None = None, clean: bool = True) -> Path:
    if dataset not in {*DATASETS, "full_pilot", "diverse_full"}:
        raise ValueError(f"unsupported dataset {dataset!r}; use toy, humaneval, mbpp, bugsinpy, stress, action_balanced, full_pilot, or diverse_full")
    if clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    if dataset in {"full_pilot", "diverse_full"}:
        manifest = []
        manifest.extend(_write_tasks(out, "humaneval", HUMANEVAL_PILOT_TASKS, 20, offset=0))
        manifest.extend(_write_tasks(out, "mbpp", MBPP_PILOT_TASKS, 20, offset=len(manifest)))
        manifest.extend(_write_tasks(out, "bugsinpy", BUGSINPY_PILOT_TASKS, 10, offset=len(manifest)))
        if dataset == "diverse_full":
            manifest.extend(_write_tasks(out, "stress", STRESS_PILOT_TASKS, offset=len(manifest)))
            manifest.extend(_write_tasks(out, "action_balanced", ACTION_BALANCED_PILOT_TASKS, offset=len(manifest)))
        if limit is not None:
            manifest = manifest[:limit]
    else:
        manifest = _write_tasks(out, dataset, DATASETS[dataset], limit)
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def prepare_dataset(
    out: Path,
    dataset: str = "toy",
    limit: int | None = None,
    clean: bool = True,
    source_root: Path | None = None,
    cache_dir: Path | None = None,
    limit_per_source: int | None = None,
    validate: bool = False,
) -> Path:
    if dataset in REAL_DATASETS:
        return prepare_real_dataset(
            out,
            dataset,
            limit=limit,
            limit_per_source=limit_per_source,
            source_root=source_root,
            cache_dir=cache_dir,
            validate=validate,
            clean=clean,
        )
    return prepare_builtin_dataset(out, dataset, limit=limit, clean=clean)


def load_subjects(root: Path, split: str | None = None) -> list[ProgramSubject]:
    manifest = root / "manifest.json"
    if manifest.exists():
        records = json.loads(manifest.read_text(encoding="utf-8"))
        subjects = [ProgramSubject.from_json(record, root) for record in records]
    else:
        subjects = []
        for path in sorted(root.rglob("subject.json")):
            subjects.append(ProgramSubject.from_json(json.loads(path.read_text(encoding="utf-8")), path.parent))
    if split:
        subjects = [s for s in subjects if s.metadata.get("split") == split]
    return subjects
