"""Action executors for the Python MVP."""

from __future__ import annotations

import ast
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

from .actions import Action
from .llm_cache import LLMCache
from .models import ExecutionResult, ProgramSubject


def _load_entrypoint(subject: ProgramSubject):
    _ensure_source_exists(subject)
    spec = importlib.util.spec_from_file_location(f"testgenrl_subject_{subject.id}", subject.path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {subject.path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, subject.entrypoint)


def _infer_values(subject: ProgramSubject) -> list[list[Any]]:
    _ensure_source_exists(subject)
    source = subject.path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    values: set[Any] = {0, 1, -1, "", "a", "abc"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, str, bool)):
            values.add(node.value)
    fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == subject.entrypoint), None)
    if fn is None:
        return [[0]]
    args: list[list[Any]] = []
    for arg in fn.args.args:
        ann = ast.unparse(arg.annotation) if arg.annotation is not None else ""
        if "str" in ann:
            args.append(["", "a", "abc"])
        elif "list" in ann:
            args.append([[], [0], [1, -1, 2]])
        else:
            args.append([v for v in values if isinstance(v, int)][:5] or [0, 1, -1])
    return args or [[0]]


def _ensure_source_exists(subject: ProgramSubject) -> None:
    if subject.path.exists():
        return
    source = subject.metadata.get("source_code")
    if isinstance(source, str) and source:
        subject.path.parent.mkdir(parents=True, exist_ok=True)
        subject.path.write_text(source, encoding="utf-8")


def _product(options: list[list[Any]], limit: int = 24) -> list[list[Any]]:
    out: list[list[Any]] = [[]]
    for vals in options:
        out = [prefix + [v] for prefix in out for v in vals]
        out = out[:limit]
    return out[:limit]


def _metadata_cases(subject: ProgramSubject, key: str) -> list[dict[str, Any]] | None:
    cases = subject.metadata.get(key)
    if isinstance(cases, list):
        return [case for case in cases if isinstance(case, dict)]
    return None


def _run_cases(subject: ProgramSubject, cases: list[dict[str, Any]]) -> ExecutionResult:
    start = time.time()
    result = ExecutionResult(action="oracle", cost={"runtime_seconds": 0.0})
    seen: set[str] = set()
    try:
        fn = _load_entrypoint(subject)
    except Exception as exc:
        result.invalid_tests = len(cases) or 1
        result.observations["load_error"] = repr(exc)
        result.cost["runtime_seconds"] = time.time() - start
        return result
    for case in cases:
        if case.get("oracle") == "check":
            oracle_code = subject.metadata.get("oracle_code")
            if not isinstance(oracle_code, str) or "def check" not in oracle_code:
                result.invalid_tests += 1
                result.observations["oracle_error"] = "missing callable check oracle"
                continue
            try:
                namespace: dict[str, Any] = {}
                exec(oracle_code, namespace)
                check = namespace.get("check")
                if not callable(check):
                    raise TypeError("oracle_code did not define callable check")
                check(fn)
                result.generated_tests.append({"oracle": "check", "status": "passed"})
            except AssertionError as exc:
                result.failures += 1
                result.generated_tests.append({"oracle": "check", "error": repr(exc)})
                result.observations.setdefault("failures", []).append({"case": case, "error": repr(exc)})
            except Exception as exc:
                result.invalid_tests += 1
                result.generated_tests.append({"oracle": "check", "error": repr(exc)})
                result.observations["oracle_error"] = repr(exc)
            continue
        key = json.dumps(case.get("args", []), sort_keys=True, default=str)
        if key in seen:
            result.duplicate_tests += 1
            continue
        seen.add(key)
        try:
            actual = fn(*case.get("args", []))
            if "expected" in case and actual != case["expected"]:
                result.failures += 1
                result.observations.setdefault("failures", []).append({"case": case, "actual": actual})
            result.generated_tests.append({"args": case.get("args", []), "actual": actual})
        except Exception as exc:
            result.failures += 1
            result.generated_tests.append({"args": case.get("args", []), "error": repr(exc)})
            result.observations.setdefault("failures", []).append({"case": case, "error": repr(exc)})
    result.coverage_delta = min(0.25, 0.02 * max(1, len(result.generated_tests)))
    result.residual_risk_delta = min(0.15, 0.01 * max(1, len(result.generated_tests)))
    result.cost["runtime_seconds"] = time.time() - start
    return result


class ExecutorSuite:
    def __init__(
        self,
        cache_dir: Path,
        llm_provider: str = "stub",
        llm_model: str = "stub",
        cache_mode: str = "live",
    ):
        self.llm = LLMCache(cache_dir, provider_name=llm_provider, model=llm_model, cache_mode=cache_mode)

    def execute(self, action: Action, subject: ProgramSubject) -> ExecutionResult:
        if action == Action.GENERATE_BOUNDARY_TESTS:
            cases = _metadata_cases(subject, "boundary_cases") or [{"args": args} for args in _product(_infer_values(subject), limit=16)]
            res = _run_cases(subject, cases)
            res.action = action.name
            res.cost["runtime_seconds"] = max(float(res.cost.get("runtime_seconds", 0.0)), 0.05)
            return res
        if action == Action.PROPERTY_BASED_FUZZ:
            options = _infer_values(subject)
            cases = _metadata_cases(subject, "fuzz_cases") or [{"args": args} for args in _product([vals + vals for vals in options], limit=32)]
            res = _run_cases(subject, cases)
            res.action = action.name
            res.cost["fuzz_iterations"] = len(cases)
            res.cost["runtime_seconds"] = max(float(res.cost.get("runtime_seconds", 0.0)), 0.15)
            return res
        if action == Action.LLM_SYNTHESIZE_INSTRUCTIONS:
            prompt = f"Emit compact JSON test-generation instructions for {subject.entrypoint} in {subject.path.name}."
            record = self.llm.complete(prompt)
            try:
                parsed = json.loads(record["text"])
                valid = isinstance(parsed, dict) and "instructions" in parsed
            except json.JSONDecodeError:
                valid = False
            return ExecutionResult(
                action=action.name,
                generated_tests=[],
                coverage_delta=0.02 if valid else 0.0,
                uncertainty_delta=0.1 if valid else 0.0,
                cost={"tokens": record["prompt_tokens"] + record["completion_tokens"]},
                observations={
                    "schema_valid": valid,
                    "llm_cache_key": record["key"],
                    "llm_provider": record["provider"],
                    "llm_model": record["model"],
                },
            )
        if action in {Action.LLM_GENERATE_ASSERTIONS, Action.REFINE_UNCERTAIN_ORACLE}:
            prompt = f"Suggest concise assertions for {subject.id}:{subject.entrypoint}."
            record = self.llm.complete(prompt)
            assertion_cases = _metadata_cases(subject, "llm_assertion_cases")
            result = ExecutionResult(
                action=action.name,
                uncertainty_delta=0.2,
                residual_risk_delta=0.05,
                cost={"tokens": record["prompt_tokens"] + record["completion_tokens"]},
                observations={
                    "llm_cache_key": record["key"],
                    "llm_provider": record["provider"],
                    "llm_model": record["model"],
                    "refined": action == Action.REFINE_UNCERTAIN_ORACLE,
                },
            )
            if assertion_cases:
                oracle = _run_cases(subject, assertion_cases)
                result.generated_tests = oracle.generated_tests
                result.failures = oracle.failures
                result.invalid_tests = oracle.invalid_tests
                result.flaky_tests = oracle.flaky_tests
                result.duplicate_tests = oracle.duplicate_tests
                result.coverage_delta = max(result.coverage_delta, oracle.coverage_delta)
                result.residual_risk_delta = max(result.residual_risk_delta, oracle.residual_risk_delta)
                result.cost["runtime_seconds"] = max(float(oracle.cost.get("runtime_seconds", 0.0)), 0.05)
                result.observations.update(oracle.observations)
            return result
        if action == Action.EXPAND_COUNTEREXAMPLE:
            res = _run_cases(subject, _metadata_cases(subject, "counterexample_cases") or subject.tests)
            res.action = action.name
            res.cost["fuzz_iterations"] = 3
            res.cost["runtime_seconds"] = max(float(res.cost.get("runtime_seconds", 0.0)), 0.1)
            return res
        if action == Action.RETEST_REPAIR_CONTEXT:
            res = _run_cases(subject, subject.tests)
            res.action = action.name
            res.observations["repair_event"] = True
            return res
        return ExecutionResult(action=action.name)
