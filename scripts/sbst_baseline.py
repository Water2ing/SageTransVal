"""Classical SBST baseline runner (eval-plan E5).

Drives Pynguin on Python source programs and EvoSuite on Java source programs,
extracts the generated inputs (not the assertions), and feeds them through the
existing SageTransVal differential harness as a new validation mode named
``sbst_coverage_driven``.

Pynguin: installed via pip. Run requires `PYNGUIN_DANGER_AWARE=1` because
Pynguin executes the SUT during generation.

EvoSuite: requires a local jar (`tools/evosuite-1.2.0.jar`). The runner shells
out with a strict per-subject budget and parses the generated JUnit tests.

Both produce JSON dumps of (mode, subject, generated_inputs, generation_seconds).
The downstream evaluator (`sagetransval.full_pipeline._cases_for_mode`) is
extended in this commit to recognize the ``sbst_coverage_driven`` mode and load
the cached generated inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sagetransval.schema import load_subject

PYNGUIN_BUDGET = int(os.environ.get("PYNGUIN_BUDGET_SECONDS", "60"))
EVOSUITE_BUDGET = int(os.environ.get("EVOSUITE_BUDGET_SECONDS", "60"))


def run_pynguin(subject_dir: pathlib.Path, out_dir: pathlib.Path) -> dict[str, Any]:
    """Run Pynguin on the Python source. Returns a record dict."""
    subject = load_subject(subject_dir / "metadata.json")
    if subject.source_language != "python":
        return {"subject": subject.id, "skipped": True, "reason": "not python"}
    # Locate the pynguin entry point: prefer one in the same venv as the current interpreter
    pynguin_bin = pathlib.Path(sys.executable).parent / "pynguin"
    if not pynguin_bin.exists():
        pynguin_bin = pathlib.Path(shutil.which("pynguin") or "")
    if not pynguin_bin.exists():
        return {"subject": subject.id, "skipped": True, "reason": "pynguin not installed"}
    work = out_dir / subject.id
    work.mkdir(parents=True, exist_ok=True)
    # Pynguin works on a module path; copy source as a fresh module.
    module_dir = work / "module"
    module_dir.mkdir(exist_ok=True)
    shutil.copy(pathlib.Path(subject.source_path), module_dir / "sut.py")
    out_tests = work / "tests"
    out_tests.mkdir(exist_ok=True)
    env = dict(os.environ, PYNGUIN_DANGER_AWARE="1")
    t0 = time.time()
    try:
        proc = subprocess.run(
            [
                str(pynguin_bin),
                "--project-path", str(module_dir),
                "--output-path", str(out_tests),
                "--module-name", "sut",
                "--maximum-search-time", str(PYNGUIN_BUDGET),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=PYNGUIN_BUDGET + 60,
        )
        rc = proc.returncode
        stderr = proc.stderr
    except subprocess.TimeoutExpired:
        return {"subject": subject.id, "skipped": True, "reason": "pynguin_timeout"}
    elapsed = time.time() - t0
    generated_inputs = _extract_pynguin_inputs(out_tests, subject.entrypoint)
    return {
        "subject": subject.id,
        "tool": "pynguin",
        "mode": "sbst_coverage_driven",
        "generated_inputs": generated_inputs,
        "generation_seconds": elapsed,
        "rc": rc,
        "stderr_tail": stderr[-400:] if stderr else "",
    }


def _extract_pynguin_inputs(out_tests: pathlib.Path, entrypoint: str) -> list[dict[str, Any]]:
    """Extract function-call inputs from Pynguin's generated pytest file.

    Pynguin emits ``test_case_*`` functions that assign locals first, then call
    ``module_0.{entrypoint}(...)``. We walk the AST: for each test function,
    build a name->literal mapping from `name = <literal>` assignments and
    resolve the call args by looking up names in that mapping.
    """
    import ast
    inputs: list[dict[str, Any]] = []
    for test_file in out_tests.rglob("test_*.py"):
        try:
            tree = ast.parse(test_file.read_text())
        except SyntaxError:
            continue
        for func in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
            locals_: dict[str, Any] = {}
            for stmt in func.body:
                # Capture literal-valued local assignments first.
                if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                        and isinstance(stmt.targets[0], ast.Name)
                        and not isinstance(stmt.value, ast.Call)):
                    val = _safe_eval(stmt.value, locals_)
                    if val is not _UNRESOLVED:
                        locals_[stmt.targets[0].id] = val
                    continue
                # Direct call without assignment, or assignment to a call.
                call_node = None
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                    call_node = stmt.value
                elif isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                    call_node = stmt.value
                if call_node is None:
                    continue
                if not (isinstance(call_node.func, ast.Attribute)
                        and call_node.func.attr == entrypoint):
                    continue
                positional: list[Any] = []
                ok = True
                for arg in call_node.args:
                    v = _safe_eval(arg, locals_)
                    if v is _UNRESOLVED:
                        ok = False
                        break
                    positional.append(v)
                kwargs: dict[str, Any] = {}
                if ok:
                    for kw in call_node.keywords:
                        v = _safe_eval(kw.value, locals_)
                        if v is _UNRESOLVED or kw.arg is None:
                            ok = False
                            break
                        kwargs[kw.arg] = v
                if ok and isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
                    # Track the call's return name so subsequent assignments can ignore it
                    locals_[stmt.targets[0].id] = _UNRESOLVED
                if ok:
                    inputs.append({"args": {"__positional__": positional, **kwargs}})
    return inputs


_UNRESOLVED = object()


def _safe_eval(node, locals_):
    """Evaluate an AST node against locals_ if it is a literal or known name."""
    import ast
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return locals_.get(node.id, _UNRESOLVED)
    if isinstance(node, (ast.List, ast.Tuple)):
        items = []
        for elt in node.elts:
            v = _safe_eval(elt, locals_)
            if v is _UNRESOLVED:
                return _UNRESOLVED
            items.append(v)
        return items if isinstance(node, ast.List) else tuple(items)
    if isinstance(node, ast.Dict):
        out: dict = {}
        for k, v in zip(node.keys, node.values):
            kk = _safe_eval(k, locals_)
            vv = _safe_eval(v, locals_)
            if kk is _UNRESOLVED or vv is _UNRESOLVED:
                return _UNRESOLVED
            out[kk] = vv
        return out
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _safe_eval(node.operand, locals_)
        return -v if v is not _UNRESOLVED else _UNRESOLVED
    return _UNRESOLVED


def _parse_python_call_args(text: str) -> dict[str, Any] | None:
    """Parse positional args of a function call into a kwargs-style dict.

    Returns None if any arg cannot be statically evaluated. We deliberately use
    ``ast.literal_eval`` rather than ``eval`` to refuse anything non-literal.
    """
    import ast
    if not text:
        return {}
    try:
        wrapped = f"f({text})"
        call = ast.parse(wrapped, mode="eval").body
        if not isinstance(call, ast.Call):
            return None
        positional: list[Any] = []
        for arg in call.args:
            positional.append(ast.literal_eval(arg))
        kwargs: dict[str, Any] = {}
        for kw in call.keywords:
            if kw.arg is None:
                return None
            kwargs[kw.arg] = ast.literal_eval(kw.value)
        # Coerce positional to keyword form using a sentinel; caller knows entrypoint.
        result: dict[str, Any] = {"__positional__": positional, **kwargs}
        return result
    except (SyntaxError, ValueError):
        return None


def run_evosuite(subject_dir: pathlib.Path, out_dir: pathlib.Path,
                 evosuite_jar: pathlib.Path) -> dict[str, Any]:
    """Run EvoSuite on the Java target/source. Returns a record dict."""
    subject = load_subject(subject_dir / "metadata.json")
    if not evosuite_jar.exists():
        return {"subject": subject.id, "skipped": True, "reason": "evosuite_jar_missing"}
    if not shutil.which("java"):
        return {"subject": subject.id, "skipped": True, "reason": "no_java"}
    # EvoSuite operates on a target Java class; we run it on the translated Java target.
    # If target_path is None (translation not yet performed) we skip.
    if subject.target_path is None or not pathlib.Path(subject.target_path).exists():
        return {"subject": subject.id, "skipped": True, "reason": "no_java_target"}
    work = out_dir / subject.id
    work.mkdir(parents=True, exist_ok=True)
    classpath_dir = work / "classes"
    classpath_dir.mkdir(exist_ok=True)
    # Compile target to a class first.
    java_src = pathlib.Path(subject.target_path)
    shutil.copy(java_src, work / "Solution.java")
    compile_proc = subprocess.run(
        ["javac", "-d", str(classpath_dir), str(work / "Solution.java")],
        capture_output=True, text=True, timeout=60,
    )
    if compile_proc.returncode != 0:
        return {"subject": subject.id, "skipped": True, "reason": "javac_failed",
                "stderr_tail": compile_proc.stderr[-400:]}
    # Run EvoSuite on the compiled class.
    t0 = time.time()
    try:
        proc = subprocess.run(
            [
                "java", "-jar", str(evosuite_jar),
                "-class", "Solution",
                "-projectCP", str(classpath_dir),
                "-Dsearch_budget=" + str(EVOSUITE_BUDGET),
                "-Dassertions=false",
            ],
            cwd=str(work),
            capture_output=True, text=True,
            timeout=EVOSUITE_BUDGET + 120,
        )
        rc = proc.returncode
        stderr = proc.stderr
    except subprocess.TimeoutExpired:
        return {"subject": subject.id, "skipped": True, "reason": "evosuite_timeout"}
    elapsed = time.time() - t0
    generated_inputs = _extract_evosuite_inputs(work, subject.entrypoint)
    return {
        "subject": subject.id,
        "tool": "evosuite",
        "mode": "sbst_coverage_driven",
        "generated_inputs": generated_inputs,
        "generation_seconds": elapsed,
        "rc": rc,
        "stderr_tail": stderr[-400:] if stderr else "",
    }


def _extract_evosuite_inputs(work: pathlib.Path, entrypoint: str) -> list[dict[str, Any]]:
    """Extract Solution.{entrypoint}(...) inputs from EvoSuite-generated test sources."""
    inputs: list[dict[str, Any]] = []
    pattern = re.compile(rf"Solution\.{re.escape(entrypoint)}\((.*?)\)", re.DOTALL)
    for test_file in (work / "evosuite-tests").rglob("Solution_ESTest.java"):
        text = test_file.read_text(errors="replace")
        for m in pattern.finditer(text):
            arg_text = m.group(1).strip()
            args = _parse_java_call_args(arg_text)
            if args is not None:
                inputs.append({"args": args})
    return inputs


def _parse_java_call_args(text: str) -> dict[str, Any] | None:
    """Very small Java-literal parser. Only handles ints, longs, doubles, booleans,
    string literals, and one-level array literals like ``new int[]{1,2,3}``.
    Returns None on anything more elaborate; callers will skip the test.
    """
    if not text.strip():
        return {"__positional__": []}
    # Split at top-level commas only.
    parts: list[str] = []
    depth = 0
    cur = []
    for ch in text:
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())

    out: list[Any] = []
    for p in parts:
        v = _eval_java_literal(p)
        if v is _UNPARSEABLE:
            return None
        out.append(v)
    return {"__positional__": out}


_UNPARSEABLE = object()


def _eval_java_literal(p: str) -> Any:
    p = p.strip().rstrip("L").rstrip("F").rstrip("D")
    if p == "true":
        return True
    if p == "false":
        return False
    if p == "null":
        return None
    if (p.startswith('"') and p.endswith('"')):
        return p[1:-1].encode("utf-8").decode("unicode_escape")
    if re.fullmatch(r"-?\d+", p):
        return int(p)
    if re.fullmatch(r"-?\d+\.\d+([eE][+-]?\d+)?", p):
        return float(p)
    m = re.match(r"new\s+\w+\[\]\s*\{(.*)\}", p, re.DOTALL)
    if m:
        sub = m.group(1).strip()
        if not sub:
            return []
        return [_eval_java_literal(x) for x in sub.split(",")]
    return _UNPARSEABLE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", choices=["pynguin", "evosuite"], required=True)
    ap.add_argument("--subjects-root", type=pathlib.Path, required=True,
                    help="Directory containing one folder per subject with metadata.json")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--evosuite-jar", type=pathlib.Path, default=ROOT / "tools" / "evosuite-1.2.0.jar")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    n = 0
    for subject_dir in sorted(args.subjects_root.iterdir()):
        if not (subject_dir / "metadata.json").exists():
            continue
        if args.tool == "pynguin":
            rec = run_pynguin(subject_dir, args.out)
        else:
            rec = run_evosuite(subject_dir, args.out, args.evosuite_jar)
        records.append(rec)
        print(f"  [{args.tool}] {rec['subject']} {'OK' if not rec.get('skipped') else 'SKIP:'+rec.get('reason','')} inputs={len(rec.get('generated_inputs', []))}")
        n += 1
        if args.limit and n >= args.limit:
            break
    summary_path = args.out / f"sbst_{args.tool}_summary.jsonl"
    with summary_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
