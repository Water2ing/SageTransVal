"""JavaScript execution adapter (eval-plan E4).

Drives ``scripts/js_executor.js`` for both source-side and target-side execution
of JS subjects. Returns an :class:`sagetransval.models.Observation` matching the
existing adapter contract so the differential oracle treats JS like any other
language.

Public entry point: :func:`execute_js`. Async functions are not supported in v1.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import tempfile
from typing import Any, Mapping

from .models import Observation
from .sandbox import SandboxLimits, run_sandboxed


SCRIPT_DIR = pathlib.Path(__file__).resolve().parents[1] / "scripts"
JS_EXECUTOR = SCRIPT_DIR / "js_executor.js"


def execute_js(source_path: pathlib.Path, entrypoint: str, parameter_order: list[str],
               input_args: Mapping[str, Any], *, sandbox: bool = False,
               wall_seconds: int = 10) -> Observation:
    """Run a JS source through the executor."""
    if not JS_EXECUTOR.exists():
        return Observation(kind="error", error="js_executor.js missing")
    with tempfile.TemporaryDirectory() as tmp:
        input_path = pathlib.Path(tmp) / "input.json"
        input_path.write_text(json.dumps(dict(input_args)))
        env = {
            "SAGE_SOURCE": str(source_path),
            "SAGE_ENTRYPOINT": entrypoint,
            "SAGE_INPUT_JSON": str(input_path),
            "SAGE_PARAM_ORDER": ",".join(parameter_order),
            "PATH": "/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin",
        }
        argv = ["node", str(JS_EXECUTOR)]
        if sandbox:
            res = run_sandboxed(argv, env=env, limits=SandboxLimits(wall_seconds=wall_seconds))
            stdout = res.stdout
            stderr = res.stderr
            timed_out = res.timed_out
        else:
            try:
                proc = subprocess.run(argv, env=env, capture_output=True, text=True,
                                      timeout=wall_seconds)
                stdout = proc.stdout
                stderr = proc.stderr
                timed_out = False
            except subprocess.TimeoutExpired:
                return Observation(kind="timeout", timeout=True)

        if timed_out:
            return Observation(kind="timeout", timeout=True)
        # Take the last non-empty line as the JSON envelope (Node may emit warnings before).
        last = ""
        for line in stdout.splitlines():
            line = line.strip()
            if line:
                last = line
        try:
            payload = json.loads(last)
        except json.JSONDecodeError:
            return Observation(kind="error", error="malformed_js_output", stdout=stdout, stderr=stderr)
        if payload.get("exception_class"):
            return Observation(
                kind="exception",
                value=payload["exception_class"],
                error=payload.get("exception_message"),
                stdout=stdout, stderr=stderr,
            )
        return Observation(
            kind="value",
            value=payload.get("return_value"),
            stdout=stdout, stderr=stderr,
        )
