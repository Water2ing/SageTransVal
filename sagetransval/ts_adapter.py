"""TypeScript execution adapter (eval-plan E4).

Drives ``scripts/ts_executor.js`` which compiles the TS source with ``tsc``
and re-invokes the JS executor on the compiled artifact. Returns an
:class:`Observation` matching the existing adapter contract.

Gating: requires ``typescript`` installed in the repo (``npm install -D typescript``).
The CompileError observation is reported normally; an absent tsc is reported as
``ConfigError`` rather than crashing.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import tempfile
from typing import Any, Mapping

from .models import Observation


SCRIPT_DIR = pathlib.Path(__file__).resolve().parents[1] / "scripts"
TS_EXECUTOR = SCRIPT_DIR / "ts_executor.js"


def execute_ts(source_path: pathlib.Path, entrypoint: str, parameter_order: list[str],
               input_args: Mapping[str, Any], *, wall_seconds: int = 15) -> Observation:
    if not TS_EXECUTOR.exists():
        return Observation(kind="error", error="ts_executor.js missing")
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
        try:
            proc = subprocess.run(
                ["node", str(TS_EXECUTOR)],
                env=env, capture_output=True, text=True,
                timeout=wall_seconds,
            )
        except subprocess.TimeoutExpired:
            return Observation(kind="timeout", timeout=True)
        last = ""
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line:
                last = line
        try:
            payload = json.loads(last)
        except json.JSONDecodeError:
            return Observation(kind="error", error="malformed_ts_output", stdout=proc.stdout, stderr=proc.stderr)
        if payload.get("exception_class"):
            return Observation(
                kind="exception",
                value=payload["exception_class"],
                error=payload.get("exception_message"),
                stdout=proc.stdout, stderr=proc.stderr,
            )
        return Observation(
            kind="value", value=payload.get("return_value"),
            stdout=proc.stdout, stderr=proc.stderr,
        )
