"""
Kotlin execution adapter for the Java->Kotlin pilot (SageTransVal eval-plan extension).

This is a scaffold matching the existing language-adapter contract in
`sagetransval.adapters`. It compiles a generated Kotlin wrapper with `kotlinc`,
runs it on the JVM, and returns the same Observation record shape as the
existing Python/Java/C/Rust adapters.

The harness is intentionally narrow: function-level Kotlin translations of
the existing Java targets, with the same observation contract. Compile errors,
runtime errors, and timeouts are explicit observation classes.

Status: scaffold. To enable end-to-end runs:
  1. Install kotlinc 1.9+ on the eval host.
  2. Convert at least 10 existing Java targets to Kotlin manually (or via LLM)
     and place them under data/proposal1_kotlin_pilot/<subject>/target.kt.
  3. Add an entry to `sagetransval.adapters.LANGUAGE_ADAPTERS` mapping
     "kotlin" -> KotlinAdapter().
  4. Run `python -m sagetransval.evaluate --pair java_kotlin --pilot`.

No translations are produced in this session (no API access).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

KOTLIN_TIMEOUT_SECONDS = 30


@dataclass
class Observation:
    """Observation record matching the existing harness contract."""
    return_value: Any = None
    exception_class: str | None = None
    exception_message: str | None = None
    stdout: str = ""
    stderr: str = ""
    compile_error: bool = False
    runtime_error: bool = False
    timeout: bool = False


KOTLIN_WRAPPER_TEMPLATE = """
import java.io.File

{target_kotlin}

fun main() {{
    val inputJson = File(System.getenv("SAGE_INPUT_JSON") ?: error("missing SAGE_INPUT_JSON")).readText()
    val input = com.google.gson.Gson().fromJson(inputJson, Map::class.java) as Map<String, Any?>
    try {{
        val result = {entrypoint}({arg_unpack})
        val out = mapOf("return_value" to result, "exception_class" to null)
        println(com.google.gson.Gson().toJson(out))
    }} catch (e: Throwable) {{
        val out = mapOf(
            "return_value" to null,
            "exception_class" to (e::class.simpleName ?: "Throwable"),
            "exception_message" to (e.message ?: "")
        )
        println(com.google.gson.Gson().toJson(out))
    }}
}}
"""


def _kotlinc_available() -> bool:
    try:
        subprocess.run(["kotlinc", "-version"], capture_output=True, check=False, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def execute_kotlin(target_source: str, entrypoint: str, parameter_names: list[str],
                   input_dict: dict) -> Observation:
    """Execute a Kotlin target with one input. Returns a normalized Observation.

    The wrapper expects the target_source to define a top-level `fun {entrypoint}(...)`.
    """
    if not _kotlinc_available():
        return Observation(compile_error=True, stderr="kotlinc not installed")

    arg_unpack = ", ".join(f"input[\"{p}\"] as {{Any?}}" for p in parameter_names).replace("{Any?}", "Any?")
    wrapper = KOTLIN_WRAPPER_TEMPLATE.format(
        target_kotlin=target_source,
        entrypoint=entrypoint,
        arg_unpack=arg_unpack,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        src = tmp_path / "Main.kt"
        src.write_text(wrapper)
        jar = tmp_path / "main.jar"
        compile_proc = subprocess.run(
            ["kotlinc", str(src), "-include-runtime", "-d", str(jar)],
            capture_output=True, text=True, timeout=60,
        )
        if compile_proc.returncode != 0:
            return Observation(compile_error=True, stderr=compile_proc.stderr)
        input_path = tmp_path / "input.json"
        input_path.write_text(json.dumps(input_dict))
        try:
            run_proc = subprocess.run(
                ["java", "-jar", str(jar)],
                capture_output=True, text=True, timeout=KOTLIN_TIMEOUT_SECONDS,
                env={"SAGE_INPUT_JSON": str(input_path), "PATH": ""},
            )
        except subprocess.TimeoutExpired:
            return Observation(timeout=True)
        if run_proc.returncode != 0:
            return Observation(runtime_error=True, stderr=run_proc.stderr, stdout=run_proc.stdout)
        try:
            payload = json.loads(run_proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            return Observation(runtime_error=True, stderr="malformed kotlin output", stdout=run_proc.stdout)
        return Observation(
            return_value=payload.get("return_value"),
            exception_class=payload.get("exception_class"),
            exception_message=payload.get("exception_message"),
            stdout=run_proc.stdout,
            stderr=run_proc.stderr,
        )


# TODO: register adapter in sagetransval.adapters.LANGUAGE_ADAPTERS
# from sagetransval.adapters import LANGUAGE_ADAPTERS
# LANGUAGE_ADAPTERS["kotlin"] = execute_kotlin
