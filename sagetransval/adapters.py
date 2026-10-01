"""Language adapters that emit normalized differential observations."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import tempfile
import textwrap
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Protocol

from .models import Observation


class LanguageAdapter(Protocol):
    def run(self, path: Path, entrypoint: str, args: list[Any], timeout: float = 5.0) -> Observation:
        ...


class PythonAdapter:
    def run(self, path: Path, entrypoint: str, args: list[Any], timeout: float = 5.0) -> Observation:
        def _raise_timeout(_signum: int, _frame: Any) -> None:
            raise TimeoutError("python execution timed out")

        previous_handler = signal.getsignal(signal.SIGALRM)
        try:
            spec = importlib.util.spec_from_file_location(f"sagetransval_{path.stem}", path)
            if spec is None or spec.loader is None:
                return Observation(kind="error", error="import_spec_failed")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            fn = getattr(module, entrypoint)
            stdout = StringIO()
            with redirect_stdout(stdout):
                signal.signal(signal.SIGALRM, _raise_timeout)
                signal.setitimer(signal.ITIMER_REAL, timeout)
                value = fn(*args)
                signal.setitimer(signal.ITIMER_REAL, 0)
            return Observation(kind="return", value=value, stdout=stdout.getvalue(), exit_code=0)
        except TimeoutError as exc:
            return Observation(kind="timeout", error=str(exc), timeout=True, exit_code=124)
        except Exception as exc:
            return Observation(kind="exception", value=type(exc).__name__, error=str(exc), exit_code=1)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)


class JavaAdapter:
    def run(self, path: Path, entrypoint: str, args: list[Any], timeout: float = 5.0) -> Observation:
        del entrypoint
        with tempfile.TemporaryDirectory(prefix="sagetransval_java_") as tmp_name:
            tmp = Path(tmp_name)
            solution = tmp / "Solution.java"
            shutil.copy(path, solution)
            (tmp / "Harness.java").write_text(_java_harness(args), encoding="utf-8")
            try:
                compile_proc = subprocess.run(
                    ["javac", "Solution.java", "Harness.java"],
                    cwd=tmp,
                    text=True,
                    capture_output=True,
                    timeout=_compile_timeout(timeout),
                )
            except subprocess.TimeoutExpired as exc:
                return Observation(kind="timeout", stdout=_timeout_text(exc.stdout), stderr=_timeout_text(exc.stderr), timeout=True, exit_code=124)
            if compile_proc.returncode != 0:
                return Observation(
                    kind="compile_error",
                    stderr=compile_proc.stderr,
                    error=_last_line(compile_proc.stderr),
                    exit_code=compile_proc.returncode,
                )
            return _run_process(["java", "Harness"], tmp, timeout)


class CAdapter:
    def __init__(self) -> None:
        self._compiled: dict[tuple[str, str, tuple[str, ...]], Path | Observation] = {}
        self._temps: list[tempfile.TemporaryDirectory[str]] = []

    def __del__(self) -> None:
        for tmp in getattr(self, "_temps", []):
            tmp.cleanup()

    def run(self, path: Path, entrypoint: str, args: list[Any], timeout: float = 5.0) -> Observation:
        compiled = self._compile(path, entrypoint, args, timeout)
        if isinstance(compiled, Observation):
            return compiled
        obs = _run_process([str(compiled), *_argv_args(args)], compiled.parent, timeout)
        # Post-process UBSan output: a non-zero exit whose stderr names
        # "runtime error: ... undefined" comes from the C side hitting UB
        # (signed overflow, shift-by-width, INT_MIN/-1, etc.). Promote it to a
        # first-class ``ub`` observation so the oracle's UB-safe equivalence
        # rule can avoid spurious mismatches with a defined Rust target.
        if obs.kind == "runtime_error" and obs.stderr and "runtime error" in obs.stderr.lower():
            return Observation(kind="ub", value=_first_ubsan_message(obs.stderr), stdout=obs.stdout, stderr=obs.stderr, exit_code=obs.exit_code)
        return obs

    def _compile(self, path: Path, entrypoint: str, args: list[Any], timeout: float) -> Path | Observation:
        shape = _arg_shape(args)
        key = (str(path.resolve()), entrypoint, shape)
        if key in self._compiled:
            return self._compiled[key]
        tmp_handle = tempfile.TemporaryDirectory(prefix="sagetransval_c_")
        self._temps.append(tmp_handle)
        tmp = Path(tmp_handle.name)
        source = tmp / "source.c"
        source_text = path.read_text(encoding="utf-8")
        source.write_text(source_text, encoding="utf-8")
        ret_kind = _c_return_kind(source_text, entrypoint)
        setup, call_args = _c_cli_call(shape)
        if ret_kind == "bool":
            call = f'printf("RETURN:%s\\n", {entrypoint}({call_args}) ? "true" : "false");'
        elif ret_kind == "string":
            call = f'const char *out = {entrypoint}({call_args}); printf("RETURN:%s\\n", out ? out : "");'
        else:
            call = f'printf("RETURN:%d\\n", {entrypoint}({call_args}));'
        (tmp / "harness.c").write_text(
            '#include <stdio.h>\n#include <stdbool.h>\n#include <stdlib.h>\n#include <string.h>\n'
            "static int parse_int_array(const char *text, int *out, int max) { "
            "if (text == NULL || text[0] == '\\0') return 0; int n = 0; const char *p = text; "
            "while (*p && n < max) { char *end; out[n++] = (int)strtol(p, &end, 10); if (*end != ',') break; p = end + 1; } return n; }\n"
            '#include "source.c"\n'
            f"int main(int argc, char **argv) {{ (void)argc; {setup}{call} return 0; }}\n",
            encoding="utf-8",
        )
        compiler = shutil.which("clang") or shutil.which("gcc")
        if compiler is None:
            self._compiled[key] = Observation(kind="compile_error", error="missing_c_compiler")
            return self._compiled[key]
        cmd = [compiler, "harness.c", "-o", "harness"]
        # Opt-in UBSan: compiling with sanitizers lets us flag C-side UB
        # (signed overflow, shift-by-width, INT_MIN/-1, etc.) so the oracle's
        # UB-safe equivalence rule can distinguish "C is UB, Rust is defined"
        # from a true mismatch. Falls back to a plain compile when the flag is
        # rejected (rare on macOS gcc-as-clang).
        use_ubsan = os.environ.get("SAGETRANSVAL_C_UBSAN", "0") == "1"
        if use_ubsan:
            cmd = [compiler, "-fsanitize=undefined", "-fno-sanitize-recover=all", "harness.c", "-o", "harness"]
        try:
            compile_proc = subprocess.run(cmd, cwd=tmp, text=True, capture_output=True, timeout=_compile_timeout(timeout))
            if use_ubsan and compile_proc.returncode != 0 and "sanitize" in compile_proc.stderr.lower():
                # Sanitizer toolchain missing -- retry without it so the run continues.
                compile_proc = subprocess.run([compiler, "harness.c", "-o", "harness"], cwd=tmp, text=True, capture_output=True, timeout=_compile_timeout(timeout))
        except subprocess.TimeoutExpired as exc:
            self._compiled[key] = Observation(kind="timeout", stdout=_timeout_text(exc.stdout), stderr=_timeout_text(exc.stderr), timeout=True, exit_code=124)
            return self._compiled[key]
        if compile_proc.returncode != 0:
            self._compiled[key] = Observation(kind="compile_error", stderr=compile_proc.stderr, error=_last_line(compile_proc.stderr))
            return self._compiled[key]
        self._compiled[key] = tmp / "harness"
        return self._compiled[key]


class RustAdapter:
    def __init__(self) -> None:
        self._compiled: dict[tuple[str, str, tuple[str, ...]], Path | Observation] = {}
        self._temps: list[tempfile.TemporaryDirectory[str]] = []

    def __del__(self) -> None:
        for tmp in getattr(self, "_temps", []):
            tmp.cleanup()

    def run(self, path: Path, entrypoint: str, args: list[Any], timeout: float = 5.0) -> Observation:
        compiled = self._compile(path, entrypoint, args, timeout)
        if isinstance(compiled, Observation):
            return compiled
        return _run_process([str(compiled), *_argv_args(args)], compiled.parent, timeout)

    def _compile(self, path: Path, entrypoint: str, args: list[Any], timeout: float) -> Path | Observation:
        shape = _arg_shape(args)
        key = (str(path.resolve()), entrypoint, shape)
        if key in self._compiled:
            return self._compiled[key]
        tmp_handle = tempfile.TemporaryDirectory(prefix="sagetransval_rust_")
        self._temps.append(tmp_handle)
        tmp = Path(tmp_handle.name)
        (tmp / "solution.rs").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        setup, call_args = _rust_cli_call(shape)
        (tmp / "harness.rs").write_text(
            "mod solution;\n"
            "fn parse_i32_list(text: &str) -> Vec<i32> { if text.is_empty() { Vec::new() } else { text.split(',').filter(|s| !s.is_empty()).map(|s| s.parse::<i32>().unwrap()).collect() } }\n"
            f"fn main() {{ let argv: Vec<String> = std::env::args().collect(); {setup}println!(\"RETURN:{{}}\", solution::{entrypoint}({call_args})); }}\n",
            encoding="utf-8",
        )
        try:
            # ``-C overflow-checks=on`` keeps the v2 integer-semantics subjects
            # honest: C produces a (possibly UB-tainted) value, idiomatic Rust
            # panics on overflow, and the oracle records the divergence.
            compile_proc = subprocess.run(
                ["rustc", "-C", "overflow-checks=on", "harness.rs", "-o", "harness"],
                cwd=tmp,
                text=True,
                capture_output=True,
                timeout=_compile_timeout(timeout),
            )
        except subprocess.TimeoutExpired as exc:
            self._compiled[key] = Observation(kind="timeout", stdout=_timeout_text(exc.stdout), stderr=_timeout_text(exc.stderr), timeout=True, exit_code=124)
            return self._compiled[key]
        if compile_proc.returncode != 0:
            self._compiled[key] = Observation(kind="compile_error", stderr=compile_proc.stderr, error=_last_line(compile_proc.stderr))
            return self._compiled[key]
        self._compiled[key] = tmp / "harness"
        return self._compiled[key]


def adapter_for(language: str) -> LanguageAdapter:
    language = language.lower()
    if language == "python":
        return PythonAdapter()
    if language == "java":
        return JavaAdapter()
    if language == "c":
        return CAdapter()
    if language == "rust":
        return RustAdapter()
    raise ValueError(f"unsupported language: {language}")


def _run_process(cmd: list[str], cwd: Path, timeout: float) -> Observation:
    proc = subprocess.Popen(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        return Observation(kind="timeout", stdout=stdout or "", stderr=stderr or "", timeout=True, exit_code=124)
    returncode = proc.returncode
    stdout = stdout or ""
    stderr = stderr or ""
    stdout = stdout.strip()
    if returncode != 0:
        return Observation(kind="runtime_error", stdout=stdout, stderr=stderr, exit_code=returncode)
    value = None
    for line in stdout.splitlines():
        if line.startswith("RETURN:"):
            value = line.removeprefix("RETURN:").strip()
            break
        if line.startswith("EXCEPTION:"):
            return Observation(kind="exception", value=line.removeprefix("EXCEPTION:").strip(), stdout=stdout, exit_code=0)
    return Observation(kind="return", value=value, stdout=stdout, stderr=stderr, exit_code=0)


def _compile_timeout(timeout: float) -> float:
    return max(timeout, 20.0)


def _timeout_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _java_harness(args: list[Any]) -> str:
    arg_exprs = ", ".join(_java_literal(value) for value in args)
    return textwrap.dedent(
        f"""
        import java.util.*;
        public class Harness {{
          public static void main(String[] args) {{
            try {{
              Object value = Solution.solve({arg_exprs});
              if (value != null && value.getClass().isArray()) {{
                if (value instanceof int[]) System.out.println("RETURN:" + Arrays.toString((int[]) value));
                else if (value instanceof long[]) System.out.println("RETURN:" + Arrays.toString((long[]) value));
                else if (value instanceof boolean[]) System.out.println("RETURN:" + Arrays.toString((boolean[]) value));
                else System.out.println("RETURN:" + Arrays.deepToString((Object[]) value));
              }} else {{
                System.out.println("RETURN:" + String.valueOf(value));
              }}
            }} catch (Throwable t) {{
              System.out.println("EXCEPTION:" + t.getClass().getSimpleName());
            }}
          }}
        }}
        """
    )


def _java_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list) and all(isinstance(item, int) for item in value):
        return "new int[]{" + ",".join(str(item) for item in value) + "}"
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return "new String[]{" + ",".join(json.dumps(item) for item in value) + "}"
    raise ValueError(f"unsupported Java harness literal: {value!r}")


def _c_args(args: list[Any]) -> str:
    return _c_call(args)[1]


def _rust_args(args: list[Any]) -> str:
    return ", ".join(_rust_literal(value) for value in args)


def _arg_shape(args: list[Any]) -> tuple[str, ...]:
    shape = []
    for value in args:
        if isinstance(value, bool):
            shape.append("bool")
        elif isinstance(value, int):
            shape.append("int")
        elif isinstance(value, str):
            shape.append("string")
        elif isinstance(value, list) and all(isinstance(item, int) for item in value):
            shape.append("int_array")
        else:
            raise ValueError(f"unsupported argument shape: {value!r}")
    return tuple(shape)


def _argv_args(args: list[Any]) -> list[str]:
    values = []
    for value in args:
        if isinstance(value, bool):
            values.append("1" if value else "0")
        elif isinstance(value, int):
            values.append(str(value))
        elif isinstance(value, str):
            values.append(value)
        elif isinstance(value, list) and all(isinstance(item, int) for item in value):
            values.append(",".join(str(item) for item in value))
        else:
            raise ValueError(f"unsupported command-line argument: {value!r}")
    return values


def _c_cli_call(shape: tuple[str, ...]) -> tuple[str, str]:
    setup: list[str] = []
    exprs: list[str] = []
    argv_idx = 1
    for idx, kind in enumerate(shape):
        if kind == "bool":
            setup.append(f"int x{idx} = atoi(argv[{argv_idx}]) != 0; ")
            exprs.append(f"x{idx}")
        elif kind == "int":
            setup.append(f"int x{idx} = atoi(argv[{argv_idx}]); ")
            exprs.append(f"x{idx}")
        elif kind == "string":
            setup.append(f"const char *x{idx} = argv[{argv_idx}]; ")
            exprs.append(f"x{idx}")
        elif kind == "int_array":
            setup.append(f"int x{idx}[256]; int x{idx}_len = parse_int_array(argv[{argv_idx}], x{idx}, 256); ")
            exprs.extend([f"x{idx}", f"x{idx}_len"])
        argv_idx += 1
    return "".join(setup), ", ".join(exprs)


def _rust_cli_call(shape: tuple[str, ...]) -> tuple[str, str]:
    setup: list[str] = []
    exprs: list[str] = []
    argv_idx = 1
    for idx, kind in enumerate(shape):
        if kind == "bool":
            setup.append(f"let x{idx}: bool = argv[{argv_idx}] == \"1\" || argv[{argv_idx}].eq_ignore_ascii_case(\"true\"); ")
            exprs.append(f"x{idx}")
        elif kind == "int":
            setup.append(f"let x{idx}: i32 = argv[{argv_idx}].parse().unwrap(); ")
            exprs.append(f"x{idx}")
        elif kind == "string":
            setup.append(f"let x{idx}: &str = argv[{argv_idx}].as_str(); ")
            exprs.append(f"x{idx}")
        elif kind == "int_array":
            setup.append(f"let x{idx}: Vec<i32> = parse_i32_list(&argv[{argv_idx}]); ")
            exprs.append(f"&x{idx}")
        argv_idx += 1
    return "".join(setup), ", ".join(exprs)


def _c_call(args: list[Any]) -> tuple[str, str]:
    setup: list[str] = []
    exprs: list[str] = []
    for idx, value in enumerate(args):
        if isinstance(value, bool):
            exprs.append("1" if value else "0")
        elif isinstance(value, int):
            exprs.append(str(value))
        elif isinstance(value, str):
            setup.append(f"const char *x{idx} = {_c_string(value)}; ")
            exprs.append(f"x{idx}")
        elif isinstance(value, list) and all(isinstance(item, int) for item in value):
            values = ",".join(str(item) for item in value) if value else "0"
            setup.append(f"int x{idx}[] = {{{values}}}; int x{idx}_len = {len(value)}; ")
            exprs.extend([f"x{idx}", f"x{idx}_len"])
        else:
            raise ValueError(f"unsupported C harness literal: {value!r}")
    return "".join(setup), ", ".join(exprs)


def _c_string(value: str) -> str:
    return json.dumps(value)


def _c_return_kind(source: str, entrypoint: str) -> str:
    pattern = rf"(?:^|[\s;])([A-Za-z_][A-Za-z0-9_\s\*]*?)\s+{re_escape(entrypoint)}\s*\("
    match = re_search(pattern, source)
    ret = " ".join(match.group(1).replace("*", " * ").split()).lower() if match else ""
    if ret in {"bool", "_bool"} or ret.endswith(" bool"):
        return "bool"
    if "char *" in ret or ret.endswith("char*"):
        return "string"
    return "int"


def _rust_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list) and all(isinstance(item, int) for item in value):
        return "&[" + ",".join(str(item) for item in value) + "]"
    raise ValueError(f"unsupported Rust harness literal: {value!r}")


def re_search(pattern: str, text: str):
    import re

    return re.search(pattern, text)


def re_escape(text: str) -> str:
    import re

    return re.escape(text)


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _first_ubsan_message(stderr: str) -> str:
    """Extract the first UBSan ``runtime error:`` message for the observation value."""
    for line in stderr.splitlines():
        stripped = line.strip()
        if "runtime error" in stripped.lower():
            return stripped
    return "undefined_behavior"
