"""KLEE preprocessor (eval-plan E7).

Reads a SubjectV2 C source, inserts ``klee_make_symbolic`` annotations for the
declared parameter types, and emits a self-contained C wrapper that calls the
entry point with symbolic inputs. The resulting source compiles to LLVM bitcode
via ``clang -emit-llvm -c -g -O0 -Xclang -disable-O0-optnone``.

Designed to run inside the ``sage-klee:3.0`` Docker image where klee + clang are
installed; works standalone on the host as well if clang and KLEE are on PATH.

Usage (inside or outside the container):
    python klee_preprocess.py \
        --subject /work/data/crust_bench/<subject_id>/metadata.json \
        --out /work/bitcode/<subject_id>.bc
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import textwrap


def _klee_decl(param_name: str, type_spec: str) -> tuple[str, str]:
    """Return (declaration_line, klee_make_symbolic_call) for one parameter."""
    if type_spec == "i32":
        return f"int32_t {param_name};", f'klee_make_symbolic(&{param_name}, sizeof({param_name}), "{param_name}");'
    if type_spec == "i64":
        return f"int64_t {param_name};", f'klee_make_symbolic(&{param_name}, sizeof({param_name}), "{param_name}");'
    if type_spec == "u32":
        return f"uint32_t {param_name};", f'klee_make_symbolic(&{param_name}, sizeof({param_name}), "{param_name}");'
    if type_spec == "u64":
        return f"uint64_t {param_name};", f'klee_make_symbolic(&{param_name}, sizeof({param_name}), "{param_name}");'
    if type_spec == "f64":
        return f"double {param_name};", f'klee_make_symbolic(&{param_name}, sizeof({param_name}), "{param_name}");'
    if type_spec == "bool":
        return f"int {param_name};", f'klee_make_symbolic(&{param_name}, sizeof({param_name}), "{param_name}");'
    if type_spec == "string":
        return f"char {param_name}[16]; {param_name}[15] = 0;", \
               f'klee_make_symbolic({param_name}, 15, "{param_name}");'
    if type_spec.startswith("array<"):
        return f"int64_t {param_name}[8];", \
               f'klee_make_symbolic({param_name}, sizeof({param_name}), "{param_name}");'
    raise NotImplementedError(f"unsupported parameter type for KLEE: {type_spec}")


def build_wrapper(source: str, entrypoint: str, parameter_types: dict[str, str]) -> str:
    decls = []
    calls = []
    args_list = []
    for name, t in parameter_types.items():
        decl, call = _klee_decl(name, t)
        decls.append("    " + decl)
        calls.append("    " + call)
        args_list.append(name)
    wrapper = textwrap.dedent("""\
        #include <stdint.h>
        #include <stdio.h>
        #include <klee/klee.h>
        {source}
        int main() {{
        {decls}
        {calls}
            {entrypoint}({args});
            return 0;
        }}
    """).format(
        source=source,
        decls="\n".join(decls),
        calls="\n".join(calls),
        entrypoint=entrypoint,
        args=", ".join(args_list),
    )
    return wrapper


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--clang", default="clang")
    args = ap.parse_args()
    with args.subject.open() as f:
        meta = json.load(f)
    base = args.subject.parent
    source = (base / meta["source_path"]).read_text()
    wrapper_src = build_wrapper(source, meta["entrypoint"], meta.get("parameter_types", {}))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    wrapper_path = args.out.with_suffix(".c")
    wrapper_path.write_text(wrapper_src)
    proc = subprocess.run(
        [args.clang, "-emit-llvm", "-c", "-g", "-O0",
         "-Xclang", "-disable-O0-optnone",
         "-I", "/usr/include/klee",
         str(wrapper_path), "-o", str(args.out)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return proc.returncode
    print(f"bitcode: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
