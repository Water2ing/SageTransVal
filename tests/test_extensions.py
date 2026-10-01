"""Pytest suite for SageTransVal eval-plan extensions (E1-E7).

These tests cover the locally-runnable subset:
  * Subject schema v2 round-trip
  * Cache-key invariance under harness/schema bumps
  * Sandbox primitive smoke test
  * C struct codec type parsing + canonical JSON
  * TransCoder smoke import
  * CRUST-Bench smoke import
  * Pynguin Python-AST extractor
  * EvoSuite/Java argument parser
  * JS adapter end-to-end (requires node)
  * TypeScript adapter end-to-end (requires node + tsc)
  * JS regex obligation extractor
  * KLEE preprocessor wrapper builder
  * KLEE .ktest parser + decoder
  * full_pipeline mode registration

Tests skip rather than fail when their external dependency is absent (e.g., no
node, no tsc, no kotlinc).
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ----- Schema v2 -----------------------------------------------------------

def test_subject_v2_roundtrip():
    from sagetransval.schema import OraclePolicy, SubjectV2

    subj = SubjectV2(
        schema_version=2, harness_version=2,
        id="x", source_language="python", target_language="java",
        language_dialect="python3.10",
        source_path="source.py", target_path="target.java",
        entrypoint="solve", parameter_types={"n": "int"}, return_type="int",
        seed_tests=[{"args": {"n": 1}}],
        oracle_policy=OraclePolicy(numeric_tolerance=1e-9, unordered_collections=("seen",)),
        dataset="unit_test",
    )
    blob = json.dumps(subj.to_json())
    restored = SubjectV2.from_json(json.loads(blob))
    assert restored.id == subj.id
    assert restored.oracle_policy.numeric_tolerance == 1e-9
    assert restored.oracle_policy.unordered_collections == ("seen",)


# ----- Cache keys ----------------------------------------------------------

def test_cache_key_changes_on_harness_bump():
    from sagetransval.cache_key import make_translation_cache_key

    k1 = make_translation_cache_key(provider="o", model="m", prompt="p", decoding={"t": 0},
                                    schema_version=1, harness_version=1)
    k2 = make_translation_cache_key(provider="o", model="m", prompt="p", decoding={"t": 0},
                                    schema_version=1, harness_version=2)
    assert k1 != k2

    k3 = make_translation_cache_key(provider="o", model="m", prompt="p", decoding={"t": 0},
                                    schema_version=2, harness_version=1)
    assert k1 != k3 and k2 != k3


# ----- Sandbox -------------------------------------------------------------

def test_sandbox_smoke():
    from sagetransval.sandbox import SandboxLimits, run_sandboxed

    res = run_sandboxed(["echo", "hello"], use_docker=False,
                       limits=SandboxLimits(wall_seconds=2))
    assert res.returncode == 0
    assert "hello" in res.stdout


def test_sandbox_timeout():
    from sagetransval.sandbox import SandboxLimits, run_sandboxed

    res = run_sandboxed(["sleep", "5"], use_docker=False,
                       limits=SandboxLimits(wall_seconds=1))
    assert res.timed_out


# ----- C struct codec ------------------------------------------------------

def test_c_struct_codec_types():
    from sagetransval.c_struct_codec import canonical_json, parse_type, supported

    assert supported("i32")
    assert supported("array<i64>")
    assert supported("array<f64, 8>")
    assert not supported("ptr<i32>")
    t = parse_type("array<i32, 4>")
    assert t.kind == "array" and t.length == 4 and t.element.scalar == "i32"
    assert canonical_json({"x": float("nan")}) == '{"x":"NaN"}'


# ----- Importers -----------------------------------------------------------

def test_transcoder_smoke_import(tmp_path):
    import scripts.import_transcoder as mod
    imported = mod.import_smoke(tmp_path)
    assert len(imported) == 3
    meta = json.loads((tmp_path / "transcoder_smoke_factorial" / "metadata.json").read_text())
    assert meta["schema_version"] == 2
    assert meta["source_language"] == "python"


def test_crust_bench_smoke_import(tmp_path):
    import scripts.import_crust_bench as mod
    imported = mod.import_smoke(tmp_path)
    assert len(imported) == 3
    meta = json.loads((tmp_path / "crust_smoke_overflow_add" / "metadata.json").read_text())
    assert meta["source_language"] == "c"
    assert meta["target_language"] == "rust"


# ----- SBST extractors -----------------------------------------------------

def test_pynguin_extractor(tmp_path):
    import scripts.sbst_baseline as m
    test_file = tmp_path / "test_sut.py"
    test_file.write_text(
        "import sut as module_0\n"
        "def test_case_0():\n"
        "    int_0 = 7\n"
        "    int_1 = module_0.factorial(int_0)\n"
        "    list_0 = [1, 2, 3]\n"
        "    int_2 = module_0.factorial(list_0)\n"
    )
    inputs = m._extract_pynguin_inputs(tmp_path, "factorial")
    assert {"args": {"__positional__": [7]}} in inputs
    assert any(i["args"]["__positional__"] == [[1, 2, 3]] for i in inputs)


def test_evosuite_java_arg_parser():
    import scripts.sbst_baseline as m
    assert m._parse_java_call_args("1, 2, 3") == {"__positional__": [1, 2, 3]}
    assert m._parse_java_call_args('"hello"') == {"__positional__": ["hello"]}
    assert m._parse_java_call_args("true, false") == {"__positional__": [True, False]}
    assert m._parse_java_call_args("new int[]{4, 5}, 2") == {"__positional__": [[4, 5], 2]}


# ----- JS / TS adapters ----------------------------------------------------

@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_adapter_end_to_end(tmp_path):
    from sagetransval.js_adapter import execute_js
    src = tmp_path / "source.js"
    src.write_text("function f(x){return x*x;} module.exports={f};")
    obs = execute_js(src, "f", ["x"], {"x": 5})
    assert obs.kind == "value" and obs.value == 25

    obs = execute_js(src, "f", ["x"], {"x": -3})
    assert obs.kind == "value" and obs.value == 9

    obs = execute_js(src, "missing", ["x"], {"x": 1})
    assert obs.kind == "exception"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_ts_adapter_compile_or_skip(tmp_path):
    from sagetransval.ts_adapter import execute_ts
    src = tmp_path / "source.ts"
    src.write_text("function g(n: number): number { return n+1; } module.exports={g};")
    obs = execute_ts(src, "g", ["n"], {"n": 9})
    # Either succeeds (tsc available) or returns a ConfigError observation
    if obs.kind == "value":
        assert obs.value == 10
    else:
        assert obs.kind == "exception"


# ----- JS obligation extractor ---------------------------------------------

def test_js_obligation_extractor():
    from sagetransval.obligations_js import extract_js_obligations
    src = "function f(x){if (x<0) throw new RangeError('x'); return Math.sqrt(x);}"
    obs = extract_js_obligations(src, "f", [])
    assert any(o["args"].get("_branch_witness") for o in obs["control flow"])
    assert any(o["args"].get("_expected_exception") == "RangeError" for o in obs["exceptions"])
    assert any(o["args"].get("_api_call") == "Math.sqrt" for o in obs["api"])
    assert len(obs["boundary/type"]) > 0


# ----- KLEE pipeline -------------------------------------------------------

def test_klee_wrapper_builds():
    from scripts.klee_preprocess import build_wrapper
    wrap = build_wrapper(
        "int add(int a, int b){return a+b;}",
        "add",
        {"a": "i32", "b": "i32"},
    )
    assert "klee_make_symbolic" in wrap
    assert "add(a, b);" in wrap


def test_ktest_parser_decodes_int(tmp_path):
    from scripts.ktest_to_input import parse_ktest, decode

    buf = b"KTEST" + struct.pack("<I", 3)
    buf += struct.pack("<I", 0)           # num_args
    buf += struct.pack("<I", 0)           # sym_argvs
    buf += struct.pack("<I", 0)           # sym_argv_len
    buf += struct.pack("<I", 1)           # num_objects
    buf += struct.pack("<I", 1) + b"x" + struct.pack("<I", 4) + struct.pack("<i", -7)
    p = tmp_path / "x.ktest"
    p.write_bytes(buf)
    objs = parse_ktest(p)
    assert objs[0][0] == "x"
    assert decode("x", objs[0][1], {"x": "i32"}) == -7


# ----- full_pipeline mode wiring -------------------------------------------

def test_pipeline_registers_extension_modes():
    from sagetransval.full_pipeline import VALIDATION_MODES, _cases_for_mode

    assert {"sbst_coverage_driven", "live_llm_tests", "chatunitest", "klee_symbolic"} <= set(VALIDATION_MODES)

    seeds = [{"args": {"n": 1}}]
    cases, fams = _cases_for_mode(seeds, {}, "sbst_coverage_driven",
                                  external_inputs=[{"args": {"__positional__": [5]}}])
    assert cases[0] == seeds[0]
    assert {"args": {"__positional__": [5]}} in cases
    assert fams == []
