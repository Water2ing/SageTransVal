from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sagetransval.adapters import CAdapter, JavaAdapter, RustAdapter
from sagetransval.cli import main as cli_main
from sagetransval.full_pipeline import prepare_subjects, write_repair_packets
from sagetransval.models import TranslationSubjectRecord
from sagetransval.obligations import extract_c_obligations, extract_python_obligations


class SageTransValFullTests(unittest.TestCase):
    def test_windows_manifest_paths_resolve_on_linux(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            target = root / "nested" / "source.py"
            target.parent.mkdir()
            target.write_text("def solve(): return 1\n", encoding="utf-8")
            record = TranslationSubjectRecord.from_json(
                {"id": "demo", "source_language": "python", "target_language": "java",
                 "source_path": "nested\\source.py", "target_path": None, "entrypoint": "solve",
                 "seed_tests": []}, base=root)
            self.assertEqual(Path(record.source_path), target)

    def test_subject_schema_round_trips(self) -> None:
        record = TranslationSubjectRecord(
            id="p2j_demo",
            source_language="python",
            target_language="java",
            source_path="source.py",
            target_path="Solution.java",
            entrypoint="solve",
            seed_tests=[{"args": [1]}],
            metadata={"dataset": "fixture"},
        )
        loaded = TranslationSubjectRecord.from_json(record.to_json())
        self.assertEqual(loaded.id, "p2j_demo")
        self.assertEqual(loaded.seed_tests[0]["args"], [1])

    def test_java_adapter_compiles_and_runs_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            path = Path(tmp_name) / "Solution.java"
            path.write_text(
                "public class Solution { public static Object solve(int x) { return x < 0 ? -x : x; } }\n",
                encoding="utf-8",
            )
            obs = JavaAdapter().run(path, "solve", [-3])
        self.assertEqual(obs.comparable(), ("return", "3"))

    def test_rust_adapter_compiles_and_runs_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            path = Path(tmp_name) / "solution.rs"
            path.write_text("pub fn solve(x: i32) -> i32 { if x < 0 { -x } else { x } }\n", encoding="utf-8")
            obs = RustAdapter().run(path, "solve", [-4])
        self.assertEqual(obs.comparable(), ("return", "4"))

    def test_c_and_rust_adapters_handle_bool_string_and_array_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            c_bool = root / "bool.c"
            c_bool.write_text("#include <stdbool.h>\nbool solve(int x) { return x % 2 == 0; }\n", encoding="utf-8")
            self.assertEqual(CAdapter().run(c_bool, "solve", [2]).comparable(), ("return", "true"))

            c_string = root / "string.c"
            c_string.write_text("int solve(const char *s) { int n = 0; while (s[n]) n++; return n; }\n", encoding="utf-8")
            r_string = root / "solution.rs"
            r_string.write_text("pub fn solve(s: &str) -> i32 { s.len() as i32 }\n", encoding="utf-8")
            self.assertEqual(CAdapter().run(c_string, "solve", ["abc"]).comparable(), ("return", "3"))
            self.assertEqual(RustAdapter().run(r_string, "solve", ["abc"]).comparable(), ("return", "3"))

            c_array = root / "array.c"
            c_array.write_text("int solve(const int *xs, int n) { int s = 0; for (int i = 0; i < n; i++) s += xs[i]; return s; }\n", encoding="utf-8")
            r_array = root / "array.rs"
            r_array.write_text("pub fn solve(xs: &[i32]) -> i32 { xs.iter().sum() }\n", encoding="utf-8")
            self.assertEqual(CAdapter().run(c_array, "solve", [[1, 2, 3]]).comparable(), ("return", "6"))
            self.assertEqual(RustAdapter().run(r_array, "solve", [[1, 2, 3]]).comparable(), ("return", "6"))

    def test_prepare_c_rust_100_has_signature_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            report = prepare_subjects("c_rust_100", Path(tmp_name) / "c_rust_100", limit=100)
            manifest = json.loads((Path(tmp_name) / "c_rust_100" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(report["imported"], 100)
        self.assertEqual(report["real_imported"], 60)
        self.assertEqual(report["diagnostic_generated"], 40)
        self.assertEqual(len(manifest["subjects"]), 100)
        for subject in manifest["subjects"]:
            self.assertEqual(subject["source_language"], "c")
            self.assertEqual(subject["target_language"], "rust")
            self.assertEqual(subject["entrypoint"], "solve")
            self.assertTrue(subject["seed_tests"])
            self.assertIn("signature", subject["metadata"])

    def test_cli_prepare_c_rust_v2_writes_portable_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            subjects = Path(tmp_name) / "c_rust_v2"
            cli_main(["prepare", "--dataset", "c_rust_v2", "--limit", "3", "--out", str(subjects)])
            manifest = json.loads((subjects / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["subjects"]), 3)
        for subject in manifest["subjects"]:
            self.assertEqual(subject["source_language"], "c")
            self.assertEqual(subject["target_language"], "rust")
            self.assertFalse(Path(subject["source_path"]).is_absolute())

    def test_obligation_extractors_emit_expected_families(self) -> None:
        py = "def solve(x):\n    if x < 0:\n        return 0\n    return 10 // x\n"
        c = "int solve(int x) { if (x < 0) return 0; return 10 / x; }\n"
        py_obligations = extract_python_obligations(py, "solve", [{"args": [1]}])
        c_obligations = extract_c_obligations(c, [{"args": [1]}])
        self.assertIn("control flow", py_obligations)
        self.assertIn("exceptions", c_obligations)

    def test_stub_pipeline_prepare_translate_validate_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            subjects = root / "subjects"
            translated = root / "translated"
            results = root / "results"
            tables = root / "tables"
            figures = root / "figures"
            cli_main(["prepare", "--dataset", "c_rust", "--limit", "1", "--out", str(subjects)])
            cli_main(["translate", "--subjects", str(subjects), "--out", str(translated), "--provider", "stub"])
            cli_main(["validate", "--subjects", str(translated), "--out", str(results)])
            cli_main(["aggregate", "--results", str(results), "--out", str(tables), "--figures", str(figures)])
            manifest = json.loads((translated / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["subjects"]), 1)
            self.assertTrue((tables / "results_tables.tex").exists())
            self.assertTrue((figures / "proposal1_full_detection.tex").exists())

    def test_stub_pipeline_c_rust_100_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            subjects = root / "subjects"
            translated = root / "translated"
            results = root / "results"
            tables = root / "tables"
            figures = root / "figures"
            cli_main(["prepare", "--dataset", "c_rust_100", "--limit", "10", "--out", str(subjects)])
            cli_main(["translate", "--subjects", str(subjects), "--out", str(translated), "--provider", "stub"])
            cli_main(["validate", "--subjects", str(translated), "--out", str(results)])
            cli_main(["aggregate", "--results", str(results), "--out", str(tables), "--figures", str(figures)])
            manifest = json.loads((translated / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["subjects"]), 10)
            self.assertTrue((tables / "results_tables.tex").exists())

    def test_repair_packets_emit_claimed_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            results = root / "results"
            packets = root / "packets"
            results.mkdir()
            row = {
                "subject": "demo",
                "pair": "python->java",
                "mode": "source_obligation_guided_repair",
                "first_mismatch": 0,
                "mismatches": 1,
                "repair_packet": True,
                "mismatch_examples": [
                    {
                        "index": 0,
                        "obligation_id": "boundary_type:000",
                        "case": {"args": [1], "obligation_id": "boundary_type:000"},
                        "source": {"kind": "return", "value": 1},
                        "target": {"kind": "return", "value": 2},
                    }
                ],
            }
            (results / "validation_events.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            count = write_repair_packets(results, packets)
            packet = json.loads((packets / "repair_packets.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(count, 1)
        self.assertEqual(
            set(packet),
            {
                "source_span",
                "target_span",
                "normalized_input",
                "source_observation",
                "target_observation",
                "obligation_id",
                "error_class",
                "normalization_policy",
            },
        )
        self.assertEqual(packet["error_class"], "value_mismatch")


if __name__ == "__main__":
    unittest.main()
