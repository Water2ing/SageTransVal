from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from testgenrl.cli import main


class IntegrationTests(unittest.TestCase):
    def test_fixed_random_train_eval_report_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            subjects = tmp_path / "subjects"
            runs = tmp_path / "runs"
            cache = tmp_path / "llm_cache"
            model = tmp_path / "models" / "dqn.json"
            results = tmp_path / "results"
            report = tmp_path / "paper_tables"

            main(["prepare", "--dataset", "toy", "--out", str(subjects)])
            main(["run", "--planner", "fixed", "--subjects", str(subjects), "--split", "test", "--cache-dir", str(cache), "--out", str(runs)])
            main(["run", "--planner", "random", "--subjects", str(subjects), "--split", "test", "--seed", "0", "--cache-dir", str(cache), "--out", str(runs)])
            main(["train", "--runs", str(runs), "--out", str(model)])
            main(["eval", "--model", str(model), "--subjects", str(subjects), "--split", "test", "--cache-mode", "replay", "--cache-dir", str(cache), "--out", str(results)])
            main(["report", "--runs", str(runs), str(results), "--out", str(report)])

            self.assertTrue((report / "metrics.csv").exists())
            metrics = (report / "metrics.csv").read_text(encoding="utf-8")
            self.assertIn("provider", metrics)
            self.assertIn("model", metrics)
            self.assertTrue((runs / "random" / "seed_0").exists())

    def test_real_python_cli_smoke_flow_with_stub(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw = tmp_path / "raw"
            raw.mkdir()
            he = {
                "task_id": "HumanEval/0",
                "prompt": "def inc(x: int) -> int:\n",
                "entry_point": "inc",
                "canonical_solution": "    return x + 1\n",
                "test": "def check(candidate):\n    assert candidate(1) == 2\n",
            }
            (raw / "HumanEval.jsonl.gz").write_bytes(gzip.compress(json.dumps(he).encode("utf-8")))
            mbpp = [
                {
                    "task_id": 1,
                    "prompt": "increment",
                    "code": "def inc2(x):\n    return x + 2\n",
                    "test_imports": [],
                    "test_list": ["assert inc2(1) == 3"],
                }
            ]
            (raw / "sanitized-mbpp.json").write_text(json.dumps(mbpp), encoding="utf-8")

            subjects = tmp_path / "real_subjects"
            runs = tmp_path / "runs"
            main(
                [
                    "prepare",
                    "--dataset",
                    "real_python",
                    "--source-root",
                    str(raw),
                    "--limit-per-source",
                    "1",
                    "--validate",
                    "--out",
                    str(subjects),
                ]
            )
            main(["run", "--planner", "fixed", "--subjects", str(subjects), "--split", "test", "--out", str(runs)])
            self.assertTrue((subjects / "import_report.json").exists())

    def test_replay_mode_fails_on_missing_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            subjects = tmp_path / "subjects"
            main(["prepare", "--dataset", "toy", "--out", str(subjects)])
            with self.assertRaisesRegex(RuntimeError, "cache miss"):
                main(
                    [
                        "run",
                        "--planner",
                        "fixed",
                        "--subjects",
                        str(subjects),
                        "--split",
                        "test",
                        "--cache-mode",
                        "replay",
                        "--out",
                        str(tmp_path / "runs"),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
