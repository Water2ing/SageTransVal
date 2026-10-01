from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.aggregate_exp import collect, grouped_stats, write_figures, write_tables
from scripts.real_python_exp import ensure_strict_real_manifest


class ExperimentScriptTests(unittest.TestCase):
    def test_aggregator_handles_seeded_and_deterministic_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results" / "stub" / "medium" / "h16"
            fixed = root / "fixed" / "subject_a"
            random = root / "random" / "seed_0" / "subject_a"
            for path, seed in [(fixed, None), (random, 0)]:
                path.mkdir(parents=True)
                event = {
                    "subject_id": "subject_a",
                    "step": 0,
                    "action": "GENERATE_BOUNDARY_TESTS",
                    "result": {
                        "cost": {"tokens": 0, "runtime_seconds": 0.01},
                        "failures": 1,
                        "generated_tests": [{"args": [0]}],
                        "coverage_delta": 0.25,
                        "duplicate_tests": 0,
                        "invalid_tests": 0,
                        "flaky_tests": 0,
                    },
                    "reward": 10.0,
                    "budget": {"runtime_seconds": 29.9},
                    "provider": "stub",
                    "model": "stub",
                    "seed": seed,
                    "subject_metadata": {"dataset": "humaneval", "profile": "full_pilot"},
                }
                (path / "events.jsonl").write_text(__import__("json").dumps(event) + "\n", encoding="utf-8")
            rows = collect(Path(tmp) / "results")
            self.assertEqual(rows[0]["horizon"], "16")
            self.assertEqual(rows[0]["dataset"], "humaneval")
            self.assertEqual(rows[0]["profile"], "full_pilot")
            stats = grouped_stats(rows, ["provider", "model", "dataset", "budget", "horizon", "planner", "seed"], ["mp1k", "return"])
            planners = {(row["planner"], row["seed"]) for row in stats}
            self.assertIn(("fixed", ""), planners)
            self.assertIn(("random", "0"), planners)

    def test_table_and_plot_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "paper_tables" / "exp"
            rows = []
            for planner in ["existing", "fixed", "random", "greedy", "bandit", "dqn"]:
                rows.append(
                    {
                        "provider": "stub",
                        "model": "stub",
                        "dataset": "humaneval",
                        "budget": "medium",
                        "horizon": "16",
                        "planner": planner,
                        "seed": "",
                        "subjects": 1,
                        "episodes": 1,
                        "ttfm_mean": 0.0,
                        "tokfm_mean": 0.0,
                        "mp1k_mean": 1.0,
                        "mpm_mean": 1.0,
                        "coverage_mean": 0.5,
                        "false_reassurance_mean": 0.0,
                        "budget_exhausted_mean": 0.0,
                        "duplicate_rate_mean": 0.0,
                        "return_mean": 1.0,
                    }
                )
            budget_rows = rows + [
                {**rows[-1], "budget": "low", "episodes": 1},
                {**rows[-1], "budget": "high", "episodes": 1},
            ]
            raw_rows = [
                {
                    "provider": "stub",
                    "model": "stub",
                    "dataset": "humaneval",
                    "budget": "medium",
                    "horizon": "16",
                    "planner": planner,
                    "seed": "",
                    "subject": "subject_a",
                    "failures": 1,
                    "ttfm": 0.0,
                    "tokfm": 0.0,
                    "false_reassurance": 0.0,
                    "reward_by_step": {0: 1.0},
                    "actions": {"GENERATE_BOUNDARY_TESTS": 1, "STOP": 1},
                }
                for planner in ["existing", "fixed", "random", "greedy", "bandit", "dqn"]
            ]
            write_tables(out, rows, budget_rows, raw_rows)
            fig_dir = Path(tmp) / "figures"
            write_figures(out, rows, raw_rows, fig_dir)
            self.assertTrue((out / "results_tables.tex").exists())
            self.assertNotIn("todoresult", (out / "results_tables.tex").read_text(encoding="utf-8"))
            self.assertTrue((fig_dir / "failure_efficiency.tex").exists())

    def test_stress_table_is_separate_and_omits_mp1k(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "paper_tables" / "stress"
            raw_rows = []
            metrics_rows = []
            for planner in ["greedy", "bandit", "dqn"]:
                raw_rows.append(
                    {
                        "profile": "stress",
                        "provider": "stub",
                        "model": "stub",
                        "dataset": "stress",
                        "budget": "medium",
                        "horizon": "16",
                        "planner": planner,
                        "seed": "",
                        "subject": f"subject_{planner}",
                        "failures": 1,
                        "ttfm": 1.0,
                        "tokfm": 0.0,
                        "coverage": 0.3,
                        "duplicate_rate": 0.2,
                        "llm_action_share": 0.1,
                        "false_reassurance": 0.0,
                        "return": 2.0,
                        "reward_by_step": {0: 1.0},
                        "actions": {"GENERATE_BOUNDARY_TESTS": 1},
                    }
                )
                metrics_rows.append(
                    {
                        "profile": "stress",
                        "provider": "stub",
                        "model": "stub",
                        "dataset": "stress",
                        "budget": "medium",
                        "horizon": "16",
                        "planner": planner,
                        "seed": "",
                        "subjects": 1,
                        "episodes": 1,
                        "coverage_mean": 0.3,
                        "duplicate_rate_mean": 0.2,
                        "llm_action_share_mean": 0.1,
                        "return_mean": 2.0,
                    }
                )
            write_tables(out, metrics_rows, metrics_rows, raw_rows)
            tex = (out / "results_tables.tex").read_text(encoding="utf-8")
            self.assertIn("tab:stress-separation", tex)
            self.assertNotIn("Failures / 1K tokens", tex)

    def test_strict_real_manifest_guard_accepts_only_humaneval_and_mbpp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "subjects"
            root.mkdir()
            rows = [
                {"id": "h", "metadata": {"dataset": "humaneval_real"}},
                {"id": "m", "metadata": {"dataset": "mbpp_real"}},
            ]
            (root / "manifest.json").write_text(__import__("json").dumps(rows), encoding="utf-8")
            ensure_strict_real_manifest(root)

    def test_strict_real_manifest_guard_rejects_proxy_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "subjects"
            root.mkdir()
            rows = [
                {"id": "h", "metadata": {"dataset": "humaneval_real"}},
                {"id": "m", "metadata": {"dataset": "mbpp_real"}},
                {"id": "b", "metadata": {"dataset": "bugsinpy_real"}},
            ]
            (root / "manifest.json").write_text(__import__("json").dumps(rows), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "bugsinpy_real"):
                ensure_strict_real_manifest(root)


if __name__ == "__main__":
    unittest.main()
