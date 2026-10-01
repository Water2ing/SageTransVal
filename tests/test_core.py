from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from testgenrl.actions import Action, action_mask, masked_argmax
from testgenrl.cli import build_parser
from testgenrl.dqn import ReplayBuffer, masked_double_dqn_target
from testgenrl.environment import TestGenEnvironment
from testgenrl.executors import ExecutorSuite
import testgenrl.llm_cache as llm_cache_module
from testgenrl.llm_cache import LLMCache, _api_key
from testgenrl.models import Budget, Transition
from testgenrl.planners import planner_from_name
from testgenrl.reward import compute_reward
from testgenrl.state import STATE_KEYS, EvidenceState
from testgenrl.subjects import load_subjects, prepare_builtin_dataset, prepare_dataset


def _write_sample_real_sources(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    he_rows = [
        {
            "task_id": f"HumanEval/{idx}",
            "prompt": f"def add_{idx}(x: int) -> int:\n",
            "entry_point": f"add_{idx}",
            "canonical_solution": f"    return x + {idx}\n",
            "test": f"def check(candidate):\n    assert candidate(1) == {idx + 1}\n",
        }
        for idx in range(3)
    ]
    (root / "HumanEval.jsonl.gz").write_bytes(gzip.compress("\n".join(json.dumps(row) for row in he_rows).encode("utf-8")))
    mbpp_rows = [
        {
            "task_id": idx,
            "prompt": "add one",
            "code": f"def plus_{idx}(x):\n    return x + {idx}\n",
            "test_imports": [],
            "test_list": [f"assert plus_{idx}(1) == {idx + 1}"],
        }
        for idx in range(3)
    ]
    (root / "sanitized-mbpp.json").write_text(json.dumps(mbpp_rows), encoding="utf-8")


class CoreTests(unittest.TestCase):
    def test_state_vector_fixed_length(self) -> None:
        state = EvidenceState()
        vector = state.vector(Budget())
        self.assertEqual(len(vector), len(STATE_KEYS))

    def test_action_mask_rules(self) -> None:
        mask = action_mask({"failures_found": 0, "repair_events": 0, "uncertain_oracles": 0}, {"tokens": 0, "runtime_seconds": 10, "fuzz_iterations": 10})
        self.assertFalse(mask[int(Action.LLM_SYNTHESIZE_INSTRUCTIONS)])
        self.assertFalse(mask[int(Action.EXPAND_COUNTEREXAMPLE)])
        self.assertFalse(mask[int(Action.RETEST_REPAIR_CONTEXT)])
        self.assertTrue(mask[int(Action.STOP)])

    def test_reward_components(self) -> None:
        from testgenrl.models import ExecutionResult

        reward, components = compute_reward(ExecutionResult(action="x", failures=1, coverage_delta=0.5, invalid_tests=1, cost={"tokens": 100}))
        self.assertGreater(reward, 0)
        self.assertIn("failure", components)
        self.assertLess(components["bad_tests"], 0)

    def test_replay_buffer(self) -> None:
        transition = Transition([0.0], 0, 1.0, [1.0], [True] * len(Action), [True] * len(Action), {}, False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.json"
            buf = ReplayBuffer()
            buf.add(transition)
            buf.save(path)
            loaded = ReplayBuffer.load(path)
            self.assertEqual(len(loaded.sample(4)), 1)

    def test_masked_target_ignores_invalid_actions(self) -> None:
        target = masked_double_dqn_target(
            reward=1.0,
            gamma=0.5,
            online_next_q=[100.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0],
            target_next_q=[100.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 0.0],
            next_mask=[False, True, False, False, False, False, False, True],
        )
        self.assertEqual(target, 11.0)

    def test_llm_cache_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LLMCache(Path(tmp))
            a = cache.complete("hello", model="stub")
            b = cache.complete("hello", model="stub")
            self.assertEqual(a["key"], b["key"])
            self.assertEqual(a["text"], b["text"])

    def test_llm_cache_key_includes_provider_and_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            openai = LLMCache(Path(tmp), provider_name="openai", model="gpt-4o-mini", provider=lambda *_: "ok")
            gemini = LLMCache(Path(tmp), provider_name="gemini", model="gemini-2.5-flash-lite", provider=lambda *_: "ok")
            self.assertNotEqual(openai.key("hello", "gpt-4o-mini"), gemini.key("hello", "gemini-2.5-flash-lite"))

    def test_llm_replay_mode_raises_on_cache_miss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = LLMCache(Path(tmp), cache_mode="replay")
            with self.assertRaisesRegex(RuntimeError, "cache miss"):
                cache.complete("missing")

    def test_llm_cache_record_does_not_store_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-secret", "GEMINI_API_KEY": "gem-test-secret"}):
                cache = LLMCache(
                    Path(tmp),
                    provider_name="openai",
                    model="gpt-4o-mini",
                    provider=lambda *_: "secret-free response",
                )
                record = cache.complete("hello")
            text = "\n".join(str(value) for value in record.values())
            self.assertNotIn("OPENAI_API_KEY", text)
            self.assertNotIn("GEMINI_API_KEY", text)
            self.assertNotIn("sk-test-secret", text)
            self.assertNotIn("gem-test-secret", text)

    def test_api_key_falls_back_to_markdown_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            key_file = Path(tmp) / "openai.md"
            key_file.write_text("# local key\nOPENAI_API_KEY=sk-file-secret\n", encoding="utf-8")
            old = llm_cache_module.DEFAULT_KEY_FILES["OPENAI_API_KEY"]
            llm_cache_module.DEFAULT_KEY_FILES["OPENAI_API_KEY"] = key_file
            try:
                with patch.dict("os.environ", {}, clear=True):
                    self.assertEqual(_api_key("OPENAI_API_KEY"), "sk-file-secret")
            finally:
                llm_cache_module.DEFAULT_KEY_FILES["OPENAI_API_KEY"] = old

    def test_cli_parses_llm_options_and_seed(self) -> None:
        args = build_parser().parse_args(
            [
                "run",
                "--planner",
                "fixed",
                "--subjects",
                "data/subjects",
                "--out",
                "runs/x",
                "--llm-provider",
                "openai",
                "--llm-model",
                "gpt-4o-mini",
                "--cache-mode",
                "replay",
                "--seed",
                "4",
            ]
        )
        self.assertEqual(args.llm_provider, "openai")
        self.assertEqual(args.llm_model, "gpt-4o-mini")
        self.assertEqual(args.cache_mode, "replay")
        self.assertEqual(args.seed, 4)

    def test_cli_parses_real_dataset_prepare_options(self) -> None:
        args = build_parser().parse_args(
            [
                "prepare",
                "--dataset",
                "real_python_strict",
                "--out",
                "data/real",
                "--source-root",
                "data/raw",
                "--cache-dir",
                ".cache/raw",
                "--limit-per-source",
                "2",
                "--validate",
            ]
        )
        self.assertEqual(args.dataset, "real_python_strict")
        self.assertEqual(args.limit_per_source, 2)
        self.assertTrue(args.validate)

    def test_prepare_real_dataset_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "humaneval"
            prepare_builtin_dataset(root, "humaneval", limit=20)
            subjects = load_subjects(root)
            self.assertEqual(len(subjects), 20)
            self.assertEqual({s.metadata["dataset"] for s in subjects}, {"humaneval"})

            root = Path(tmp) / "mbpp"
            prepare_builtin_dataset(root, "mbpp", limit=20)
            subjects = load_subjects(root)
            self.assertEqual(len(subjects), 20)
            self.assertEqual({s.metadata["dataset"] for s in subjects}, {"mbpp"})

            root = Path(tmp) / "bugsinpy"
            prepare_builtin_dataset(root, "bugsinpy", limit=10)
            subjects = load_subjects(root)
            self.assertEqual(len(subjects), 10)
            self.assertEqual({s.metadata["dataset"] for s in subjects}, {"bugsinpy"})

    def test_prepare_real_humaneval_smoke_validates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            _write_sample_real_sources(raw)
            root = Path(tmp) / "humaneval_real"
            prepare_dataset(root, "humaneval_real", limit=2, validate=True, source_root=raw)
            subjects = load_subjects(root)
            self.assertEqual(len(subjects), 2)
            self.assertEqual({s.metadata["dataset"] for s in subjects}, {"humaneval_real"})
            self.assertEqual({s.metadata["oracle_type"] for s in subjects}, {"callable_check"})
            self.assertTrue((root / "import_report.json").exists())

    def test_prepare_real_mbpp_smoke_validates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            _write_sample_real_sources(raw)
            root = Path(tmp) / "mbpp_real"
            prepare_dataset(root, "mbpp_real", limit=2, validate=True, source_root=raw)
            subjects = load_subjects(root)
            self.assertEqual(len(subjects), 2)
            self.assertEqual({s.metadata["dataset"] for s in subjects}, {"mbpp_real"})
            self.assertEqual({s.metadata["profile"] for s in subjects}, {"real_python"})

    def test_prepare_combined_real_python_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            _write_sample_real_sources(raw)
            root = Path(tmp) / "real_python"
            prepare_dataset(root, "real_python", limit_per_source=2, validate=True, source_root=raw)
            subjects = load_subjects(root)
            counts = {}
            for subject in subjects:
                counts[subject.metadata["dataset"]] = counts.get(subject.metadata["dataset"], 0) + 1
            self.assertEqual(counts, {"humaneval_real": 2, "mbpp_real": 2, "bugsinpy_real": 2})
            report = json.loads((root / "import_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["imported"], 6)

    def test_prepare_real_python_strict_excludes_bugsinpy_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            _write_sample_real_sources(raw)
            root = Path(tmp) / "real_python_strict"
            prepare_dataset(root, "real_python_strict", limit_per_source=2, validate=True, source_root=raw)
            subjects = load_subjects(root)
            counts = {}
            for subject in subjects:
                counts[subject.metadata["dataset"]] = counts.get(subject.metadata["dataset"], 0) + 1
                self.assertEqual(subject.metadata["profile"], "real_python")
            self.assertEqual(counts, {"humaneval_real": 2, "mbpp_real": 2})
            self.assertNotIn("bugsinpy_real", counts)

    def test_callable_check_oracle_executes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            _write_sample_real_sources(raw)
            root = Path(tmp) / "humaneval_real"
            prepare_dataset(root, "humaneval_real", limit=1, validate=True, source_root=raw)
            subject = load_subjects(root)[0]
            result = ExecutorSuite(Path(tmp) / "cache").execute(Action.EXPAND_COUNTEREXAMPLE, subject)
            self.assertEqual(result.invalid_tests, 0)
            self.assertEqual(result.failures, 0)
            self.assertEqual(result.generated_tests[0]["status"], "passed")

    def test_prepare_full_pilot_subjects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "full"
            prepare_builtin_dataset(root, "full_pilot")
            subjects = load_subjects(root)
            self.assertEqual(len(subjects), 50)
            counts = {}
            for subject in subjects:
                counts[subject.metadata["dataset"]] = counts.get(subject.metadata["dataset"], 0) + 1
            self.assertEqual(counts, {"humaneval": 20, "mbpp": 20, "bugsinpy": 10})

    def test_prepare_stress_subjects_marks_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "stress"
            prepare_builtin_dataset(root, "stress")
            subjects = load_subjects(root)
            self.assertGreaterEqual(len(subjects), 12)
            self.assertEqual({s.metadata["profile"] for s in subjects}, {"stress"})
            self.assertIn("stress_category", subjects[0].metadata)

    def test_prepare_diverse_full_subjects_has_all_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "diverse"
            prepare_builtin_dataset(root, "diverse_full")
            subjects = load_subjects(root)
            profiles = {s.metadata["profile"] for s in subjects}
            self.assertEqual(profiles, {"full_pilot", "stress", "action_balanced"})
            self.assertEqual(len(subjects), 74)

    def test_stress_repeated_boundary_has_duplicate_penalty_and_saturation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "stress"
            prepare_builtin_dataset(root, "stress")
            subject = next(s for s in load_subjects(root) if s.id == "stress_boundary_zero_division")
            env = TestGenEnvironment(Path(tmp) / "cache")
            env.run_episode(subject, planner_from_name("greedy"), Budget(horizon=2), Path(tmp) / "run")
            events = [
                __import__("json").loads(line)
                for line in (Path(tmp) / "run" / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(events[0]["action"], Action.GENERATE_BOUNDARY_TESTS.name)
            self.assertEqual(events[1]["action"], Action.GENERATE_BOUNDARY_TESTS.name)
            self.assertGreater(events[1]["result"]["duplicate_tests"], 0)
            self.assertGreater(events[0]["result"]["coverage_delta"], 0)
            self.assertEqual(events[1]["result"]["coverage_delta"], 0.0)
            self.assertLess(events[1]["reward"], events[0]["reward"])

    def test_action_balanced_uses_stress_style_saturation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "action_balanced"
            prepare_builtin_dataset(root, "action_balanced")
            subject = next(s for s in load_subjects(root) if s.id == "action_balanced_boundary_mod_zero")
            env = TestGenEnvironment(Path(tmp) / "cache")
            env.run_episode(subject, planner_from_name("greedy"), Budget(horizon=2), Path(tmp) / "run")
            events = [
                __import__("json").loads(line)
                for line in (Path(tmp) / "run" / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertGreater(events[1]["result"]["duplicate_tests"], 0)
            self.assertEqual(events[1]["result"]["coverage_delta"], 0.0)

    def test_executor_returns_structured_result_on_runtime_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "subjects"
            prepare_builtin_dataset(root)
            subject = next(s for s in load_subjects(root) if s.id == "first_char_buggy")
            result = ExecutorSuite(Path(tmp) / "cache").execute(Action.GENERATE_BOUNDARY_TESTS, subject)
            self.assertEqual(result.action, Action.GENERATE_BOUNDARY_TESTS.name)
            self.assertGreaterEqual(result.failures, 1)


if __name__ == "__main__":
    unittest.main()
