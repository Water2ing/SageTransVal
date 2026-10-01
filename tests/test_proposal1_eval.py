from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sagetransval import SUBJECTS, evaluate_subjects, write_outputs


class Proposal1EvaluationTests(unittest.TestCase):
    def test_proposal1_evaluation_has_expected_modes_and_detection(self) -> None:
        rows = evaluate_subjects()
        self.assertEqual(len(rows), len(SUBJECTS) * 5)
        by_mode = {}
        for row in rows:
            by_mode.setdefault(row["mode"], []).append(row)
        self.assertEqual(set(by_mode), {"existing", "random", "direct_llm_tests", "analysis_guided", "analysis_guided_repair"})
        self.assertEqual(sum(row["detected"] for row in by_mode["existing"]), 0)
        self.assertEqual(sum(row["detected"] for row in by_mode["analysis_guided"]), len(SUBJECTS))

    def test_proposal1_outputs_are_generated_without_placeholders(self) -> None:
        rows = evaluate_subjects()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "tables"
            figures = Path(tmp) / "figures"
            write_outputs(rows, out, figures)
            tex = (out / "results_tables.tex").read_text(encoding="utf-8")
            self.assertIn("tab:detection", tex)
            self.assertNotIn("todoresult", tex.lower())
            self.assertTrue((figures / "proposal1_detection.tex").exists())


if __name__ == "__main__":
    unittest.main()
