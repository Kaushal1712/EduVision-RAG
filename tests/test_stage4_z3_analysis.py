"""
Unit tests for experiments/stage4_faithfulness/z3/z3_analyze.py helpers (offline).

Run:  venv/bin/python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments" / "stage4_faithfulness" / "z3"))

from z3_analyze import confusion, contains_claim, crosstab, score_auroc  # noqa: E402


class TestAnalysisHelpers(unittest.TestCase):

    def test_confusion_and_kappa(self):
        pairs = [(True, True)] * 6 + [(False, True)] * 2 + [(True, False)] * 1 + [(False, False)] * 19
        c = confusion(pairs)
        self.assertEqual((c["tp"], c["fp"], c["fn"], c["tn"]), (6, 2, 1, 19))
        self.assertEqual((c["precision"], c["recall"]), (0.75, 0.8571))
        # po = 25/28, pe = (8*7 + 20*21)/28^2
        po, pe = 25 / 28, (8 * 7 + 20 * 21) / 28 ** 2
        self.assertAlmostEqual(c["cohen_kappa"], round((po - pe) / (1 - pe), 4))

    def test_confusion_perfect_and_empty(self):
        self.assertEqual(confusion([(True, True), (False, False)])["cohen_kappa"], 1.0)
        self.assertIsNone(confusion([])["agreement"])

    def test_contains_claim(self):
        claim = 'This is mentioned in the video where it states, "we can actually set it such that whenever we click'
        self.assertTrue(contains_claim(claim, [claim + ' it will open" [Video: "07_Forms" @ 01:38].']))
        self.assertFalse(contains_claim("SEO stands for search engine optimization", ["It improves visibility"]))

    def test_crosstab_and_auroc(self):
        rows = [{"a": "x", "b": "p", "s": 0.9}, {"a": "x", "b": "q", "s": 0.1}, {"a": "y", "b": "p", "s": 0.8}]
        self.assertEqual(crosstab(rows, lambda r: r["a"], lambda r: r["b"]), {"x": {"p": 1, "q": 1}, "y": {"p": 1}})
        self.assertEqual(score_auroc(rows, lambda r: r["b"] == "p", lambda r: r["s"])["auroc"], 1.0)


if __name__ == "__main__":
    unittest.main()
