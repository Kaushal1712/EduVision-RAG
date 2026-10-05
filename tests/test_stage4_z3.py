"""
Unit tests for the pure helpers of experiments/stage4_faithfulness/z3_prepare_annotation.py (no model).

Run:  venv/bin/python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments" / "stage4_faithfulness"))

from z3_prepare_annotation import exact_citation, fence, render_sheet, spread, strata_tags, z2_category  # noqa: E402


def row(rank_d=None, rank_l=None, cited=True, flagged=False, hall=False, lang="en", fu=False, cat="in_scope"):
    r = {"id": "EVB-1", "repeat": 0, "sentence_index": 0, "judge_flagged_sentence": flagged,
         "judge_hallucination": hall, "language": lang, "followup": fu, "category": cat}
    if cited:
        r["citations"] = ["[Video: x @ 00:01]"]
        r["dense"] = {"rank": rank_d}
        r["lexical"] = {"rank": rank_l}
    return r


class TestCategories(unittest.TestCase):

    def test_z2_category(self):
        self.assertEqual(z2_category(row(1, 1)), "aligned")
        self.assertEqual(z2_category(row(2, 3)), "misaligned")
        self.assertEqual(z2_category(row(1, 2)), "mixed")
        self.assertEqual(z2_category(row(cited=False)), "uncited")

    def test_disagreement_tags(self):
        self.assertIn("z2_judge_disagree", strata_tags(row(2, 2), "fully_supported"))
        self.assertNotIn("z2_judge_disagree", strata_tags(row(2, 2), "partially_supported"))
        self.assertIn("z2_judge_disagree", strata_tags(row(1, 1, flagged=True), "partially_supported"))
        tags = strata_tags(row(1, 1, lang="hinglish", fu=True), "fully_supported")
        self.assertIn("hinglish", tags)
        self.assertIn("followup", tags)
        self.assertNotIn("ordinary_english", tags)

    def test_spread_is_even_and_deterministic(self):
        rows = [{"id": f"E{i}", "repeat": 0, "sentence_index": 0, "v": i} for i in range(10)]
        self.assertEqual([r["v"] for r in spread(rows, lambda r: r["v"], 4)], [0, 3, 6, 9])
        self.assertEqual(len(spread(rows[:2], lambda r: r["v"], 4)), 2)


class TestRendering(unittest.TestCase):

    def test_exact_citation_keeps_closing_bracket(self):
        self.assertEqual(exact_citation('A [Video: "T" @ 00:01].', '[Video: "T" @ 00:01'), '[Video: "T" @ 00:01]')

    def test_fence_survives_backticks(self):
        self.assertTrue(fence("use ```code```").startswith("````text"))

    def test_sheet_preserves_text_and_has_no_labels(self):
        s = {"sample_id": "S01", "query_id": "EVB-1", "language": "en", "followup": False,
             "question_category": "in_scope", "question": "Q?", "prior_conversation": [],
             "answer": "Use `<link>` tag [Video: \"T\" @ 00:01].", "claim": "Use `<link>` tag",
             "citations": ['[Video: "T" @ 00:01]'], "cited_evidence": ["E1"],
             "evidence": [{"label": "E1", "video_filename": "t.mp4", "time": "00:01 → 00:09",
                           "text": "the link tag goes in head", "cited_by_claim": True}]}
        md = render_sheet([s])
        for text in (s["answer"], s["claim"], s["evidence"][0]["text"]):
            self.assertIn(text, md)
        self.assertNotIn("dense", md.lower())
        self.assertNotIn("judge", md.lower())


if __name__ == "__main__":
    unittest.main()
