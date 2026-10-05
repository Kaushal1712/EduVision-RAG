"""
Unit tests for the deterministic parts of eval/answer_eval.py.

Run:  venv/bin/python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from answer_eval import (citation_metrics, evidence_metrics, is_refusal, parse_citations,  # noqa: E402
                         ts_to_seconds, video_for_title)
from bench_data import DEFAULT_MANIFEST_PATH, load_json  # noqa: E402

MANIFEST = load_json(DEFAULT_MANIFEST_PATH)
T2 = "02_your_first_html_website_sigma_web_development_course_tutorial_2"
T7 = "07_forms_and_input_tags_in_html_sigma_web_development_course_tutorial_7"


class TestCitations(unittest.TestCase):

    def test_timestamps(self):
        self.assertEqual(ts_to_seconds("02:27"), 147)
        self.assertEqual(ts_to_seconds("02:27.3"), 147.3)

    def test_title_mapping(self):
        self.assertEqual(video_for_title("02_Your First HTML Website ｜ Sigma Web Development Course - Tutorial #2.mp4", MANIFEST), T2)
        self.assertEqual(video_for_title("Tutorial #7", MANIFEST), T7)
        self.assertEqual(video_for_title("Tutorial 7", MANIFEST), T7)
        self.assertIsNone(video_for_title("Some other video", MANIFEST))

    def test_parse_and_score(self):
        ans = ('Use a link tag [Video: "02_Your First HTML Website ｜ Sigma Web Development Course - Tutorial #2.mp4" @ 15:01]. '
               'Also see [Video: "Tutorial #7" @ 01:00.5].')
        cites = parse_citations(ans, MANIFEST)
        self.assertEqual([(c["video_id"], c["time_s"]) for c in cites], [(T2, 901.0), (T7, 60.5)])
        gold = [{"video_id": T2, "start_time": 889.7, "end_time": 939.7}]
        m = citation_metrics(cites, gold)
        self.assertTrue(m["citation_hit"])
        self.assertEqual(m["citation_start_error_s"], 11.3)
        miss = citation_metrics(cites, [{"video_id": T2, "start_time": 100, "end_time": 120}])
        self.assertFalse(miss["citation_hit"])
        self.assertTrue(miss["citation_in_gold_video"])

    def test_tolerance_window(self):
        gold = [{"video_id": T2, "start_time": 100, "end_time": 120}]
        self.assertTrue(citation_metrics([{"video_id": T2, "time_s": 95.0}], gold)["citation_hit"])
        self.assertFalse(citation_metrics([{"video_id": T2, "time_s": 94.0}], gold)["citation_hit"])

    def test_no_citations(self):
        m = citation_metrics([], [{"video_id": T2, "start_time": 1, "end_time": 2}])
        self.assertEqual((m["n_citations"], m["citation_hit"], m["citation_start_error_s"]), (0, False, None))


class TestEvidenceAndRefusal(unittest.TestCase):

    def test_evidence_metrics(self):
        gold = [{"video_id": T2, "start_time": 100, "end_time": 120}]
        src = [{"video_id": T2, "start_time": 110, "end_time": 130}, {"video_id": T7, "start_time": 110, "end_time": 130},
               {"video_id": T2, "start_time": 120, "end_time": 130}]
        m = evidence_metrics(src, gold)
        self.assertTrue(m["evidence_has_gold"])
        self.assertAlmostEqual(m["evidence_precision"], 1 / 3, places=4)
        self.assertEqual(evidence_metrics([], gold), {"n_evidence": 0, "evidence_has_gold": False, "evidence_precision": None})

    def test_refusal(self):
        self.assertTrue(is_refusal({"not_found": True, "answer": ""}))
        self.assertTrue(is_refusal({"not_found": False, "answer": "I could not find this topic in the provided course material."}))
        self.assertFalse(is_refusal({"not_found": False, "answer": "Use the link tag."}))


if __name__ == "__main__":
    unittest.main()
