"""
Unit tests for experiments/stage4_faithfulness/z1_citation_audit.py (offline citation audit).

Run:  venv/bin/python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments" / "stage4_faithfulness"))

from z1_citation_audit import classify, display_seconds, extract_citations, sentence_coverage  # noqa: E402

MANIFEST = {"videos": [{"video_id": "02_v", "tutorial_number": 2, "video_filename": "02_Your First HTML.mp4"},
                       {"video_id": "03_v", "tutorial_number": 3, "video_filename": "03_Basic Structure.mp4"}]}
EVIDENCE = [{"video_id": "02_v", "start_time": 43.4, "end_time": 55.2},
            {"video_id": "02_v", "start_time": 120.0, "end_time": 131.0}]


def cit(text):
    return extract_citations(text, MANIFEST)[0]


class TestExtract(unittest.TestCase):

    def test_plain_and_range_citations(self):
        cs = extract_citations('A [Video: "02_Your First HTML.mp4" @ 00:43]. B [Video: "Tutorial #3" @ 02:50 → 03:05].',
                               MANIFEST)
        self.assertEqual([(c["video_id"], c["time_s"], c["end_s"]) for c in cs],
                         [("02_v", 43.0, None), ("03_v", 170.0, 185.0)])

    def test_unknown_title_is_unresolved(self):
        self.assertIsNone(cit('[Video: "Some other course" @ 01:00]')["video_id"])


class TestClassify(unittest.TestCase):

    def test_valid_with_tolerance_and_positions(self):
        # 00:43 is the displayed (truncated) start of the 43.4 s chunk
        self.assertEqual(classify(cit('[Video: "Tutorial #2" @ 00:43]'), EVIDENCE, []),
                         {"class": "valid_evidence", "position": "exact_chunk_start"})
        self.assertEqual(classify(cit('[Video: "Tutorial #2" @ 00:50]'), EVIDENCE, [])["position"], "interior")
        self.assertEqual(classify(cit('[Video: "Tutorial #2" @ 00:56]'), EVIDENCE, [])["class"], "valid_evidence")
        self.assertEqual(classify(cit('[Video: "Tutorial #2" @ 00:57]'), EVIDENCE, [])["class"], "same_video_outside")

    def test_other_classes(self):
        self.assertEqual(classify(cit('[Video: "Tutorial #3" @ 00:43]'), EVIDENCE, [])["class"], "video_not_in_evidence")
        self.assertEqual(classify(cit('[Video: "Nope" @ 00:43]'), EVIDENCE, [])["class"], "unresolved_title")
        prior = [cit('[Video: "Tutorial #3" @ 05:00]')]
        self.assertEqual(classify(cit('[Video: "Tutorial #3" @ 05:01]'), EVIDENCE, prior)["class"], "valid_prior_answer")

    def test_range_needs_both_ends_inside_evidence(self):
        self.assertEqual(classify(cit('[Video: "Tutorial #2" @ 00:44 → 00:55]'), EVIDENCE, [])["class"], "valid_evidence")
        self.assertEqual(classify(cit('[Video: "Tutorial #2" @ 00:44 → 01:30]'), EVIDENCE, [])["class"],
                         "same_video_outside")

    def test_display_seconds_matches_retriever_format(self):
        self.assertEqual(display_seconds(147.28, 3.0), 147.2)
        self.assertEqual(display_seconds(147.98, 12.0), 147.0)


class TestSentenceCoverage(unittest.TestCase):

    def test_mp4_does_not_split_and_trailing_citation_attaches(self):
        a = ('HTML is a markup language for pages [Video: "02_Your First HTML.mp4" @ 00:43]. '
             'CSS styles the page and layout. [Video: "Tutorial #2" @ 02:00] '
             'It was shown with an example too.')
        self.assertEqual(sentence_coverage(a), {"sentences": 3, "cited": 2})

    def test_short_fragments_ignored(self):
        self.assertEqual(sentence_coverage("Yes. It is used for styling the whole page."), {"sentences": 1, "cited": 0})


if __name__ == "__main__":
    unittest.main()
