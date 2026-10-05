"""
Unit tests for the pure logic of experiments/stage4_faithfulness/z2_alignment_proxy.py (no model).

Run:  venv/bin/python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments" / "stage4_faithfulness"))

from z1_citation_audit import extract_citations, sentence_coverage  # noqa: E402
from z2_alignment_proxy import (  # noqa: E402
    alignment, bin_counts, lexical_overlap, map_claims, resolve, segment,
)

MANIFEST = {"videos": [{"video_id": "02_v", "tutorial_number": 2, "video_filename": "02_Your First HTML.mp4"}]}


class TestSegment(unittest.TestCase):

    def test_trailing_citation_goes_to_preceding_sentence_and_matches_z1(self):
        a = ('HTML is a markup language for pages [Video: "02_Your First HTML.mp4" @ 00:43]. '
             'CSS styles the page and layout. [Video: "Tutorial #2" @ 02:00] '
             'It was shown with an example too.')
        sents, amb = segment(a)
        self.assertEqual([s["cits"] for s in sents], [[0], [1], []])
        self.assertEqual(sents[0]["text"], "HTML is a markup language for pages")
        self.assertEqual(amb["leading_citation_reassigned"], 1)
        z1 = sentence_coverage(a)
        self.assertEqual(z1, {"sentences": len(sents), "cited": sum(bool(s["cits"]) for s in sents)})

    def test_mid_sentence_and_range_citations(self):
        a = ('The install is shown [Video: "Tutorial #2" @ 02:50 → 03:05] and then the editor opens '
             'with a welcome screen.')
        sents, amb = segment(a)
        self.assertEqual(len(sents), 1)
        self.assertEqual(sents[0]["cits"], [0])
        self.assertNotIn("03:05", sents[0]["text"])
        self.assertEqual(amb["mid_sentence_citation"], 1)

    def test_list_items_and_short_fragments(self):
        sents, amb = segment("Steps:\n- Open the folder in VS Code now.\n- Done [Video: \"Tutorial #2\" @ 00:43]")
        self.assertEqual([s["text"] for s in sents], ["Open the folder in VS Code now"])
        self.assertEqual(amb["list_item_sentences"], 1)
        self.assertEqual(amb["citation_in_short_fragment_dropped"], 1)


class TestResolve(unittest.TestCase):

    EV = [{"video_id": "02_v", "start_time": 40.0, "end_time": 52.0},
          {"video_id": "02_v", "start_time": 51.3, "end_time": 63.0},     # overlaps the first
          {"video_id": "02_v", "start_time": 100.0, "end_time": 110.0}]

    def cit(self, text):
        return extract_citations(text, MANIFEST)[0]

    def test_overlap_resolved_by_displayed_start(self):
        self.assertEqual(resolve(self.cit('[Video: "Tutorial #2" @ 00:51]'), self.EV),
                         {"chunks": [1], "overlap_ambiguous": False})

    def test_overlap_without_exact_start_is_ambiguous(self):
        self.assertEqual(resolve(self.cit('[Video: "Tutorial #2" @ 00:52]'), self.EV),
                         {"chunks": [0, 1], "overlap_ambiguous": True})

    def test_range_keeps_all_overlapping_chunks(self):
        self.assertEqual(resolve(self.cit('[Video: "Tutorial #2" @ 00:45 → 01:45]'), self.EV)["chunks"], [0, 1, 2])


class TestScores(unittest.TestCase):

    def test_lexical_overlap_ignores_stopwords(self):
        # content tokens {link, tag, goes, head}; "goes" is not in the chunk
        self.assertEqual(lexical_overlap("The link tag goes in the head", "put the link tag inside head"), 0.75)
        self.assertAlmostEqual(lexical_overlap("link tag stylesheet", "the link tag"), 2 / 3)
        self.assertIsNone(lexical_overlap("it is the", "anything"))

    def test_alignment_rank_and_margin(self):
        a = alignment([0.5, 0.7, 0.6], cited=[2])
        self.assertEqual((a["cited"], a["best"], a["rank"]), (0.6, 0.7, 2))
        self.assertAlmostEqual(a["margin"], 0.1)
        self.assertEqual(alignment([0.5, 0.7, 0.6], cited=[0, 1])["rank"], 1)    # max over cited chunks

    def test_bins(self):
        self.assertEqual(bin_counts([0.0, 0.01, 0.03, 0.2], (0.0, 0.02, 0.05, 0.10), exact_zero=True),
                         {"0": 1, "<0.02": 1, "[0.02,0.05)": 1, "[0.05,0.1)": 0, ">=0.1": 1})

    def test_map_claims(self):
        idx, unmatched = map_claims(["link tag goes in head section", "totally different claim words"],
                                    ["The link tag goes in the head section", "CSS colours text"])
        self.assertEqual((idx, unmatched), ([0, None], 1))


if __name__ == "__main__":
    unittest.main()
