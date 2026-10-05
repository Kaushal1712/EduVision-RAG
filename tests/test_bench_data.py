"""
Unit tests for eval/bench_data.py (benchmark validation) and a consistency check
of the committed benchmark against the committed corpus manifest.

Run:  venv/bin/python -m unittest discover -s tests -v
"""

import copy
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from bench_data import (  # noqa: E402
    DEFAULT_BENCHMARK_PATH,
    DEFAULT_MANIFEST_PATH,
    gold_spans,
    is_scored,
    load_json,
    validate_benchmark,
)

MANIFEST = {"videos": [
    {"video_id": "01_a", "tutorial_number": 1, "video_filename": "01_a.mp4", "duration_s": 600.0, "sha256": "x"},
    {"video_id": "02_b", "tutorial_number": 2, "video_filename": "02_b.mp4", "duration_s": 300.0, "sha256": "y"},
]}


def item(**overrides):
    base = {
        "id": "Q1", "split": "dev", "query": "what is html?", "history": [], "followup": False,
        "category": "in_scope", "language": "en", "phrasing": "exact_term", "difficulty": "easy",
        "gold": [{"video_id": "01_a", "start_time": 10.0, "end_time": 20.0}],
        "annotation": {"status": "draft"}, "source": None,
    }
    base.update(overrides)
    return base


def bench(*items):
    return {"benchmark_version": "test", "queries": list(items)}


class TestValidation(unittest.TestCase):

    def assertInvalid(self, b, fragment):
        errors = validate_benchmark(b, MANIFEST)
        self.assertTrue(any(fragment in e for e in errors), f"expected {fragment!r} in {errors}")

    def test_valid_minimal(self):
        oos = item(id="Q2", category="out_of_scope", phrasing=None, gold=[])
        na = item(id="Q3", gold=[], annotation={"status": "needs_annotation"})
        fu = item(id="Q4", followup=True,
                  history=[{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])
        self.assertEqual(validate_benchmark(bench(item(), oos, na, fu), MANIFEST), [])

    def test_duplicate_ids(self):
        self.assertInvalid(bench(item(), item()), "duplicate id")

    def test_out_of_scope_with_gold(self):
        self.assertInvalid(bench(item(category="out_of_scope", phrasing=None)), "must not have gold")

    def test_scored_item_without_gold(self):
        self.assertInvalid(bench(item(gold=[])), "needs at least one gold span")

    def test_needs_annotation_with_gold(self):
        self.assertInvalid(bench(item(annotation={"status": "needs_annotation"})), "must have empty gold")

    def test_unknown_video(self):
        self.assertInvalid(bench(item(gold=[{"video_id": "99_x", "start_time": 0, "end_time": 5}])),
                           "not in the corpus manifest")

    def test_span_beyond_duration(self):
        self.assertInvalid(bench(item(gold=[{"video_id": "02_b", "start_time": 290, "end_time": 302}])),
                           "after the video's duration")

    def test_span_within_tolerance_is_ok(self):
        b = bench(item(gold=[{"video_id": "02_b", "start_time": 290, "end_time": 300.5}]))
        self.assertEqual(validate_benchmark(b, MANIFEST), [])

    def test_inverted_or_missing_times(self):
        self.assertInvalid(bench(item(gold=[{"video_id": "01_a", "start_time": 20, "end_time": 10}])),
                           "invalid gold span")
        self.assertInvalid(bench(item(gold=[{"video_id": "01_a", "start_time": 20}])), "invalid gold span")

    def test_overlapping_spans_must_be_merged(self):
        spans = [{"video_id": "01_a", "start_time": 10, "end_time": 20},
                 {"video_id": "01_a", "start_time": 20, "end_time": 30}]
        self.assertInvalid(bench(item(gold=spans)), "merge them")

    def test_same_times_different_videos_ok(self):
        spans = [{"video_id": "01_a", "start_time": 10, "end_time": 20},
                 {"video_id": "02_b", "start_time": 10, "end_time": 20}]
        self.assertEqual(validate_benchmark(bench(item(gold=spans)), MANIFEST), [])

    def test_followup_flag_must_match_history(self):
        self.assertInvalid(bench(item(followup=True)), "followup must be true exactly")
        self.assertInvalid(bench(item(history=[{"role": "user", "content": "a"}])), "followup must be true exactly")

    def test_bad_history_turn(self):
        self.assertInvalid(bench(item(followup=True, history=[{"role": "system", "content": "a"}])),
                           "history turns")

    def test_enums(self):
        self.assertInvalid(bench(item(split="train")), "split")
        self.assertInvalid(bench(item(language="fr")), "language")
        self.assertInvalid(bench(item(phrasing=None)), "phrasing")
        self.assertInvalid(bench(item(annotation={"status": "done"})), "annotation.status")

    def test_empty_query(self):
        self.assertInvalid(bench(item(query="   ")), "non-empty string")

    def test_is_scored(self):
        self.assertTrue(is_scored(item()))
        self.assertTrue(is_scored(item(annotation={"status": "verified"})))
        self.assertTrue(is_scored(item(annotation={"status": "video_checked"})))
        self.assertFalse(is_scored(item(gold=[], annotation={"status": "needs_annotation"})))
        self.assertFalse(is_scored(item(category="near_miss_oos", phrasing=None, gold=[])))


class TestCommittedBenchmark(unittest.TestCase):
    """The committed benchmark must always validate and keep its coverage guarantees."""

    @classmethod
    def setUpClass(cls):
        cls.bench = load_json(DEFAULT_BENCHMARK_PATH)
        cls.manifest = load_json(DEFAULT_MANIFEST_PATH)

    def test_validates(self):
        self.assertEqual(validate_benchmark(self.bench, self.manifest), [])

    def test_size(self):
        self.assertGreaterEqual(len(self.bench["queries"]), 120)
        self.assertLessEqual(len(self.bench["queries"]), 180)

    def test_every_video_has_scored_gold(self):
        covered = {g.video_id for q in self.bench["queries"] if is_scored(q) for g in gold_spans(q)}
        self.assertEqual(covered, {v["video_id"] for v in self.manifest["videos"]})

    def test_splits_cover_each_category(self):
        cats = Counter((q["split"], q["category"]) for q in self.bench["queries"])
        for split in ("dev", "test"):
            for cat in ("in_scope", "out_of_scope", "near_miss_oos"):
                self.assertGreater(cats[(split, cat)], 0, f"no {cat} items in {split}")

    def test_required_query_kinds_present(self):
        qs = self.bench["queries"]
        self.assertTrue(any(q["language"] == "hinglish" for q in qs))
        self.assertTrue(any(q["followup"] and is_scored(q) for q in qs))
        for phrasing in ("exact_term", "paraphrase", "rare_term"):
            self.assertTrue(any(q["phrasing"] == phrasing for q in qs))
        for difficulty in ("easy", "medium", "hard"):
            self.assertTrue(any(q["difficulty"] == difficulty for q in qs))

    def test_unresolved_items_are_documented(self):
        for q in self.bench["queries"]:
            if q["annotation"]["status"] == "needs_annotation":
                self.assertTrue((q["annotation"].get("notes") or "").strip(), f"{q['id']} lacks a reason")

    def test_changed_gold_keeps_audit_trail(self):
        for q in self.bench["queries"]:
            ver = q["annotation"].get("verification")
            if ver and ver["change"] not in ("confirmed", "located"):
                self.assertIn("previous_gold", q["annotation"], q["id"])

    def test_gold_never_references_chunk_ids(self):
        for q in self.bench["queries"]:
            for g in q["gold"]:
                self.assertEqual(set(g), {"video_id", "start_time", "end_time"}, q["id"])

    def test_benchmark_not_mutated_by_loading(self):
        before = copy.deepcopy(self.bench)
        validate_benchmark(self.bench, self.manifest)
        self.assertEqual(before, self.bench)


if __name__ == "__main__":
    unittest.main()
