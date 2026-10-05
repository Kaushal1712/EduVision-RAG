"""
Unit tests for eval/bench_metrics.py (timestamp-anchored retrieval metrics).

Run:  venv/bin/python -m unittest discover -s tests -v
"""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from bench_metrics import (  # noqa: E402
    GoldSpan,
    RetrievedChunk,
    auroc,
    covered_fraction,
    first_relevant_start_error,
    is_valid_interval,
    judge_retrieved,
    mean,
    ndcg_at_k,
    overlap_seconds,
    percentile,
    rate,
    recall_at_k,
    reciprocal_rank,
)

V1, V2 = "01_video", "02_video"


def chunk(rank, start, end, video=V1, chunk_id=None, sim=0.6):
    return RetrievedChunk(rank, chunk_id or f"{video}_{start}", video, start, end, sim)


class TestOverlap(unittest.TestCase):

    def test_partial_overlap(self):
        self.assertEqual(overlap_seconds(10, 20, 15, 30), 5)

    def test_containment(self):
        self.assertEqual(overlap_seconds(10, 40, 15, 20), 5)
        self.assertEqual(overlap_seconds(15, 20, 10, 40), 5)

    def test_touching_endpoints_is_not_overlap(self):
        self.assertEqual(overlap_seconds(10, 20, 20, 30), 0)

    def test_disjoint(self):
        self.assertEqual(overlap_seconds(10, 20, 25, 30), 0)

    def test_judge_requires_same_video(self):
        gold = [GoldSpan(V1, 100, 120)]
        judged = judge_retrieved([chunk(1, 105, 115, video=V2)], gold)
        self.assertEqual(judged[0]["matched_gold"], [])

    def test_judge_touching_chunk_not_relevant(self):
        gold = [GoldSpan(V1, 100, 120)]
        judged = judge_retrieved([chunk(1, 90, 100), chunk(2, 120, 130), chunk(3, 99, 101)], gold)
        self.assertEqual([j["matched_gold"] for j in judged], [[], [], [0]])

    def test_min_overlap_threshold(self):
        gold = [GoldSpan(V1, 100, 120)]
        r = [chunk(1, 118, 130)]   # 2 s overlap
        self.assertEqual(judge_retrieved(r, gold, min_overlap_s=0.0)[0]["matched_gold"], [0])
        self.assertEqual(judge_retrieved(r, gold, min_overlap_s=2.0)[0]["matched_gold"], [])
        self.assertEqual(judge_retrieved(r, gold, min_overlap_s=1.5)[0]["matched_gold"], [0])

    def test_chunk_spanning_two_gold_spans(self):
        gold = [GoldSpan(V1, 100, 110), GoldSpan(V1, 115, 125)]
        judged = judge_retrieved([chunk(1, 105, 120)], gold)
        self.assertEqual(judged[0]["matched_gold"], [0, 1])


class TestInvalidAndDuplicate(unittest.TestCase):

    def test_valid_interval_rules(self):
        self.assertTrue(is_valid_interval(V1, 0.0, 1.0))
        self.assertFalse(is_valid_interval("", 0.0, 1.0))
        self.assertFalse(is_valid_interval(None, 0.0, 1.0))
        self.assertFalse(is_valid_interval(V1, 5.0, 5.0))
        self.assertFalse(is_valid_interval(V1, 6.0, 5.0))
        self.assertFalse(is_valid_interval(V1, -1.0, 5.0))
        self.assertFalse(is_valid_interval(V1, float("nan"), 5.0))
        self.assertFalse(is_valid_interval(V1, 0.0, float("inf")))
        self.assertFalse(is_valid_interval(V1, "0", 5.0))
        self.assertFalse(is_valid_interval(V1, True, 5.0))

    def test_invalid_metadata_earns_no_credit(self):
        gold = [GoldSpan(V1, 100, 120)]
        retrieved = [
            chunk(1, 0.0, 0.0),                              # zero-length (retriever's default for missing times)
            RetrievedChunk(2, "c2", "", 105, 115, 0.6),      # missing video_id
            chunk(3, float("nan"), 115),
            chunk(4, 105, 115),
        ]
        judged = judge_retrieved(retrieved, gold)
        self.assertEqual([j["invalid"] for j in judged], [True, True, True, False])
        matches = [j["matched_gold"] for j in judged]
        self.assertEqual(reciprocal_rank(matches, 10), 0.25)

    def test_duplicate_chunk_earns_no_credit(self):
        gold = [GoldSpan(V1, 100, 120), GoldSpan(V1, 200, 220)]
        retrieved = [chunk(1, 105, 115, chunk_id="a"), chunk(2, 105, 115, chunk_id="a"),
                     chunk(3, 205, 215, chunk_id="b")]
        judged = judge_retrieved(retrieved, gold)
        self.assertEqual([j["duplicate"] for j in judged], [False, True, False])
        self.assertEqual([j["matched_gold"] for j in judged], [[0], [], [1]])


class TestRankingMetrics(unittest.TestCase):

    def test_empty_retrieval(self):
        self.assertEqual(recall_at_k([], 1, 10), 0.0)
        self.assertEqual(reciprocal_rank([], 10), 0.0)
        self.assertEqual(ndcg_at_k([], 1, 10), 0.0)
        self.assertIsNone(first_relevant_start_error([], [], [GoldSpan(V1, 0, 1)], 10))

    def test_recall_is_span_based(self):
        matches = [[], [0], [0], [1]]
        self.assertEqual(recall_at_k(matches, 2, 1), 0.0)
        self.assertEqual(recall_at_k(matches, 2, 2), 0.5)
        self.assertEqual(recall_at_k(matches, 2, 3), 0.5)   # second hit on span 0 adds nothing
        self.assertEqual(recall_at_k(matches, 2, 4), 1.0)

    def test_recall_cutoff_respected(self):
        matches = [[]] * 5 + [[0]]
        self.assertEqual(recall_at_k(matches, 1, 5), 0.0)
        self.assertEqual(recall_at_k(matches, 1, 10), 1.0)

    def test_reciprocal_rank(self):
        self.assertEqual(reciprocal_rank([[0]], 10), 1.0)
        self.assertEqual(reciprocal_rank([[], [], [0]], 10), 1 / 3)
        self.assertEqual(reciprocal_rank([[], [], [0]], 2), 0.0)

    def test_ndcg_known_value(self):
        matches = [[], [0], [0], [1]]
        dcg = 1 / math.log2(3) + 1 / math.log2(5)     # ranks 2 and 4 cover new spans; rank 3 is redundant
        idcg = 1 / math.log2(2) + 1 / math.log2(3)
        self.assertAlmostEqual(ndcg_at_k(matches, 2, 10), dcg / idcg)

    def test_ndcg_perfect_and_bounded(self):
        self.assertEqual(ndcg_at_k([[0], [1]], 2, 10), 1.0)
        self.assertEqual(ndcg_at_k([[0], [0], [0]], 1, 10), 1.0)   # redundancy cannot exceed 1
        self.assertLessEqual(ndcg_at_k([[0, 1], [], []], 2, 10), 1.0)

    def test_ndcg_is_granularity_invariant(self):
        # Same top hit, one coarse chunk vs. three fine chunks of the same span.
        self.assertEqual(ndcg_at_k([[0]], 1, 10), ndcg_at_k([[0], [0], [0]], 1, 10))

    def test_metrics_reject_no_gold(self):
        with self.assertRaises(ValueError):
            recall_at_k([[0]], 0, 10)
        with self.assertRaises(ValueError):
            ndcg_at_k([[0]], 0, 10)

    def test_start_error(self):
        gold = [GoldSpan(V1, 100, 150), GoldSpan(V1, 300, 320)]
        retrieved = [chunk(1, 10, 20), chunk(2, 130, 140), chunk(3, 300, 310)]
        matches = [j["matched_gold"] for j in judge_retrieved(retrieved, gold)]
        self.assertEqual(first_relevant_start_error(retrieved, matches, gold, 10), 30)
        self.assertIsNone(first_relevant_start_error(retrieved, matches, gold, 1))

    def test_start_error_uses_closest_matched_span(self):
        gold = [GoldSpan(V1, 100, 110), GoldSpan(V1, 115, 125)]
        retrieved = [chunk(1, 108, 120)]
        matches = [j["matched_gold"] for j in judge_retrieved(retrieved, gold)]
        self.assertEqual(first_relevant_start_error(retrieved, matches, gold, 10), 7)


class TestCoveredFraction(unittest.TestCase):

    def test_full_partial_none(self):
        gold = [GoldSpan(V1, 100, 120)]
        self.assertEqual(covered_fraction(gold, {V1: [(90, 130)]}), 1.0)
        self.assertEqual(covered_fraction(gold, {V1: [(110, 130)]}), 0.5)
        self.assertEqual(covered_fraction(gold, {V2: [(100, 120)]}), 0.0)
        self.assertEqual(covered_fraction(gold, {}), 0.0)

    def test_overlapping_intervals_not_double_counted(self):
        gold = [GoldSpan(V1, 100, 120)]
        self.assertEqual(covered_fraction(gold, {V1: [(100, 112), (108, 116)]}), 0.8)

    def test_weighted_by_span_length(self):
        gold = [GoldSpan(V1, 0, 10), GoldSpan(V2, 0, 30)]
        self.assertEqual(covered_fraction(gold, {V1: [(0, 10)]}), 0.25)

    def test_rejects_empty_gold(self):
        with self.assertRaises(ValueError):
            covered_fraction([], {V1: [(0, 1)]})


class TestAbstentionAndAggregation(unittest.TestCase):

    def test_auroc(self):
        self.assertEqual(auroc([0.9, 0.8], [0.1, 0.2]), 1.0)
        self.assertEqual(auroc([0.1], [0.9]), 0.0)
        self.assertEqual(auroc([0.5], [0.5]), 0.5)
        self.assertEqual(auroc([0.9, 0.8], [0.7, 0.8]), 0.875)
        self.assertEqual(auroc([0.6], [float("-inf")]), 1.0)

    def test_auroc_undefined_without_both_classes(self):
        self.assertIsNone(auroc([], [0.1]))
        self.assertIsNone(auroc([0.1], []))

    def test_rate(self):
        self.assertEqual(rate(1, 4), 0.25)
        self.assertIsNone(rate(0, 0))

    def test_mean_and_percentile(self):
        self.assertIsNone(mean([]))
        self.assertEqual(mean([1, 2, 3]), 2)
        self.assertIsNone(percentile([], 50))
        self.assertEqual(percentile([4, 1, 3, 2], 50), 2.5)
        self.assertAlmostEqual(percentile([1, 2, 3, 4], 90), 3.7)
        self.assertEqual(percentile([7], 90), 7)


if __name__ == "__main__":
    unittest.main()
