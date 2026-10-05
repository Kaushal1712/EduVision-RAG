"""
Unit tests for eval/run_benchmark.py using a fake retrieval system (no model, no
ChromaDB, no network).

Run:  venv/bin/python -m unittest discover -s tests -v
"""

import json
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from run_benchmark import (  # noqa: E402
    evaluate_item,
    summarize_abstention,
    summarize_all,
    summarize_exact_agreement,
    summarize_ranking,
    unclear_rate_at_k,
)

V1, V2 = "01_a", "02_b"
THRESHOLD = 0.5


@dataclass
class FakeResult:
    chunk_id: str
    video_id: str
    start_time: float
    end_time: float
    similarity: float
    text_en: str = "some transcript text"

    @property
    def below_threshold(self) -> bool:
        return self.similarity < THRESHOLD


def item(qid="Q1", category="in_scope", gold=None, history=None, status="draft", split="dev", query="what is html?"):
    return {
        "id": qid, "split": split, "query": query, "history": history or [], "followup": bool(history),
        "category": category, "language": "en", "phrasing": "exact_term" if category == "in_scope" else None,
        "difficulty": "easy", "annotation": {"status": status},
        "gold": [] if gold is None else gold, "source": None,
    }


def gold(video, start, end):
    return {"video_id": video, "start_time": start, "end_time": end}


class FakeSystem:
    def __init__(self, results):
        self.results = results
        self.retrieve_calls = []
        self.rewrite_calls = []

    def retrieve(self, q):
        self.retrieve_calls.append(q)
        return list(self.results)

    def rewrite(self, q, history):
        self.rewrite_calls.append((q, history))
        return "rewritten: " + q

    @staticmethod
    def validate(q):
        return None if len(q) >= 3 else "Query is too short."

    def run(self, it, **kw):
        return evaluate_item(it, self.retrieve, self.rewrite, self.validate, k_max=10, **kw)


class TestEvaluateItem(unittest.TestCase):

    def test_scored_item_metrics(self):
        sys_ = FakeSystem([
            FakeResult("c1", V2, 10, 20, 0.70),
            FakeResult("c2", V1, 95, 105, 0.65),
            FakeResult("c3", V1, 104, 114, 0.40),
        ])
        rec = sys_.run(item(gold=[gold(V1, 100, 130)]))
        m = rec["metrics"]
        self.assertEqual(m["recall@5"], 1.0)
        self.assertEqual(m["mrr@10"], 0.5)
        self.assertEqual(m["start_error_s"], 5.0)
        self.assertEqual(m["video_hit@1"], 0.0)
        self.assertEqual(rec["gate"], "answered")
        self.assertEqual(rec["top1_similarity"], 0.7)
        self.assertEqual([r["matched_gold"] for r in rec["retrieved"]], [[], [0], [0]])

    def test_out_of_scope_has_no_ranking_metrics(self):
        rec = FakeSystem([FakeResult("c1", V1, 0, 10, 0.3)]).run(item(category="out_of_scope"))
        self.assertIsNone(rec["metrics"])
        self.assertEqual(rec["gate"], "refused")

    def test_needs_annotation_is_not_scored(self):
        rec = FakeSystem([FakeResult("c1", V1, 0, 10, 0.6)]).run(item(status="needs_annotation"))
        self.assertIsNone(rec["metrics"])
        self.assertEqual(rec["gate"], "answered")

    def test_empty_retrieval(self):
        rec = FakeSystem([]).run(item(gold=[gold(V1, 0, 10)]))
        self.assertEqual(rec["metrics"]["recall@10"], 0.0)
        self.assertEqual(rec["metrics"]["ndcg@10"], 0.0)
        self.assertIsNone(rec["metrics"]["start_error_s"])
        self.assertIsNone(rec["top1_similarity"])
        self.assertEqual(rec["gate"], "refused")

    def test_validation_failure_skips_retrieval(self):
        sys_ = FakeSystem([FakeResult("c1", V1, 0, 10, 0.9)])
        rec = sys_.run(item(query="x", gold=[gold(V1, 0, 10)]))
        self.assertEqual(sys_.retrieve_calls, [])
        self.assertEqual(rec["validation_error"], "Query is too short.")
        self.assertEqual(rec["gate"], "refused")
        self.assertEqual(rec["metrics"]["recall@10"], 0.0)

    def test_followup_uses_rewrite_and_history(self):
        hist = [{"role": "user", "content": "what is html?"}, {"role": "assistant", "content": "HTML is..."}]
        sys_ = FakeSystem([FakeResult("c1", V1, 0, 10, 0.6)])
        rec = sys_.run(item(history=hist, query="why is it important?", gold=[gold(V1, 0, 10)]))
        self.assertEqual(sys_.rewrite_calls, [("why is it important?", hist)])
        self.assertEqual(sys_.retrieve_calls, ["rewritten: why is it important?"])
        self.assertTrue(rec["rewritten"])

    def test_single_turn_never_rewrites(self):
        sys_ = FakeSystem([])
        sys_.run(item(gold=[gold(V1, 0, 10)]))
        self.assertEqual(sys_.rewrite_calls, [])

    def test_unclear_text_flag_and_usable_recall(self):
        sys_ = FakeSystem([FakeResult("c1", V1, 0, 10, 0.6, text_en="[unclear audio]"),
                           FakeResult("c2", V2, 0, 10, 0.55)])
        rec = sys_.run(item(gold=[gold(V1, 5, 8)]))
        self.assertTrue(rec["retrieved"][0]["unclear_text"])
        self.assertEqual(rec["metrics"]["recall@5"], 1.0)
        self.assertEqual(rec["metrics"]["recall@5_usable_text"], 0.0)

    def test_duplicates_reported(self):
        sys_ = FakeSystem([FakeResult("c1", V1, 0, 10, 0.6), FakeResult("c1", V1, 0, 10, 0.6)])
        rec = sys_.run(item(gold=[gold(V1, 0, 10)]))
        self.assertEqual([r["duplicate"] for r in rec["retrieved"]], [False, True])
        self.assertEqual(rec["metrics"]["ndcg@10"], 1.0)

    def test_exact_topk_overlap(self):
        sys_ = FakeSystem([FakeResult("c1", V1, 0, 10, 0.6), FakeResult("c2", V1, 10, 20, 0.5)])
        rec = sys_.run(item(gold=[gold(V1, 0, 10)]), exact_topk_fn=lambda q: ["c1", "c3"])
        self.assertEqual(rec["exact_topk_overlap"], 0.5)
        self.assertIsNone(sys_.run(item(gold=[gold(V1, 0, 10)]))["exact_topk_overlap"])

    def test_transcript_coverage_buckets(self):
        sys_ = FakeSystem([FakeResult("c1", V1, 0, 10, 0.6)])
        it = item(gold=[gold(V1, 100, 120)])
        self.assertEqual(sys_.run(it, usable_text_intervals={V1: [(0, 200)]})["transcript_coverage"], "full")
        self.assertEqual(sys_.run(it, usable_text_intervals={V1: [(110, 200)]})["transcript_coverage"], "partial")
        rec = sys_.run(it, usable_text_intervals={V2: [(0, 200)]})
        self.assertEqual((rec["transcript_coverage"], rec["gold_usable_text_coverage"]), ("none", 0.0))
        self.assertIsNone(sys_.run(item(category="out_of_scope"), usable_text_intervals={})["transcript_coverage"])

    def test_serialization_is_deterministic(self):
        results = [FakeResult("c1", V1, 0.1234567891, 10.0, 0.61234567891)]
        a = FakeSystem(results).run(item(gold=[gold(V1, 0, 10)]))
        b = FakeSystem(results).run(item(gold=[gold(V1, 0, 10)]))
        dump = lambda r: json.dumps(r, sort_keys=True, ensure_ascii=False)
        self.assertEqual(dump(a), dump(b))
        self.assertEqual(a["retrieved"][0]["similarity"], 0.612346)


class TestSummaries(unittest.TestCase):

    def _records(self):
        hit = FakeSystem([FakeResult("c1", V1, 0, 10, 0.7)])
        miss = FakeSystem([FakeResult("c9", V2, 0, 10, 0.45)])
        oos_hi = FakeSystem([FakeResult("c5", V1, 0, 10, 0.55)])
        oos_lo = FakeSystem([FakeResult("c6", V1, 0, 10, 0.30)])
        return [
            hit.run(item("A", gold=[gold(V1, 0, 10)], split="dev")),
            miss.run(item("B", gold=[gold(V1, 0, 10)], split="test")),
            hit.run(item("C", status="needs_annotation", split="test")),
            oos_hi.run(item("D", category="near_miss_oos", split="dev")),
            oos_lo.run(item("E", category="out_of_scope", split="test")),
        ]

    def test_ranking_summary(self):
        s = summarize_ranking(self._records())
        self.assertEqual(s["n"], 2)
        self.assertEqual(s["recall@10"], 0.5)
        self.assertEqual(s["start_error_s"]["n"], 1)

    def test_abstention_summary(self):
        s = summarize_abstention(self._records())
        self.assertEqual(s["n_in_scope"], 3)          # needs_annotation items still count as in-scope
        self.assertEqual(s["false_answer_rate"], 0.5)  # D answered, E refused
        self.assertEqual(s["false_answer_rate_near"], 1.0)
        self.assertEqual(s["false_answer_rate_far"], 0.0)
        self.assertAlmostEqual(s["false_refusal_rate"], 1 / 3, places=6)   # B refused
        # in-scope top1: 0.7, 0.45, 0.7 ; oos: 0.55, 0.30 → wins 2+2+1+1+... = (2 + 1 + 2) / 6
        self.assertAlmostEqual(s["auroc_top1_similarity"], 5 / 6, places=6)

    def test_exact_agreement_summary(self):
        recs = [{"exact_topk_overlap": 1.0}, {"exact_topk_overlap": 0.9}, {"exact_topk_overlap": None}]
        s = summarize_exact_agreement(recs)
        self.assertEqual(s["n"], 2)
        self.assertEqual(s["fraction_identical_set"], 0.5)
        self.assertEqual(s["mean_overlap"], 0.95)

    def test_unclear_rate(self):
        recs = [
            {"retrieved": [{"unclear_text": True}, {"unclear_text": False}]},
            {"retrieved": [{"unclear_text": False}] * 5},
            {"retrieved": []},                                  # ignored: nothing retrieved
        ]
        self.assertEqual(unclear_rate_at_k(recs, 5), 0.1)       # (1/5 + 0/5) / 2

    def test_summarize_all_splits(self):
        recs = self._records()
        items_by_id = {"A": item("A", gold=[gold(V1, 0, 10)]), "B": item("B", gold=[gold(V1, 0, 10)])}
        s = summarize_all(recs, items_by_id)
        self.assertEqual(s["dev"]["ranking"]["n"], 1)
        self.assertEqual(s["test"]["ranking"]["n"], 1)
        self.assertEqual(s["all"]["by"]["video_of_first_gold_span"][V1]["n"], 2)


if __name__ == "__main__":
    unittest.main()
