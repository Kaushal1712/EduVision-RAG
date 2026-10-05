"""
Unit tests for eval/compare_results.py (paired CONTROL vs CANDIDATE comparison).

Run:  venv/bin/python -m unittest discover -s tests -v
"""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from compare_results import compare, outcome, paired_bootstrap_ci  # noqa: E402


def rec(qid, split, cov, r10, mrr=None, err=None):
    return {"id": qid, "split": split, "transcript_coverage": cov, "phrasing": "exact_term", "language": "en",
            "metrics": {"recall@5": r10, "recall@10": r10, "mrr@10": r10 if mrr is None else mrr,
                        "ndcg@10": r10, "start_error_s": err}}


def result(label, records, index_path="data/vector_db"):
    cfg = {"chroma_db_path": index_path, "active_collection": "c", "retrieval_top_k": 10}
    summary = {s: {"abstention": {}, "retrieved_unclear_rate@5": 0.0, "exact_search_agreement": {}}
               for s in ("dev", "test", "all")}
    return {"label": label, "benchmark": {"sha256": "b", "version": "v"}, "summary": summary,
            "system": {"code_sha256": {"pipeline.py": "x"}, "config": cfg, "index": {"sha256": label}},
            "per_query": records}


class TestCompare(unittest.TestCase):

    def test_outcome_classes(self):
        hit, miss = {"recall@10": 0.5}, {"recall@10": 0.0}
        self.assertEqual(outcome(miss, hit), "recovered")
        self.assertEqual(outcome(hit, miss), "lost")
        self.assertEqual(outcome(hit, hit), "both_hit")
        self.assertEqual(outcome(miss, miss), "both_miss")

    def test_bootstrap_is_deterministic_and_brackets_mean(self):
        deltas = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.5]
        a, b = paired_bootstrap_ci(deltas), paired_bootstrap_ci(deltas)
        self.assertEqual(a, b)
        self.assertLessEqual(a[0], sum(deltas) / len(deltas))
        self.assertGreaterEqual(a[1], sum(deltas) / len(deltas))
        self.assertIsNone(paired_bootstrap_ci([]))
        self.assertEqual(paired_bootstrap_ci([0.2, 0.2]), (0.2, 0.2))

    def test_slices_come_from_control(self):
        control = result("ctl", [rec("A", "test", "none", 0.0, err=None), rec("B", "test", "full", 1.0, err=10.0)])
        cand_recs = [rec("A", "test", "full", 1.0, err=3.0), rec("B", "test", "full", 1.0, err=2.0)]
        res = compare(control, result("cand", cand_recs, "experiments/x"), ["test"])
        block = res["splits"]["test"]
        self.assertEqual(block["by_control_coverage"]["none"]["n"], 1)
        self.assertEqual(block["by_control_coverage"]["none"]["recall@10"]["delta"], 1.0)
        out = block["overall"]["outcomes_hit@10"]
        self.assertEqual((out["recovered"]["ids"], out["both_hit"]["ids"]), (["A"], ["B"]))
        ts = block["overall"]["start_error_s"]["paired_both_hit"]
        self.assertEqual((ts["n"], ts["improved"]), (1, 1))

    def test_refuses_different_benchmark_or_code(self):
        control = result("ctl", [rec("A", "test", "full", 1.0)])
        other = copy.deepcopy(control)
        other["benchmark"]["sha256"] = "different"
        with self.assertRaises(SystemExit):
            compare(control, other, ["test"])
        other = copy.deepcopy(control)
        other["system"]["config"]["retrieval_top_k"] = 5
        with self.assertRaises(SystemExit):
            compare(control, other, ["test"])

    def test_index_path_difference_is_allowed(self):
        control = result("ctl", [rec("A", "test", "full", 1.0)])
        res = compare(control, result("cand", [rec("A", "test", "full", 0.0)], "experiments/y"), ["test"])
        self.assertEqual(res["splits"]["test"]["overall"]["outcomes_hit@10"]["lost"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
