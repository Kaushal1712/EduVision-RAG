"""
Unit tests for the pure diagnostics in experiments/stage3_retrieval/run_stage3.py.

Run:  venv/bin/python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments" / "stage3_retrieval"))

from bm25 import BM25, tokenize  # noqa: E402
from answer_compare_stage3 import per_query_changes, record_score, repeat_consistency  # noqa: E402
from answer_eval_stage3 import dense_order_evidence  # noqa: E402
from colbert import colbert_rerank, colbert_score, vecs_sha256  # noqa: E402
from fusion import rrf_fuse  # noqa: E402
from m3sparse import SparseIndex, encode_sparse, weights_sha256  # noqa: E402
from compare_stage3 import (  # noqa: E402
    check_comparable, evidence_hit_changes, gate_changes, list_changes, rank_changes, union_recall,
)
from run_stage3 import (  # noqa: E402
    evidence_metrics, evidence_vs_answer_eval, latency_summary, pool_recall, ranking_agreement,
    replayed_rewrites, summarize_evidence, summarize_pool,
)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from bench_metrics import GoldSpan  # noqa: E402


def chunk(cid, below=False, matched=()):
    return {"chunk_id": cid, "below_threshold": below, "matched_gold": list(matched)}


def record(qid, retrieved, query="q", rewritten=False):
    return {"id": qid, "retrieved": retrieved, "retrieval_query": query, "rewritten": rewritten}


class TestEvidenceMetrics(unittest.TestCase):

    def test_skips_below_threshold_and_caps_in_ranked_order(self):
        rec = record("a", [chunk("c1", below=True, matched=[0]), chunk("c2"), chunk("c3", matched=[1]),
                           chunk("c4"), chunk("c5", matched=[1]), chunk("c6", matched=[0])])
        ev = evidence_metrics(rec, n_gold=2, max_evidence=3)
        self.assertEqual(ev["evidence_ids"], ["c2", "c3", "c4"])
        self.assertEqual(ev["evidence_hit"], 1.0)
        self.assertEqual(ev["evidence_recall"], 0.5)          # gold 0 only in c1 (below) and c6 (cut)
        self.assertAlmostEqual(ev["evidence_precision"], 1 / 3, places=5)

    def test_gate_refused_query_has_empty_evidence(self):
        rec = record("a", [chunk("c1", below=True, matched=[0])])
        ev = evidence_metrics(rec, n_gold=1, max_evidence=5)
        self.assertEqual((ev["evidence_n"], ev["evidence_hit"], ev["evidence_recall"]), (0, 0.0, 0.0))
        self.assertIsNone(ev["evidence_precision"])

    def test_query_level_dense_gate_refusal_empties_evidence(self):
        rec = record("a", [chunk("c1", matched=[0])])
        rec["gate"] = "refused"
        ev = evidence_metrics(rec, n_gold=1, max_evidence=5)
        self.assertEqual((ev["evidence_n"], ev["evidence_hit"], ev["evidence_recall"]), (0, 0.0, 0.0))
        self.assertIsNone(ev["evidence_precision"])

    def test_summary_ignores_unscored(self):
        recs = [{"evidence": {"evidence_hit": 1.0, "evidence_recall": 1.0, "evidence_precision": 0.5,
                              "evidence_n": 2}},
                {"evidence": {"evidence_hit": 0.0, "evidence_recall": 0.0, "evidence_precision": None,
                              "evidence_n": 0}},
                {"evidence": None}]
        s = summarize_evidence(recs)
        self.assertEqual((s["n"], s["evidence_hit"], s["mean_evidence_precision"]), (2, 0.5, 0.5))


class TestAgreement(unittest.TestCase):

    def test_ranking_agreement(self):
        ref = {"a": record("a", [chunk("x"), chunk("y")]), "b": record("b", [chunk("x"), chunk("y")])}
        recs = [record("a", [chunk("x"), chunk("y")]), record("b", [chunk("y"), chunk("x")]),
                record("c", [chunk("z")])]
        out = ranking_agreement(recs, ref)
        self.assertEqual((out["n"], out["identical_ranking"], out["identical_set"]), (2, 1, 2))
        self.assertEqual([d["id"] for d in out["differing"]], ["b"])

    def test_evidence_vs_answer_eval_uses_repeat_zero_in_scope(self):
        recs = [{"id": "a", "evidence": {"evidence_hit": 1.0}}, {"id": "b", "evidence": {"evidence_hit": 0.0}},
                {"id": "c", "evidence": None}]
        ans = [{"id": "a", "repeat": 0, "category": "in_scope", "evidence_has_gold": True},
               {"id": "a", "repeat": 1, "category": "in_scope", "evidence_has_gold": False},
               {"id": "b", "repeat": 0, "category": "in_scope", "evidence_has_gold": True}]
        out = evidence_vs_answer_eval(recs, ans)
        self.assertEqual((out["n"], out["agree"]), (2, 1))
        self.assertEqual(out["disagree"][0]["id"], "b")

    def test_latency_summary(self):
        s = latency_summary([0.1, 0.2, 0.3, 0.4])
        self.assertEqual((s["n"], s["max_s"]), (4, 0.4))
        self.assertAlmostEqual(s["mean_s"], 0.25)


class Res:
    def __init__(self, cid, video, start, end):
        self.chunk_id, self.video_id, self.start_time, self.end_time, self.similarity = cid, video, start, end, 0.5


class TestPool(unittest.TestCase):

    def test_pool_recall_counts_gold_found_deeper_in_pool(self):
        gold = [GoldSpan("v1", 100.0, 110.0), GoldSpan("v2", 0.0, 5.0)]
        pool = [Res(f"c{i}", "v9", 0.0, 1.0) for i in range(30)]
        pool[3] = Res("hit1", "v1", 105.0, 115.0)      # rank 4
        pool[24] = Res("hit2", "v2", 4.0, 9.0)         # rank 25
        out = pool_recall(pool, gold, ks=(10, 20, 50))
        self.assertEqual(out, {"recall@10": 0.5, "recall@20": 0.5, "recall@50": 1.0})

    def test_summarize_pool(self):
        recs = [{"pool": {"recall@10": 0.0, "recall@50": 1.0}}, {"pool": {"recall@10": 0.5, "recall@50": 0.5}},
                {"pool": None}]
        s = summarize_pool(recs)
        self.assertEqual((s["n"], s["recall@10"], s["hit@10"], s["hit@50"]), (2, 0.25, 0.5, 1.0))
        self.assertIsNone(summarize_pool([{"pool": None}]))


class TestReplayedRewrites(unittest.TestCase):

    def test_same_query_different_history_maps_per_item(self):
        items = [{"id": "a", "query": "how does it work? ", "history": [{"q": "flexbox"}]},
                 {"id": "b", "query": "how does it work?", "history": [{"q": "grid"}]},
                 {"id": "c", "query": "what is html", "history": []}]
        control = [{"id": "a", "retrieval_query": "how does flexbox work"},
                   {"id": "b", "retrieval_query": "how does grid work"},
                   {"id": "c", "retrieval_query": "what is html"}]
        fn = replayed_rewrites(items, control)
        self.assertEqual(fn("how does it work?", [{"q": "grid"}]), "how does grid work")
        self.assertEqual(fn("how does it work?", [{"q": "flexbox"}]), "how does flexbox work")
        with self.assertRaises(KeyError):
            fn("unseen", [])


def run(gate="answered", top1=0.6, ids=("x", "y"), ev=("x",), hit=1.0, qid="a", cat="in_scope"):
    return {"id": qid, "category": cat, "gate": gate, "top1_similarity": top1,
            "retrieved": [{"chunk_id": i} for i in ids], "evidence": {"evidence_ids": list(ev), "evidence_hit": hit}}


class TestCompareStage3(unittest.TestCase):

    def result(self, **cfg):
        base = {"similarity_threshold": 0.44, "retrieval_top_k": 10, "max_llm_evidence": 5,
                "embedding_model": "m", "query_encoding": "q", "active_collection": "c", "retrieval_method": "hnsw"}
        base.update(cfg)
        return {"benchmark": {"sha256": "b"}, "split": "dev",
                "system": {"index": {"sha256": "i"}, "code_sha256": {"p": "x"}, "config": base}}

    def test_retrieval_method_may_differ_but_threshold_may_not(self):
        self.assertEqual(check_comparable(self.result(), self.result(retrieval_method="exact")), [])
        self.assertEqual(check_comparable(self.result(), self.result(similarity_threshold=0.5)),
                         ["config similarity_threshold differs"])

    def test_gate_changes_and_near_threshold(self):
        pairs = [(run(qid="a"), run(qid="a", top1=0.6000001)),
                 (run(qid="b", top1=0.442), run(qid="b", gate="refused", top1=0.439, cat="out_of_scope"))]
        g = gate_changes(pairs, 0.44)
        self.assertEqual([c["id"] for c in g["changed"]], ["b"])
        self.assertEqual(g["near_threshold_ids"], ["b"])
        self.assertAlmostEqual(g["max_abs_top1_similarity_diff"], 0.003, places=6)

    def test_list_changes_and_evidence_hits(self):
        pairs = [(run(qid="a"), run(qid="a")), (run(qid="b"), run(qid="b", ids=("y", "x"))),
                 (run(qid="c"), run(qid="c", ids=("x", "z"), hit=0.0))]
        lc = list_changes(pairs, lambda r: [x["chunk_id"] for x in r["retrieved"]])
        self.assertEqual((lc["identical"], lc["reordered"], lc["different_set"]), (1, ["b"], ["c"]))
        self.assertEqual(evidence_hit_changes(pairs), {"gained": [], "lost": ["c"]})


class TestBM25(unittest.TestCase):

    def test_tokenize_splits_markup_and_punctuation(self):
        self.assertEqual(tokenize("Use <hr> and z-index: 10!"), ["use", "hr", "and", "z", "index", "10"])

    def test_scores_match_hand_computed_okapi(self):
        import math
        docs = ["css grid layout", "css flexbox flexbox", "html table"]
        bm = BM25(docs, k1=1.2, b=0.75)
        avgdl = 8 / 3
        idf = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))           # df("flexbox") = 1
        expected = idf * 2 * 2.2 / (2 + 1.2 * (1 - 0.75 + 0.75 * 3 / avgdl))
        s = bm.scores("flexbox")
        self.assertAlmostEqual(s[1], expected, places=10)
        self.assertEqual((s[0], s[2]), (0.0, 0.0))

    def test_rare_term_outweighs_common_term_and_unknown_terms_score_zero(self):
        bm = BM25(["css grid", "css box", "css padding", "css margin"])
        s = bm.scores("css grid")
        self.assertEqual(int(s.argmax()), 0)
        self.assertGreater(bm.idf["grid"], bm.idf["css"])
        self.assertTrue((bm.scores("kya hai") == 0).all())


class TestUnionRecall(unittest.TestCase):

    def test_union_counts_gold_only_the_variant_found(self):
        a = {"id": "q", "n_gold": 2, "retrieved": [chunk("c1", matched=[0]), chunk("c2")]}
        b = {"id": "q", "n_gold": 2, "retrieved": [dict(chunk("c3", matched=[1]), rank=1, similarity=0.43),
                                                   dict(chunk("c1", matched=[0]), rank=2, similarity=0.6)]}
        u = union_recall([(a, b)], k=2, threshold=0.44)
        self.assertEqual((u["control_recall"], u["union_recall"], u["ids_with_gold_added_by_variant"]),
                         (0.5, 1.0, ["q"]))
        self.assertEqual(u["added_gold_detail"]["q"],
                         [{"variant_rank": 1, "dense_cosine": 0.43, "passes_threshold": False}])
        self.assertEqual(u["ids_with_added_gold_passing_threshold"], [])
        self.assertEqual(union_recall([(a, b)], k=1, threshold=0.44)["union_recall"], 1.0)


class FakeM3:
    def encode(self, texts, batch_size, max_length, return_dense, return_sparse, return_colbert_vecs):
        assert return_sparse and not return_dense and not return_colbert_vecs
        import numpy as np
        table = {"css grid": {"10": np.float32(0.5), "20": np.float32(0.25)}, "grid": {"20": np.float32(0.5)}}
        return {"lexical_weights": [table[t] for t in texts]}


class TestM3Sparse(unittest.TestCase):

    def test_scores_are_sums_of_shared_token_weight_products(self):
        idx = SparseIndex([{"10": 0.5, "20": 0.25}, {"30": 1.0}, {"20": 0.1}])
        s = idx.scores({"20": 0.5, "99": 1.0})
        self.assertEqual(list(s), [0.125, 0.0, 0.05])

    def test_encode_sparse_returns_plain_floats_and_hash_is_stable(self):
        w = encode_sparse(FakeM3(), ["css grid", "grid"], batch_size=16)
        self.assertEqual(w, [{"10": 0.5, "20": 0.25}, {"20": 0.5}])
        self.assertIsInstance(w[0]["10"], float)
        self.assertEqual(weights_sha256(w), weights_sha256([dict(reversed(list(w[0].items()))), w[1]]))


class TestRRF(unittest.TestCase):

    def test_scores_follow_the_formula(self):
        fused = dict(rrf_fuse([[7, 8], [8, 9]], k=60))
        self.assertAlmostEqual(fused[8], 1 / 62 + 1 / 61)
        self.assertAlmostEqual(fused[7], 1 / 61)
        self.assertAlmostEqual(fused[9], 1 / 62)

    def test_item_in_both_lists_beats_single_list_top_item(self):
        self.assertEqual([i for i, _ in rrf_fuse([[1, 2, 3], [3, 4, 5]], k=20)][:2], [3, 1])

    def test_ties_broken_by_first_ranking_then_item(self):
        # 5: dense 1 + bm25 2; 6: dense 2 + bm25 1 → equal scores; dense order wins.
        self.assertEqual([i for i, _ in rrf_fuse([[5, 6], [6, 5]], k=40)], [5, 6])
        # items only in the second list tie among themselves → ordered by item id after dense items
        self.assertEqual([i for i, _ in rrf_fuse([[1], [9, 3]], k=10)], [1, 9, 3])
        self.assertEqual([i for i, _ in rrf_fuse([[1], [], [4], [2]], k=10)], [1, 2, 4])

    def test_smaller_k_rewards_top_ranks_more(self):
        # dense-only rank 1 vs an item at rank 3 in both lists
        def winner(k):
            return rrf_fuse([[1, 9, 3], [8, 7, 3]], k)[0][0]
        self.assertEqual(winner(1), 1)
        self.assertEqual(winner(60), 3)

    def test_rejects_non_positive_k(self):
        with self.assertRaises(ValueError):
            rrf_fuse([[1]], k=0)


class TestColBERT(unittest.TestCase):

    def test_maxsim_averaged_over_query_tokens(self):
        import numpy as np
        q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        p = np.array([[1.0, 0.0], [0.6, 0.8]], dtype=np.float32)
        self.assertAlmostEqual(colbert_score(q, p), (1.0 + 0.8) / 2, places=6)

    def test_rerank_orders_by_score_and_keeps_dense_order_on_ties(self):
        import numpy as np
        q = np.array([[1.0, 0.0]], dtype=np.float32)
        vecs = {3: np.array([[0.5, 0.0]]), 7: np.array([[0.9, 0.0]]), 9: np.array([[0.5, 0.0]])}
        self.assertEqual([i for i, _ in colbert_rerank(q, [3, 9, 7], vecs)], [7, 3, 9])
        self.assertEqual([i for i, _ in colbert_rerank(q, [9, 3, 7], vecs)], [7, 9, 3])

    def test_hash_depends_on_shape_and_values(self):
        import numpy as np
        a = [np.zeros((2, 2), dtype=np.float32)]
        self.assertNotEqual(vecs_sha256(a), vecs_sha256([np.zeros((1, 4), dtype=np.float32)]))
        self.assertEqual(vecs_sha256(a), vecs_sha256([np.zeros((2, 2), dtype=np.float32)]))


class TestRankChanges(unittest.TestCase):

    def test_classifies_first_gold_rank_moves(self):
        mk = lambda qid, ranks: {"id": qid, "retrieved": [dict(chunk(f"c{r}", matched=[0] if r in ranks else []),
                                                                rank=r) for r in range(1, 6)]}
        pairs = [(mk("a", [3]), mk("a", [1])), (mk("b", [2]), mk("b", [4])), (mk("c", [2]), mk("c", [])),
                 (mk("d", []), mk("d", [5])), (mk("e", [1]), mk("e", [1]))]
        out = rank_changes(pairs)
        self.assertEqual([r["id"] for r in out["improved"]], ["a", "d"])
        self.assertEqual([r["id"] for r in out["worsened"]], ["b", "c"])
        self.assertEqual(out["unchanged"], 1)


def ans(qid, rep, label, cat="in_scope", ev=("c1",), gold=True, cite=True):
    refused = label == "refused"
    return {"id": qid, "repeat": rep, "category": cat, "language": "en", "followup": False, "refused": refused,
            "judge": {"correctness": "n/a" if refused else label, "grounding": "fully_supported",
                      "hallucination": False},
            "evidence_has_gold": gold, "citation_hit": cite, "sources_used": [{"chunk_id": c} for c in ev]}


class TestAnswerCompare(unittest.TestCase):

    def test_record_score(self):
        self.assertEqual([record_score(ans("a", 0, x)) for x in ("correct", "partially_correct", "incorrect", "refused")],
                         [2, 1, 0, 0])
        self.assertEqual(record_score(ans("o", 0, "refused", cat="out_of_scope")), 2)
        self.assertEqual(record_score(ans("o", 0, "correct", cat="out_of_scope")), 0)

    def test_per_query_changes_uses_mean_over_repeats(self):
        ref = [ans("a", 0, "correct"), ans("a", 1, "partially_correct"), ans("b", 0, "correct"), ans("b", 1, "correct"),
               ans("c", 0, "refused", gold=False), ans("c", 1, "refused", gold=False)]
        cand = [ans("a", 0, "correct"), ans("a", 1, "correct"), ans("b", 0, "refused"), ans("b", 1, "correct"),
                ans("c", 0, "correct"), ans("c", 1, "correct")]
        out = per_query_changes(ref, cand)
        self.assertEqual([r["id"] for r in out["better"]], ["a", "c"])
        self.assertEqual([r["id"] for r in out["worse"]], ["b"])
        self.assertEqual([r["id"] for r in out["major_better"]], ["c"])     # 0 → 2
        self.assertEqual([r["id"] for r in out["major_worse"]], ["b"])      # 2 → 1
        self.assertEqual(out["evidence_has_gold"], {"gained": ["c"], "lost": []})

    def test_repeat_consistency(self):
        recs = [ans("a", 0, "correct"), ans("a", 1, "correct"), ans("b", 0, "correct", ev=("x",)),
                ans("b", 1, "incorrect", ev=("y",))]
        out = repeat_consistency(recs)
        self.assertEqual((out["n"], out["same_outcome_label"], out["same_evidence_chunks"]), (2, 1, 1))
        self.assertEqual(out["different_evidence_ids"], ["b"])


class TestDenseOrderEvidence(unittest.TestCase):

    def test_selected_set_unchanged_only_order_changes(self):
        from dataclasses import dataclass

        @dataclass
        class R:
            chunk_id: str
            similarity: float
            below_threshold: bool
            rank: int = 0

        colbert = [R("a", 0.50, False), R("b", 0.40, True), R("c", 0.70, False), R("d", 0.60, False),
                   R("e", 0.65, False), R("f", 0.90, False)]
        out = dense_order_evidence(colbert, max_evidence=3)
        # evidence set = first 3 above threshold in ColBERT order: a, c, d → dense order c, d, a
        self.assertEqual([r.chunk_id for r in out], ["c", "d", "a", "b", "e", "f"])
        self.assertEqual([r.rank for r in out], [1, 2, 3, 4, 5, 6])
        selected = [r.chunk_id for r in out if not r.below_threshold][:3]
        self.assertEqual(sorted(selected), ["a", "c", "d"])            # same SET as before
        self.assertEqual([r.rank for r in colbert], [0] * 6)          # inputs not mutated

    def test_ties_keep_colbert_order_and_refused_lists_unchanged(self):
        from dataclasses import dataclass

        @dataclass
        class R:
            chunk_id: str
            similarity: float
            below_threshold: bool
            rank: int = 0

        out = dense_order_evidence([R("x", 0.5, False), R("y", 0.5, False)], max_evidence=5)
        self.assertEqual([r.chunk_id for r in out], ["x", "y"])
        out = dense_order_evidence([R("x", 0.3, True), R("y", 0.2, True)], max_evidence=5)
        self.assertEqual([r.chunk_id for r in out], ["x", "y"])


if __name__ == "__main__":
    unittest.main()
