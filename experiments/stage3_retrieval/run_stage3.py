"""
experiments/stage3_retrieval/run_stage3.py
───────────────────────────────────────────
Stage 3: retrieval experiments against the LOCKED Stage 2.6 candidate, DEV only.

Variants:
  control_s26   the frozen Stage 2.6 control: the unchanged production retrieve() (ChromaDB HNSW)
                on the lv2g_translate index at the locked threshold. Follow-up rewrites run live.
  exact_dense   brute-force exact cosine over the same stored embeddings, same query encoder
                (retriever.encode_query). Separates HNSW approximation from later changes and
                gives the candidate-pool ceiling (recall@20/@50) for reranking.
  bm25          Okapi BM25 over the indexed chunk text (bm25.py), ranked by BM25 alone; only
                chunks with a positive score are returned. Diagnostic of what lexical matching
                contributes.
  m3sparse      BGE-M3 learned sparse weights (m3sparse.py) from the production query model,
                ranked by lexical-matching score alone; positive scores only. Passage weights
                are recomputed every run and saved under artifacts/ with their SHA-256.
  rrf_bm25_k20 / _k40 / _k60
                Reciprocal Rank Fusion (fusion.py) of the exact dense top-50 and the BM25 top-50
                (positive scores), with the RRF constant k fixed in advance to 20, 40 or 60.
  colbert_dense20
                the exact dense top-20 reordered by BGE-M3 ColBERT (MaxSim) score (colbert.py);
                top-10 of the reordered list returned. colbert_dense50: the same over the dense
                top-50 (diagnostic only).

Gate (all variants): the Stage 2.6 dense-cosine gate. Every returned chunk carries its exact
BGE-M3 dense cosine as `similarity`, so the generator's per-chunk evidence filter is unchanged.
The query-level gate is "exact dense top-1 >= threshold" — for dense rankings this is what
generator.generate already does; for other rankings it keeps the gate independent of the
ranking (the list-based gate is kept as gate_from_returned_list).

  venv/bin/python experiments/stage3_retrieval/run_stage3.py --variant control_s26
  venv/bin/python experiments/stage3_retrieval/run_stage3.py --variant exact_dense|bm25|m3sparse|rrf_bm25_k20|colbert_dense20|... [--run-tag r2]
Writes eval/results/stage3_<variant>_dev[_<run-tag>].json.

Every variant other than the control replays the control run's follow-up rewrites (same item →
same retrieval query), so the comparison isolates retrieval and needs no OpenAI call.

Scoring reuses eval/run_benchmark.py (evaluate_item, summarize) unchanged. Additions:
  * coverage slices: transcript_coverage is replaced by the Stage 1.5 production-index slice
    (as eval/compare_results.py and the Stage 2 reports use); the run's own value is kept as
    transcript_coverage_self.
  * evidence: the chunks generator._select_evidence would pass to the LLM — the first
    MAX_LLM_EVIDENCE retrieved chunks at or above the threshold, in ranked order.
    evidence_hit = some evidence chunk overlaps gold; evidence_recall = share of gold spans
    matched by evidence; evidence_precision = share of evidence chunks that match gold.
  * latency of each retrieve() call (warm; model load timed separately).
  * agreement with a reference run's rankings (ChromaDB HNSW can differ between processes).
  * pool (exact_dense): recall@10/@20/@50 of the exact top-50 — the most any reranker of that
    pool could reach.
  * agreement of evidence_hit with the evidence_has_gold recorded by eval/answer_eval.py for
    the same configuration, which checks the evidence metric against what ask() really passed.

TEST is never retrieved: only DEV items are run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "eval"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EVAL_DIR))

from bench_data import (  # noqa: E402
    DEFAULT_BENCHMARK_PATH, DEFAULT_MANIFEST_PATH, gold_spans, is_scored, load_json, sha256_file,
    validate_benchmark,
)
from bench_metrics import RetrievedChunk, auroc, judge_retrieved, mean, percentile, recall_at_k  # noqa: E402
from run_benchmark import (  # noqa: E402
    EVAL_CODE_FILES, SYSTEM_CODE_FILES, _r, evaluate_item, summarize,
)

RESULTS = EVAL_DIR / "results"
LOCKED_SELECTION = ROOT / "experiments" / "stage2_6_abstention" / "locked_selection.json"
INDEX_DIR = ROOT / "experiments" / "stage2_transcription" / "artifacts" / "lv2g_translate" / "vector_db"
COLLECTION = "lv2g_translate"
# The index the Stage 2/2.6 results were produced on; its fingerprint must not change.
INDEX_REFERENCE_RUN = RESULTS / "stage2_lv2g_translate.json"
# Stage 1.5 production-index coverage slices (full / partial / none).
COVERAGE_REFERENCE_RUN = RESULTS / "stage1_5_baseline_bench-v1.1.json"
# Answer-level DEV control for the locked configuration (t0.44 + guard).
ANSWER_CONTROL = RESULTS / "answers_s26_cand_t044_guard_dev.json"
CONTROL_RUN = RESULTS / "stage3_control_s26_dev.json"
SPLIT = "dev"
RRF_KS = (20, 40, 60)          # predeclared sweep; not extended after seeing results
RRF_LIST_DEPTH = 50            # items taken from each input ranking
COLBERT_POOLS = (20, 50)       # 20 = the experiment; 50 = diagnostic
VARIANTS = (("control_s26", "exact_dense", "bm25", "m3sparse") + tuple(f"rrf_bm25_k{k}" for k in RRF_KS)
            + tuple(f"colbert_dense{n}" for n in COLBERT_POOLS))
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
POOL_SIZE = 50
POOL_KS = (10, 20, 50)


# ── Pure diagnostics ──────────────────────────────────────────────────────────

def evidence_metrics(record: dict, n_gold: int, max_evidence: int) -> dict:
    """What generator._select_evidence would pass to the LLM, scored against gold."""
    evidence = ([] if record.get("gate") == "refused" else
                [x for x in record["retrieved"] if not x["below_threshold"]][:max_evidence])
    matched = [x["matched_gold"] for x in evidence]
    covered = {g for m in matched for g in m}
    return {
        "evidence_ids":       [x["chunk_id"] for x in evidence],
        "evidence_n":         len(evidence),
        "evidence_hit":       float(bool(covered)),
        "evidence_recall":    _r(len(covered) / n_gold) if n_gold else None,
        "evidence_precision": _r(sum(bool(m) for m in matched) / len(evidence)) if evidence else None,
    }


def summarize_evidence(records: Sequence[dict]) -> dict:
    scored = [r for r in records if r.get("evidence")]
    prec = [r["evidence"]["evidence_precision"] for r in scored if r["evidence"]["evidence_precision"] is not None]
    return {
        "n":                       len(scored),
        "evidence_hit":            _r(mean([r["evidence"]["evidence_hit"] for r in scored])),
        "evidence_recall":         _r(mean([r["evidence"]["evidence_recall"] for r in scored])),
        "mean_evidence_precision": _r(mean(prec)),
        "mean_evidence_n":         _r(mean([r["evidence"]["evidence_n"] for r in scored])),
    }


def pool_recall(pool: Sequence, gold: Sequence, ks: Sequence[int] = POOL_KS) -> dict:
    """recall@k over a ranked candidate pool (objects with RetrievalResult attributes)."""
    chunks = [RetrievedChunk(rank=i + 1, chunk_id=r.chunk_id, video_id=r.video_id,
                             start_time=r.start_time, end_time=r.end_time, similarity=r.similarity)
              for i, r in enumerate(pool)]
    matches = [j["matched_gold"] for j in judge_retrieved(chunks, gold)]
    return {f"recall@{k}": _r(recall_at_k(matches, len(gold), k)) for k in ks}


def summarize_pool(records: Sequence[dict]) -> Optional[dict]:
    scored = [r for r in records if r.get("pool")]
    if not scored:
        return None
    out: dict = {"n": len(scored)}
    for key in scored[0]["pool"]:
        vals = [r["pool"][key] for r in scored]
        out[key] = _r(mean(vals))
        out[f"hit{key[len('recall'):]}"] = _r(mean([float(v > 0) for v in vals]))
    return out


def replayed_rewrites(items: Sequence[dict], control_records: Sequence[dict]):
    """rewrite_fn that returns the control run's retrieval query for the same item."""
    by_id = {r["id"]: r["retrieval_query"] for r in control_records}
    table: dict[tuple[str, str], str] = {}
    for item in items:
        key = (item["query"].strip(), json.dumps(item["history"], sort_keys=True))
        if table.get(key, by_id[item["id"]]) != by_id[item["id"]]:
            raise ValueError(f"conflicting replayed rewrites for {key}")
        table[key] = by_id[item["id"]]

    def rewrite_fn(query: str, history: list) -> str:
        return table[(query, json.dumps(history, sort_keys=True))]
    return rewrite_fn


def latency_summary(latencies: Sequence[float]) -> dict:
    return {"n": len(latencies), "mean_s": _r(mean(latencies)), "p50_s": _r(percentile(latencies, 50)),
            "p95_s": _r(percentile(latencies, 95)), "max_s": _r(max(latencies)) if latencies else None}


def ranking_agreement(records: Sequence[dict], reference: dict[str, dict]) -> dict:
    """Per-query comparison of retrieved chunk_id lists with a reference run."""
    same_list, same_set, differ = 0, 0, []
    for r in records:
        ref = reference.get(r["id"])
        if ref is None:
            continue
        a = [x["chunk_id"] for x in r["retrieved"]]
        b = [x["chunk_id"] for x in ref["retrieved"]]
        same_list += a == b
        same_set += set(a) == set(b)
        if a != b:
            differ.append({"id": r["id"], "rewritten": r["rewritten"],
                           "same_query_text": r["retrieval_query"] == ref["retrieval_query"],
                           "set_overlap": _r(len(set(a) & set(b)) / max(len(a), 1))})
    n = sum(r["id"] in reference for r in records)
    return {"n": n, "identical_ranking": same_list, "identical_set": same_set, "differing": differ}


def evidence_vs_answer_eval(records: Sequence[dict], answer_records: Sequence[dict]) -> dict:
    """evidence_hit here vs evidence_has_gold recorded by answer_eval (repeat 0, in-scope)."""
    ans = {a["id"]: a for a in answer_records if a["repeat"] == 0 and a["category"] == "in_scope"}
    agree, disagree = 0, []
    for r in records:
        a = ans.get(r["id"])
        if a is None or not r.get("evidence"):
            continue
        if bool(r["evidence"]["evidence_hit"]) == bool(a["evidence_has_gold"]):
            agree += 1
        else:
            disagree.append({"id": r["id"], "stage3": r["evidence"]["evidence_hit"],
                             "answer_eval": a["evidence_has_gold"]})
    return {"n": agree + len(disagree), "agree": agree, "disagree": disagree}


# ── Run ───────────────────────────────────────────────────────────────────────

def _bind_environment(threshold: float) -> None:
    """Point config/settings.py at the locked index before any project module imports it."""
    wanted = {"CHROMA_DB_PATH": str(INDEX_DIR), "ACTIVE_COLLECTION": COLLECTION,
              "SIMILARITY_THRESHOLD": f"{threshold}"}
    for k, v in wanted.items():
        if k in os.environ and os.environ[k] != v:
            raise SystemExit(f"{k} is set to {os.environ[k]!r}; Stage 3 requires {v!r}")
        os.environ[k] = v
    if "config.settings" in sys.modules:
        raise SystemExit("config.settings was imported before the environment was bound")


class _Corpus:
    """The collection's chunks in a fixed order (sorted ids), with exact dense cosine scoring."""

    def __init__(self, collection_name: str, threshold: float):
        import numpy as np
        from ingestion.indexer import get_chroma_client
        from retrieval.retriever import RetrievalResult, _fmt_timestamp, encode_query
        self._np, self._RetrievalResult, self._fmt, self._encode = np, RetrievalResult, _fmt_timestamp, encode_query
        got = get_chroma_client().get_collection(collection_name).get(
            include=["embeddings", "metadatas", "documents"])
        order = sorted(range(len(got["ids"])), key=lambda i: got["ids"][i])   # fixed tie order
        self.ids = [got["ids"][i] for i in order]
        self.metas = [got["metadatas"][i] for i in order]
        self.docs = [got["documents"][i] for i in order]
        emb = np.asarray([got["embeddings"][i] for i in order], dtype=np.float64)
        self.emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        self.threshold = threshold

    def dense_sims(self, q: str):
        v = self._np.asarray(self._encode(q), dtype=self._np.float64)
        return self.emb @ (v / self._np.linalg.norm(v))

    def result(self, i: int, rank: int, similarity: float):
        """A RetrievalResult built field-for-field as retriever.retrieve() builds it."""
        m = self.metas[i]
        start, end = float(m.get("start_time", 0.0)), float(m.get("end_time", 0.0))
        return self._RetrievalResult(
            chunk_id=self.ids[i], rank=rank, similarity=similarity, text_en=self.docs[i],
            text_raw=m.get("text_raw", self.docs[i]), video_id=m.get("video_id", ""),
            video_filename=m.get("video_filename", ""), start_time=start, end_time=end,
            start_time_fmt=self._fmt(start, end - start), end_time_fmt=self._fmt(end, end - start),
            source_segment_ids=json.loads(m.get("source_segment_ids", "[]")),
            language=m.get("language", ""), chunk_index=int(m.get("chunk_index", 0)),
            below_threshold=similarity < self.threshold)


def _exact_dense_retriever(corpus: _Corpus, top_k: int, pool_size: int):
    """
    Exact cosine ranking. Returns (retrieve_fn, pools, extras): retrieve_fn(q) gives the top_k;
    pools[q] keeps the top pool_size; extras[q] holds per-query scores for the record.
    """
    import numpy as np
    pools: dict[str, list] = {}
    extras: dict[str, dict] = {}

    def retrieve_fn(q: str) -> list:
        sims = corpus.dense_sims(q)
        top = np.argsort(-sims, kind="stable")[:pool_size]
        pools[q] = [corpus.result(int(i), n + 1, float(sims[i])) for n, i in enumerate(top)]
        extras[q] = {"dense_top1": float(sims[top[0]])}
        return pools[q][:top_k]
    return retrieve_fn, pools, extras


def _bm25_retriever(corpus: _Corpus, top_k: int, pool_size: int):
    """BM25 ranking of chunks with a positive score; similarity = the chunk's exact dense cosine."""
    import numpy as np
    from bm25 import BM25
    index = BM25(corpus.docs)
    pools: dict[str, list] = {}
    extras: dict[str, dict] = {}

    def retrieve_fn(q: str) -> list:
        sims = corpus.dense_sims(q)               # gate and evidence filter only, not the ranking
        t0 = time.perf_counter()
        scores = index.scores(q)
        top = [int(i) for i in np.argsort(-scores, kind="stable")[:pool_size] if scores[i] > 0]
        scoring_s = time.perf_counter() - t0
        pools[q] = [corpus.result(i, n + 1, float(sims[i])) for n, i in enumerate(top)]
        extras[q] = {"dense_top1": float(sims.max()), "bm25_top1": float(scores[top[0]]) if top else 0.0,
                     "bm25_n_positive": int((scores > 0).sum()), "bm25_scoring_s": scoring_s}
        return pools[q][:top_k]
    return retrieve_fn, pools, extras


def _m3sparse_retriever(corpus: _Corpus, top_k: int, pool_size: int):
    """BGE-M3 lexical-matching ranking of chunks with a positive score; similarity = exact dense cosine."""
    import numpy as np
    from m3sparse import PASSAGE_BATCH_SIZE, SparseIndex, encode_sparse, weights_sha256
    from retrieval.retriever import _get_model
    model = _get_model()
    t0 = time.perf_counter()
    passage_weights = encode_sparse(model, corpus.docs, PASSAGE_BATCH_SIZE)
    index = SparseIndex(passage_weights)
    pools: dict[str, list] = {}
    extras: dict[str, dict] = {}

    def retrieve_fn(q: str) -> list:
        sims = corpus.dense_sims(q)               # gate and evidence filter only, not the ranking
        t1 = time.perf_counter()
        qw = encode_sparse(model, [q.strip()], batch_size=1)[0]
        t2 = time.perf_counter()
        scores = index.scores(qw)
        top = [int(i) for i in np.argsort(-scores, kind="stable")[:pool_size] if scores[i] > 0]
        t3 = time.perf_counter()
        pools[q] = [corpus.result(i, n + 1, float(sims[i])) for n, i in enumerate(top)]
        extras[q] = {"dense_top1": float(sims.max()), "m3sparse_top1": float(scores[top[0]]) if top else 0.0,
                     "m3sparse_n_positive": int((scores > 0).sum()), "m3sparse_query_encoding_s": t2 - t1,
                     "m3sparse_scoring_s": t3 - t2}
        return pools[q][:top_k]
    retrieve_fn.index_info = {"passage_encoding_s": time.perf_counter() - t0, "n_passages": len(passage_weights),
                              "passage_weights_sha256": weights_sha256(passage_weights),
                              "mean_tokens_per_passage": float(np.mean([len(w) for w in passage_weights]))}
    retrieve_fn.passage_weights = dict(zip(corpus.ids, passage_weights))
    return retrieve_fn, pools, extras


def _rrf_bm25_retriever(rrf_k: int):
    """Factory: RRF of exact dense and BM25 rankings; similarity = exact dense cosine."""
    def factory(corpus: _Corpus, top_k: int, pool_size: int):
        import numpy as np
        from bm25 import BM25
        from fusion import rrf_fuse
        index = BM25(corpus.docs)
        pools: dict[str, list] = {}
        extras: dict[str, dict] = {}

        def retrieve_fn(q: str) -> list:
            t0 = time.perf_counter()
            sims = corpus.dense_sims(q)
            dense_rank = [int(i) for i in np.argsort(-sims, kind="stable")[:RRF_LIST_DEPTH]]
            t1 = time.perf_counter()
            scores = index.scores(q)
            bm25_rank = [int(i) for i in np.argsort(-scores, kind="stable")[:RRF_LIST_DEPTH] if scores[i] > 0]
            t2 = time.perf_counter()
            fused = rrf_fuse([dense_rank, bm25_rank], rrf_k)[:pool_size]
            t3 = time.perf_counter()
            pools[q] = [corpus.result(i, n + 1, float(sims[i])) for n, (i, _) in enumerate(fused)]
            top_ids = {i for i, _ in fused[:top_k]}
            extras[q] = {"dense_top1": float(sims[dense_rank[0]]), "rrf_top1": fused[0][1],
                         "top_k_from_bm25_only": len(top_ids - set(dense_rank[:top_k])),
                         "dense_s": t1 - t0, "bm25_s": t2 - t1, "fusion_s": t3 - t2}
            return pools[q][:top_k]
        return retrieve_fn, pools, extras
    return factory


def _colbert_dense_retriever(depth: int):
    """Factory: exact dense top-`depth` reordered by ColBERT; similarity = exact dense cosine."""
    def factory(corpus: _Corpus, top_k: int, pool_size: int):
        import numpy as np
        from colbert import PASSAGE_BATCH_SIZE, colbert_rerank, encode_colbert, vecs_sha256
        from retrieval.retriever import _get_model
        model = _get_model()
        t0 = time.perf_counter()
        passage_vecs = encode_colbert(model, corpus.docs, PASSAGE_BATCH_SIZE)
        build_s = time.perf_counter() - t0
        pools: dict[str, list] = {}
        extras: dict[str, dict] = {}
        parity: list[float] = []

        def retrieve_fn(q: str) -> list:
            t1 = time.perf_counter()
            sims = corpus.dense_sims(q)
            dense_rank = [int(i) for i in np.argsort(-sims, kind="stable")[:depth]]
            t2 = time.perf_counter()
            qv = encode_colbert(model, [q.strip()], batch_size=1)[0]
            t3 = time.perf_counter()
            reranked = colbert_rerank(qv, dense_rank, passage_vecs)
            t4 = time.perf_counter()
            if not parity:   # first call: check the NumPy score against FlagEmbedding's own
                parity.extend(abs(sc - float(model.colbert_score(qv, passage_vecs[i]))) for i, sc in reranked)
            pools[q] = [corpus.result(i, n + 1, float(sims[i])) for n, (i, _) in enumerate(reranked)]
            extras[q] = {"dense_top1": float(sims[dense_rank[0]]), "colbert_top1": reranked[0][1],
                         "top_k_from_outside_dense_top_k": sum(1 for i, _ in reranked[:top_k]
                                                                if dense_rank.index(i) >= top_k),
                         "dense_rank_of_new_top1": dense_rank.index(reranked[0][0]) + 1,
                         "dense_s": t2 - t1, "colbert_query_s": t3 - t2, "colbert_score_s": t4 - t3}
            return pools[q][:top_k]
        retrieve_fn.index_info = {
            "passage_encoding_s": build_s, "n_passages": len(passage_vecs),
            "passage_vecs_sha256": vecs_sha256(passage_vecs),
            "mean_tokens_per_passage": float(np.mean([v.shape[0] for v in passage_vecs])),
            "passage_vecs_mb_float32": float(sum(v.nbytes for v in passage_vecs) / 2**20),
            "candidates_per_query": depth,
        }
        retrieve_fn.parity = parity
        retrieve_fn.pool_ks = tuple(k for k in POOL_KS if k <= depth)
        return retrieve_fn, pools, extras
    return factory


RETRIEVAL_DESCRIPTIONS = {
    "exact_dense": "exact cosine over stored embeddings (float64), retriever.encode_query",
    "bm25": "Okapi BM25 over indexed text_en (bm25.py: [a-z0-9]+ tokens, k1=1.2, b=0.75, no stemming/stop words), "
            "positive scores only",
    "m3sparse": "BGE-M3 lexical weights (production query model, fp32; passages max_length 512 batch 16, "
                "queries max_length 512), score = sum of shared-token weight products, positive scores only",
    **{f"rrf_bm25_k{k}": f"RRF (k={k}) of exact dense top-{RRF_LIST_DEPTH} and BM25 top-{RRF_LIST_DEPTH} "
                         "(positive scores; bm25.py settings unchanged); ties by dense rank" for k in RRF_KS},
    **{f"colbert_dense{n}": f"exact dense top-{n} reordered by BGE-M3 ColBERT MaxSim (colbert.py; production query "
                            "model, fp32, passages max_length 512 batch 16); ties by dense rank" for n in COLBERT_POOLS},
}


LEXICAL_PARAMS = {"bm25": "k1=1.2, b=0.75 (bm25.py)", "m3sparse": "BGE-M3 sparse head (m3sparse.py)"}


def _lexical_diagnostics(records: Sequence[dict], top_k: int, prefix: str) -> dict:
    """Lexical-ranking diagnostics: score separability for abstention, empty/short lists, cost."""
    run = [r for r in records if r.get("variant_scores")]
    score = lambda rs: [r["variant_scores"][f"{prefix}_top1"] for r in rs]
    ins = [r for r in run if r["category"] == "in_scope"]
    near = [r for r in run if r["category"] == "near_miss_oos"]
    oos = near + [r for r in run if r["category"] == "out_of_scope"]
    out = {
        "params": LEXICAL_PARAMS[prefix],
        f"auroc_top1_{prefix}": _r(auroc(score(ins), score(oos))),
        f"auroc_top1_{prefix}_near": _r(auroc(score(ins), score(near))),
        "queries_with_fewer_than_top_k_positive": sorted(
            r["id"] for r in run if r["variant_scores"][f"{prefix}_n_positive"] < top_k),
        "gate_from_returned_list_differs": sorted(r["id"] for r in run if r["gate_from_returned_list"] != r["gate"]),
        "scoring_latency": latency_summary([r["variant_scores"][f"{prefix}_scoring_s"] for r in run]),
    }
    if f"{prefix}_query_encoding_s" in run[0]["variant_scores"]:
        out["query_encoding_latency"] = latency_summary([r["variant_scores"][f"{prefix}_query_encoding_s"]
                                                         for r in run])
    return out


def _rrf_diagnostics(records: Sequence[dict]) -> dict:
    """Per-component cost and how much of the fused top-k BM25 alone brought in."""
    run = [r for r in records if r.get("variant_scores")]
    v = lambda key: [r["variant_scores"][key] for r in run]
    return {
        "latency": {key: latency_summary(v(key)) for key in ("dense_s", "bm25_s", "fusion_s")},
        "mean_top_k_from_bm25_only": _r(mean(v("top_k_from_bm25_only"))),
        "gate_from_returned_list_differs": sorted(r["id"] for r in run if r["gate_from_returned_list"] != r["gate"]),
    }


def _colbert_diagnostics(records: Sequence[dict], parity: Sequence[float]) -> dict:
    run = [r for r in records if r.get("variant_scores")]
    v = lambda key: [r["variant_scores"][key] for r in run]
    return {
        "latency": {key: latency_summary(v(key)) for key in ("dense_s", "colbert_query_s", "colbert_score_s")},
        "parity_max_abs_diff_vs_flagembedding_colbert_score": max(parity) if parity else None,
        "mean_top_k_from_outside_dense_top_k": _r(mean(v("top_k_from_outside_dense_top_k"))),
        "top1_changed": sum(x != 1 for x in v("dense_rank_of_new_top1")),
        "gate_from_returned_list_differs": sorted(r["id"] for r in run if r["gate_from_returned_list"] != r["gate"]),
    }


def _sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--run-tag", help="suffix for a repeat run, e.g. r2")
    args = p.parse_args()

    locked = load_json(LOCKED_SELECTION)
    threshold = float(locked["similarity_threshold"])
    _bind_environment(threshold)

    bench = load_json(DEFAULT_BENCHMARK_PATH)
    errors = validate_benchmark(bench, load_json(DEFAULT_MANIFEST_PATH))
    if errors:
        raise SystemExit(f"benchmark invalid: {errors[:3]}")
    items = [q for q in bench["queries"] if q["split"] == SPLIT]

    import config.settings as s
    from run_benchmark import (_exact_topk_function, _git_state, _index_fingerprint, _rel_hashes,
                               _system_config, _system_functions, _usable_text_intervals, _versions)
    assert s.SIMILARITY_THRESHOLD == threshold and s.ACTIVE_COLLECTION == COLLECTION

    index = _index_fingerprint(COLLECTION)
    expected_sha = load_json(INDEX_REFERENCE_RUN)["system"]["index"]["sha256"]
    if index["sha256"] != expected_sha:
        raise SystemExit(f"index fingerprint {index['sha256']} != locked {expected_sha}")

    retrieve_fn, rewrite_fn, validate_fn = _system_functions(s.RETRIEVAL_TOP_K, threshold, COLLECTION)
    pools: dict[str, list] = {}
    extras: dict[str, dict] = {}
    if args.variant == "control_s26":
        experiment = {"retrieval": "production retrieve(): BGE-M3 dense, ChromaDB HNSW top-k, unchanged",
                      "followup_rewrites": "live (pipeline._rewrite_query_for_retrieval)"}
    else:
        factory = {"exact_dense": _exact_dense_retriever, "bm25": _bm25_retriever,
                   "m3sparse": _m3sparse_retriever,
                   **{f"rrf_bm25_k{k}": _rrf_bm25_retriever(k) for k in RRF_KS},
                   **{f"colbert_dense{n}": _colbert_dense_retriever(n) for n in COLBERT_POOLS}}[args.variant]
        retrieve_fn, pools, extras = factory(_Corpus(COLLECTION, threshold), s.RETRIEVAL_TOP_K, POOL_SIZE)
        rewrite_fn = replayed_rewrites(items, load_json(CONTROL_RUN)["per_query"])
        experiment = {"retrieval": f"{RETRIEVAL_DESCRIPTIONS[args.variant]}; "
                                   f"top-{s.RETRIEVAL_TOP_K} returned, top-{POOL_SIZE} kept as pool",
                      "followup_rewrites": f"replayed from {CONTROL_RUN.relative_to(ROOT)}",
                      "control_run_sha256": _sha256_text(CONTROL_RUN)}
        if hasattr(retrieve_fn, "index_info"):
            experiment["index_build"] = {k: _r(v) if isinstance(v, float) else v
                                         for k, v in retrieve_fn.index_info.items()}
    t = time.perf_counter()
    retrieve_fn("warm-up query: what is HTML")
    model_load_s = time.perf_counter() - t
    exact_fn = _exact_topk_function(s.RETRIEVAL_TOP_K, COLLECTION)
    usable = _usable_text_intervals(COLLECTION)
    coverage_s15 = {r["id"]: r["transcript_coverage"] for r in load_json(COVERAGE_REFERENCE_RUN)["per_query"]}

    timings: list[float] = []

    def timed_retrieve(q: str):
        t0 = time.perf_counter()
        out = retrieve_fn(q)
        timings.append(time.perf_counter() - t0)
        return out

    started = datetime.now(timezone.utc)
    records = []
    for item in items:
        before = len(timings)
        rec = evaluate_item(item, timed_retrieve, rewrite_fn, validate_fn, k_max=s.RETRIEVAL_TOP_K,
                            exact_topk_fn=exact_fn, usable_text_intervals=usable)
        rec["retrieval_latency_s"] = _r(timings[-1]) if len(timings) > before else None
        rec["transcript_coverage_self"] = rec["transcript_coverage"]
        rec["transcript_coverage"] = coverage_s15[rec["id"]] if rec["metrics"] is not None else None
        ex = extras.get(rec["retrieval_query"]) if rec["validation_error"] is None else None
        if ex is not None:
            rec["variant_scores"] = {k: _r(v) for k, v in ex.items()}
            rec["gate_from_returned_list"] = rec["gate"]
            rec["top1_similarity"] = _r(ex["dense_top1"])
            rec["gate"] = "answered" if ex["dense_top1"] >= threshold else "refused"
        rec["dense_gate"] = rec["gate"]
        rec["n_gold"] = len(gold_spans(item)) if is_scored(item) else None
        rec["evidence"] = (evidence_metrics(rec, len(gold_spans(item)), s.MAX_LLM_EVIDENCE)
                           if is_scored(item) else None)
        pool = pools.get(rec["retrieval_query"])
        rec["pool"] = (pool_recall(pool, gold_spans(item), getattr(retrieve_fn, "pool_ks", POOL_KS))
                       if pool is not None and is_scored(item) else None)
        records.append(rec)

    reference = {r["id"]: r for r in load_json(INDEX_REFERENCE_RUN)["per_query"] if r["split"] == SPLIT}
    answer_records = load_json(ANSWER_CONTROL)["records"]
    summary = summarize(records)
    summary["evidence"] = summarize_evidence(records)
    summary["evidence_by_language"] = {
        lang: summarize_evidence([r for r in records if r["language"] == lang]) for lang in ("en", "hinglish")}
    summary["pool"] = summarize_pool(records)
    summary["latency"] = latency_summary([r["retrieval_latency_s"] for r in records
                                          if r["retrieval_latency_s"] is not None])

    config = _system_config()
    if args.variant != "control_s26":
        config["retrieval_method"] = experiment["retrieval"]
        config["followup_rewrite"] = experiment["followup_rewrites"]
    me = Path(__file__)
    result = {
        "label": f"stage3_{args.variant}_{SPLIT}" + (f"_{args.run_tag}" if args.run_tag else ""),
        "variant": args.variant,
        "split": SPLIT,
        "benchmark": {"version": bench["benchmark_version"],
                      "path": str(DEFAULT_BENCHMARK_PATH.relative_to(ROOT)),
                      "sha256": sha256_file(DEFAULT_BENCHMARK_PATH), "n_queries_run": len(items)},
        "system": {"config": config, "index": index, "code_sha256": _rel_hashes(SYSTEM_CODE_FILES)},
        "experiment": {
            **experiment,
            "gate": "dense-cosine gate at the locked threshold (generator.generate)",
            "locked_selection_sha256": _sha256_text(LOCKED_SELECTION),
            "code_sha256": {str(f.relative_to(ROOT)): _sha256_text(f) for f in sorted(me.parent.glob("*.py"))},
            "slices": "transcript_coverage = Stage 1.5 production-index slice",
        },
        "evaluation": {"k_max": s.RETRIEVAL_TOP_K, "code_sha256": _rel_hashes(EVAL_CODE_FILES),
                       "git": _git_state(), "versions": _versions()},
        "summary": {SPLIT: summary},
        "diagnostics": {
            "ranking_agreement_with_stage2_run": ranking_agreement(records, reference),
            "evidence_vs_answer_eval": evidence_vs_answer_eval(records, answer_records),
            "model_load_and_first_query_s": _r(model_load_s),
            **({args.variant: _lexical_diagnostics(records, s.RETRIEVAL_TOP_K, args.variant)}
               if args.variant in LEXICAL_PARAMS else {}),
            **({"rrf": _rrf_diagnostics(records)} if args.variant.startswith("rrf_") else {}),
            **({"colbert": _colbert_diagnostics(records, retrieve_fn.parity)}
               if args.variant.startswith("colbert_") else {}),
        },
        "per_query": records,
        "run_info": {"started_at": started.isoformat(timespec="seconds")},
    }
    if hasattr(retrieve_fn, "passage_weights"):
        ARTIFACTS.mkdir(exist_ok=True)
        art = ARTIFACTS / f"{result['label']}_passage_weights.json"
        art.write_text(json.dumps(retrieve_fn.passage_weights, sort_keys=True) + "\n")
        result["experiment"]["index_build"]["artifact"] = str(art.relative_to(ROOT))
    out = RESULTS / f"{result['label']}.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
