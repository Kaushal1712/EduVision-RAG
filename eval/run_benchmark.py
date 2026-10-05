"""
eval/run_benchmark.py
──────────────────────
Stage 1: run the timestamp-anchored benchmark against the CURRENT retrieval system
and write a reproducible, machine-readable result file.

═══════════════════════════════════════════════════════════════
WHAT IS MEASURED
═══════════════════════════════════════════════════════════════

The retrieval half of pipeline.ask(), reproduced call-for-call:

  query.strip() → validate_query()
    → _rewrite_query_for_retrieval(query, history)      (follow-ups only)
    → retrieve(top_k=RETRIEVAL_TOP_K,
               similarity_threshold=SIMILARITY_THRESHOLD,
               collection_name=ACTIVE_COLLECTION)

Generation is not run. Ranking metrics need only the ranked chunks, and the
no-LLM refusal gate (generate() returns not_found when no chunk is at or above
the threshold) is fully determined by the retrieved similarities.
If pipeline.ask() changes how it calls retrieval, update _system_functions().

Follow-up rewriting calls OpenAI (temperature 0) when the query contains a
trigger pronoun; the rewritten text is recorded per query so runs can be compared.

ChromaDB's HNSW search is approximate, and its results for a few queries were
observed to differ between processes. Each query is therefore also compared with
an exact brute-force top-k over the collection's stored embeddings
(exact_topk_overlap). This is a diagnostic of the current system only; the
ranking metrics always score what the system actually returned.

Metric definitions live in eval/bench_metrics.py; the benchmark format in
eval/benchmark/README.md.

═══════════════════════════════════════════════════════════════
USAGE
═══════════════════════════════════════════════════════════════

  cd <project_root>
  venv/bin/python eval/run_benchmark.py                    # → eval/results/baseline.json
  venv/bin/python eval/run_benchmark.py --label stage2_x   # → eval/results/stage2_x.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

_EVAL_DIR = Path(__file__).resolve().parent
_ROOT = _EVAL_DIR.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_EVAL_DIR))

from bench_data import (  # noqa: E402
    DEFAULT_BENCHMARK_PATH,
    DEFAULT_MANIFEST_PATH,
    gold_spans,
    is_scored,
    load_json,
    sha256_file,
    validate_benchmark,
)
from bench_metrics import (  # noqa: E402
    RetrievedChunk,
    auroc,
    covered_fraction,
    first_relevant_start_error,
    judge_retrieved,
    mean,
    ndcg_at_k,
    percentile,
    rate,
    recall_at_k,
    reciprocal_rank,
)

RESULT_SCHEMA_VERSION = 1
RESULTS_DIR: Path = _EVAL_DIR / "results"
UNCLEAR_MARKER = "[unclear audio]"   # written by ingestion/normalizer.py for untranslatable chunks
FLOAT_DIGITS = 6

# Files whose content defines "the evaluation code" and "the system under test".
EVAL_CODE_FILES = ["eval/bench_metrics.py", "eval/bench_data.py", "eval/run_benchmark.py"]
SYSTEM_CODE_FILES = ["pipeline.py", "retrieval/retriever.py", "config/settings.py", "ingestion/indexer.py"]


def _r(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(float(x), FLOAT_DIGITS)


# ── Per-query evaluation (pure, given the injected system functions) ──────────

def evaluate_item(
    item: dict,
    retrieve_fn: Callable[[str], Sequence],
    rewrite_fn: Callable[[str, list], str],
    validate_fn: Callable[[str], Optional[str]],
    k_max: int,
    min_overlap_s: float = 0.0,
    exact_topk_fn: Optional[Callable[[str], list[str]]] = None,
    usable_text_intervals: Optional[dict] = None,
) -> dict:
    """
    Run one benchmark item through the system and score it.

    retrieve_fn returns objects with the RetrievalResult attributes
    (chunk_id, video_id, start_time, end_time, similarity, below_threshold, text_en).
    exact_topk_fn, if given, returns the exact nearest-neighbour chunk_ids for a query.
    usable_text_intervals, if given, maps video_id → [(start, end)] of indexed chunks whose
    text is usable (not [unclear audio]); it yields the transcript-coverage diagnostic.
    """
    query = item["query"].strip()
    validation_error = validate_fn(query)
    retrieval_query = query
    results: Sequence = []
    if validation_error is None:
        retrieval_query = rewrite_fn(query, item["history"]) if item["history"] else query
        results = list(retrieve_fn(retrieval_query))

    gold = gold_spans(item)
    chunks = [
        RetrievedChunk(rank=i + 1, chunk_id=r.chunk_id, video_id=r.video_id,
                       start_time=r.start_time, end_time=r.end_time, similarity=r.similarity)
        for i, r in enumerate(results)
    ]
    judged = judge_retrieved(chunks, gold, min_overlap_s)
    matches = [j["matched_gold"] for j in judged]
    unclear = [UNCLEAR_MARKER in (r.text_en or "") for r in results]

    answered = validation_error is None and any(not r.below_threshold for r in results)
    exact_overlap = None
    if exact_topk_fn is not None and validation_error is None:
        exact = exact_topk_fn(retrieval_query)
        exact_overlap = _r(len({r.chunk_id for r in results} & set(exact)) / len(exact)) if exact else None
    record = {
        "id":               item["id"],
        "split":            item["split"],
        "category":         item["category"],
        "followup":         item["followup"],
        "language":         item["language"],
        "phrasing":         item["phrasing"],
        "difficulty":       item["difficulty"],
        "status":           item["annotation"]["status"],
        "query":            item["query"],
        "retrieval_query":  retrieval_query,
        "rewritten":        retrieval_query != query,
        "validation_error": validation_error,
        "gate":             "answered" if answered else "refused",
        "top1_similarity":  _r(results[0].similarity) if results else None,
        "exact_topk_overlap": exact_overlap,
        "retrieved": [
            {
                "rank":            c.rank,
                "chunk_id":        c.chunk_id,
                "video_id":        c.video_id,
                "start_time":      _r(c.start_time),
                "end_time":        _r(c.end_time),
                "similarity":      _r(c.similarity),
                "below_threshold": bool(r.below_threshold),
                "unclear_text":    u,
                "matched_gold":    j["matched_gold"],
                "invalid":         j["invalid"],
                "duplicate":       j["duplicate"],
            }
            for c, r, u, j in zip(chunks, results, unclear, judged)
        ],
        "metrics": None,
        "gold_usable_text_coverage": None,
        "transcript_coverage": None,
    }

    if is_scored(item) and usable_text_intervals is not None:
        frac = covered_fraction(gold, usable_text_intervals)
        record["gold_usable_text_coverage"] = _r(frac)
        record["transcript_coverage"] = "none" if frac == 0 else ("full" if frac >= 0.999 else "partial")

    if is_scored(item):
        n = len(gold)
        usable = [m if not u else [] for m, u in zip(matches, unclear)]
        gold_videos = {g.video_id for g in gold}
        record["metrics"] = {
            "recall@5":              _r(recall_at_k(matches, n, 5)),
            "recall@10":             _r(recall_at_k(matches, n, 10)),
            "mrr@10":                _r(reciprocal_rank(matches, 10)),
            "ndcg@10":               _r(ndcg_at_k(matches, n, 10)),
            "start_error_s":         _r(first_relevant_start_error(chunks, matches, gold, k_max)),
            # Diagnostics (not headline metrics):
            "video_hit@1":           float(bool(chunks) and chunks[0].video_id in gold_videos),
            "recall@5_usable_text":  _r(recall_at_k(usable, n, 5)),
        }
    return record


# ── Aggregation ───────────────────────────────────────────────────────────────

RANKING_KEYS = ["recall@5", "recall@10", "mrr@10", "ndcg@10", "video_hit@1", "recall@5_usable_text"]
# transcript_coverage: share of the gold time the production index has usable text for
# ("none" / "partial" / "full") — the slice a transcription change should move.
BREAKDOWNS = ["language", "phrasing", "difficulty", "followup", "status", "transcript_coverage"]


def summarize_ranking(records: Sequence[dict]) -> dict:
    scored = [r for r in records if r["metrics"] is not None]
    out: dict = {"n": len(scored)}
    for key in RANKING_KEYS:
        out[key] = _r(mean([r["metrics"][key] for r in scored]))
    errs = [r["metrics"]["start_error_s"] for r in scored if r["metrics"]["start_error_s"] is not None]
    out["start_error_s"] = {
        "n": len(errs), "mean": _r(mean(errs)),
        "median": _r(percentile(errs, 50)), "p90": _r(percentile(errs, 90)),
    }
    return out


def summarize_abstention(records: Sequence[dict]) -> dict:
    """Gate-level abstention at the CURRENT threshold. Nothing is calibrated here."""
    in_scope = [r for r in records if r["category"] == "in_scope"]
    far = [r for r in records if r["category"] == "out_of_scope"]
    near = [r for r in records if r["category"] == "near_miss_oos"]
    oos = far + near

    def score(r: dict) -> float:
        return r["top1_similarity"] if r["top1_similarity"] is not None else float("-inf")

    def answered(rs: Sequence[dict]) -> int:
        return sum(r["gate"] == "answered" for r in rs)

    return {
        "n_in_scope":                len(in_scope),
        "n_out_of_scope":            len(far),
        "n_near_miss_oos":           len(near),
        "false_answer_rate":         _r(rate(answered(oos), len(oos))),
        "false_answer_rate_far":     _r(rate(answered(far), len(far))),
        "false_answer_rate_near":    _r(rate(answered(near), len(near))),
        "false_refusal_rate":        _r(rate(len(in_scope) - answered(in_scope), len(in_scope))),
        "auroc_top1_similarity":     _r(auroc([score(r) for r in in_scope], [score(r) for r in oos])),
        "auroc_top1_similarity_near": _r(auroc([score(r) for r in in_scope], [score(r) for r in near])),
    }


def summarize_exact_agreement(records: Sequence[dict]) -> dict:
    """How often the system's top-k set equals the exact nearest-neighbour top-k set."""
    ov = [r["exact_topk_overlap"] for r in records if r.get("exact_topk_overlap") is not None]
    return {
        "n":                      len(ov),
        "mean_overlap":           _r(mean(ov)),
        "fraction_identical_set": _r(rate(sum(o == 1.0 for o in ov), len(ov))),
    }


def unclear_rate_at_k(records: Sequence[dict], k: int) -> Optional[float]:
    """Mean share of the top-k slots filled by chunks whose indexed text is [unclear audio]."""
    shares = [sum(x["unclear_text"] for x in r["retrieved"][:k]) / k for r in records if r["retrieved"]]
    return _r(mean(shares))


def summarize(records: Sequence[dict]) -> dict:
    out = {"ranking": summarize_ranking(records), "abstention": summarize_abstention(records),
           "exact_search_agreement": summarize_exact_agreement(records),
           "retrieved_unclear_rate@5": unclear_rate_at_k(records, 5), "by": {}}
    for field in BREAKDOWNS:
        groups: dict[str, list] = {}
        for r in records:
            if r["metrics"] is not None:
                groups.setdefault(str(r[field]), []).append(r)
        out["by"][field] = {g: summarize_ranking(rs) for g, rs in sorted(groups.items())}
    return out


def summarize_all(records: Sequence[dict], items_by_id: dict) -> dict:
    summary = {"all": summarize(records)}
    for split in ("dev", "test"):
        summary[split] = summarize([r for r in records if r["split"] == split])
    per_video: dict[str, list] = {}
    for r in records:
        if r["metrics"] is not None:
            per_video.setdefault(items_by_id[r["id"]]["gold"][0]["video_id"], []).append(r)
    summary["all"]["by"]["video_of_first_gold_span"] = {v: summarize_ranking(rs) for v, rs in sorted(per_video.items())}
    return summary


# ── System binding and provenance ─────────────────────────────────────────────

def _system_functions(top_k: int, threshold: float, collection: str):
    """Bind the real system exactly as pipeline.ask() calls it."""
    from pipeline import _rewrite_query_for_retrieval, validate_query
    from retrieval.retriever import retrieve

    def retrieve_fn(q: str):
        return retrieve(query=q, top_k=top_k, similarity_threshold=threshold,
                        video_id_filter=None, collection_name=collection)
    return retrieve_fn, _rewrite_query_for_retrieval, validate_query


def _exact_topk_function(top_k: int, collection: str) -> Callable[[str], list[str]]:
    """Exact cosine top-k over the collection's stored embeddings (diagnostic only)."""
    import numpy as np
    from ingestion.indexer import get_chroma_client
    from retrieval.retriever import encode_query

    got = get_chroma_client().get_collection(collection).get(include=["embeddings"])
    ids = got["ids"]
    emb = np.asarray(got["embeddings"], dtype=np.float64)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    def exact_fn(q: str) -> list[str]:
        v = np.asarray(encode_query(q), dtype=np.float64)
        sims = emb @ (v / np.linalg.norm(v))
        return [ids[i] for i in np.argsort(-sims, kind="stable")[:top_k]]
    return exact_fn


def _index_fingerprint(collection_name: str) -> dict:
    import numpy as np
    from ingestion.indexer import get_chroma_client
    col = get_chroma_client().get_collection(collection_name)
    got = col.get(include=["metadatas", "documents", "embeddings"])
    order = sorted(range(len(got["ids"])), key=lambda i: got["ids"][i])
    h = hashlib.sha256()
    for i in order:
        m = got["metadatas"][i]
        h.update(json.dumps([got["ids"][i], m.get("video_id"), m.get("start_time"), m.get("end_time"),
                             got["documents"][i]], ensure_ascii=False).encode("utf-8"))
        h.update(np.asarray(got["embeddings"][i], dtype=np.float32).tobytes())
    return {
        "collection":      collection_name,
        "count":           col.count(),
        "distance_metric": (col.metadata or {}).get("hnsw:space"),
        "unclear_chunks":  sum(UNCLEAR_MARKER in (d or "") for d in got["documents"]),
        "sha256":          h.hexdigest(),
        "fingerprint_of":  "sorted ids + video_id + start/end + document text + float32 embeddings",
    }


def _usable_text_intervals(collection_name: str) -> dict[str, list[tuple[float, float]]]:
    """video_id → [(start, end)] of indexed chunks whose document is not [unclear audio]."""
    from ingestion.indexer import get_chroma_client
    got = get_chroma_client().get_collection(collection_name).get(include=["metadatas", "documents"])
    out: dict[str, list[tuple[float, float]]] = {}
    for m, d in zip(got["metadatas"], got["documents"]):
        if UNCLEAR_MARKER not in (d or ""):
            out.setdefault(m["video_id"], []).append((float(m["start_time"]), float(m["end_time"])))
    return out


def _git_state() -> dict:
    def git(*args: str) -> Optional[str]:
        try:
            return subprocess.run(["git", *args], cwd=_ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


def _versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version
    out = {"python": platform.python_version()}
    for pkg in ("chromadb", "FlagEmbedding", "torch", "transformers", "numpy", "openai"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def _system_config() -> dict:
    import config.settings as s
    from pipeline import _get_openai_client_if_available
    db_path = Path(s.CHROMA_DB_PATH)
    db_path = (db_path if db_path.is_absolute() else _ROOT / db_path).resolve()
    return {
        # CHROMA_DB_PATH / ACTIVE_COLLECTION can be overridden by environment variables
        # (config/settings.py), which is how experimental indexes are evaluated with the
        # unchanged production retrieval code.
        "chroma_db_path":        str(db_path.relative_to(_ROOT)) if db_path.is_relative_to(_ROOT) else str(db_path),
        "active_collection":     s.ACTIVE_COLLECTION,
        "embedding_model":       s.BGE_MODEL,
        "query_encoding":        "BGE-M3 dense vector, no instruction prefix, max_length=512, fp32",
        "passage_embeddings":    "BGE-M3 dense vectors of the raw transcript chunk text (v1 embeddings reused by indexer_v2)",
        "retrieval_method":      "dense nearest-neighbour search, ChromaDB HNSW",
        "similarity":            "1 - ChromaDB cosine distance",
        "retrieval_top_k":       s.RETRIEVAL_TOP_K,
        "max_llm_evidence":      s.MAX_LLM_EVIDENCE,
        "similarity_threshold":  s.SIMILARITY_THRESHOLD,
        "refusal_gate":          "no LLM call (not_found) when no retrieved chunk has similarity >= threshold",
        "followup_rewrite":      f"{s.OPENAI_MODEL} at temperature 0, only when the query matches pipeline._FOLLOWUP_RE",
        "rewrite_llm_available": _get_openai_client_if_available() is not None,
        "chunking":              "5 Whisper segments/chunk, 1-segment overlap, hard break on gaps > 5 s",
        "whisper_model":         s.WHISPER_MODEL,
    }


def _rel_hashes(paths: Sequence[str]) -> dict:
    return {p: sha256_file(_ROOT / p) for p in paths}


# ── Main ──────────────────────────────────────────────────────────────────────

def print_summary(summary: dict, threshold: float) -> None:
    print()
    print(f"{'split':<6} {'n':>4} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'nDCG10':>7} "
          f"{'vid@1':>6} {'R@5use':>7} {'ts-med':>7}")
    for split in ("all", "dev", "test"):
        rk = summary[split]["ranking"]
        med = rk["start_error_s"]["median"]
        print(f"{split:<6} {rk['n']:>4} {rk['recall@5']:>6.3f} {rk['recall@10']:>6.3f} {rk['mrr@10']:>6.3f} "
              f"{rk['ndcg@10']:>7.3f} {rk['video_hit@1']:>6.3f} {rk['recall@5_usable_text']:>7.3f} "
              f"{(f'{med:.1f}s' if med is not None else '-'):>7}")
    ab = summary["all"]["abstention"]
    print(f"\nAbstention gate at threshold {threshold} (not calibrated): "
          f"false-answer {ab['false_answer_rate']:.3f} (far {ab['false_answer_rate_far']:.3f}, "
          f"near {ab['false_answer_rate_near']:.3f}), false-refusal {ab['false_refusal_rate']:.3f}, "
          f"AUROC {ab['auroc_top1_similarity']:.3f}")
    ex = summary["all"]["exact_search_agreement"]
    print(f"Top-k identical to exact search for {ex['fraction_identical_set']:.3f} of {ex['n']} queries "
          f"(mean overlap {ex['mean_overlap']:.3f})")
    print(f"Share of top-5 slots holding [unclear audio] chunks: {summary['all']['retrieved_unclear_rate@5']:.3f}")
    cov = summary["all"]["by"]["transcript_coverage"]
    print("By usable-transcript coverage of the gold span: " + ", ".join(
        f"{k} n={v['n']} R@10={v['recall@10']:.3f} MRR={v['mrr@10']:.3f}" for k, v in cov.items()))


def main() -> int:
    p = argparse.ArgumentParser(description="Run the Stage 1 benchmark against the current system.")
    p.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK_PATH))
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    p.add_argument("--label", default="baseline", help="result name → eval/results/<label>.json")
    p.add_argument("--min-overlap-s", type=float, default=0.0)
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)

    bench_path = Path(args.benchmark).resolve()
    bench = load_json(bench_path)
    manifest = load_json(Path(args.manifest))
    errors = validate_benchmark(bench, manifest)
    if errors:
        for e in errors:
            print(f"  ERROR {e}")
        print(f"Benchmark is invalid ({len(errors)} errors); refusing to score it.")
        return 1

    import config.settings as s
    started = datetime.now(timezone.utc)
    t0 = time.time()
    retrieve_fn, rewrite_fn, validate_fn = _system_functions(
        s.RETRIEVAL_TOP_K, s.SIMILARITY_THRESHOLD, s.ACTIVE_COLLECTION)
    exact_fn = _exact_topk_function(s.RETRIEVAL_TOP_K, s.ACTIVE_COLLECTION)
    usable = _usable_text_intervals(s.ACTIVE_COLLECTION)

    records = []
    for n, item in enumerate(bench["queries"], start=1):
        records.append(evaluate_item(item, retrieve_fn, rewrite_fn, validate_fn,
                                     k_max=s.RETRIEVAL_TOP_K, min_overlap_s=args.min_overlap_s,
                                     exact_topk_fn=exact_fn, usable_text_intervals=usable))
        if n % 25 == 0:
            print(f"  {n}/{len(bench['queries'])} queries")

    items_by_id = {q["id"]: q for q in bench["queries"]}
    result = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "label": args.label,
        "benchmark": {
            "version":  bench["benchmark_version"],
            "path":     str(bench_path.relative_to(_ROOT)),
            "sha256":   sha256_file(bench_path),
            "n_queries": len(bench["queries"]),
            "n_scored": sum(is_scored(q) for q in bench["queries"]),
            "annotation_status_counts": dict(sorted(
                Counter(q["annotation"]["status"] for q in bench["queries"]).items())),
        },
        "system": {
            "config": _system_config(),
            "index":  _index_fingerprint(s.ACTIVE_COLLECTION),
            "code_sha256": _rel_hashes(SYSTEM_CODE_FILES),
        },
        "evaluation": {
            "k_max":         s.RETRIEVAL_TOP_K,
            "min_overlap_s": args.min_overlap_s,
            "code_sha256":   _rel_hashes(EVAL_CODE_FILES),
            "git":           _git_state(),
            "versions":      _versions(),
        },
        "summary":   summarize_all(records, items_by_id),
        "per_query": records,
        # Everything above is deterministic for a given benchmark, index, code and
        # rewrite output; run_info is not.
        "run_info": {
            "started_at": started.isoformat(timespec="seconds"),
            "duration_s": round(time.time() - t0, 1),
            "platform":   platform.platform(),
        },
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{args.label}.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print_summary(result["summary"], s.SIMILARITY_THRESHOLD)
    print(f"\nWrote {out.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
