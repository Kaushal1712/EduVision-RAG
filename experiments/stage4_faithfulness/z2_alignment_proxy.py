"""
experiments/stage4_faithfulness/z2_alignment_proxy.py
──────────────────────────────────────────────────────
Stage 4, Z2: offline claim–citation alignment PROXY on existing DEV answers. No API calls.

  venv/bin/python experiments/stage4_faithfulness/z2_alignment_proxy.py [--out PATH]
Default output: experiments/stage4_faithfulness/z2_alignment_proxy.json

For every factual answer sentence (Z1's segmentation and citation-to-sentence rule), the sentence is
scored against every evidence chunk the model was given:
  dense    cosine of BGE-M3 dense vectors (production query model, retrieval.retriever._get_model:
           BAAI/bge-m3 fp32, cached; sentences and chunks encoded the same way)
  lexical  share of the sentence's content tokens that occur in the chunk (lower-cased [a-z0-9]+,
           a fixed English stop-word list removed)
For cited sentences it records the cited chunk's score, the best provided chunk's score, the cited
chunk's rank among the provided chunks and the best-minus-cited margin, for both signals.

This is a semantic-similarity PROXY. A high score does not mean a sentence is supported, and a low
score does not mean it is unsupported. It is meant to rank cases for human review, not to replace it.

Controls and diagnostics:
  * shuffled control: each cited sentence is also scored against a chunk from another query's
    evidence (a known-misaligned citation), which shows whether the proxy separates the two at all.
  * uncited sentences: best provided-chunk score only; not counted as citation failures.
  * follow-ups: the prior assistant turn is scored separately (prior_context_dense), since rule 7
    allows answers to build on it.
  * judge relation: each judge `unsupported_claims` string is mapped to the answer sentence with the
    highest token Jaccard (>= JACCARD_MIN); unmatched claims are counted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(HERE))

from answer_eval import _CITATION_RE, RESULTS_DIR  # noqa: E402
from bench_data import DEFAULT_BENCHMARK_PATH, DEFAULT_MANIFEST_PATH, load_json  # noqa: E402
from bench_metrics import auroc, percentile  # noqa: E402
from z1_citation_audit import (  # noqa: E402
    RANGE_RE, RUNS, SENTENCE_MIN_WORDS, TOLERANCE_S, display_seconds, extract_citations, sentence_coverage,
)

MAX_LENGTH = 512
BATCH_SIZE = 16
JACCARD_MIN = 0.5
KNOWN_CASES = ("EVB-018", "EVB-054", "EVB-163")
# Reporting bins (descriptive; not decision thresholds).
MARGIN_BINS = (0.0, 0.02, 0.05, 0.10)
DENSE_BEST_BINS = (0.45, 0.55, 0.65)
LEX_BEST_BINS = (0.25, 0.50, 0.75)
STOPWORDS = frozenset("""
a an the and or but if then so of to in on at by for with from as is are was were be been being it its
this that these those you your we our they their he she his her i me my can could will would should may
might do does did done have has had not no yes also just very more most such into about which what when
where how why who whom there here than too only any all some each other both same own while through
""".split())
_PH = re.compile("⁣(\\d+)⁤")
_TOKEN = re.compile(r"[a-z0-9]+")


# ── Pure functions (unit-tested) ──────────────────────────────────────────────

def segment(answer: str) -> tuple[list[dict], dict]:
    """
    Split an answer into sentences with the indices of the citations attached to each, using Z1's rule:
    a citation at the start of a part belongs to the previous sentence; parts under SENTENCE_MIN_WORDS
    words are not factual sentences. Returns (sentences, ambiguity counts). Sentence dicts:
    {"text", "cits": [citation indices in extract_citations order], "factual", "list_item"}.
    """
    out, pos, n = [], 0, 0
    for m in _CITATION_RE.finditer(answer or ""):
        rng = RANGE_RE.match(answer, m.end())
        end = rng.end() if rng else m.end()
        if end < len(answer) and answer[end] == "]":
            end += 1
        start = m.start()
        out.append(answer[pos:start] + f"⁣{n}⁤")
        pos, n = end, n + 1
    masked = "".join(out) + (answer or "")[pos:]
    amb = {"leading_citation_reassigned": 0, "leading_citation_without_previous": 0,
           "citation_in_short_fragment_dropped": 0, "mid_sentence_citation": 0,
           "list_item_sentences": 0}
    sentences: list[dict] = []
    for raw in (p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", masked)):
        if not raw:
            continue
        lead = []
        while True:
            m = _PH.match(raw)
            if not m:
                break
            lead.append(int(m.group(1)))
            raw = raw[m.end():].lstrip(" ,;")
        if lead:
            if sentences:
                sentences[-1]["cits"].extend(lead)
                amb["leading_citation_reassigned"] += 1
            else:
                amb["leading_citation_without_previous"] += 1
        inner = [int(x) for x in _PH.findall(raw)]
        text = re.sub(r"\s+", " ", _PH.sub(" ", raw)).strip(" .-*•")
        words = len(text.split())
        if words < SENTENCE_MIN_WORDS:
            amb["citation_in_short_fragment_dropped"] += len(inner)
            continue
        tail = raw[raw.rfind("⁤") + 1:] if inner else ""
        if inner and len(tail.split()) >= SENTENCE_MIN_WORDS:
            amb["mid_sentence_citation"] += 1
        is_list = bool(re.match(r"^(\d+[.)]|[-*•])\s", raw))
        amb["list_item_sentences"] += is_list
        sentences.append({"text": text, "cits": inner + ([] if sentences or not lead else lead),
                          "factual": True, "list_item": is_list})
    return sentences, amb


def resolve(cit: dict, evidence: Sequence[dict]) -> dict:
    """
    Evidence chunks (indices into `evidence`) a citation points to. With overlapping chunks several
    may cover the cited time: a chunk whose displayed start equals the cited time wins; otherwise all
    covering chunks are kept and the citation is marked ambiguous. A range citation keeps every chunk
    overlapping [start, end].
    """
    vid, t, t_end = cit["video_id"], cit["time_s"], cit["end_s"]
    if vid is None:
        return {"chunks": [], "overlap_ambiguous": False}
    lo, hi = (t, t_end) if t_end is not None else (t, t)
    cover = [i for i, c in enumerate(evidence) if c["video_id"] == vid
             and c["start_time"] - TOLERANCE_S <= hi and lo <= c["end_time"] + TOLERANCE_S]
    if t_end is None and len(cover) > 1:
        exact = [i for i in cover if abs(display_seconds(evidence[i]["start_time"],
                                                         evidence[i]["end_time"] - evidence[i]["start_time"]) - t) < 1e-6]
        if len(exact) == 1:
            return {"chunks": exact, "overlap_ambiguous": False}
        return {"chunks": cover, "overlap_ambiguous": True}
    return {"chunks": cover, "overlap_ambiguous": False}


def content_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS}


def lexical_overlap(sentence: str, chunk: str) -> Optional[float]:
    """Share of the sentence's content tokens present in the chunk (None if the sentence has none)."""
    s = content_tokens(sentence)
    return len(s & content_tokens(chunk)) / len(s) if s else None


def alignment(scores: Sequence[float], cited: Sequence[int]) -> dict:
    """Cited-chunk score (max over cited chunks), best score, cited rank (1 = best), margin."""
    best = max(scores)
    c = max(scores[i] for i in cited)
    rank = 1 + sum(s > c for s in scores)
    return {"cited": c, "best": best, "rank": rank, "margin": best - c}


def bin_counts(values: Sequence[float], edges: Sequence[float], exact_zero: bool = False) -> dict:
    labels, counts = [], []
    if exact_zero:
        labels.append("0")
        counts.append(sum(v <= 1e-12 for v in values))
        values = [v for v in values if v > 1e-12]
    lo = None
    for e in edges[1:] if exact_zero else edges:
        labels.append(f"<{e}" if lo is None else f"[{lo},{e})")
        counts.append(sum((lo is None or v >= lo) and v < e for v in values))
        lo = e
    labels.append(f">={lo}")
    counts.append(sum(v >= lo for v in values))
    return dict(zip(labels, counts))


def jaccard(a: str, b: str) -> float:
    x, y = content_tokens(a), content_tokens(b)
    return len(x & y) / len(x | y) if x | y else 0.0


def map_claims(claims: Sequence[str], sentences: Sequence[str]) -> tuple[list[Optional[int]], int]:
    """Index of the answer sentence each judge claim refers to (Jaccard >= JACCARD_MIN), and #unmatched."""
    out = []
    for c in claims:
        best = max(range(len(sentences)), key=lambda i: jaccard(c, sentences[i]), default=None)
        out.append(best if best is not None and jaccard(c, sentences[best]) >= JACCARD_MIN else None)
    return out, sum(x is None for x in out)


def quantiles(values: Sequence[float]) -> Optional[dict]:
    if not values:
        return None
    return {f"p{q}": round(percentile(values, q), 4) for q in (10, 25, 50, 75, 90)} | {"n": len(values)}


# ── Scoring ───────────────────────────────────────────────────────────────────

def make_dense(model) -> Callable[[Sequence[str]], dict[str, list[float]]]:
    import numpy as np
    cache: dict[str, np.ndarray] = {}

    def encode(texts: Sequence[str]) -> None:
        todo = sorted({t for t in texts if t not in cache})
        if todo:
            vecs = model.encode(todo, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, return_dense=True,
                                return_sparse=False, return_colbert_vecs=False)["dense_vecs"]
            for t, v in zip(todo, vecs):
                v = np.asarray(v, dtype=np.float64)
                cache[t] = v / np.linalg.norm(v)

    def cos(a: str, b: str) -> float:
        return float(cache[a] @ cache[b])
    encode.cos = cos
    return encode


def analyse_run(records: Sequence[dict], bench: dict, manifest: dict, dense) -> tuple[list[dict], dict, int]:
    answered = [r for r in records if not r["refused"]]
    texts = []
    prepared = []
    amb_total: dict[str, int] = {}
    for r in answered:
        cits = extract_citations(r["answer"], manifest)
        sents, amb = segment(r["answer"])
        for k, v in amb.items():
            amb_total[k] = amb_total.get(k, 0) + v
        z1 = sentence_coverage(r["answer"])
        if z1 != {"sentences": len(sents), "cited": sum(bool(s["cits"]) for s in sents)}:
            raise SystemExit(f"{r['id']}: segmentation disagrees with Z1 {z1}")
        prior = [h["content"] for h in bench[r["id"]]["history"] if h["role"] == "assistant"]
        texts += [s["text"] for s in sents] + [c["text_en"] for c in r["sources_used"]] + prior
        prepared.append((r, cits, sents, prior))
    dense(texts)

    # Shuffled control: a chunk from the next record (cyclic) with a different query and video.
    pool = [(p[0]["id"], c) for p in prepared for c in p[0]["sources_used"]]
    rows, unmatched_total = [], 0
    for n, (r, cits, sents, prior) in enumerate(prepared):
        ev = r["sources_used"]
        item = bench[r["id"]]
        j = r.get("judge") or {}
        claim_map, unmatched = map_claims(j.get("unsupported_claims") or [], [s["text"] for s in sents])
        flagged = {i for i in claim_map if i is not None}
        unmatched_total += unmatched
        for k, s in enumerate(sents):
            d = [dense.cos(s["text"], c["text_en"]) for c in ev]
            lx = [lexical_overlap(s["text"], c["text_en"]) for c in ev]
            row = {"run_record": f"{r['id']}#{r['repeat']}", "id": r["id"], "repeat": r["repeat"],
                   "sentence_index": k, "text": s["text"], "language": r["language"], "followup": r["followup"],
                   "phrasing": item["phrasing"], "category": r["category"],
                   "judge_hallucination": j.get("hallucination"), "judge_flagged_sentence": k in flagged,
                   "list_item": s["list_item"], "n_evidence": len(ev),
                   "dense_best": max(d) if d else None, "lex_best": max((x for x in lx if x is not None), default=None),
                   "prior_context_dense": max((dense.cos(s["text"], p) for p in prior), default=None)}
            if s["cits"]:
                res = [resolve(cits[i], ev) for i in s["cits"]]
                chunks = sorted({c for x in res for c in x["chunks"]})
                row["citations"] = [cits[i]["raw"] for i in s["cits"]]
                row["cited_chunk_ids"] = [ev[c]["chunk_id"] for c in chunks]
                row["overlap_ambiguous"] = any(x["overlap_ambiguous"] for x in res)
                row["distinct_cited_chunks"] = len(chunks)
                if chunks:
                    row["dense"] = alignment(d, chunks)
                    if all(x is not None for x in lx):
                        row["lexical"] = alignment(lx, chunks)
                    vids = {ev[c]["video_id"] for c in chunks}
                    for off in range(1, len(pool)):
                        qid, other = pool[(n * 7 + k + off) % len(pool)]
                        if qid != r["id"] and other["video_id"] not in vids:
                            row["shuffled_dense"] = dense.cos(s["text"], other["text_en"])   # pool texts were encoded
                            row["shuffled_lexical"] = lexical_overlap(s["text"], other["text_en"])
                            break
            rows.append(row)
    return rows, amb_total, unmatched_total


def summarize(rows: Sequence[dict]) -> dict:
    cited = [r for r in rows if r.get("dense")]
    uncited = [r for r in rows if "citations" not in r]
    lex = [r for r in cited if r.get("lexical")]
    dm = [r["dense"]["margin"] for r in cited]
    both_first = sum(r["dense"]["rank"] == 1 and r.get("lexical", {}).get("rank") == 1 for r in cited)
    sh = [r for r in cited if r.get("shuffled_dense") is not None]
    return {
        "sentences": len(rows), "cited_sentences": len(cited), "uncited_sentences": len(uncited),
        "dense_rank_of_cited": {"1": sum(r["dense"]["rank"] == 1 for r in cited),
                                "2": sum(r["dense"]["rank"] == 2 for r in cited),
                                ">=3": sum(r["dense"]["rank"] >= 3 for r in cited)},
        "lexical_rank_of_cited": {"1": sum(r["lexical"]["rank"] == 1 for r in lex),
                                  "2": sum(r["lexical"]["rank"] == 2 for r in lex),
                                  ">=3": sum(r["lexical"]["rank"] >= 3 for r in lex)},
        "cited_first_by_both_signals": both_first,
        "dense_margin_bins": bin_counts(dm, MARGIN_BINS, exact_zero=True),
        "dense_cited_quantiles": quantiles([r["dense"]["cited"] for r in cited]),
        "dense_best_quantiles_cited_sentences": quantiles([r["dense"]["best"] for r in cited]),
        "dense_margin_quantiles": quantiles(dm),
        "lexical_cited_quantiles": quantiles([r["lexical"]["cited"] for r in lex]),
        "lexical_margin_quantiles": quantiles([r["lexical"]["margin"] for r in lex]),
        "dense_best_bins_cited": bin_counts([r["dense"]["best"] for r in cited], DENSE_BEST_BINS),
        "lexical_best_bins_cited": bin_counts([r["lexical"]["best"] for r in lex], LEX_BEST_BINS),
        "uncited_dense_best_quantiles": quantiles([r["dense_best"] for r in uncited if r["dense_best"] is not None]),
        "uncited_lexical_best_quantiles": quantiles([r["lex_best"] for r in uncited if r["lex_best"] is not None]),
        "shuffled_control": {
            "n": len(sh),
            "dense_shuffled_quantiles": quantiles([r["shuffled_dense"] for r in sh]),
            "auroc_dense_cited_vs_shuffled": round(auroc([r["dense"]["cited"] for r in sh],
                                                         [r["shuffled_dense"] for r in sh]), 4) if sh else None,
            "auroc_lexical_cited_vs_shuffled": round(auroc([r["lexical"]["cited"] for r in sh if r.get("lexical")],
                                                           [r["shuffled_lexical"] for r in sh if r.get("lexical")]), 4)
            if sh else None,
        },
        "overlap_ambiguous_citations": sum(bool(r.get("overlap_ambiguous")) for r in cited),
        "sentences_citing_multiple_chunks": sum(r.get("distinct_cited_chunks", 0) > 1 for r in cited),
    }


def judge_relation(rows: Sequence[dict]) -> dict:
    """Sentence level: judge-flagged unsupported sentences vs others (in-scope answers)."""
    ins = [r for r in rows if r["category"] == "in_scope"]
    fl = [r for r in ins if r["judge_flagged_sentence"]]
    ok = [r for r in ins if not r["judge_flagged_sentence"]]
    d = lambda rs, f: [f(r) for r in rs if f(r) is not None]
    best = lambda r: r["dense_best"]
    cited = lambda r: r["dense"]["cited"] if r.get("dense") else None
    out = {"flagged_sentences": len(fl), "other_sentences": len(ok),
           "flagged_cited": sum("citations" in r for r in fl), "other_cited": sum("citations" in r for r in ok),
           "dense_best_quantiles": {"flagged": quantiles(d(fl, best)), "other": quantiles(d(ok, best))},
           "auroc_dense_best_other_vs_flagged": round(auroc(d(ok, best), d(fl, best)), 4) if fl and ok else None,
           "auroc_dense_cited_other_vs_flagged": round(auroc(d(ok, cited), d(fl, cited)), 4)
           if d(fl, cited) and d(ok, cited) else None,
           "lex_best_quantiles": {"flagged": quantiles(d(fl, lambda r: r["lex_best"])),
                                  "other": quantiles(d(ok, lambda r: r["lex_best"]))}}
    # Record level: hallucination-flagged answers vs others, worst cited margin per answer.
    by_rec: dict[str, list[dict]] = {}
    for r in ins:
        by_rec.setdefault(r["run_record"], []).append(r)
    worst = {k: max((x["dense"]["margin"] for x in v if x.get("dense")), default=None) for k, v in by_rec.items()}
    hall = {k: v[0]["judge_hallucination"] for k, v in by_rec.items()}
    out["record_worst_dense_margin_quantiles"] = {
        "hallucination": quantiles([w for k, w in worst.items() if w is not None and hall[k] is True]),
        "no_hallucination": quantiles([w for k, w in worst.items() if w is not None and hall[k] is False])}
    return out


SLICES = {
    "all": lambda r: True, "en": lambda r: r["language"] == "en", "hinglish": lambda r: r["language"] == "hinglish",
    "followup": lambda r: r["followup"], "exact_term": lambda r: r["phrasing"] == "exact_term",
    "rare_term": lambda r: r["phrasing"] == "rare_term", "paraphrase": lambda r: r["phrasing"] == "paraphrase",
    "judge_hallucination": lambda r: r["judge_hallucination"] is True,
    "judge_no_hallucination": lambda r: r["judge_hallucination"] is False,
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(HERE / "z2_alignment_proxy.json"))
    args = p.parse_args()
    bench = {q["id"]: q for q in load_json(DEFAULT_BENCHMARK_PATH)["queries"]}
    manifest = load_json(DEFAULT_MANIFEST_PATH)
    from retrieval.retriever import _get_model
    dense = make_dense(_get_model())
    out: dict = {"stage": "4", "experiment": "Z2 claim-citation alignment proxy (offline, no API)",
                 "proxy_note": "semantic/lexical similarity proxy; not a support judgement and not ground truth",
                 "model": "BAAI/bge-m3 dense via retrieval.retriever._get_model (fp32), max_length 512",
                 "inputs": {n: f"eval/results/answers_{l}.json" for n, l in RUNS.items()},
                 "runs": {}}
    for name, label in RUNS.items():
        data = load_json(RESULTS_DIR / f"answers_{label}.json")
        if data["split"] != "dev" or any(r["split"] != "dev" for r in data["records"]):
            raise SystemExit(f"{label}: not DEV-only")
        rows, amb, unmatched = analyse_run(data["records"], bench, manifest, dense)
        out["runs"][name] = {
            "segmentation_ambiguity": amb,
            "slices": {s: summarize([r for r in rows if f(r)]) for s, f in SLICES.items()},
            "judge_relation": judge_relation(rows),
            "judge_unsupported_claims_unmatched": unmatched,
            "known_cases": [r for r in rows if r["id"] in KNOWN_CASES],
            "sentences": rows,
        }
    scores = json.dumps({n: [(r["run_record"], r["sentence_index"], r.get("dense"), r.get("lexical"))
                             for r in v["sentences"]] for n, v in out["runs"].items()}, sort_keys=True)
    out["scores_sha256"] = hashlib.sha256(scores.encode()).hexdigest()
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    for n, v in out["runs"].items():
        s = v["slices"]["all"]
        print(n, json.dumps({k: s[k] for k in ("sentences", "cited_sentences", "dense_rank_of_cited",
                                                "lexical_rank_of_cited", "dense_margin_bins", "shuffled_control")}))
    print("scores_sha256", out["scores_sha256"], "→", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
