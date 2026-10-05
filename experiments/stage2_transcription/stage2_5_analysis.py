"""
experiments/stage2_transcription/stage2_5_analysis.py
──────────────────────────────────────────────────────
Stage 2.5 analyses, recomputed from the committed result files (no models, no network):

  1. chunking/boundary variants of the candidate, DEV only (selection evidence);
  2. timestamp error split by whether the gold span starts exactly at a production chunk
     boundary (Stage 1 drafted gold from production chunk times; Stage 1.5 kept them);
  3. top-1 similarity and refusal-gate decisions by query language.

  venv/bin/python experiments/stage2_transcription/stage2_5_analysis.py
Writes eval/results/stage2_5_analysis.json.
"""

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

from compare_results import compare_group  # noqa: E402

RES = ROOT / "eval" / "results"
CONTROL = "stage1_5_baseline_bench-v1.1"
CANDIDATE = "stage2_lv2g_translate"
VARIANTS = {"w5o1 gap5 (Stage 2 candidate, production chunking)": CANDIDATE,
            "w3o1": "stage2_5_lv2g_translate_w3o1", "w4o1": "stage2_5_lv2g_translate_w4o1",
            "w5o2": "stage2_5_lv2g_translate_w5o2", "w5o1 gap3": "stage2_5_lv2g_translate_w5o1_gap3",
            "w5o1 gap2": "stage2_5_lv2g_translate_w5o1_gap2"}


def load(name):
    return {r["id"]: r for r in json.loads((RES / f"{name}.json").read_text())["per_query"]}


def main() -> None:
    bench = {q["id"]: q for q in json.loads((ROOT / "eval/benchmark/eduvision_bench_v1.json").read_text())["queries"]}
    ctl, cand = load(CONTROL), load(CANDIDATE)
    out: dict = {}

    # 1. variants, DEV only
    out["chunking_variants_dev"] = {}
    for label, name in VARIANTS.items():
        k = load(name)
        g = compare_group([(ctl[i], k[i]) for i in ctl if ctl[i]["metrics"] and ctl[i]["split"] == "dev"])
        out["chunking_variants_dev"][label] = {
            "recall@5": g["recall@5"]["candidate"], "recall@10": g["recall@10"]["candidate"],
            "mrr@10": g["mrr@10"]["candidate"], "ndcg@10": g["ndcg@10"]["candidate"],
            "start_error_s": g["start_error_s"]["candidate_all_hits"],
            "recovered_vs_control": g["outcomes_hit@10"]["recovered"]["n"], "lost_vs_control": g["outcomes_hit@10"]["lost"]["n"]}

    # 2. timestamp error vs gold alignment with production chunk starts
    prod_starts = {}
    for f in (ROOT / "data/processed/chunks").glob("*_chunks.json"):
        d = json.loads(f.read_text())
        prod_starts[d["video_id"]] = [c["start_time"] for c in d["chunks"]]
    aligned = lambda g: any(abs(s - g["start_time"]) < 0.05 for s in prod_starts[g["video_id"]])
    spans = [g for q in bench.values() if q["category"] == "in_scope" for g in q["gold"]]
    def first(res, i):
        for x in res[i]["retrieved"][:10]:
            if x["matched_gold"]:
                g = bench[i]["gold"][x["matched_gold"][0]]
                return x["start_time"] - g["start_time"], aligned(g)
        return None
    out["timestamp_alignment"] = {"gold_spans_starting_at_a_production_chunk_start": sum(map(aligned, spans)),
                                  "gold_spans_total": len(spans), "by_split": {}}
    for split in ("dev", "test", "all"):
        block = {}
        for subset, want in (("aligned", True), ("not_aligned", False)):
            for sysname, res in (("control", ctl), ("candidate", cand)):
                e = [first(res, i) for i in ctl if ctl[i]["metrics"] and (split == "all" or ctl[i]["split"] == split)]
                e = [d for d, a in (x for x in e if x) if a == want]
                block[f"{subset}/{sysname}"] = {"n": len(e), "abs_median_s": round(st.median(abs(d) for d in e), 2) if e else None,
                                                "starts_before_answer": round(sum(d < 0 for d in e) / len(e), 3) if e else None}
        out["timestamp_alignment"]["by_split"][split] = block

    # 3. similarity / gate by language
    out["similarity_by_language"] = {}
    for lang in ("en", "hinglish"):
        for cat in ("in_scope", "out_of_scope_or_near_miss"):
            rows = [(r, cand[r["id"]]) for r in ctl.values()
                    if r["language"] == lang and ((r["category"] == "in_scope") == (cat == "in_scope"))]
            out["similarity_by_language"][f"{lang}/{cat}"] = {
                "n": len(rows),
                "top1_median_control": round(st.median(a["top1_similarity"] for a, _ in rows), 4),
                "top1_median_candidate": round(st.median(b["top1_similarity"] for _, b in rows), 4),
                "gate_refused_control": sum(a["gate"] == "refused" for a, _ in rows),
                "gate_refused_candidate": sum(b["gate"] == "refused" for _, b in rows)}

    (RES / "stage2_5_analysis.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
