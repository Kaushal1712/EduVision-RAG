"""
experiments/stage3_retrieval/answer_compare_stage3.py
──────────────────────────────────────────────────────
Stage 3: compare judged answer runs (eval/answer_eval.py output) — summaries, slices and
per-query changes. The headline metrics are answer_eval.summarize, unchanged.

  venv/bin/python experiments/stage3_retrieval/answer_compare_stage3.py \
      --reference s26_cand_t044_guard_dev --rerun s3_control_dev --candidate s3_colbert20_dev \
      --out eval/results/stage3_answer_quality_dev.json

Per-query outcome score, averaged over repeats:
  in-scope:     correct 2, partially_correct 1, incorrect / refused 0
  out-of-scope: refused 2, answered 0
A query is better / worse when its mean score differs. Comparing the control re-run with the
frozen control (identical configuration) gives the noise floor of these per-query counts:
GPT-4o-mini runs at temperature 0.2, so repeats are not identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from answer_eval import RESULTS_DIR, summarize  # noqa: E402
from bench_data import DEFAULT_BENCHMARK_PATH, load_json  # noqa: E402

CORRECTNESS_SCORE = {"correct": 2, "partially_correct": 1}
PHRASINGS = ("exact_term", "rare_term", "paraphrase")


def record_score(rec: dict) -> int:
    if rec["category"] != "in_scope":
        return 2 if rec["refused"] else 0
    if rec["refused"]:
        return 0
    return CORRECTNESS_SCORE.get((rec.get("judge") or {}).get("correctness"), 0)


def by_query(records: Sequence[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in sorted(records, key=lambda r: r["repeat"]):
        out.setdefault(r["id"], []).append(r)
    return out


def describe(recs: Sequence[dict]) -> dict:
    j = lambda r, k: (r.get("judge") or {}).get(k)
    return {
        "score": sum(record_score(r) for r in recs) / len(recs),
        "labels": ["refused" if r["refused"] else (j(r, "correctness") if r["category"] == "in_scope" else "answered")
                   for r in recs],
        "grounding": [j(r, "grounding") for r in recs],
        "hallucination": [j(r, "hallucination") for r in recs],
        "evidence_has_gold": [r.get("evidence_has_gold") for r in recs],
        "citation_hit": [r.get("citation_hit") for r in recs],
        "evidence_ids": [[s["chunk_id"] for s in r["sources_used"]] for r in recs],
    }


def per_query_changes(ref: Sequence[dict], cand: Sequence[dict]) -> dict:
    a, b = by_query(ref), by_query(cand)
    better, worse, same = [], [], []
    for qid in sorted(a):
        da, db = describe(a[qid]), describe(b[qid])
        row = {"id": qid, "language": a[qid][0]["language"], "followup": a[qid][0]["followup"],
               "category": a[qid][0]["category"], "reference": {k: v for k, v in da.items() if k != "evidence_ids"},
               "candidate": {k: v for k, v in db.items() if k != "evidence_ids"},
               "evidence_changed": da["evidence_ids"][0] != db["evidence_ids"][0]}
        (better if db["score"] > da["score"] else worse if db["score"] < da["score"] else same).append(row)
    flips = lambda key: {
        "gained": [qid for qid in sorted(a) if not any(describe(a[qid])[key]) and any(describe(b[qid])[key])],
        "lost": [qid for qid in sorted(a) if all(describe(a[qid])[key]) and not all(describe(b[qid])[key])
                 and a[qid][0]["category"] == "in_scope"],
    }
    return {"better": better, "worse": worse, "unchanged": len(same),
            "major_worse": [r for r in worse if r["reference"]["score"] - r["candidate"]["score"] >= 1],
            "major_better": [r for r in better if r["candidate"]["score"] - r["reference"]["score"] >= 1],
            "evidence_has_gold": flips("evidence_has_gold"), "citation_hit": flips("citation_hit")}


def repeat_consistency(records: Sequence[dict]) -> dict:
    """Do repeat 0 and repeat 1 agree (same outcome label / same evidence chunks)?"""
    q = by_query(records)
    pairs = [v for v in q.values() if len(v) == 2]
    d = [(describe([x]), describe([y])) for x, y in pairs]
    return {"n": len(pairs),
            "same_outcome_label": sum(x["labels"] == y["labels"] for x, y in d),
            "same_evidence_chunks": sum(x["evidence_ids"] == y["evidence_ids"] for x, y in d),
            "different_evidence_ids": sorted(p[0]["id"] for p, (x, y) in zip(pairs, d)
                                             if x["evidence_ids"] != y["evidence_ids"])}


def slices(records: Sequence[dict], phrasing: dict[str, str]) -> dict:
    out = {"all": summarize(records),
           "by_language": {lang: summarize([r for r in records if r["language"] == lang]) for lang in ("en", "hinglish")},
           "followup": summarize([r for r in records if r["followup"]]),
           "by_phrasing": {ph: summarize([r for r in records if phrasing.get(r["id"]) == ph]) for ph in PHRASINGS}}
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--rerun", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--out")
    args = p.parse_args()
    phrasing = {q["id"]: q["phrasing"] for q in load_json(DEFAULT_BENCHMARK_PATH)["queries"]}
    runs = {name: load_json(RESULTS_DIR / f"answers_{label}.json")
            for name, label in (("reference", args.reference), ("rerun", args.rerun), ("candidate", args.candidate))}
    for name, data in runs.items():
        if data["split"] != "dev" or any(r["split"] != "dev" for r in data["records"]):
            raise SystemExit(f"{name}: not a DEV-only run")
        if any(not r.get("judge") for r in data["records"]):
            raise SystemExit(f"{name}: unjudged records")
    systems = {n: {k: v for k, v in d["system"].items() if k not in ("chroma_db_path",)} for n, d in runs.items()}
    if len({json.dumps(v, sort_keys=True) for v in systems.values()}) != 1:
        raise SystemExit(f"generation settings differ: {systems}")
    judges = {n: sorted({r["judge_model"] for r in d["records"]}) for n, d in runs.items()}
    if len({tuple(v) for v in judges.values()}) != 1:
        raise SystemExit(f"judge models differ: {judges}")

    rec = {n: d["records"] for n, d in runs.items()}
    out = {
        "labels": {"reference": args.reference, "rerun": args.rerun, "candidate": args.candidate},
        "generation_settings": systems["reference"], "judge_model": judges["reference"],
        "candidate_stage3": runs["candidate"].get("stage3"), "rerun_stage3": runs["rerun"].get("stage3"),
        "summary": {n: slices(r, phrasing) for n, r in rec.items()},
        "per_query": {
            "candidate_vs_reference": per_query_changes(rec["reference"], rec["candidate"]),
            "candidate_vs_rerun": per_query_changes(rec["rerun"], rec["candidate"]),
            "noise_floor_rerun_vs_reference": per_query_changes(rec["reference"], rec["rerun"]),
        },
        "repeat_consistency": {n: repeat_consistency(r) for n, r in rec.items()},
    }
    keys = ["false_refusal_rate", "false_answer_rate", "correct", "correct_or_partial", "incorrect",
            "answered_fully_grounded", "hallucination_rate_in_scope", "hallucination_rate_oos", "evidence_has_gold",
            "mean_evidence_precision", "answered_with_citation", "answered_citation_hit", "citation_start_error_median_s"]
    print(f"{'metric':34}{'frozen control':>16}{'control rerun':>16}{'ColBERT':>12}")
    for k in keys:
        print(f"{k:34}" + "".join(f"{str(out['summary'][n]['all'][k]):>{w}}"
                                  for n, w in (("reference", 16), ("rerun", 16), ("candidate", 12))))
    for name, block in out["per_query"].items():
        print(f"{name}: better {len(block['better'])}, worse {len(block['worse'])}, unchanged {block['unchanged']}; "
              f"major better {[r['id'] for r in block['major_better']]}, major worse {[r['id'] for r in block['major_worse']]}")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
