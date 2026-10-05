"""
experiments/stage3_retrieval/compare_stage3.py
───────────────────────────────────────────────
Paired DEV comparison of two run_stage3.py results (CONTROL vs VARIANT) on the same index.

  venv/bin/python experiments/stage3_retrieval/compare_stage3.py \
      eval/results/stage3_control_s26_dev.json eval/results/stage3_<variant>_dev.json \
      [--out eval/results/stage3_comparison_<variant>_dev.json]

Unlike eval/compare_results.py (where only the index may differ), here the retrieval method
is the variable. Everything else must match: benchmark, index fingerprint, threshold, top-k,
evidence cap and the production code files.

Ranking deltas, bootstrap CIs, recovered/lost and timestamp errors come from
compare_results.compare_group. Added: evidence (what the LLM would receive), gate decisions,
ranking/evidence set and order changes, candidate-pool recall, latency, and union recall: the gold
found by the control's top-k together with the variant's top-k (what the variant adds as candidates).
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
sys.path.insert(0, str(HERE))

from compare_results import compare_group  # noqa: E402
from bench_metrics import mean  # noqa: E402
from run_stage3 import summarize_evidence, summarize_pool  # noqa: E402

MUST_MATCH_CONFIG = ("similarity_threshold", "retrieval_top_k", "max_llm_evidence", "embedding_model",
                     "query_encoding", "active_collection")
NEAR_THRESHOLD = 0.005   # top-1 similarities this close to the threshold are listed


def check_comparable(control: dict, variant: dict) -> list[str]:
    problems = []
    if control["benchmark"]["sha256"] != variant["benchmark"]["sha256"]:
        problems.append("benchmark files differ")
    if control["system"]["index"]["sha256"] != variant["system"]["index"]["sha256"]:
        problems.append("index fingerprints differ")
    if control["system"]["code_sha256"] != variant["system"]["code_sha256"]:
        problems.append("production code differs")
    if control["split"] != variant["split"]:
        problems.append("splits differ")
    for key in MUST_MATCH_CONFIG:
        if control["system"]["config"].get(key) != variant["system"]["config"].get(key):
            problems.append(f"config {key} differs")
    return problems


def gate_changes(pairs: Sequence[tuple[dict, dict]], threshold: float) -> dict:
    """All queries (in-scope and OOS): gate decisions and top-1 dense similarity."""
    changed = [{"id": a["id"], "category": a["category"], "control": a["gate"], "variant": b["gate"]}
               for a, b in pairs if a["gate"] != b["gate"]]
    diffs = [abs(a["top1_similarity"] - b["top1_similarity"]) for a, b in pairs
             if a["top1_similarity"] is not None and b["top1_similarity"] is not None]
    near = sorted({a["id"] for a, b in pairs for r in (a, b)
                   if r["top1_similarity"] is not None and abs(r["top1_similarity"] - threshold) < NEAR_THRESHOLD})
    return {"n": len(pairs), "changed": changed, "max_abs_top1_similarity_diff": max(diffs) if diffs else None,
            "near_threshold_ids": near}


def list_changes(pairs: Sequence[tuple[dict, dict]], get) -> dict:
    """Classify per-query lists as identical, same set in another order, or a different set."""
    out = {"identical": 0, "reordered": [], "different_set": []}
    for a, b in pairs:
        x, y = get(a), get(b)
        if x == y:
            out["identical"] += 1
        elif set(x) == set(y):
            out["reordered"].append(a["id"])
        else:
            out["different_set"].append(a["id"])
    out["n"] = len(pairs)
    return out


def evidence_hit_changes(pairs: Sequence[tuple[dict, dict]]) -> dict:
    gained = [a["id"] for a, b in pairs if not a["evidence"]["evidence_hit"] and b["evidence"]["evidence_hit"]]
    lost = [a["id"] for a, b in pairs if a["evidence"]["evidence_hit"] and not b["evidence"]["evidence_hit"]]
    return {"gained": gained, "lost": lost}


def union_recall(pairs: Sequence[tuple[dict, dict]], k: int, threshold: float) -> dict:
    """
    Recall of the union of both top-k lists, against the control's top-k alone. For each query
    where the variant adds gold, the variant chunks carrying that gold, with their dense cosine:
    only those at or above the threshold could pass the generator's evidence filter.
    """
    ctrl, union, added, detail = [], [], [], {}
    for a, b in pairs:
        n = a.get("n_gold")
        if not n:
            continue
        ca = {g for x in a["retrieved"][:k] for g in x["matched_gold"]}
        cb = {g for x in b["retrieved"][:k] for g in x["matched_gold"]}
        ctrl.append(len(ca) / n)
        union.append(len(ca | cb) / n)
        if cb - ca:
            added.append(a["id"])
            detail[a["id"]] = [{"variant_rank": x["rank"], "dense_cosine": x["similarity"],
                                "passes_threshold": x["similarity"] >= threshold}
                               for x in b["retrieved"][:k] if set(x["matched_gold"]) - ca]
    usable = [i for i, d in detail.items() if any(x["passes_threshold"] for x in d)]
    return {"k": k, "n": len(ctrl), "control_recall": mean(ctrl), "union_recall": mean(union),
            "ids_with_gold_added_by_variant": added, "added_gold_detail": detail,
            "ids_with_added_gold_passing_threshold": usable}


def first_relevant_rank(record: dict) -> int | None:
    return next((x["rank"] for x in record["retrieved"] if x["matched_gold"]), None)


def rank_changes(pairs: Sequence[tuple[dict, dict]]) -> dict:
    """Per scored query: rank of the first gold chunk in the top-k, control vs variant (None = miss)."""
    rows = [{"id": a["id"], "control": first_relevant_rank(a), "variant": first_relevant_rank(b)}
            for a, b in pairs]
    better = [r for r in rows if r["control"] != r["variant"] and r["variant"] is not None
              and (r["control"] is None or r["variant"] < r["control"])]
    worsened = [r for r in rows if r["control"] != r["variant"] and r not in better]
    return {"improved": better, "worsened": worsened, "unchanged": sum(r["control"] == r["variant"] for r in rows)}


def compare_stage3(control: dict, variant: dict) -> dict:
    problems = check_comparable(control, variant)
    if problems:
        raise SystemExit("Refusing to compare: " + "; ".join(problems))
    var = {r["id"]: r for r in variant["per_query"]}
    pairs_all = [(c, var[c["id"]]) for c in control["per_query"]]
    scored = [(c, k) for c, k in pairs_all if c["metrics"] is not None]
    split = control["split"]
    cs, ks = control["summary"][split], variant["summary"][split]

    out: dict = {
        "control": {"label": control["label"], "experiment": control["experiment"]},
        "variant": {"label": variant["label"], "experiment": variant["experiment"]},
        "split": split, "index_sha256": control["system"]["index"]["sha256"],
        "slices_from": "Stage 1.5 production-index coverage (transcript_coverage in run_stage3 results)",
        "overall": compare_group(scored), "by_coverage_s15": {}, "by_language": {}, "by_phrasing": {},
        "by_followup": {},
    }
    for field, target in (("transcript_coverage", "by_coverage_s15"), ("language", "by_language"),
                          ("phrasing", "by_phrasing"), ("followup", "by_followup")):
        for g in sorted({c[field] for c, _ in scored}, key=str):
            out[target][str(g)] = compare_group([p for p in scored if p[0][field] == g])
    out["evidence"] = {
        "control": cs["evidence"], "variant": ks["evidence"],
        "by_language": {lang: {"control": summarize_evidence([c for c, _ in scored if c["language"] == lang]),
                               "variant": summarize_evidence([k for _, k in scored if k["language"] == lang])}
                        for lang in ("en", "hinglish")},
        "hit_changes": evidence_hit_changes(scored),
        "list_changes": list_changes(scored, lambda r: r["evidence"]["evidence_ids"]),
    }
    out["top10_changes"] = list_changes(pairs_all, lambda r: [x["chunk_id"] for x in r["retrieved"]])
    out["gate"] = gate_changes(pairs_all, control["system"]["config"]["similarity_threshold"])
    out["abstention"] = {"control": cs["abstention"], "variant": ks["abstention"]}
    out["pool_variant"] = {
        "overall": ks.get("pool"),
        "by_language": {g: summarize_pool([k for _, k in scored if k["language"] == g]) for g in ("en", "hinglish")},
        "by_phrasing": {str(g): summarize_pool([k for _, k in scored if k["phrasing"] == g])
                        for g in sorted({k["phrasing"] for _, k in scored}, key=str)},
        "by_coverage_s15": {str(g): summarize_pool([k for _, k in scored if k["transcript_coverage"] == g])
                            for g in sorted({k["transcript_coverage"] for _, k in scored}, key=str)},
    }
    out["first_relevant_rank_changes"] = rank_changes(scored)
    out["union"] = {f"top{k}": union_recall(scored, k, control["system"]["config"]["similarity_threshold"])
                    for k in (5, 10)}
    out["latency"] = {"control": cs["latency"], "variant": ks["latency"]}
    return out


def print_report(res: dict) -> None:
    o = res["overall"]
    print(f"{res['control']['label']} → {res['variant']['label']}  (n={o['n']})")
    for key in ("recall@5", "recall@10", "mrr@10", "ndcg@10"):
        m = o[key]
        print(f"  {key:<10} {m['control']:.4f} → {m['candidate']:.4f}  Δ {m['delta']:+.4f}  CI {m['delta_ci95']}")
    print("  Hit@10 outcomes: " + ", ".join(f"{k} {v['n']}" for k, v in o["outcomes_hit@10"].items()))
    ts = o["start_error_s"]["paired_both_hit"]
    print(f"  start error (both hit, n={ts['n']}): median {ts['control_median']} → {ts['candidate_median']}")
    for lang, g in res["by_language"].items():
        print(f"  {lang:<10} R@10 {g['recall@10']['control']:.3f}→{g['recall@10']['candidate']:.3f}  "
              f"MRR {g['mrr@10']['control']:.3f}→{g['mrr@10']['candidate']:.3f}")
    e = res["evidence"]
    print(f"  evidence recall {e['control']['evidence_recall']} → {e['variant']['evidence_recall']}; "
          f"changes {e['hit_changes']}; lists: identical {e['list_changes']['identical']}/{e['list_changes']['n']}")
    print(f"  top-10 identical {res['top10_changes']['identical']}/{res['top10_changes']['n']}; "
          f"gate changes {res['gate']['changed']}; max |Δ top1 sim| {res['gate']['max_abs_top1_similarity_diff']}")
    print(f"  pool {res['pool_variant']['overall']}")
    u = res["union"]["top10"]
    print(f"  union top-10 recall {u['control_recall']:.4f} → {u['union_recall']:.4f}; "
          f"gold added for {u['ids_with_gold_added_by_variant']}; "
          f"passing the dense threshold: {u['ids_with_added_gold_passing_threshold']}")
    print(f"  latency p50/p95 {res['latency']['control']['p50_s']}/{res['latency']['control']['p95_s']} → "
          f"{res['latency']['variant']['p50_s']}/{res['latency']['variant']['p95_s']} s")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("control")
    p.add_argument("variant")
    p.add_argument("--out")
    args = p.parse_args()
    res = compare_stage3(json.loads(Path(args.control).read_text()), json.loads(Path(args.variant).read_text()))
    print_report(res)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
