"""
experiments/stage2_6_abstention/threshold_sweep.py
───────────────────────────────────────────────────
Stage 2.6: retrieval-gate behaviour of the Stage 2 candidate index across similarity thresholds.

The gate (generation/generator.py) refuses without an LLM call when no retrieved chunk has
similarity >= threshold. Ranking does not depend on the threshold, so the gate decision for
any threshold follows from the per-query top-1 similarity already stored by run_benchmark.py.

  venv/bin/python experiments/stage2_6_abstention/threshold_sweep.py [--split dev]
Writes eval/results/stage2_6_threshold_sweep_<split>.json.

"Effective" recall counts a query as 0 when the gate refuses it.
Coverage slices come from the control run (as in eval/compare_results.py).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

from bench_metrics import auroc  # noqa: E402

RES = ROOT / "eval" / "results"
RUNS = {"control": "stage1_5_baseline_bench-v1.1", "candidate": "stage2_lv2g_translate"}
THRESHOLDS = [round(0.40 + 0.01 * i, 2) for i in range(21)]


def sweep(per_query: list[dict], coverage: dict[str, str]) -> dict:
    ins = [r for r in per_query if r["category"] == "in_scope"]
    oos = [r for r in per_query if r["category"] != "in_scope"]
    rate = lambda rs, f: round(sum(1 for r in rs if f(r)) / len(rs), 4) if rs else None
    out = {"n_in_scope": len(ins), "n_oos": len(oos),
           "auroc_top1": round(auroc([r["top1_similarity"] for r in ins], [r["top1_similarity"] for r in oos]), 4),
           "thresholds": {}}
    for t in THRESHOLDS:
        passes = lambda r: r["top1_similarity"] is not None and r["top1_similarity"] >= t
        scored = [r for r in ins if r["metrics"]]
        eff = lambda rs, k: round(sum(r["metrics"][k] for r in rs if passes(r)) / len(rs), 4) if rs else None
        out["thresholds"][f"{t:.2f}"] = {
            "gate_false_answer_rate_oos": rate(oos, passes),
            "gate_false_refusal_rate": rate(ins, lambda r: not passes(r)),
            "gate_false_refusal_en": rate([r for r in ins if r["language"] == "en"], lambda r: not passes(r)),
            "gate_false_refusal_hinglish": rate([r for r in ins if r["language"] == "hinglish"], lambda r: not passes(r)),
            "refused_in_scope_ids": sorted(r["id"] for r in ins if not passes(r)),
            "passed_oos_ids": sorted(r["id"] for r in oos if passes(r)),
            "effective_recall@5": eff(scored, "recall@5"), "effective_recall@10": eff(scored, "recall@10"),
            "effective_recall@10_by_coverage": {c: eff([r for r in scored if coverage[r["id"]] == c], "recall@10")
                                                for c in ("full", "partial", "none")},
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    args = p.parse_args()
    control = json.loads((RES / f"{RUNS['control']}.json").read_text())["per_query"]
    coverage = {r["id"]: r["transcript_coverage"] for r in control}
    result = {"split": args.split, "note": "gate-level only; the LLM can still refuse after the gate passes"}
    for name, run in RUNS.items():
        pq = [r for r in json.loads((RES / f"{run}.json").read_text())["per_query"]
              if args.split == "all" or r["split"] == args.split]
        result[name] = sweep(pq, coverage)
    out = RES / f"stage2_6_threshold_sweep_{args.split}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    c = result["candidate"]
    print(f"candidate {args.split}: AUROC {c['auroc_top1']}")
    print(f"{'T':>5} {'FA oos':>7} {'FR':>6} {'FR en':>6} {'FR hi':>6} {'effR@5':>7} {'effR@10':>8}  none/partial/full")
    for t, v in c["thresholds"].items():
        cov = v["effective_recall@10_by_coverage"]
        print(f"{t:>5} {v['gate_false_answer_rate_oos']:>7} {v['gate_false_refusal_rate']:>6} {v['gate_false_refusal_en']:>6} "
              f"{v['gate_false_refusal_hinglish']:>6} {v['effective_recall@5']:>7} {v['effective_recall@10']:>8}  "
              f"{cov['none']}/{cov['partial']}/{cov['full']}")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
