"""
eval/compare_results.py
────────────────────────
Paired comparison of two run_benchmark.py result files (CONTROL vs CANDIDATE) on the
same benchmark.

  venv/bin/python eval/compare_results.py eval/results/<control>.json eval/results/<candidate>.json \
      [--splits dev test all] [--out eval/results/<comparison>.json]

Reports, per split:
  * ranking metrics and their deltas, with a seeded paired-bootstrap 95% CI;
  * the same on the transcript-coverage slices (full / partial / none). Slices are
    always taken from the CONTROL run, i.e. how much usable text the production index
    has for each gold span. The candidate's own coverage would move the slice
    boundaries with the system under test.
  * per-query outcomes on Hit@10: recovered (control miss → candidate hit), lost,
    both hit, both miss; plus MRR improvements/regressions;
  * timestamp error on queries that both systems hit;
  * abstention-gate rates (an index change can shift similarity scores).

Refuses to compare runs whose benchmark file or retrieval/system code differ: the only
intended difference is the index.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_metrics import mean, percentile  # noqa: E402

KEYS = ["recall@5", "recall@10", "mrr@10", "ndcg@10"]
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 2026
TIMESTAMP_TOLERANCE_S = 1.0   # changes smaller than this count as "same"


def _r(x: Optional[float], d: int = 4) -> Optional[float]:
    return None if x is None else round(x, d)


def hit10(m: dict) -> bool:
    return m["recall@10"] > 0


def paired_bootstrap_ci(deltas: Sequence[float], resamples: int = BOOTSTRAP_RESAMPLES,
                        seed: int = BOOTSTRAP_SEED) -> Optional[tuple[float, float]]:
    """95% percentile CI of the mean paired difference."""
    if not deltas:
        return None
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def outcome(c: dict, k: dict) -> str:
    hc, hk = hit10(c), hit10(k)
    if hc and hk:
        return "both_hit"
    if hk:
        return "recovered"
    if hc:
        return "lost"
    return "both_miss"


def compare_group(pairs: Sequence[tuple[dict, dict]]) -> dict:
    """pairs: (control_record, candidate_record) for scored queries."""
    out: dict = {"n": len(pairs)}
    if not pairs:
        return out
    for key in KEYS:
        c = [a["metrics"][key] for a, _ in pairs]
        k = [b["metrics"][key] for _, b in pairs]
        deltas = [y - x for x, y in zip(c, k)]
        ci = paired_bootstrap_ci(deltas)
        out[key] = {"control": _r(mean(c)), "candidate": _r(mean(k)), "delta": _r(mean(deltas)),
                    "delta_ci95": [_r(ci[0]), _r(ci[1])] if ci else None}
    for name, fn in (("hit@5", lambda m: m["recall@5"] > 0), ("hit@10", hit10)):
        out[name] = {"control": _r(mean([float(fn(a["metrics"])) for a, _ in pairs])),
                     "candidate": _r(mean([float(fn(b["metrics"])) for _, b in pairs]))}

    classes = {"recovered": [], "lost": [], "both_hit": [], "both_miss": []}
    for a, b in pairs:
        classes[outcome(a["metrics"], b["metrics"])].append(a["id"])
    out["outcomes_hit@10"] = {k: {"n": len(v), "ids": v} for k, v in classes.items()}
    out["mrr_changes"] = {
        "improved": sum(b["metrics"]["mrr@10"] > a["metrics"]["mrr@10"] for a, b in pairs),
        "worsened": sum(b["metrics"]["mrr@10"] < a["metrics"]["mrr@10"] for a, b in pairs),
        "unchanged": sum(b["metrics"]["mrr@10"] == a["metrics"]["mrr@10"] for a, b in pairs),
    }

    def ts_stats(errs: list[float]) -> dict:
        return {"n": len(errs), "median": _r(percentile(errs, 50), 2), "mean": _r(mean(errs), 2),
                "p90": _r(percentile(errs, 90), 2)}
    c_err = [a["metrics"]["start_error_s"] for a, _ in pairs if a["metrics"]["start_error_s"] is not None]
    k_err = [b["metrics"]["start_error_s"] for _, b in pairs if b["metrics"]["start_error_s"] is not None]
    both = [(a["metrics"]["start_error_s"], b["metrics"]["start_error_s"]) for a, b in pairs
            if a["metrics"]["start_error_s"] is not None and b["metrics"]["start_error_s"] is not None]
    out["start_error_s"] = {
        "control_all_hits": ts_stats(c_err), "candidate_all_hits": ts_stats(k_err),
        "paired_both_hit": {
            "n": len(both),
            "control_median": _r(percentile([x for x, _ in both], 50), 2),
            "candidate_median": _r(percentile([y for _, y in both], 50), 2),
            "improved": sum(y < x - TIMESTAMP_TOLERANCE_S for x, y in both),
            "regressed": sum(y > x + TIMESTAMP_TOLERANCE_S for x, y in both),
            "same_within_1s": sum(abs(y - x) <= TIMESTAMP_TOLERANCE_S for x, y in both),
        },
    }
    return out


def compare(control: dict, candidate: dict, splits: Sequence[str]) -> dict:
    problems = []
    if control["benchmark"]["sha256"] != candidate["benchmark"]["sha256"]:
        problems.append("benchmark files differ")
    if control["system"]["code_sha256"] != candidate["system"]["code_sha256"]:
        problems.append("system code differs")
    cfg_c = {k: v for k, v in control["system"]["config"].items() if k not in ("chroma_db_path", "active_collection")}
    cfg_k = {k: v for k, v in candidate["system"]["config"].items() if k not in ("chroma_db_path", "active_collection")}
    if cfg_c != cfg_k:
        problems.append(f"system config differs beyond the index: {set(cfg_c.items()) ^ set(cfg_k.items())}")
    if problems:
        raise SystemExit("Refusing to compare: " + "; ".join(problems))

    cand = {r["id"]: r for r in candidate["per_query"]}
    pairs_all = [(c, cand[c["id"]]) for c in control["per_query"] if c["metrics"] is not None]
    result: dict = {
        "control": {"label": control["label"], "index": control["system"]["index"],
                    "config": control["system"]["config"]},
        "candidate": {"label": candidate["label"], "index": candidate["system"]["index"],
                      "config": candidate["system"]["config"]},
        "benchmark": control["benchmark"]["version"],
        "slices_from": "control transcript_coverage",
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "splits": {},
    }
    for split in splits:
        pairs = [p for p in pairs_all if split == "all" or p[0]["split"] == split]
        block = {"overall": compare_group(pairs), "by_control_coverage": {}, "by_phrasing": {}, "by_language": {}}
        for field, target in (("transcript_coverage", "by_control_coverage"), ("phrasing", "by_phrasing"),
                              ("language", "by_language")):
            for g in sorted({p[0][field] for p in pairs}, key=str):
                block[target][str(g)] = compare_group([p for p in pairs if p[0][field] == g])
        cs, ks = control["summary"].get(split), candidate["summary"].get(split)
        if cs and ks:
            block["abstention"] = {"control": cs["abstention"], "candidate": ks["abstention"]}
            block["retrieved_unclear_rate@5"] = {"control": cs["retrieved_unclear_rate@5"],
                                                 "candidate": ks["retrieved_unclear_rate@5"]}
            block["exact_search_agreement"] = {"control": cs["exact_search_agreement"],
                                               "candidate": ks["exact_search_agreement"]}
        result["splits"][split] = block
    return result


def print_report(res: dict) -> None:
    for split, block in res["splits"].items():
        print(f"\n=== {split.upper()} ===")
        rows: list[tuple[str, dict]] = [("overall", block["overall"])] + \
            [(f"cov={g}", v) for g, v in block["by_control_coverage"].items()]
        print(f"{'group':<14}{'n':>4}  " + "  ".join(f"{k:>24}" for k in KEYS))
        for name, g in rows:
            if g["n"] == 0:
                continue
            cells = []
            for k in KEYS:
                m = g[k]
                cells.append(f"{m['control']:.3f}→{m['candidate']:.3f} ({m['delta']:+.3f})")
            print(f"{name:<14}{g['n']:>4}  " + "  ".join(f"{c:>24}" for c in cells))
        o = block["overall"]["outcomes_hit@10"]
        print("Hit@10 outcomes: " + ", ".join(f"{k} {v['n']}" for k, v in o.items()))
        ts = block["overall"]["start_error_s"]["paired_both_hit"]
        print(f"Start error on {ts['n']} queries hit by both: median {ts['control_median']}s → "
              f"{ts['candidate_median']}s; improved {ts['improved']}, regressed {ts['regressed']}, "
              f"same {ts['same_within_1s']}")
        for k in ("recall@10", "mrr@10"):
            print(f"  {k} delta 95% CI: {block['overall'][k]['delta_ci95']}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("control")
    p.add_argument("candidate")
    p.add_argument("--splits", nargs="*", default=["dev", "test", "all"])
    p.add_argument("--out")
    args = p.parse_args()
    res = compare(json.loads(Path(args.control).read_text()), json.loads(Path(args.candidate).read_text()),
                  args.splits)
    print_report(res)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
