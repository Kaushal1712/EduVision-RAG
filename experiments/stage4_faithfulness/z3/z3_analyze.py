"""
experiments/stage4_faithfulness/z3/z3_analyze.py
─────────────────────────────────────────────────
Z3 analysis: compare the (AI-assisted, NOT human-verified) Z3 labels with the Z2 alignment proxy and
the existing LLM judge. Offline; no API calls; reads only existing files; never edits labels.

  venv/bin/python experiments/stage4_faithfulness/z3/z3_analyze.py
Writes experiments/stage4_faithfulness/z3/z3_analysis.json.

Binary views used for agreement (all defined before looking at results):
  Z3 claim problem       A in {Partially supported, Unsupported, Contradicted}
  Z3 strict problem      A in {Unsupported, Contradicted}
  Z3 citation problem    B == "Points to evidence but does not support the claim" (cited claims only)
  Z2 misaligned          cited chunk ranked 2nd or lower by BOTH dense and lexical (Z2 category)
  Z2 not-best (dense)    cited chunk ranked 2nd or lower by dense
  Judge flag             the sentence was matched to one of the judge's unsupported_claims: Z2's word-Jaccard
                         mapping (>= 0.5) OR one text containing the other's first 60 characters. The
                         containment rule was added after inspection found one miss (S24 / EVB-054, Jaccard
                         0.43); Z2-mapping-only results are kept as judge_vs_z3_z2_mapping_only.
The sample is stratified and over-represents disagreements and known cases, so these rates describe
this set only, not DEV as a whole.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Optional, Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "eval"))

from bench_metrics import auroc  # noqa: E402

LABELS = HERE / "z3_labels.csv"
SET = HERE / "z3_annotation_set.json"
NOT_SUPPORT = "Points to evidence but does not support the claim"
CORRECT = "Correctly supports the claim"
PROBLEM = {"Partially supported", "Unsupported", "Contradicted"}
STRICT = {"Unsupported", "Contradicted"}


# ── Pure helpers (unit-tested) ────────────────────────────────────────────────

def confusion(pairs: Sequence[tuple[bool, bool]]) -> dict:
    """pairs: (reference, predicted). Reference = Z3."""
    tp = sum(r and p for r, p in pairs)
    fp = sum((not r) and p for r, p in pairs)
    fn = sum(r and (not p) for r, p in pairs)
    tn = sum((not r) and (not p) for r, p in pairs)
    n = len(pairs)
    po = (tp + tn) / n if n else None
    pe = (((tp + fp) * (tp + fn)) + ((fn + tn) * (fp + tn))) / (n * n) if n else None
    kappa = (po - pe) / (1 - pe) if n and pe != 1 else None
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "agreement": round(po, 4) if po is not None else None,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "cohen_kappa": round(kappa, 4) if kappa is not None else None}


def crosstab(rows: Sequence[dict], a: Callable[[dict], str], b: Callable[[dict], str]) -> dict:
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        out.setdefault(a(r), Counter())[b(r)] += 1
    return {k: dict(v) for k, v in sorted(out.items())}


def score_auroc(rows: Sequence[dict], positive: Callable[[dict], bool], score: Callable[[dict], Optional[float]]):
    pos = [score(r) for r in rows if positive(r) and score(r) is not None]
    neg = [score(r) for r in rows if not positive(r) and score(r) is not None]
    return {"n_pos": len(pos), "n_neg": len(neg), "auroc": round(auroc(pos, neg), 4) if pos and neg else None}


# ── Analysis ──────────────────────────────────────────────────────────────────

def contains_claim(claim: str, judge_claims: Sequence[str]) -> bool:
    c = claim[:60].lower()
    return any(c in j.lower() or j[:60].lower() in claim.lower() for j in judge_claims)


def load() -> list[dict]:
    labels = {r["sample_id"]: r for r in csv.DictReader(LABELS.open(newline=""))}
    rows = []
    for s in json.loads(SET.read_text())["samples"]:
        lab = labels[s["sample_id"]]
        if lab["claim"] != s["claim"]:
            raise SystemExit(f"{s['sample_id']}: claim text differs between labels and annotation set")
        rows.append({**s, "A": lab["A_claim_support"], "B": lab["B_citation_status"], "note": lab["C_note"]})
    return rows


def main() -> int:
    rows = load()
    cited = [r for r in rows if r["citations"]]
    jmap = lambda r: bool(r["judge"]["this_sentence_matched_to_unsupported_claim"])
    jflag = lambda r: jmap(r) or contains_claim(r["claim"], r["judge"]["unsupported_claims"] or [])
    out: dict = {
        "status": "Z3 labels are AI-assisted and NOT human-verified; this is not ground truth.",
        "n_samples": len(rows), "n_cited": len(cited),
        "A_distribution": dict(Counter(r["A"] for r in rows)),
        "B_distribution": dict(Counter(r["B"] for r in rows)),
        "A_by_slice": {
            "cited": dict(Counter(r["A"] for r in cited)),
            "uncited": dict(Counter(r["A"] for r in rows if not r["citations"])),
            "hinglish": dict(Counter(r["A"] for r in rows if r["language"] == "hinglish")),
            "followup": dict(Counter(r["A"] for r in rows if r["followup"])),
            "out_of_scope_question": dict(Counter(r["A"] for r in rows if r["question_category"] != "in_scope")),
        },
        "A_by_z2_category": crosstab(rows, lambda r: r["z2"]["category"], lambda r: r["A"]),
        "B_by_z2_category": crosstab(cited, lambda r: r["z2"]["category"], lambda r: r["B"]),
        "A_by_judge_flag": crosstab(rows, lambda r: "judge_flagged" if jflag(r) else "not_flagged", lambda r: r["A"]),
        "A_by_judge_grounding": crosstab(rows, lambda r: str(r["judge"]["grounding"]), lambda r: r["A"]),
        "A_by_judge_hallucination_answer": crosstab(rows, lambda r: str(r["judge"]["hallucination"]), lambda r: r["A"]),
    }
    # Z3 vs Z2 — citation alignment (cited claims only)
    out["z2_vs_z3_citation"] = {
        "z2_misaligned_vs_z3_citation_problem": confusion([(r["B"] == NOT_SUPPORT, r["z2"]["category"] == "misaligned")
                                                           for r in cited]),
        "z2_dense_not_best_vs_z3_citation_problem": confusion([(r["B"] == NOT_SUPPORT, r["z2"]["dense_rank_of_cited"] >= 2)
                                                               for r in cited]),
        "auroc_dense_margin_for_z3_citation_problem": score_auroc(cited, lambda r: r["B"] == NOT_SUPPORT,
                                                                  lambda r: r["z2"]["dense_margin"]),
        "auroc_low_cited_dense_for_z3_citation_problem": score_auroc(cited, lambda r: r["B"] == NOT_SUPPORT,
                                                                     lambda r: -r["z2"]["cited_dense"]),
    }
    # Z3 vs Z2 — claim support (all claims; Z2 has only similarity, so best-chunk scores are used)
    out["z2_vs_z3_claim"] = {
        "auroc_low_best_dense_for_z3_problem": score_auroc(rows, lambda r: r["A"] in PROBLEM, lambda r: -r["z2"]["best_dense"]),
        "auroc_low_best_lexical_for_z3_problem": score_auroc(rows, lambda r: r["A"] in PROBLEM,
                                                             lambda r: -(r["z2"]["best_lexical"] or 0.0)),
        "z2_misaligned_vs_z3_problem_cited": confusion([(r["A"] in PROBLEM, r["z2"]["category"] == "misaligned")
                                                        for r in cited]),
    }
    # Z3 vs judge
    out["judge_vs_z3"] = {
        "judge_flag_vs_z3_problem": confusion([(r["A"] in PROBLEM, jflag(r)) for r in rows]),
        "judge_flag_vs_z3_strict_problem": confusion([(r["A"] in STRICT, jflag(r)) for r in rows]),
        "answer_hallucination_vs_z3_problem": confusion([(r["A"] in PROBLEM, r["judge"]["hallucination"] is True)
                                                         for r in rows]),
        "answer_not_fully_grounded_vs_z3_problem": confusion([(r["A"] in PROBLEM, r["judge"]["grounding"] != "fully_supported")
                                                              for r in rows]),
    }
    out["judge_vs_z3_z2_mapping_only"] = {
        "judge_flag_vs_z3_problem": confusion([(r["A"] in PROBLEM, jmap(r)) for r in rows]),
        "changed_by_containment_rule": [r["sample_id"] for r in rows if jflag(r) != jmap(r)]}
    brief = lambda r: {"sample_id": r["sample_id"], "query_id": r["query_id"], "A": r["A"], "B": r["B"],
                       "z2_category": r["z2"]["category"], "z2_dense_rank": r["z2"]["dense_rank_of_cited"],
                       "z2_dense_margin": r["z2"]["dense_margin"], "judge_flag": jflag(r),
                       "judge_grounding": r["judge"]["grounding"], "claim": r["claim"], "note": r["note"]}
    out["disagreements"] = {
        "z2_misaligned_but_z3_citation_ok": [brief(r) for r in cited if r["z2"]["category"] == "misaligned" and r["B"] == CORRECT],
        "z2_not_misaligned_but_z3_citation_problem": [brief(r) for r in cited if r["z2"]["category"] != "misaligned"
                                                      and r["B"] == NOT_SUPPORT],
        "judge_flagged_but_z3_supported": [brief(r) for r in rows if jflag(r) and r["A"] == "Supported"],
        "judge_not_flagged_but_z3_problem": [brief(r) for r in rows if not jflag(r) and r["A"] in PROBLEM],
    }
    out["citation_problem_cases"] = [brief(r) | {"cited_evidence": r["cited_evidence"],
                                                 "z2_best_dense_chunk": r["z2"]["best_dense_chunk"],
                                                 "z2_best_lexical_chunk": r["z2"]["best_lexical_chunk"],
                                                 "judge_hallucination_answer": r["judge"]["hallucination"]}
                                     for r in cited if r["B"] == NOT_SUPPORT]
    (HERE / "z3_analysis.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    for k in ("A_distribution", "B_distribution", "B_by_z2_category", "A_by_judge_flag"):
        print(k, json.dumps(out[k]))
    print("z2 vs z3 citation", json.dumps(out["z2_vs_z3_citation"]))
    print("z2 vs z3 claim", json.dumps(out["z2_vs_z3_claim"]))
    print("judge vs z3", json.dumps(out["judge_vs_z3"]))
    for k, v in out["disagreements"].items():
        print(k, [(d["sample_id"], d["query_id"], d["A"], d["B"][:12], d["z2_category"], d["judge_flag"]) for d in v])
    for c in out["citation_problem_cases"]:
        print("CIT", c["sample_id"], c["query_id"], c["A"], "cited", c["cited_evidence"], "z2", c["z2_category"],
              "rank", c["z2_dense_rank"], "margin", c["z2_dense_margin"], "best d/l", c["z2_best_dense_chunk"],
              c["z2_best_lexical_chunk"], "judge_flag", c["judge_flag"], c["judge_grounding"], "|", c["note"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
