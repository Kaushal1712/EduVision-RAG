"""
experiments/stage4_faithfulness/z3_prepare_annotation.py
─────────────────────────────────────────────────────────
Stage 4, Z3: prepare the human verification set (no labels). Offline; no API calls.

  venv/bin/python experiments/stage4_faithfulness/z3_prepare_annotation.py
Writes, under experiments/stage4_faithfulness/z3/:
  z3_annotation_sheet.md   what the annotator reads: question, prior conversation, full answer, the claim,
                           its citation, and every evidence chunk given to the model (original text)
  z3_labels.csv            the empty label sheet the annotator fills in
  z3_metadata.csv          Z2 scores and existing-judge output per sample (kept out of the sheet so
                           they do not anchor the labels)
  z3_annotation_set.json   everything above in one machine-readable file

Samples are factual sentences from Z2 (z2_alignment_proxy.json), chosen by fixed rules per stratum
(see SELECTION below), at most one sentence per query unless a stratum rule says otherwise. The order
in the sheet is shuffled with a fixed seed so strata are not grouped.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Optional, Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(HERE))

from answer_eval import RESULTS_DIR  # noqa: E402
from bench_data import DEFAULT_BENCHMARK_PATH, DEFAULT_MANIFEST_PATH, load_json  # noqa: E402
from z1_citation_audit import RUNS, extract_citations  # noqa: E402
from z2_alignment_proxy import lexical_overlap, resolve, segment  # noqa: E402

OUT = HERE / "z3"
Z2_PATH = HERE / "z2_alignment_proxy.json"
SEED = 2026
MIN_HINGLISH = 5
MIN_FOLLOWUP = 4

# Known cases from Z1/Z2 (run, query id, repeat, sentence index) and why they are included.
KNOWN = [
    ("frozen_s26_guard", "EVB-018", 0, 2, "Z1/Z2 known case: code example cited to a chunk that does not show it; judge: fully supported"),
    ("frozen_s26_guard", "EVB-018", 0, 0, "Z1/Z2 known case: uncited <link>-tag explanation in the same answer (second sentence of EVB-018, kept deliberately)"),
    ("rerun_s3_control", "EVB-054", 0, 1, "Z1/Z2 known case: quoted sentence the judge flagged as unsupported"),
    ("frozen_s26_guard", "EVB-163", 0, 0, "Z1/Z2 known case: range citation; judge flags the stated location"),
    ("frozen_s26_guard", "EVB-134", 0, 2, "Z2 known case: model-added code example, most misaligned by both signals; judge did not flag"),
    ("frozen_s26_guard", "EVB-086", 1, 2, "Z2 known case: model-added example cited to an unrelated-looking chunk; judge did not flag"),
]
# Quotas per stratum (primary reason), filled in this order from the frozen Stage 2.6 run.
QUOTAS = {"z2_misaligned": 4, "z2_aligned": 6, "z2_mixed": 3, "uncited": 7, "judge_flagged": 5,
          "disagree_judge_flags_z2_aligned": 2}
SELECTION_RULES = {
    "z2_misaligned": "cited chunk ranked 2nd or lower by both dense and lexical; highest dense margin first",
    "z2_aligned": "cited chunk ranked 1st by both signals; spread evenly over the cited dense-score distribution",
    "z2_mixed": "cited chunk ranked 1st by exactly one signal; spread over the dense margin",
    "uncited": "factual sentence with no citation; spread evenly over the best dense-score distribution",
    "judge_flagged": "sentence the judge listed in unsupported_claims; spread over the dense best score",
    "disagree_judge_flags_z2_aligned": "judge-flagged sentence whose cited chunk Z2 ranks 1st by both signals",
    "hinglish_topup": "Hinglish query, filled up to the minimum",
    "followup_topup": "follow-up query, filled up to the minimum",
    "control_positive": "near-verbatim support: cited chunk ranked 1st by both signals with lexical overlap 1.0",
    "control_out_of_scope": "sentence from an answer to an out-of-scope question (the model answered it)",
    "control_weak_evidence": "cited sentence whose best provided chunk has the lowest dense score",
}


# ── Pure helpers (unit-tested) ────────────────────────────────────────────────

def z2_category(row: dict) -> str:
    if "citations" not in row:
        return "uncited"
    if not row.get("dense") or not row.get("lexical"):
        return "cited_unscored"
    d, lx = row["dense"]["rank"], row["lexical"]["rank"]
    if d == 1 and lx == 1:
        return "aligned"
    if d >= 2 and lx >= 2:
        return "misaligned"
    return "mixed"


def strata_tags(row: dict, judge_grounding: Optional[str]) -> list[str]:
    cat = z2_category(row)
    tags = [f"z2_{cat}"]
    if row["judge_flagged_sentence"]:
        tags.append("judge_flagged_sentence")
    if row["judge_hallucination"] is True:
        tags.append("judge_hallucination_answer")
    if (cat == "misaligned" and not row["judge_flagged_sentence"] and judge_grounding == "fully_supported") or \
            (cat == "aligned" and row["judge_flagged_sentence"]):
        tags.append("z2_judge_disagree")
    tags.append("hinglish" if row["language"] == "hinglish" else "english")
    if row["followup"]:
        tags.append("followup")
    if row["category"] != "in_scope":
        tags.append("out_of_scope_question")
    if row["language"] == "en" and not row["followup"] and row["category"] == "in_scope":
        tags.append("ordinary_english")
    return tags


def spread(rows: Sequence[dict], key, n: int) -> list[dict]:
    """n rows evenly spaced over the sorted values of key (deterministic; ties by id order)."""
    s = sorted(rows, key=lambda r: (key(r), r["id"], r["repeat"], r["sentence_index"]))
    if len(s) <= n:
        return list(s)
    return [s[round(i * (len(s) - 1) / (n - 1))] for i in range(n)] if n > 1 else [s[len(s) // 2]]


def exact_citation(answer: str, raw: str) -> str:
    """The citation as written in the answer, including its closing bracket."""
    return raw + "]" if raw + "]" in answer else raw


def mmss(seconds: float, duration: float) -> str:
    from retrieval.retriever import _fmt_timestamp
    return _fmt_timestamp(seconds, duration)


# ── Selection ─────────────────────────────────────────────────────────────────

def select(z2: dict, answers: dict) -> list[dict]:
    rows = {n: v["sentences"] for n, v in z2["runs"].items()}
    key = lambda run, r: (run, r["id"], r["repeat"], r["sentence_index"])
    by_key = {key(n, r): r for n, rs in rows.items() for r in rs}
    grounding = lambda run, r: (answers[run][(r["id"], r["repeat"])].get("judge") or {}).get("grounding")
    chosen: list[dict] = []
    used_ids: set[str] = set()

    def take(run: str, r: dict, reason: str, stratum: str) -> None:
        chosen.append({"run": run, "row": r, "stratum": stratum, "reason": reason,
                       "tags": strata_tags(r, grounding(run, r))})
        used_ids.add(r["id"])

    for run, qid, rep, idx, reason in KNOWN:
        take(run, by_key[(run, qid, rep, idx)], reason, "known_case")

    run = "frozen_s26_guard"
    free = lambda: [r for r in rows[run] if r["id"] not in used_ids and r["category"] == "in_scope"]
    pick = {
        "z2_misaligned": lambda: sorted([r for r in free() if z2_category(r) == "misaligned"],
                                        key=lambda r: (-r["dense"]["margin"], r["id"])),
        "z2_aligned": lambda: spread([r for r in free() if z2_category(r) == "aligned"],
                                     lambda r: r["dense"]["cited"], 40),
        "z2_mixed": lambda: spread([r for r in free() if z2_category(r) == "mixed"],
                                   lambda r: r["dense"]["margin"], 20),
        "uncited": lambda: spread([r for r in free() if z2_category(r) == "uncited"],
                                  lambda r: r["dense_best"], 40),
        "judge_flagged": lambda: spread([r for r in free() if r["judge_flagged_sentence"]],
                                        lambda r: r["dense_best"], 20),
        "disagree_judge_flags_z2_aligned": lambda: [r for r in free() if r["judge_flagged_sentence"]
                                                    and z2_category(r) == "aligned"],
    }
    for stratum, n in QUOTAS.items():
        cands = pick[stratum]()
        step = max(1, len(cands) // n) if stratum in ("z2_aligned", "z2_mixed", "uncited", "judge_flagged") else 1
        got = 0
        for r in cands[::step] + cands:
            if got == n:
                break
            if r["id"] in used_ids:
                continue
            take(run, r, SELECTION_RULES[stratum], stratum)
            got += 1

    for tag, minimum, stratum in (("hinglish", MIN_HINGLISH, "hinglish_topup"),
                                  ("followup", MIN_FOLLOWUP, "followup_topup")):
        have = sum(tag in c["tags"] for c in chosen)
        cands = spread([r for r in free() if (r["language"] == "hinglish" if tag == "hinglish" else r["followup"])],
                       lambda r: r["dense_best"], 20)
        for r in cands:
            if have >= minimum:
                break
            if r["id"] not in used_ids:
                take(run, r, SELECTION_RULES[stratum], stratum)
                have += 1

    pos = [r for r in free() if z2_category(r) == "aligned" and r["lexical"]["cited"] == 1.0]
    if pos:
        take(run, sorted(pos, key=lambda r: (-r["dense"]["cited"], r["id"]))[0], SELECTION_RULES["control_positive"],
             "control_positive")
    oos = [r for r in rows[run] if r["category"] != "in_scope" and r["id"] not in used_ids and "citations" in r]
    if oos:
        take(run, sorted(oos, key=lambda r: (r["id"], r["repeat"], r["sentence_index"]))[0],
             SELECTION_RULES["control_out_of_scope"], "control_out_of_scope")
    weak = [r for r in free() if r.get("dense")]
    if weak:
        take(run, min(weak, key=lambda r: (r["dense"]["best"], r["id"])), SELECTION_RULES["control_weak_evidence"],
             "control_weak_evidence")
    return chosen


# ── Build ─────────────────────────────────────────────────────────────────────

def build_sample(c: dict, answers: dict, bench: dict, manifest: dict, model_cos) -> dict:
    run, r = c["run"], c["row"]
    rec = answers[run][(r["id"], r["repeat"])]
    item = bench[r["id"]]
    sents, _ = segment(rec["answer"])
    sent = sents[r["sentence_index"]]
    if sent["text"] != r["text"]:
        raise SystemExit(f"{r['id']}: sentence text mismatch with Z2")
    cits = extract_citations(rec["answer"], manifest)
    ev = rec["sources_used"]
    cited_idx = sorted({i for k in sent["cits"] for i in resolve(cits[k], ev)["chunks"]})
    dense = [model_cos(r["text"], e["text_en"]) for e in ev]
    lex = [lexical_overlap(r["text"], e["text_en"]) for e in ev]
    if r.get("dense") and abs(max(dense[i] for i in cited_idx) - r["dense"]["cited"]) > 1e-6:
        raise SystemExit(f"{r['id']}: recomputed dense score differs from Z2")
    best_d = max(range(len(ev)), key=lambda i: dense[i]) if ev else None
    best_l = max(range(len(ev)), key=lambda i: (lex[i] or 0.0)) if ev else None
    j = rec.get("judge") or {}
    evidence = [{"label": f"E{i + 1}", "video_filename": e["video_filename"],
                 "time": f"{mmss(e['start_time'], e['end_time'] - e['start_time'])} → "
                         f"{mmss(e['end_time'], e['end_time'] - e['start_time'])}",
                 "text": e["text_en"], "cited_by_claim": i in cited_idx} for i, e in enumerate(ev)]
    return {
        "query_id": r["id"], "run": run, "answers_file": f"eval/results/answers_{RUNS[run]}.json",
        "repeat": r["repeat"], "sentence_index": r["sentence_index"],
        "language": r["language"], "followup": r["followup"], "phrasing": item["phrasing"],
        "question_category": r["category"],
        "question": item["query"], "prior_conversation": item["history"], "answer": rec["answer"],
        "claim": r["text"], "citations": [exact_citation(rec["answer"], cits[k]["raw"]) for k in sent["cits"]],
        "cited_evidence": [evidence[i]["label"] for i in cited_idx], "evidence": evidence,
        "selection": {"stratum": c["stratum"], "reason": c["reason"], "tags": c["tags"]},
        "z2": {"category": z2_category(r),
               "cited_dense": round(max(dense[i] for i in cited_idx), 4) if cited_idx else None,
               "cited_lexical": round(max(lex[i] for i in cited_idx), 4) if cited_idx and None not in lex else None,
               "best_dense_chunk": evidence[best_d]["label"] if best_d is not None else None,
               "best_dense": round(dense[best_d], 4) if best_d is not None else None,
               "best_lexical_chunk": evidence[best_l]["label"] if best_l is not None else None,
               "best_lexical": round(lex[best_l], 4) if best_l is not None and lex[best_l] is not None else None,
               "dense_rank_of_cited": r.get("dense", {}).get("rank"),
               "lexical_rank_of_cited": r.get("lexical", {}).get("rank"),
               "dense_margin": round(r["dense"]["margin"], 4) if r.get("dense") else None},
        "judge": {"model": rec.get("judge_model"), "correctness": j.get("correctness"),
                  "grounding": j.get("grounding"), "hallucination": j.get("hallucination"),
                  "unsupported_claims": j.get("unsupported_claims"),
                  "this_sentence_matched_to_unsupported_claim": r["judge_flagged_sentence"]},
    }


def fence(text: str) -> str:
    ticks = "````" if "```" in text else "```"
    return f"{ticks}text\n{text}\n{ticks}"


def render_sheet(samples: Sequence[dict]) -> str:
    out = ["# Z3 annotation sheet — claim support and citation status (DEV)", "",
           "Read `README.md` first. For each sample, label the **claim** using ONLY the evidence chunks shown "
           "(and, for follow-ups, the prior conversation). Write labels in `z3_labels.csv`.", "",
           "A. Claim support: `Supported` · `Partially supported` · `Unsupported` · `Contradicted`  ",
           "B. Citation status: `Correctly supports the claim` · `Points to evidence but does not support the claim` · "
           "`No citation` · `Citation points outside provided evidence` · `Cannot determine`  ",
           "C. Note: optional, one short sentence.", ""]
    for s in samples:
        out += ["---", "", f"## {s['sample_id']} · {s['query_id']} · {s['language']}"
                f"{' · follow-up' if s['followup'] else ''}"
                f"{' · out-of-scope question' if s['question_category'] != 'in_scope' else ''}", "",
                "**Question**", "", fence(s["question"]), ""]
        if s["prior_conversation"]:
            out += ["**Prior conversation** (the model was allowed to build on it)", "",
                    fence("\n".join(f"{h['role']}: {h['content']}" for h in s["prior_conversation"])), ""]
        out += ["**Full answer**", "", fence(s["answer"]), "",
                "**Claim to label**", "", fence(s["claim"]), "",
                "**Citation attached to the claim:** "
                + (", ".join(f"`{c}` → {', '.join(s['cited_evidence']) or 'no provided chunk'}" for c in s["citations"])
                   if s["citations"] else "none"), "",
                "**Evidence given to the model**", ""]
        for e in s["evidence"]:
            out += [f"**{e['label']}** · {e['video_filename']} · [{e['time']}]"
                    f"{'  ← cited by this claim' if e['cited_by_claim'] else ''}", "", fence(e["text"]), ""]
    return "\n".join(out) + "\n"


def main() -> int:
    bench = {q["id"]: q for q in load_json(DEFAULT_BENCHMARK_PATH)["queries"]}
    manifest = load_json(DEFAULT_MANIFEST_PATH)
    z2 = load_json(Z2_PATH)
    answers = {n: {(r["id"], r["repeat"]): r for r in load_json(RESULTS_DIR / f"answers_{l}.json")["records"]}
               for n, l in RUNS.items()}
    chosen = select(z2, answers)

    from retrieval.retriever import _get_model
    from z2_alignment_proxy import make_dense
    dense = make_dense(_get_model())
    texts = [c["row"]["text"] for c in chosen] + [e["text_en"] for c in chosen
                                                  for e in answers[c["run"]][(c["row"]["id"], c["row"]["repeat"])]["sources_used"]]
    dense(texts)
    samples = [build_sample(c, answers, bench, manifest, dense.cos) for c in chosen]
    random.Random(SEED).shuffle(samples)
    for n, s in enumerate(samples, start=1):
        s["sample_id"] = f"S{n:02d}"
    samples.sort(key=lambda s: s["sample_id"])

    OUT.mkdir(exist_ok=True)
    (OUT / "z3_annotation_sheet.md").write_text(render_sheet(samples))
    with open(OUT / "z3_labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "query_id", "claim", "A_claim_support", "B_citation_status", "C_note"])
        for s in samples:
            w.writerow([s["sample_id"], s["query_id"], s["claim"], "", "", ""])
    with open(OUT / "z3_metadata.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "query_id", "run", "repeat", "sentence_index", "language", "followup", "phrasing",
                    "question_category", "stratum", "tags", "selection_reason", "z2_category", "z2_cited_dense",
                    "z2_cited_lexical", "z2_best_dense_chunk", "z2_best_dense", "z2_best_lexical_chunk",
                    "z2_best_lexical", "z2_dense_rank_of_cited", "z2_lexical_rank_of_cited", "z2_dense_margin",
                    "judge_correctness", "judge_grounding", "judge_hallucination",
                    "judge_flagged_this_sentence", "judge_unsupported_claims"])
        for s in samples:
            z, j = s["z2"], s["judge"]
            w.writerow([s["sample_id"], s["query_id"], s["run"], s["repeat"], s["sentence_index"], s["language"],
                        s["followup"], s["phrasing"], s["question_category"], s["selection"]["stratum"],
                        ";".join(s["selection"]["tags"]), s["selection"]["reason"], z["category"], z["cited_dense"],
                        z["cited_lexical"], z["best_dense_chunk"], z["best_dense"], z["best_lexical_chunk"],
                        z["best_lexical"], z["dense_rank_of_cited"], z["lexical_rank_of_cited"], z["dense_margin"],
                        j["correctness"], j["grounding"], j["hallucination"],
                        j["this_sentence_matched_to_unsupported_claim"], " | ".join(j["unsupported_claims"] or [])])
    (OUT / "z3_annotation_set.json").write_text(json.dumps(
        {"stage": "4", "experiment": "Z3 human verification set (unlabelled)", "seed": SEED,
         "selection_rules": SELECTION_RULES, "quotas": QUOTAS, "known_cases": [k[:4] for k in KNOWN],
         "samples": samples}, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(samples)} samples →", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
