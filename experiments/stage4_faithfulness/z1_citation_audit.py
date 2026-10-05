"""
experiments/stage4_faithfulness/z1_citation_audit.py
─────────────────────────────────────────────────────
Stage 4, Z1: offline citation-integrity audit of existing DEV answer files. No API calls.

  venv/bin/python experiments/stage4_faithfulness/z1_citation_audit.py
Writes experiments/stage4_faithfulness/z1_citation_integrity_audit.json (+ .md written separately).

What a citation is checked against: the evidence chunks actually passed to the model
(`sources_used` in the answer records, i.e. generator._select_evidence output) and, for
follow-ups, the citations in the prior assistant turn(s) of the benchmark history.

Citation classes (first matching rule wins):
  unresolved_title       the cited title maps to no course video (answer_eval.video_for_title)
  valid_evidence         the cited time lies inside a provided evidence chunk of that video, ±1 s
                         (for a range citation "@ a → b" both ends must lie inside provided chunks)
  valid_prior_answer     follow-up only: same video and time (±1 s) as a citation in the prior answer
  same_video_outside     the video is among the provided evidence, but no provided chunk covers the time
  video_not_in_evidence  a course video that was not among the provided evidence
This is citation/evidence INTEGRITY only: a valid citation points at text the model was given; it
says nothing about whether that text supports the sentence it is attached to.

Parsing reuses answer_eval's _CITATION_RE, ts_to_seconds and video_for_title, so citations are
the same ones answer_eval.parse_citations counts.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from answer_eval import _CITATION_RE, RESULTS_DIR, ts_to_seconds, video_for_title  # noqa: E402
from bench_data import DEFAULT_BENCHMARK_PATH, DEFAULT_MANIFEST_PATH, load_json  # noqa: E402
from compare_results import paired_bootstrap_ci  # noqa: E402

TOLERANCE_S = 1.0
RUNS = {
    "frozen_s26_guard": "s26_cand_t044_guard_dev",   # locked Stage 2.6 configuration
    "rerun_s3_control": "s3_control_dev",            # same configuration, re-run in Stage 3
    "no_rule8_t044": "s26_cand_t044_dev",            # same retrieval and threshold, no rule 8
}
CLASSES = ("valid_evidence", "valid_prior_answer", "same_video_outside", "video_not_in_evidence",
           "unresolved_title")
RANGE_RE = re.compile(r"\s*(?:→|->|–|-|to)\s*(\d{1,2}:\d{2}(?:\.\d)?)")
SENTENCE_MIN_WORDS = 4
PLACEHOLDER = "⁣CIT⁣"


# ── Pure functions (unit-tested) ──────────────────────────────────────────────

def extract_citations(answer: str, manifest: dict) -> list[dict]:
    """Citations as answer_eval parses them, plus the raw text and an optional range end."""
    out = []
    for m in _CITATION_RE.finditer(answer or ""):
        rng = RANGE_RE.match(answer, m.end())
        end_s = ts_to_seconds(rng.group(1)) if rng else None
        out.append({"raw": m.group(0) + (rng.group(0) if rng else ""), "title": m.group("title").strip(),
                    "time_s": ts_to_seconds(m.group("ts")), "end_s": end_s,
                    "video_id": video_for_title(m.group("title"), manifest)})
    return out


def _covering(video_id: str, t: float, chunks: Sequence[dict]) -> list[dict]:
    return [c for c in chunks if c["video_id"] == video_id
            and c["start_time"] - TOLERANCE_S <= t <= c["end_time"] + TOLERANCE_S]


def display_seconds(seconds: float, duration: float) -> float:
    """The time the model saw for a chunk boundary (retriever._fmt_timestamp display, in seconds)."""
    if duration < 5.0:
        return int(seconds) + int((seconds - int(seconds)) * 10) / 10
    return float(int(seconds))


def classify(cit: dict, evidence: Sequence[dict], prior: Sequence[dict]) -> dict:
    """Class of one citation plus where it lands inside the covering evidence chunk."""
    vid, t = cit["video_id"], cit["time_s"]
    if vid is None:
        return {"class": "unresolved_title", "position": None}
    hits = _covering(vid, t, evidence)
    end_ok = cit["end_s"] is None or bool(_covering(vid, cit["end_s"], evidence))
    if hits and end_ok:
        pos = "interior"
        for c in hits:
            dur = c["end_time"] - c["start_time"]
            if abs(t - display_seconds(c["start_time"], dur)) < 1e-6:
                pos = "exact_chunk_start"
                break
            if abs(t - display_seconds(c["end_time"], dur)) < 1e-6:
                pos = "exact_chunk_end"
        return {"class": "valid_evidence", "position": pos}
    if any(p["video_id"] == vid and abs(p["time_s"] - t) <= TOLERANCE_S for p in prior):
        return {"class": "valid_prior_answer", "position": None}
    if any(c["video_id"] == vid for c in evidence):
        return {"class": "same_video_outside", "position": None}
    return {"class": "video_not_in_evidence", "position": None}


def sentence_coverage(answer: str) -> dict:
    """
    HEURISTIC: split the answer into sentences (citations masked so '.mp4' does not split) and
    count sentences of >= SENTENCE_MIN_WORDS words with at least one citation. A sentence that
    starts with a citation hands it to the previous sentence ("Claim. [Video: …]").
    """
    masked = _CITATION_RE.sub(PLACEHOLDER, answer or "")
    masked = re.sub(PLACEHOLDER + RANGE_RE.pattern, PLACEHOLDER, masked)
    masked = re.sub(r"\[?" + PLACEHOLDER + r"\]?", PLACEHOLDER, masked)
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", masked) if p.strip()]
    sentences: list[dict] = []
    for p in parts:
        lead = p.startswith(PLACEHOLDER)
        text = p.replace(PLACEHOLDER, " ").strip(" .-*•")
        if lead and sentences and len(text.split()) < SENTENCE_MIN_WORDS:
            sentences[-1]["cited"] = True          # a citation standing alone after its sentence
            continue
        if lead and sentences:
            sentences[-1]["cited"] = True
            # the leading citation(s) belong to the previous sentence
            p = re.sub("^(?:" + re.escape(PLACEHOLDER) + r"\s*)+", "", p)
        sentences.append({"cited": PLACEHOLDER in p, "words": len(text.split())})
    factual = [s for s in sentences if s["words"] >= SENTENCE_MIN_WORDS]
    return {"sentences": len(factual), "cited": sum(s["cited"] for s in factual)}


# ── Audit ─────────────────────────────────────────────────────────────────────

def prior_citations(item: dict, manifest: dict) -> list[dict]:
    return [c for h in item["history"] if h["role"] == "assistant"
            for c in extract_citations(h["content"], manifest)]


def audit_records(records: Sequence[dict], bench: dict, manifest: dict) -> list[dict]:
    rows = []
    for r in records:
        if r["refused"]:
            continue
        item = bench[r["id"]]
        prior = prior_citations(item, manifest)
        cits = []
        for c in extract_citations(r["answer"], manifest):
            cits.append({**c, **classify(c, r["sources_used"], prior)})
        j = r.get("judge") or {}
        rows.append({
            "id": r["id"], "repeat": r["repeat"], "category": r["category"], "language": r["language"],
            "followup": r["followup"], "n_evidence": len(r["sources_used"]), "citations": cits,
            "coverage": sentence_coverage(r["answer"]), "prior_answer_citations": len(prior),
            "judge": {"hallucination": j.get("hallucination"), "grounding": j.get("grounding"),
                      "correctness": j.get("correctness"), "n_unsupported_claims": len(j.get("unsupported_claims") or []),
                      "unsupported_claims": j.get("unsupported_claims") or []},
        })
    return rows


def rate(n: int, d: int) -> Optional[float]:
    return round(n / d, 4) if d else None


def summarize(rows: Sequence[dict]) -> dict:
    cits = [c for r in rows for c in r["citations"]]
    cls = Counter(c["class"] for c in cits)
    pos = Counter(c["position"] for c in cits if c["class"] == "valid_evidence")
    invalid = {"same_video_outside", "video_not_in_evidence", "unresolved_title"}
    ans_invalid = [r for r in rows if any(c["class"] in invalid for c in r["citations"])]
    cov = [r["coverage"] for r in rows]
    return {
        "answered_records": len(rows),
        "records_without_citation": sum(not r["citations"] for r in rows),
        "total_citations": len(cits),
        "range_citations": sum(c["end_s"] is not None for c in cits),
        "by_class": {k: cls.get(k, 0) for k in CLASSES},
        "by_class_rate": {k: rate(cls.get(k, 0), len(cits)) for k in CLASSES},
        "valid_rate": rate(cls["valid_evidence"] + cls["valid_prior_answer"], len(cits)),
        "invalid_rate": rate(sum(cls[k] for k in invalid), len(cits)),
        "valid_evidence_position": {k: pos.get(k, 0) for k in ("exact_chunk_start", "exact_chunk_end", "interior")},
        "records_with_invalid_citation": len(ans_invalid),
        "records_with_invalid_citation_rate": rate(len(ans_invalid), len(rows)),
        "sentence_citation_coverage_HEURISTIC": rate(sum(c["cited"] for c in cov), sum(c["sentences"] for c in cov)),
        "records_fully_cited_HEURISTIC": rate(sum(c["cited"] == c["sentences"] for c in cov), len(cov)),
    }


def judge_relationship(rows: Sequence[dict]) -> dict:
    """Descriptive only: judge flags for in-scope answers with vs without an invalid citation."""
    invalid = {"same_video_outside", "video_not_in_evidence", "unresolved_title"}
    ins = [r for r in rows if r["category"] == "in_scope" and r["judge"]["grounding"] is not None]
    groups = {"with_invalid_citation": [r for r in ins if any(c["class"] in invalid for c in r["citations"])],
              "all_citations_valid": [r for r in ins if r["citations"]
                                      and not any(c["class"] in invalid for c in r["citations"])],
              "no_citation": [r for r in ins if not r["citations"]]}
    ts = re.compile(r"\d{1,2}:\d{2}|timestamp", re.I)
    out = {}
    for g, rs in groups.items():
        out[g] = {"n": len(rs),
                  "hallucination_rate": rate(sum(r["judge"]["hallucination"] is True for r in rs), len(rs)),
                  "not_fully_grounded_rate": rate(sum(r["judge"]["grounding"] != "fully_supported" for r in rs), len(rs)),
                  "has_unsupported_claims_rate": rate(sum(r["judge"]["n_unsupported_claims"] > 0 for r in rs), len(rs)),
                  "unsupported_claims_mentioning_timestamps": sum(
                      bool(ts.search(c)) for r in rs for c in r["judge"]["unsupported_claims"])}
    return out


def per_query_invalid(rows: Sequence[dict]) -> dict[str, float]:
    """Share of a query's answered repeats that contain an invalid citation."""
    invalid = {"same_video_outside", "video_not_in_evidence", "unresolved_title"}
    acc: dict[str, list[float]] = {}
    for r in rows:
        acc.setdefault(r["id"], []).append(float(any(c["class"] in invalid for c in r["citations"])))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def paired(rows_a: Sequence[dict], rows_b: Sequence[dict]) -> dict:
    a, b = per_query_invalid(rows_a), per_query_invalid(rows_b)
    ids = sorted(set(a) & set(b))
    d = [b[i] - a[i] for i in ids]
    ci = paired_bootstrap_ci(d)
    return {"n_queries_answered_in_both": len(ids), "delta_records_with_invalid_citation": round(sum(d) / len(d), 4),
            "ci95": [round(ci[0], 4), round(ci[1], 4)]}


def main() -> int:
    bench = {q["id"]: q for q in load_json(DEFAULT_BENCHMARK_PATH)["queries"]}
    manifest = load_json(DEFAULT_MANIFEST_PATH)
    rows = {}
    for name, label in RUNS.items():
        data = load_json(RESULTS_DIR / f"answers_{label}.json")
        if data["split"] != "dev" or any(r["split"] != "dev" for r in data["records"]):
            raise SystemExit(f"{label}: not DEV-only")
        rows[name] = audit_records(data["records"], bench, manifest)
    dev_followups_with_cited_prior = sorted(q["id"] for q in bench.values() if q["split"] == "dev"
                                            and q["followup"] and prior_citations(q, manifest))
    invalid = {"same_video_outside", "video_not_in_evidence", "unresolved_title"}
    examples = [{"run": n, "id": r["id"], "repeat": r["repeat"], "citation": c["raw"], "class": c["class"],
                 "cited_time_s": c["time_s"], "cited_video": c["video_id"]}
                for n in ("frozen_s26_guard", "rerun_s3_control") for r in rows[n] for c in r["citations"]
                if c["class"] in invalid]
    out = {
        "stage": "4", "experiment": "Z1 citation integrity audit (offline, no API)",
        "inputs": {n: f"eval/results/answers_{l}.json" for n, l in RUNS.items()},
        "tolerance_s": TOLERANCE_S,
        "note": "Citation integrity only; a valid citation is not evidence that the cited text supports the claim.",
        "dev_followups_whose_prior_answer_has_citations": dev_followups_with_cited_prior,
        "summary": {n: summarize(r) for n, r in rows.items()},
        "summary_by_language": {n: {lang: summarize([x for x in r if x["language"] == lang])
                                    for lang in ("en", "hinglish")} for n, r in rows.items()},
        "summary_followup": {n: summarize([x for x in r if x["followup"]]) for n, r in rows.items()},
        "summary_oos_answered": {n: summarize([x for x in r if x["category"] != "in_scope"]) for n, r in rows.items()},
        "judge_relationship_descriptive": {n: judge_relationship(r) for n, r in rows.items()},
        "paired": {"rule8_vs_no_rule8 (no_rule8 minus frozen)": paired(rows["frozen_s26_guard"], rows["no_rule8_t044"]),
                   "rerun_vs_frozen (rerun minus frozen)": paired(rows["frozen_s26_guard"], rows["rerun_s3_control"])},
        "invalid_citation_examples": examples,
        "records": rows,
    }
    path = HERE / "z1_citation_integrity_audit.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    for n, s in out["summary"].items():
        print(n, json.dumps({k: s[k] for k in ("total_citations", "by_class", "valid_rate", "invalid_rate",
                                                "valid_evidence_position", "records_with_invalid_citation_rate",
                                                "sentence_citation_coverage_HEURISTIC")}))
    print("wrote", path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
