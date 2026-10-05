"""
eval/answer_eval.py
────────────────────
Stage 2.5: downstream answer-quality evaluation of the full EduVision pipeline
(retrieval → GPT-4o-mini generation) on benchmark v1.1.

  # 1. generate answers with the unchanged production pipeline.ask()
  #    (point it at an experimental index with CHROMA_DB_PATH / ACTIVE_COLLECTION, as run_benchmark.py does)
  venv/bin/python eval/answer_eval.py generate --label control --split dev --repeats 2
  # 2. grade them with a separate, pinned judge model
  venv/bin/python eval/answer_eval.py judge --label control
  # 3. summarise one or more labels
  venv/bin/python eval/answer_eval.py summarize control candidate --split dev

Deterministic measures (no judge):
  refusal         in-scope refused (false refusal) / out-of-scope answered (false answer);
                  "refused" = the no-LLM gate fired or the answer contains "could not find"
  evidence        evidence_has_gold: some chunk passed to the LLM overlaps a gold span;
                  evidence_precision: share of those chunks that overlap a gold span
  citations       [Video: "<title>" @ MM:SS] parsed from the answer; citation_hit = a citation in
                  a gold video within [gold start − 5 s, gold end + 5 s]; citation_start_error_s
                  = |cited time − gold start| for the closest citation in a gold video
Judge measures (JUDGE_MODEL, temperature 0, JSON):
  correctness     correct / partially_correct / incorrect, against the REFERENCE: the independent
                  Stage 1.5 annotation transcript (Whisper medium, eval/benchmark/annotation_asr/)
                  of the gold spans — text from neither system under test
  grounding       fully_supported / partially_supported / unsupported by the EVIDENCE given
  hallucination   a substantive claim supported by neither EVIDENCE nor REFERENCE
  oos_behavior    for out-of-scope items
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

_EVAL_DIR = Path(__file__).resolve().parent
_ROOT = _EVAL_DIR.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_EVAL_DIR))

from bench_data import DEFAULT_BENCHMARK_PATH, DEFAULT_MANIFEST_PATH, load_json, manifest_index  # noqa: E402

RESULTS_DIR = _EVAL_DIR / "results"
ANNOTATION_ASR = _EVAL_DIR / "benchmark" / "annotation_asr"
JUDGE_MODEL = "gpt-4.1-2025-04-14"
CITATION_TOLERANCE_S = 5.0
REFERENCE_MAX_CHARS = 3000
REFUSAL_MARKER = "could not find"

_CITATION_RE = re.compile(
    r"\[Video:\s*\"?(?P<title>[^\"\]@]+?)\"?\s*@\s*(?P<ts>\d{1,2}:\d{2}(?:\.\d)?)")


# ── Deterministic helpers (unit-tested) ───────────────────────────────────────

def ts_to_seconds(ts: str) -> float:
    m, s = ts.split(":")
    return int(m) * 60 + float(s)


def video_for_title(title: str, manifest: dict) -> Optional[str]:
    """Map a cited video title to a video_id via its tutorial number."""
    for pattern in (r"#\s*(\d{1,2})\b", r"^\s*(\d{1,2})[_ ]", r"[Tt]utorial\s*(\d{1,2})\b"):
        m = re.search(pattern, title)
        if m:
            n = int(m.group(1))
            for v in manifest["videos"]:
                if v["tutorial_number"] == n:
                    return v["video_id"]
    low = title.strip().lower()
    for v in manifest["videos"]:
        if low and low[:25] in v["video_filename"].lower():
            return v["video_id"]
    return None


def parse_citations(answer: str, manifest: dict) -> list[dict]:
    out = []
    for m in _CITATION_RE.finditer(answer or ""):
        out.append({"title": m.group("title").strip(), "time_s": ts_to_seconds(m.group("ts")),
                    "video_id": video_for_title(m.group("title"), manifest)})
    return out


def citation_metrics(citations: list[dict], gold: list[dict]) -> dict:
    gold_videos = {g["video_id"] for g in gold}
    hit = any(c["video_id"] == g["video_id"]
              and g["start_time"] - CITATION_TOLERANCE_S <= c["time_s"] <= g["end_time"] + CITATION_TOLERANCE_S
              for c in citations for g in gold)
    errs = [abs(c["time_s"] - g["start_time"]) for c in citations for g in gold if c["video_id"] == g["video_id"]]
    return {"n_citations": len(citations),
            "citation_in_gold_video": any(c["video_id"] in gold_videos for c in citations),
            "citation_hit": hit,
            "citation_start_error_s": round(min(errs), 2) if errs else None}


def evidence_metrics(sources: list[dict], gold: list[dict]) -> dict:
    def overlaps(s):
        return any(s["video_id"] == g["video_id"] and min(s["end_time"], g["end_time"]) > max(s["start_time"], g["start_time"])
                   for g in gold)
    flags = [overlaps(s) for s in sources]
    return {"n_evidence": len(sources), "evidence_has_gold": any(flags),
            "evidence_precision": round(sum(flags) / len(flags), 4) if flags else None}


def is_refusal(rec: dict) -> bool:
    return bool(rec["not_found"]) or REFUSAL_MARKER in (rec["answer"] or "").lower()


# ── generate ──────────────────────────────────────────────────────────────────

def cmd_generate(args) -> None:
    import hashlib
    import config.settings as s
    import generation.generator as generator
    from pipeline import ask

    # Experiment knobs, applied at runtime only (production files are not edited):
    #   --threshold          passed to ask(similarity_threshold=...) — retrieval gate + evidence filter
    #   --system-prompt-file replaces generation.generator.SYSTEM_PROMPT for this process
    threshold = args.threshold if args.threshold is not None else s.SIMILARITY_THRESHOLD
    if args.system_prompt_file:
        generator.SYSTEM_PROMPT = Path(args.system_prompt_file).read_text()
    prompt_sha = hashlib.sha256(generator.SYSTEM_PROMPT.encode()).hexdigest()

    bench = load_json(DEFAULT_BENCHMARK_PATH)
    manifest = load_json(DEFAULT_MANIFEST_PATH)
    items = [q for q in bench["queries"] if args.split == "all" or q["split"] == args.split]
    records = []
    for rep in range(args.repeats):
        for n, q in enumerate(items, 1):
            history = [{"role": h["role"], "content": h["content"], "result": None} for h in q["history"]] or None
            t0 = time.time()
            r = ask(q["query"], chat_history=history, similarity_threshold=threshold)
            sources = [{"chunk_id": x.chunk_id, "video_id": x.video_id, "video_filename": x.video_filename,
                        "start_time": x.start_time, "end_time": x.end_time, "similarity": round(x.similarity, 4),
                        "text_en": x.text_en} for x in r.sources_used]
            rec = {"id": q["id"], "repeat": rep, "split": q["split"], "category": q["category"],
                   "language": q["language"], "followup": q["followup"], "query": q["query"],
                   "answer": r.answer, "not_found": r.not_found, "error": r.error, "query_valid": r.query_valid,
                   "sources_used": sources, "top1_similarity": round(r.retrieval_stats.top_similarity, 4)
                   if r.retrieval_stats else None, "prompt_tokens": r.prompt_tokens,
                   "completion_tokens": r.completion_tokens, "latency_s": round(time.time() - t0, 2)}
            rec["refused"] = is_refusal(rec)
            rec["citations"] = parse_citations(r.answer, manifest)
            if q["gold"]:
                rec.update(evidence_metrics(sources, q["gold"]))
                rec.update(citation_metrics(rec["citations"], q["gold"]))
            records.append(rec)
            if n % 20 == 0:
                print(f"  repeat {rep}: {n}/{len(items)}", flush=True)
    out = {"label": args.label, "split": args.split, "repeats": args.repeats,
           "benchmark": bench["benchmark_version"],
           "system": {"chroma_db_path": s.CHROMA_DB_PATH, "active_collection": s.ACTIVE_COLLECTION,
                      "generator_model": s.OPENAI_MODEL, "temperature": s.TEMPERATURE,
                      "max_llm_evidence": s.MAX_LLM_EVIDENCE, "similarity_threshold": threshold,
                      "system_prompt_file": args.system_prompt_file, "system_prompt_sha256": prompt_sha},
           "records": records}
    path = RESULTS_DIR / f"answers_{args.label}.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"Wrote {path.relative_to(_ROOT)} ({len(records)} answers)")


# ── judge ─────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """You grade answers from a question-answering assistant for a Hindi/English web-development video course.
The assistant must answer ONLY from the transcript EVIDENCE it was given, cite timestamps, and say
"I could not find this topic in the provided course material." when the evidence does not answer the question.

You receive: the QUESTION (and prior conversation for follow-ups), whether the question is IN_SCOPE of the course,
a REFERENCE transcript of the video passage(s) that answer it (an independent machine transcript, English
translation, may contain recognition errors; absent for out-of-scope questions), the EVIDENCE excerpts the
assistant saw, and the ANSWER.

Return a JSON object with exactly these keys:
- "refused": true if the answer declines / says it could not find the topic, else false.
- "correctness": for an IN_SCOPE question that was answered: "correct" (answers the question consistently with
  the REFERENCE, no material error), "partially_correct" (relevant and partly right but incomplete, vague, or with
  a minor error), or "incorrect" (wrong, contradicts the REFERENCE, or does not answer the question).
  Use "n/a" if refused or out of scope.
- "grounding": "fully_supported", "partially_supported" or "unsupported": are the answer's substantive claims
  supported by the EVIDENCE (or by prior conversation)? Use "n/a" if refused.
- "unsupported_claims": list of short strings, the substantive claims not supported by the EVIDENCE.
- "hallucination": true if the answer makes a substantive claim supported by neither the EVIDENCE nor the
  REFERENCE (for out-of-scope questions: answering from general knowledge counts).
- "oos_behavior": for out-of-scope questions: "appropriate_refusal", "answered_from_general_knowledge",
  or "answered_with_tangential_course_content"; otherwise "n/a".
- "notes": one short sentence.
Judge substance, not style or language. Ignore citation formatting."""


def reference_text(q: dict) -> str:
    parts = []
    for g in q["gold"]:
        path = ANNOTATION_ASR / f"{g['video_id']}.medium.json"
        segs = load_json(path)["segments"]
        text = " ".join(s["text"] for s in segs if s["end"] > g["start_time"] - 2 and s["start"] < g["end_time"] + 2)
        parts.append(f"[Tutorial {int(g['video_id'][:2])}, {g['start_time']:.0f}-{g['end_time']:.0f}s] {text}")
    return "\n".join(parts)[:REFERENCE_MAX_CHARS]


def cmd_judge(args) -> None:
    from generation.generator import _get_client
    client = _get_client()
    path = RESULTS_DIR / f"answers_{args.label}.json"
    data = load_json(path)
    bench = {q["id"]: q for q in load_json(DEFAULT_BENCHMARK_PATH)["queries"]}
    for n, rec in enumerate(data["records"], 1):
        if rec.get("judge") and not args.force:
            continue
        q = bench[rec["id"]]
        history = "\n".join(f"{h['role']}: {h['content']}" for h in q["history"]) or "(none)"
        evidence = "\n".join(f"[{i}] {s['video_filename'][:40]} {s['start_time']:.0f}-{s['end_time']:.0f}s: {s['text_en']}"
                             for i, s in enumerate(rec["sources_used"], 1)) or "(no evidence: the assistant refused before calling the LLM)"
        user = (f"QUESTION: {q['query']}\nPRIOR CONVERSATION:\n{history}\nIN_SCOPE: {q['category'] == 'in_scope'}\n"
                f"REFERENCE:\n{reference_text(q) if q['gold'] else '(out of scope: no reference)'}\n"
                f"EVIDENCE:\n{evidence}\nANSWER:\n{rec['answer'] or '(empty)'}")
        resp = client.chat.completions.create(
            model=args.judge_model, temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}])
        rec["judge"] = json.loads(resp.choices[0].message.content)
        rec["judge_model"] = args.judge_model
        if n % 25 == 0:
            print(f"  judged {n}/{len(data['records'])}", flush=True)
            path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(f"Judged {len(data['records'])} answers → {path.relative_to(_ROOT)}")


# ── summarize ─────────────────────────────────────────────────────────────────

def summarize(records: list[dict]) -> dict:
    ins = [r for r in records if r["category"] == "in_scope"]
    oos = [r for r in records if r["category"] != "in_scope"]
    answered = [r for r in ins if not r["refused"]]
    def frac(rs, fn):
        return round(sum(1 for r in rs if fn(r)) / len(rs), 3) if rs else None
    j = lambda r, k: (r.get("judge") or {}).get(k)
    errs = sorted(r["citation_start_error_s"] for r in answered if r.get("citation_start_error_s") is not None)
    return {
        "n_in_scope_answers": len(ins), "n_oos_answers": len(oos),
        "false_refusal_rate": frac(ins, lambda r: r["refused"]),
        "false_answer_rate": frac(oos, lambda r: not r["refused"]),
        "correct": frac(ins, lambda r: j(r, "correctness") == "correct"),
        "correct_or_partial": frac(ins, lambda r: j(r, "correctness") in ("correct", "partially_correct")),
        "incorrect": frac(ins, lambda r: j(r, "correctness") == "incorrect"),
        "answered_fully_grounded": frac(answered, lambda r: j(r, "grounding") == "fully_supported"),
        "hallucination_rate_in_scope": frac(ins, lambda r: j(r, "hallucination") is True),
        "hallucination_rate_oos": frac(oos, lambda r: j(r, "hallucination") is True),
        "evidence_has_gold": frac(ins, lambda r: r.get("evidence_has_gold")),
        "mean_evidence_precision": round(sum(r["evidence_precision"] for r in ins if r.get("evidence_precision") is not None)
                                         / max(1, sum(1 for r in ins if r.get("evidence_precision") is not None)), 3),
        "answered_with_citation": frac(answered, lambda r: r.get("n_citations", 0) > 0),
        "answered_citation_hit": frac(answered, lambda r: r.get("citation_hit")),
        "citation_start_error_median_s": errs[len(errs) // 2] if errs else None,
        "oos_behavior": dict(Counter(j(r, "oos_behavior") for r in oos)),
    }


def cmd_summarize(args) -> None:
    out = {}
    for label in args.labels:
        data = load_json(RESULTS_DIR / f"answers_{label}.json")
        recs = [r for r in data["records"] if args.split == "all" or r["split"] == args.split]
        out[label] = {"all": summarize(recs),
                      "by_language": {lang: summarize([r for r in recs if r["language"] == lang]) for lang in ("en", "hinglish")},
                      "followup": summarize([r for r in recs if r["followup"]])}
    keys = list(next(iter(out.values()))["all"])
    print(f"{'metric':34}" + "".join(f"{l[:22]:>24}" for l in out))
    for k in keys:
        print(f"{k:34}" + "".join(f"{str(out[l]['all'][k]):>24}" for l in out))
    for lang in ("en", "hinglish"):
        print(f"\n[{lang}]")
        for k in ("n_in_scope_answers", "false_refusal_rate", "correct", "correct_or_partial", "hallucination_rate_in_scope",
                  "evidence_has_gold", "answered_citation_hit"):
            print(f"  {k:32}" + "".join(f"{str(out[l]['by_language'][lang][k]):>24}" for l in out))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nWrote {args.out}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--label", required=True)
    g.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    g.add_argument("--repeats", type=int, default=1)
    g.add_argument("--threshold", type=float, help="override SIMILARITY_THRESHOLD for this run")
    g.add_argument("--system-prompt-file", help="replace the generator's SYSTEM_PROMPT for this run")
    j = sub.add_parser("judge")
    j.add_argument("--label", required=True)
    j.add_argument("--judge-model", default=JUDGE_MODEL)
    j.add_argument("--force", action="store_true")
    s = sub.add_parser("summarize")
    s.add_argument("labels", nargs="+")
    s.add_argument("--split", default="all", choices=["dev", "test", "all"])
    s.add_argument("--out")
    args = p.parse_args()
    {"generate": cmd_generate, "judge": cmd_judge, "summarize": cmd_summarize}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
