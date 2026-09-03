"""
eval/eval_followup.py
──────────────────────
Stage 16 (regression): Two-turn conversation/follow-up evaluation.

Tests the full contextual follow-up pipeline end-to-end:
  1. Turn 1  — a single, self-contained first question (establishes context)
  2. History — built exactly as app/ui.py builds it (role/content dicts)
  3. Turn 2  — an ambiguous follow-up passed to ask(..., chat_history=...)
  4. Assert  — specific, documented pass conditions per case

What is tested:
  CF1  Pronoun follow-up        "what is html?" → "why is it important?"
  CF2  Timestamp follow-up      "how do I install VS Code?" → "what is the exact timestamp?"
  CF3  Location follow-up       "what is the Live Preview extension?" → "where is it installed?"
  CF4  Self-contained no-op     "what is HTML?" → "what is CSS?" (no pronoun — rewrite must not corrupt)
  CF5  Out-of-scope boundary    "what is Python?" → "why is it useful?" (must refuse both turns)

What this file does NOT test:
  - Single-turn retrieval quality (that is evaluate.py's job)
  - Exact wording of answers (too brittle; the LLM is stochastic)
  - Ingestion, normalizer, or embedding correctness

Run:
  cd <project_root>
  source venv/bin/activate
  python eval/eval_followup.py

Requirements:
  - OPENAI_API_KEY set in .env (needed for answer generation and query rewriting)
  - ChromaDB v2 index built (run ingestion/indexer_v2.py first)
  - Internet access for the OpenAI API calls
"""

import sys
import logging
import time
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.WARNING)

from pipeline import ask  # noqa: E402 — must follow sys.path setup


# ─────────────────────────────────────────────────────────────────────────────
# TEST CASE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
#
# Each entry is a dict with:
#   id                   — unique identifier, prefix CF (Conversational Follow-up)
#   description          — one-line summary of what is being tested
#   turn1_query          — first question; establishes conversation context
#   turn1_expect_found   — True if turn 1 should produce a grounded answer
#   turn2_query          — ambiguous follow-up; uses pronouns / references turn 1
#   turn2_expect_found   — True if turn 2 should produce a grounded answer
#   turn2_require_keywords — any of these strings must appear in the answer (case-insensitive)
#   turn2_expect_video   — if set, at least one source must cite this video
#   notes                — human-readable explanation of expected behaviour

FOLLOWUP_TESTS = [
    {
        "id":          "CF1",
        "description": "Pronoun follow-up: 'it' must resolve to HTML",
        "turn1_query": "what is html?",
        "turn1_expect_found": True,
        "turn2_query": "why is it important?",
        "turn2_expect_found": True,
        "turn2_require_keywords": ["html"],
        "turn2_expect_video":    ["10_Video", "11_Semantic", "13_"],
        "notes": (
            "The query rewriter should resolve 'it' to HTML. "
            "The generator should either retrieve evidence about HTML importance "
            "or use the prior turn's HTML explanation as context. "
            "Returning not_found here is the regression this test guards against."
        ),
    },
    {
        "id":          "CF2",
        "description": "Timestamp follow-up: should cite a specific time for VS Code install",
        "turn1_query": "how do I install VS Code?",
        "turn1_expect_found": True,
        "turn2_query": "what is the exact timestamp in the video?",
        "turn2_expect_found": True,
        "turn2_require_keywords": [],   # timestamp format varies; check sources instead
        "turn2_expect_video":    "Tutorial #1",
        "notes": (
            "'what is the exact timestamp?' must be resolved to the VS Code install context. "
            "The response must not be not_found and must cite Tutorial #1 sources."
        ),
    },
    {
        "id":          "CF3",
        "description": "Location follow-up: 'where is it installed' must resolve to Live Preview",
        "turn1_query": "what is the Live Preview extension?",
        "turn1_expect_found": True,
        "turn2_query": "where is it installed?",
        "turn2_expect_found": True,
        "turn2_require_keywords": ["preview", "extension"],
        "turn2_expect_video":    "Tutorial #2",
        "notes": (
            "'it' should resolve to Live Preview extension. "
            "The answer should describe where in VS Code / Tutorial #2 it is installed."
        ),
    },
    {
        "id":          "CF4",
        "description": "Self-contained follow-up: no pronoun, rewrite must not corrupt the query",
        "turn1_query": "what is HTML?",
        "turn1_expect_found": True,
        "turn2_query": "what is CSS?",
        "turn2_expect_found": True,
        "turn2_require_keywords": ["css"],
        "turn2_expect_video":    ["14_Introduction to CSS", "15_Inline", "17_CSS"],
        "notes": (
            "'what is CSS?' is already self-contained — no pronouns. "
            "The rewriter should return it unchanged. "
            "The answer must be about CSS, not HTML."
        ),
    },
    {
        "id":          "CF5",
        "description": "Out-of-scope boundary: follow-up of a not-found topic must also refuse",
        "turn1_query": "what is Python?",
        "turn1_expect_found": False,   # Python is not in the 2-video corpus
        "turn2_query": "why is it useful?",
        "turn2_expect_found": False,   # must still refuse — no Python content in index
        "turn2_require_keywords": [],
        "turn2_expect_video":    None,
        "notes": (
            "Turn 1 returns not_found (Python not in corpus). "
            "Turn 2 'why is it useful?' with that context should NOT produce a grounded answer. "
            "The system must not hallucinate or use general knowledge about Python."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_history(turn1_query: str, turn1_answer: str) -> list:
    """
    Build a chat_history list exactly as app/ui.py does before calling ask().

    Structure mirrors the session_state.chat_history entries:
      {"role": "user"|"assistant", "content": str, "result": None}
    """
    return [
        {"role": "user",      "content": turn1_query,  "result": None},
        {"role": "assistant", "content": turn1_answer,  "result": None},
    ]


def _video_matches(expected, sources: list) -> bool:
    """
    Return True if any source filename matches the expected video label.

    expected may be:
      str  — a single label ('Tutorial #1', 'Tutorial #2', or a filename fragment)
      list — multiple acceptable labels; passes if ANY keyword matches ANY source
    """
    keywords = expected if isinstance(expected, list) else [expected]
    for s in sources:
        fn = s.video_filename.lower()
        for kw in keywords:
            kl = kw.lower()
            if kl in fn:
                return True
            if kw == "Tutorial #1" and ("01_" in fn or "tutorial_1" in fn):
                return True
            if kw == "Tutorial #2" and ("02_" in fn or "tutorial_2" in fn):
                return True
    return False


def _score_turn2(tc: dict, r2) -> tuple:
    """
    Return (grade, reason) for the follow-up (turn 2) result.
    grade in {'PASS', 'FAIL'}
    """
    expect_found = tc["turn2_expect_found"]

    # ── Out-of-scope boundary (CF5) ───────────────────────────────────────────
    if not expect_found:
        if r2.not_found or (r2.success and "could not find" in r2.answer.lower()):
            return "PASS", "Correctly refused follow-up of out-of-scope topic"
        return "FAIL", (
            f"Expected not_found but got grounded answer: {r2.answer[:120]!r}"
        )

    # ── In-scope cases: must produce a grounded answer ────────────────────────
    if r2.not_found:
        return "FAIL", (
            "Returned not_found — conversation context was not used to resolve "
            "the follow-up query. REGRESSION: contextual retrieval is broken."
        )
    if r2.error:
        return "FAIL", f"Pipeline error: {r2.error}"
    if not r2.success:
        return "FAIL", f"No answer produced (query_valid={r2.query_valid})"

    # Keyword check
    answer_lower = r2.answer.lower()
    for kw in tc.get("turn2_require_keywords", []):
        if kw.lower() not in answer_lower:
            return "FAIL", (
                f"Required keyword {kw!r} missing from answer: {r2.answer[:120]!r}"
            )

    # Expected video check
    expected_video = tc.get("turn2_expect_video")
    if expected_video and r2.sources_used:
        if not _video_matches(expected_video, r2.sources_used):
            vids = [s.video_filename[:35] for s in r2.sources_used[:2]]
            return "FAIL", (
                f"Expected source from {expected_video!r} but got: {vids}"
            )

    top_sim = r2.retrieval_stats.top_similarity if r2.retrieval_stats else 0.0
    return "PASS", (
        f"Grounded answer, {len(r2.sources_used)} sources, top_sim={top_sim:.2f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_followup_evaluation():
    print("=" * 70)
    print("EduVision RAG — Conversational Follow-up Evaluation")
    print("=" * 70)
    print(f"  Test cases : {len(FOLLOWUP_TESTS)}")
    print(f"  Each case runs 2 ask() calls: turn1 -> build history -> turn2")
    print()

    results = []
    pass_count = 0
    fail_count = 0

    for tc in FOLLOWUP_TESTS:
        cid  = tc["id"]
        desc = tc["description"]
        print(f"--- [{cid}] {desc}")
        print(f"    Turn 1 : {tc['turn1_query']!r}")
        print(f"    Turn 2 : {tc['turn2_query']!r}")
        print(f"    Notes  : {tc['notes']}")

        # ── Turn 1 ────────────────────────────────────────────────────────────
        t1_start = time.time()
        r1 = ask(tc["turn1_query"])
        t1_elapsed = time.time() - t1_start

        # Determine turn 1 answer text (mirrors ui.py history append logic)
        if r1.success:
            t1_answer = r1.answer
            t1_ok = tc["turn1_expect_found"]
            print(f"    T1 ans : {t1_answer[:120]}{'...' if len(t1_answer) > 120 else ''}")
        elif r1.not_found:
            t1_answer = "I could not find this topic in the provided course material."
            t1_ok = not tc["turn1_expect_found"]
            print(f"    T1 ans : [NOT FOUND — expected_found={tc['turn1_expect_found']}]")
        else:
            t1_answer = r1.validation_error or r1.error or ""
            t1_ok = False
            print(f"    T1 ans : [ERROR: {t1_answer}]")

        print(f"    T1 stat: {t1_elapsed:.2f}s  tokens: {r1.total_tokens}")

        if not t1_ok:
            reason = (
                f"Turn 1 prerequisite failed: "
                f"success={r1.success} not_found={r1.not_found} "
                f"(expected_found={tc['turn1_expect_found']})"
            )
            print(f"    Grade  : FAIL — {reason}")
            print()
            fail_count += 1
            results.append({"id": cid, "grade": "FAIL", "reason": reason,
                             "t1_latency": round(t1_elapsed, 2), "t2_latency": 0})
            continue

        # ── Build history (exactly as ui.py does) ─────────────────────────────
        history = _build_history(tc["turn1_query"], t1_answer)

        # ── Turn 2 ────────────────────────────────────────────────────────────
        t2_start = time.time()
        r2 = ask(tc["turn2_query"], chat_history=history)
        t2_elapsed = time.time() - t2_start

        if r2.success:
            print(f"    T2 ans : {r2.answer[:160]}{'...' if len(r2.answer) > 160 else ''}")
        elif r2.not_found:
            print(f"    T2 ans : [NOT FOUND]")
        else:
            print(f"    T2 ans : [ERROR: {r2.error}]")

        if r2.retrieval_stats:
            rs = r2.retrieval_stats
            print(f"    T2 ret : {rs.total_results} chunks, "
                  f"{rs.above_threshold} above threshold, top_sim={rs.top_similarity:.3f}")

        if r2.sources_used:
            for s in r2.sources_used[:2]:
                vid = "T1" if "01_" in s.video_filename else "T2"
                print(f"      [{vid} {s.start_time_fmt}->{s.end_time_fmt}] "
                      f"sim={s.similarity:.3f}  {s.text[:55]}...")

        print(f"    T2 stat: {t2_elapsed:.2f}s  tokens: {r2.total_tokens}")

        grade, reason = _score_turn2(tc, r2)
        grade_icon = "PASS" if grade == "PASS" else "FAIL"
        print(f"    Grade  : {grade_icon}  -- {reason}")
        print()

        if grade == "PASS":
            pass_count += 1
        else:
            fail_count += 1

        results.append({
            "id":         cid,
            "grade":      grade,
            "reason":     reason,
            "t1_latency": round(t1_elapsed, 2),
            "t2_latency": round(t2_elapsed, 2),
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(FOLLOWUP_TESTS)
    print("=" * 70)
    print("FOLLOW-UP EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Total cases : {total}")
    print(f"  PASS        : {pass_count} ({pass_count / total * 100:.0f}%)")
    print(f"  FAIL        : {fail_count} ({fail_count / total * 100:.0f}%)")
    print()
    print(f"  {'ID':<6} {'Grade':<5} Reason")
    print(f"  {'='*6} {'='*5} {'='*50}")
    for r in results:
        print(f"  {r['id']:<6} {r['grade']:<5} {r['reason'][:55]}")
    print()
    verdict = (
        "ALL FOLLOW-UP TESTS PASS"
        if fail_count == 0
        else f"{fail_count} FOLLOW-UP TEST(S) FAILED"
    )
    print(f"  OVERALL: {verdict}")
    print("=" * 70)

    return results, {"PASS": pass_count, "FAIL": fail_count}


if __name__ == "__main__":
    run_followup_evaluation()
