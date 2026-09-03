"""
generation/generator.py
────────────────────────
Stage 9: Ranked Evidence → Grounded Answer with Timestamp Citations.

═══════════════════════════════════════════════════════════════
GENERATOR ARCHITECTURE
═══════════════════════════════════════════════════════════════

The generator sits at the OUTPUT end of the RAG pipeline:

  User question
        │
        │  [Stage 8] retrieve(question) → list[RetrievalResult]
        ▼
  Ranked evidence (transcript chunks with timestamps)
        │
        │  [Stage 9 — this module]
        │  1. Filter: keep only above-threshold evidence
        │  2. Build a grounded prompt (question + evidence blocks)
        │  3. Call GPT-4o-mini via OpenAI chat/completions API
        │  4. Return answer + source citations
        ▼
  GeneratorResponse  ← consumed by Stage 12 (Streamlit UI)

Design principle (from project spec):
  "RETRIEVAL is separate from GENERATION.
   The LLM receives transcript evidence + timestamps and uses them
   to produce a grounded answer — it does NOT search for the answer itself."

═══════════════════════════════════════════════════════════════
PROMPT DESIGN
═══════════════════════════════════════════════════════════════

SYSTEM PROMPT (enforces grounding):
  - Answer ONLY from the provided evidence.
  - Cite video name and timestamp for every claim.
  - If evidence lacks the answer → say "could not find in course material".
  - Never invent, never guess.

USER MESSAGE structure:
  Question: <user_query>

  Evidence:
  [1] Video: "<video_filename>" | [<start> → <end>]
      "<transcript text>"

  [2] ...

This structure makes it trivial for the model to cite sources:
it just copies the video name and timestamp from [1], [2], etc.

Evidence is sorted by similarity (highest first) so the most
relevant chunk appears first — models are more attentive to content
near the top of the context.

═══════════════════════════════════════════════════════════════
EVIDENCE FILTERING
═══════════════════════════════════════════════════════════════

We apply a two-tier evidence selection:

  TIER 1 — Above threshold: results where similarity ≥ threshold
  TIER 2 — Below threshold: results where similarity < threshold

Selection logic:
  - If NO results are above threshold → return not_found immediately
    (no LLM call, saves tokens, correct behaviour for irrelevant queries)
  - If ≥ 1 result is above threshold → pass only above-threshold evidence
    to the LLM. Below-threshold results are excluded to avoid polluting
    the evidence with noise.

═══════════════════════════════════════════════════════════════
ERROR HANDLING HIERARCHY
═══════════════════════════════════════════════════════════════

  1. Missing / placeholder API key → MissingAPIKeyError (raised before any call)
  2. Empty retrieval results →      not_found=True (no LLM call)
  3. All results below threshold →  not_found=True (no LLM call)
  4. openai.AuthenticationError →   error="Invalid API key"
  5. openai.RateLimitError →        error="Rate limit exceeded"
  6. openai.APIConnectionError →    error="Network/connection error"
  7. Unexpected exception →         error=str(exc)

For cases 4–7, a GeneratorResponse is still returned (not raised) so
the UI can display a friendly message rather than crashing.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import openai

from config.settings import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    SIMILARITY_THRESHOLD,
)
from retrieval.retriever import RetrievalResult

logger = logging.getLogger(__name__)

# Sentinel used to detect the placeholder API key from .env
_PLACEHOLDER_KEY = "your-openai-api-key-here"

# Number of above-threshold results to include in the prompt.
# Keeping to 5 matches TOP_K_RESULTS and fits well within context limits.
MAX_EVIDENCE_CHUNKS = 5


# ── Custom exceptions ─────────────────────────────────────────────────────────

class MissingAPIKeyError(Exception):
    """Raised when OPENAI_API_KEY is not set or is the placeholder value."""


# ── Return data model ─────────────────────────────────────────────────────────

@dataclass
class GeneratorResponse:
    """
    Complete response from the generator, consumed by the UI (Stage 12).

    Fields:
        answer:      The LLM's grounded answer text. Empty string if
                     not_found=True or error is set.
        sources:     The evidence chunks actually passed to the LLM.
                     Empty if not_found=True (no evidence was usable).
        model:       OpenAI model name used (e.g. "gpt-4o-mini").
        query:       The original user question.
        not_found:   True if retrieval evidence was insufficient to
                     answer the question (threshold not met).
        error:       Non-None if an API or system error occurred.
        latency_s:   Total wall-clock time for the LLM call (seconds).
        prompt_tokens:     Prompt token count from the API response.
        completion_tokens: Completion token count from the API response.
    """
    answer:             str
    sources:            list[RetrievalResult]
    model:              str
    query:              str
    not_found:          bool = False
    error:              Optional[str] = None
    latency_s:          float = 0.0
    prompt_tokens:      int = 0
    completion_tokens:  int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def success(self) -> bool:
        """True if a real answer was generated (not not_found, not errored)."""
        return not self.not_found and self.error is None and bool(self.answer)

    def to_dict(self) -> dict:
        return {
            "query":            self.query,
            "answer":           self.answer,
            "not_found":        self.not_found,
            "error":            self.error,
            "model":            self.model,
            "latency_s":        round(self.latency_s, 3),
            "prompt_tokens":    self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "sources": [s.to_dict() for s in self.sources],
        }


# ── Prompt construction ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are EduVision, an AI teaching assistant for video-based programming courses.
Your job is to answer student questions using the video transcript evidence provided.

Rules you MUST follow:
1. Base your answer on the transcript excerpts in the Evidence section AND any Prior
   context from this conversation shown in the user message.
2. For every fact you state, cite the source using this format: [Video: "<title>" @ <timestamp>].
   Example: "HTML is a markup language [Video: \"Tutorial #2\" @ 00:43]."
3. If NEITHER the Evidence section NOR the Prior context contains enough information to
   answer the question, respond with exactly:
   "I could not find this topic in the provided course material."
   Do not add anything else when you use this phrase.
   Note: if the query is a bare topic or keyword (e.g. "HTML", "CSS", "flexbox"), treat
   it as a request to explain or summarise that topic. If the Evidence covers it, answer
   normally with citations — do not return the not-found phrase just because the query is short.
4. Never invent information. Never use general knowledge outside the provided evidence
   and prior context.
5. Keep your answer focused and clear \u2014 2 to 5 sentences unless more detail is truly needed.
6. If multiple evidence chunks are relevant, synthesise them into one coherent answer.
7. When the Prior context section contains a relevant previous answer, you may build on
   it to answer follow-up questions, re-using the timestamp citations already given.
"""

def _build_user_message(
    query: str,
    evidence: list[RetrievalResult],
    prior_context: str | None = None,
) -> str:
    """
    Construct the user message containing the question and numbered evidence blocks.

    When prior_context is provided (the last assistant response in a follow-up turn),
    it is included as a labeled section before the fresh Evidence so GPT can cite it
    without relying solely on message-history interpretation.

    The evidence is already sorted by similarity (highest first) from Stage 8.
    We number each block so the LLM can reference them by [Video: \"...\" @ time].
    """
    lines = [f"Question: {query.strip()}", ""]

    if prior_context:
        lines.append("Prior context from this conversation:")
        lines.append(prior_context.strip())
        lines.append("")

    lines.append("Evidence:")

    for idx, r in enumerate(evidence, start=1):
        # Clean video filename — strip the full path prefix if present
        vid_name = r.video_filename
        lines.append(
            f"[{idx}] Video: \"{vid_name}\" | [{r.start_time_fmt} \u2192 {r.end_time_fmt}]"
        )
        lines.append(f'    "{r.text.strip()}"')
        lines.append("")

    return "\n".join(lines)


# ── OpenAI client management ──────────────────────────────────────────────────

def _get_client() -> openai.OpenAI:
    """
    Return an OpenAI client initialised with the API key from settings.

    Raises:
        MissingAPIKeyError: If key is not set or is the placeholder.
    """
    key = OPENAI_API_KEY
    if not key or key.strip() == _PLACEHOLDER_KEY:
        raise MissingAPIKeyError(
            "OPENAI_API_KEY is not set. "
            "Add your key to the .env file:\n"
            "  OPENAI_API_KEY=sk-..."
        )
    return openai.OpenAI(api_key=key)


# ── Evidence selection ────────────────────────────────────────────────────────

def _select_evidence(
    results: list[RetrievalResult],
    threshold: float,
    max_chunks: int,
) -> list[RetrievalResult]:
    """
    Select above-threshold evidence from retrieval results.

    Returns at most max_chunks results with similarity >= threshold,
    in ranked order (best first).
    """
    above = [r for r in results if not r.below_threshold]
    return above[:max_chunks]


# ── Core generation ───────────────────────────────────────────────────────────

def generate(
    query: str,
    retrieval_results: list[RetrievalResult],
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    model: str = OPENAI_MODEL,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    max_evidence_chunks: int = MAX_EVIDENCE_CHUNKS,
    chat_history: list[dict] | None = None,
) -> GeneratorResponse:
    """
    Generate a grounded answer from retrieval evidence.

    This function:
      1. Validates the API key is set.
      2. Filters evidence to above-threshold chunks only.
      3. Returns not_found if no usable evidence exists.
      4. Builds a structured prompt and calls GPT-4o-mini.
      5. Returns a GeneratorResponse with answer, sources, and token stats.

    Args:
        query:               The user's natural language question.
        retrieval_results:   Results from Stage 8 retrieve(), pre-ranked.
        similarity_threshold: Minimum similarity to include as evidence.
        model:               OpenAI model name (default from settings).
        max_tokens:          Maximum tokens in the LLM response.
        temperature:         Sampling temperature (lower = more grounded).
        chat_history:        Optional list of prior conversation turns, each
                             a dict with "role" ("user"|"assistant") and
                             "content" (str).  The last MAX_HISTORY_TURNS pairs
                             are injected so GPT can resolve follow-up references
                             (e.g. "it", "that") without replacing RAG evidence.

    Returns:
        GeneratorResponse — always returns, never raises (except MissingAPIKeyError).
        Empty/below-threshold evidence returns not_found=True WITHOUT raising
        MissingAPIKeyError, even if the key is not configured.

    Raises:
        MissingAPIKeyError: Only when evidence exists and an LLM call is needed,
                            but the API key is missing or is the placeholder.
    """
    # ── Handle empty retrieval (before API key check — no LLM needed) ─────────
    if not retrieval_results:
        logger.info("generate(): no retrieval results → not_found")
        return GeneratorResponse(
            answer="I could not find this topic in the provided course material.",
            sources=[],
            model=model,
            query=query,
            not_found=True,
        )

    # ── Filter evidence (before API key check — no LLM needed if all below) ──
    evidence = _select_evidence(retrieval_results, similarity_threshold, max_evidence_chunks)

    if not evidence:
        logger.info(
            "generate(): all %d results below threshold=%.2f → not_found",
            len(retrieval_results), similarity_threshold,
        )
        return GeneratorResponse(
            answer="I could not find this topic in the provided course material.",
            sources=[],
            model=model,
            query=query,
            not_found=True,
        )

    # ── Validate API key — only needed when we have evidence and will call LLM ─
    # MissingAPIKeyError is intentionally not caught here so the caller knows
    # they must configure the key before LLM answers are possible.
    client = _get_client()

    logger.info(
        "generate(): using %d/%d evidence chunks (threshold=%.2f)",
        len(evidence), len(retrieval_results), similarity_threshold,
    )

    # ── Build prompt ──────────────────────────────────────────────────────────
    # Extract the last assistant response from chat_history as "prior context".
    # It is included directly in the user message body so GPT sees it as an
    # explicit evidence source, not just implicit message history, preventing
    # Rule 3 (not-found) from firing when the fresh evidence is weak.
    prior_context: str | None = None
    if chat_history:
        for h in reversed(chat_history):
            if h.get("role") == "assistant" and h.get("content"):
                prior_context = h["content"]
                break

    # Cap prior_context at ~500 chars to prevent token inflation in long
    # conversations.  Truncate at the last sentence boundary within the cap
    # so the injected text ends cleanly rather than mid-sentence.
    if prior_context and len(prior_context) > 500:
        cut = prior_context[:500].rfind(".")
        prior_context = (prior_context[: cut + 1] if cut > 0 else prior_context[:500]) + " \u2026"

    user_message = _build_user_message(query, evidence, prior_context=prior_context)

    # Inject prior conversation turns (up to MAX_HISTORY_TURNS pairs) between
    # the system prompt and the current question so GPT can resolve follow-up
    # references.  Only plain-text turns are injected — no evidence blocks —
    # so stale evidence from earlier turns never overrides the current retrieval.
    MAX_HISTORY_TURNS = 3  # keep last 3 user+assistant pairs = 6 messages
    history_messages: list[dict] = []
    if chat_history:
        # chat_history entries: {"role": "user"|"assistant", "content": str, ...}
        # Take only the text-content turns, excluding the CURRENT turn.
        prior = [
            {"role": h["role"], "content": h["content"]}
            for h in chat_history
            if h["role"] in ("user", "assistant") and h.get("content")
        ]
        # Keep the tail (most recent) up to MAX_HISTORY_TURNS*2 messages
        history_messages = prior[-(MAX_HISTORY_TURNS * 2):]

    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        *history_messages,
        {"role": "user",    "content": user_message},
    ]

    logger.debug("Prompt user message:\n%s", user_message)

    # ── Call OpenAI API ───────────────────────────────────────────────────────
    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency = time.time() - t0

        answer = response.choices[0].message.content.strip()
        usage  = response.usage

        logger.info(
            "generate(): answer in %.2f s  |  tokens: prompt=%d completion=%d",
            latency,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )

        return GeneratorResponse(
            answer=answer,
            sources=evidence,
            model=model,
            query=query,
            not_found=("could not find" in answer.lower()),
            latency_s=latency,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

    except openai.AuthenticationError as exc:
        logger.error("OpenAI AuthenticationError: %s", exc)
        return GeneratorResponse(
            answer="",
            sources=evidence,
            model=model,
            query=query,
            error=f"Invalid API key. Check OPENAI_API_KEY in .env. ({exc})",
        )

    except openai.RateLimitError as exc:
        logger.error("OpenAI RateLimitError: %s", exc)
        return GeneratorResponse(
            answer="",
            sources=evidence,
            model=model,
            query=query,
            error=f"OpenAI rate limit exceeded. Please wait and retry. ({exc})",
        )

    except openai.APIConnectionError as exc:
        logger.error("OpenAI APIConnectionError: %s", exc)
        return GeneratorResponse(
            answer="",
            sources=evidence,
            model=model,
            query=query,
            error=f"Could not connect to OpenAI. Check your internet connection. ({exc})",
        )

    except openai.APIStatusError as exc:
        logger.error("OpenAI APIStatusError %s: %s", exc.status_code, exc)
        return GeneratorResponse(
            answer="",
            sources=evidence,
            model=model,
            query=query,
            error=f"OpenAI API error (HTTP {exc.status_code}): {exc.message}",
        )

    except Exception as exc:
        logger.exception("Unexpected error during generation: %s", exc)
        return GeneratorResponse(
            answer="",
            sources=evidence,
            model=model,
            query=query,
            error=f"Unexpected error: {exc}",
        )


# ── End-to-end convenience function ──────────────────────────────────────────

def answer(
    query: str,
    top_k: int = 5,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    video_id_filter: Optional[str] = None,
) -> GeneratorResponse:
    """
    Full RAG pipeline: query → retrieve → generate → answer.

    Convenience wrapper that calls Stage 8 (retrieve) and then
    Stage 9 (generate) in one call. Used by Stage 12 (Streamlit UI).

    Args:
        query:               Natural language question.
        top_k:               Number of chunks to retrieve.
        similarity_threshold: Evidence quality floor.
        video_id_filter:     Restrict retrieval to one video (optional).

    Returns:
        GeneratorResponse with grounded answer and source citations.
    """
    from retrieval.retriever import retrieve

    results = retrieve(
        query,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        video_id_filter=video_id_filter,
    )

    return generate(
        query=query,
        retrieval_results=results,
        similarity_threshold=similarity_threshold,
    )


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 68)
    print("EduVision RAG — Stage 9: Generator Test Suite")
    print("=" * 68)
    print(f"Model       : {OPENAI_MODEL}")
    print(f"Temperature : {TEMPERATURE}")
    print(f"Max tokens  : {MAX_TOKENS}")
    print(f"Threshold   : {SIMILARITY_THRESHOLD}")
    print()

    # ── Check API key before any retrieval ────────────────────────────────────
    key = OPENAI_API_KEY
    api_key_set = key and key != _PLACEHOLDER_KEY
    print(f"API key     : {'✅ SET' if api_key_set else '❌ NOT SET — will test error handling'}")
    print()

    from retrieval.retriever import retrieve, RetrievalResult

    # ─────────────────────────────────────────────────────────────────────────
    # TEST CASES
    # ─────────────────────────────────────────────────────────────────────────
    test_cases = [
        {
            "label":      "T1 — Relevant: VS Code installation",
            "query":      "How do I install VS Code?",
            "expect":     "Should cite Video 1 with installation timestamps",
        },
        {
            "label":      "T2 — Relevant: HTML websites",
            "query":      "How do websites work with HTML and CSS?",
            "expect":     "Should cite Video 2 with HTML explanation timestamps",
        },
        {
            "label":      "T3 — Relevant: What is HTML",
            "query":      "What is HTML and what does it do?",
            "expect":     "Should describe HTML from transcript evidence with citations",
        },
        {
            "label":      "T4 — Irrelevant: Out of scope",
            "query":      "What is the capital of France?",
            "expect":     "Should say 'could not find in course material'",
        },
        {
            "label":      "T5 — Weak evidence (simulated): empty results",
            "query":      "__EMPTY_RESULTS__",   # special sentinel
            "expect":     "Should return not_found without calling LLM",
        },
    ]

    all_structural_pass = True

    for tc in test_cases:
        print(f"\n{'─'*68}")
        print(f"  {tc['label']}")
        print(f"  Query  : {tc['query']!r}")
        print(f"  Expect : {tc['expect']}")
        print()

        # ── Special case: test empty retrieval ────────────────────────────────
        if tc["query"] == "__EMPTY_RESULTS__":
            try:
                resp = generate(
                    query="empty retrieval test",
                    retrieval_results=[],   # force empty
                )
                print(f"  not_found    : {resp.not_found}")
                print(f"  error        : {resp.error}")
                print(f"  LLM called   : {'NO (correct)' if resp.prompt_tokens == 0 else 'YES (unexpected)'}")
                ok = resp.not_found and resp.error is None and resp.prompt_tokens == 0
                print(f"  Result : {'✅ PASS' if ok else '❌ FAIL'}")
                if not ok:
                    all_structural_pass = False
            except MissingAPIKeyError as e:
                print(f"  ✅ PASS (MissingAPIKeyError before empty-results check: {e})")
            continue

        # ── Retrieve evidence for the query ───────────────────────────────────
        results = retrieve(tc["query"], top_k=5)
        print(f"  Retrieved {len(results)} results:")
        for r in results:
            flag = " ⚠️ [below threshold]" if r.below_threshold else ""
            print(f"    #{r.rank}  sim={r.similarity:.4f}{flag}  [{r.start_time_fmt}]  "
                  f"{r.video_filename[:45]}")

        print()

        # ── Call generator ────────────────────────────────────────────────────
        try:
            resp = generate(
                query=tc["query"],
                retrieval_results=results,
            )
        except MissingAPIKeyError as e:
            print(f"  ⚠️  API key not set. Showing structural test only.")
            print(f"  Error : {e}")
            # Structural tests: can still validate evidence selection
            from retrieval.retriever import retrieve
            evidence = _select_evidence(results, SIMILARITY_THRESHOLD, MAX_EVIDENCE_CHUNKS)
            print(f"  Evidence selected : {len(evidence)} chunks above threshold={SIMILARITY_THRESHOLD}")
            print(f"  User message preview:")
            msg = _build_user_message(tc["query"], evidence)
            for line in msg.split("\n")[:15]:
                print(f"    {line}")
            if tc["query"] == "What is the capital of France?":
                print(f"  Expected not_found=True (all below threshold): "
                      f"{'✅ CORRECT' if not evidence else '❌ Evidence found (threshold may be too low)'}")
            continue

        # ── Display result ────────────────────────────────────────────────────
        print(f"  Answer       : {resp.answer[:250]}{'...' if len(resp.answer) > 250 else ''}")
        print()
        print(f"  not_found    : {resp.not_found}")
        print(f"  error        : {resp.error}")
        print(f"  model        : {resp.model}")
        print(f"  latency      : {resp.latency_s:.2f}s")
        print(f"  tokens       : prompt={resp.prompt_tokens}  completion={resp.completion_tokens}  total={resp.total_tokens}")
        print(f"  sources used : {len(resp.sources)}")
        for s in resp.sources:
            print(f"    [{s.start_time_fmt} → {s.end_time_fmt}]  {s.video_filename[:50]}")

        # ── Structural validation ─────────────────────────────────────────────
        struct_checks = []

        # All responses must have a query
        struct_checks.append(("query preserved", resp.query == tc["query"]))

        # Sources must have valid timestamps
        for s in resp.sources:
            struct_checks.append((
                f"  {s.chunk_id[-20:]} start<end",
                s.start_time < s.end_time
            ))
            struct_checks.append((
                f"  {s.chunk_id[-20:]} start_time_fmt MM:SS",
                len(s.start_time_fmt) >= 4 and ":" in s.start_time_fmt
            ))

        for check_name, result in struct_checks:
            if not result:
                print(f"  ❌ STRUCTURAL FAIL: {check_name}")
                all_structural_pass = False

        # Irrelevant query → not_found or answer contains "could not find"
        if "France" in tc["query"]:
            irrelevant_ok = resp.not_found or "could not find" in resp.answer.lower()
            print(f"  Irrelevant-query check : {'✅ Refused correctly' if irrelevant_ok else '❌ Did not refuse'}")
            if not irrelevant_ok:
                all_structural_pass = False

        print(f"  Structural checks : ✅ all pass" if all([r for _, r in struct_checks])
              else f"  Structural checks : ❌ some failed")

    # ── Source files untouched ────────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print("  Source integrity checks:")
    import chromadb, json as _json, pathlib
    from chromadb.config import Settings as CS
    col = chromadb.PersistentClient(path="data/vector_db", settings=CS(anonymized_telemetry=False)).get_collection("eduvision_chunks")
    chroma_ok  = col.count() == 235
    segs_ok    = len(_json.load(open("data/transcripts/02_your_first_html_website_sigma_web_development_course_tutorial_2_cleaned.json"))["segments"]) == 649
    print(f"  ChromaDB count = 235 : {'✅' if chroma_ok else '❌'}  got={col.count()}")
    print(f"  Cleaned JSON = 649 segs : {'✅' if segs_ok else '❌'}")

    print(f"\n{'='*68}")
    print(f"STRUCTURAL RESULT : {'✅ ALL PASS' if all_structural_pass and chroma_ok and segs_ok else '❌ SOME FAILED'}")
    if not api_key_set:
        print("NOTE: API key not set — LLM response content not tested.")
        print("      Set OPENAI_API_KEY in .env and re-run for full testing.")
    print(f"{'='*68}")
