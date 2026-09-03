"""
pipeline.py
────────────
Stage 10: End-to-End RAG Pipeline Orchestrator.

This module is the single entry point for the EduVision UI (Stage 12).
It wires Stage 8 (Retriever) and Stage 9 (Generator) together without
duplicating any logic from either stage.

═══════════════════════════════════════════════════════════════
WHAT THIS MODULE DOES
═══════════════════════════════════════════════════════════════

The pipeline exposes three functions to the UI:

  ask(query, ...)     → PipelineResult
    Full RAG: question → retrieve evidence → generate grounded answer.
    This is the primary function for the chat/Q&A interface.

  search(query, ...)  → list[RetrievalResult]
    Retrieval only: question → ranked transcript evidence.
    Used by the "Search" tab in the UI to browse evidence directly.

  health_check()      → HealthStatus
    Confirms ChromaDB is populated, BGE-M3 is loadable, and the API
    key is configured. Called on app startup before accepting queries.

═══════════════════════════════════════════════════════════════
WHAT THIS MODULE DOES NOT DO
═══════════════════════════════════════════════════════════════

  ✗ Does NOT re-implement retrieval logic (Stage 8 owns that)
  ✗ Does NOT re-implement generation logic (Stage 9 owns that)
  ✗ Does NOT call OpenAI directly
  ✗ Does NOT open ChromaDB directly
  ✗ Does NOT re-embed queries (Stage 8 owns that)
  ✗ Does NOT modify chunks, embeddings, or transcripts

═══════════════════════════════════════════════════════════════
DATA FLOW
═══════════════════════════════════════════════════════════════

  User question  (str)
        │
        │  [pipeline.ask()]
        │
        ├─ validate_query()          → PipelineResult(error) if invalid
        │
        ├─ Stage 8: retrieve()       → list[RetrievalResult]
        │            ↓
        │          retrieval_stats   (count, top similarity, latency)
        │
        ├─ Stage 9: generate()       → GeneratorResponse
        │            ↓
        │          answer, sources, tokens, latency
        │
        └─ PipelineResult            → consumed by Stage 12 (Streamlit)

═══════════════════════════════════════════════════════════════
PipelineResult vs GeneratorResponse
═══════════════════════════════════════════════════════════════

GeneratorResponse (Stage 9):
  answer, sources, model, not_found, error, tokens, latency_s

PipelineResult (Stage 10) adds:
  retrieval_results   — the FULL ranked list from Stage 8
                        (GeneratorResponse.sources only has the
                        above-threshold subset passed to the LLM)
  retrieval_latency_s — how long retrieval alone took
  total_latency_s     — retrieval + generation combined
  query_valid         — False if query was rejected before retrieval
  validation_error    — human-readable reason query was rejected

This distinction matters for the UI: it can show all 5 retrieved
chunks even if only 3 were passed to the LLM (below-threshold ones
are still interesting for the user to see).
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from config.settings import (
    RETRIEVAL_TOP_K,
    MAX_LLM_EVIDENCE,
    TOP_K_RESULTS,
    SIMILARITY_THRESHOLD,
    OPENAI_MODEL,
    ACTIVE_COLLECTION,
)
from generation.generator import (
    GeneratorResponse,
    MissingAPIKeyError,
    generate,
)
from retrieval.retriever import RetrievalResult, retrieve

logger = logging.getLogger(__name__)

# Maximum query length (characters). Queries longer than this are unlikely
# to be genuine questions and would waste tokens on retrieval + generation.
MAX_QUERY_LENGTH = 500

# Pronoun/contextual-reference pattern used by _rewrite_query_for_retrieval.
# A query that contains NONE of these words is treated as self-contained and
# skips the GPT rewrite call entirely, saving ~1–2 s of latency.
# The pattern is intentionally conservative: only the most common English
# pronouns and demonstratives are included so ambiguous follow-ups are never
# silently passed through without rewriting.
_FOLLOWUP_RE = re.compile(
    r"\b(it|its|this|that|these|those|they|them|their|he|she|her|his"
    r"|the video|the tutorial)\b",
    re.IGNORECASE,
)


# ── Return data models ────────────────────────────────────────────────────────

@dataclass
class RetrievalStats:
    """Summary statistics from the retrieval step."""
    total_results:    int     # total chunks returned (= top_k)
    above_threshold:  int     # chunks with similarity ≥ threshold
    below_threshold:  int     # chunks with similarity < threshold
    top_similarity:   float   # highest cosine similarity
    latency_s:        float   # retrieval wall-clock time

    @property
    def has_evidence(self) -> bool:
        """True if at least one result is above the similarity threshold."""
        return self.above_threshold > 0


@dataclass
class PipelineResult:
    """
    Complete output of one ask() call.

    Consumed by Stage 12 (Streamlit UI). All fields needed to render
    the answer, source citations, timestamps, and diagnostics are here.
    """
    # ── Query info ────────────────────────────────────────────────────────────
    query:              str           # the original user question

    # ── Answer ────────────────────────────────────────────────────────────────
    answer:             str           # LLM's grounded answer (empty if not_found/error)
    not_found:          bool          # True if evidence was insufficient
    error:              Optional[str] # non-None if a system error occurred

    # ── Sources ───────────────────────────────────────────────────────────────
    # All retrieved chunks (Stage 8 output), including below-threshold ones
    retrieval_results:  list[RetrievalResult] = field(default_factory=list)
    # Subset actually passed to the LLM (above-threshold only)
    sources_used:       list[RetrievalResult] = field(default_factory=list)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    retrieval_stats:    Optional[RetrievalStats] = None
    model:              str = OPENAI_MODEL
    prompt_tokens:      int = 0
    completion_tokens:  int = 0
    retrieval_latency_s: float = 0.0
    generation_latency_s: float = 0.0
    total_latency_s:    float = 0.0

    # ── Validation ────────────────────────────────────────────────────────────
    query_valid:        bool = True
    validation_error:   Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def success(self) -> bool:
        """True if a grounded answer was generated without errors."""
        return (self.query_valid and not self.not_found
                and self.error is None and bool(self.answer))

    def to_dict(self) -> dict:
        """Serialisable representation for logging / debugging."""
        return {
            "query":              self.query,
            "answer":             self.answer,
            "not_found":          self.not_found,
            "error":              self.error,
            "query_valid":        self.query_valid,
            "validation_error":   self.validation_error,
            "model":              self.model,
            "total_tokens":       self.total_tokens,
            "retrieval_latency_s": round(self.retrieval_latency_s, 3),
            "generation_latency_s": round(self.generation_latency_s, 3),
            "total_latency_s":    round(self.total_latency_s, 3),
            "retrieval_stats": {
                "total":          self.retrieval_stats.total_results if self.retrieval_stats else 0,
                "above_threshold": self.retrieval_stats.above_threshold if self.retrieval_stats else 0,
                "top_similarity": round(self.retrieval_stats.top_similarity, 4) if self.retrieval_stats else 0,
            } if self.retrieval_stats else None,
            "sources_used": [s.to_dict() for s in self.sources_used],
            "retrieval_results": [r.to_dict() for r in self.retrieval_results],
        }


@dataclass
class HealthStatus:
    """Result of health_check() — used on app startup."""
    chroma_ok:      bool
    chroma_count:   int
    api_key_set:    bool
    model_name:     str
    errors:         list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        """True if the system can accept queries."""
        # API key not being set is a warning, not a blocker — the UI can
        # still show retrieved evidence even without an LLM answer.
        return self.chroma_ok and len(self.errors) == 0


# ── Input validation ──────────────────────────────────────────────────────────

def validate_query(query: str) -> Optional[str]:
    """
    Validate the user query before sending it through the pipeline.

    Returns None if valid, or a human-readable error string if invalid.
    Does NOT raise — the caller handles the validation_error field.
    """
    if not query or not query.strip():
        return "Query cannot be empty. Please enter a question."

    if len(query.strip()) < 3:
        return "Query is too short. Please ask a complete question."

    if len(query) > MAX_QUERY_LENGTH:
        return (
            f"Query is too long ({len(query)} characters). "
            f"Please keep questions under {MAX_QUERY_LENGTH} characters."
        )

    return None  # valid


# ── Follow-up query rewriting for retrieval ──────────────────────────────────

def _rewrite_query_for_retrieval(
    query: str,
    chat_history: list[dict] | None,
) -> str:
    """
    Rewrite an ambiguous follow-up query into a self-contained retrieval string.

    Examples:
        history: [user: "what is html", assistant: "HTML is..."]
        query:   "why is it important?"
        returns: "why is HTML important?"

    The rewritten query is used ONLY for ChromaDB retrieval so the embedding
    can match relevant chunks.  The original query is preserved for the
    generator so the conversational answer reads naturally.

    Returns the original query unchanged when:
      - No usable history exists (first question)
      - The API key is missing (graceful degradation)
      - The rewrite call fails for any reason
    """
    if not chat_history:
        return query

    # Collect the last 2 user+assistant turns (4 messages max) for context.
    recent = [
        h for h in chat_history
        if h.get("role") in ("user", "assistant") and h.get("content")
    ][-4:]

    if not recent:
        return query

    # ── Heuristic: skip GPT call for self-contained queries ───────────────────
    # If the query contains no pronoun or contextual reference (it, this, that,
    # they, he, she, etc.) it is almost certainly self-contained and does not
    # need rewriting.  Skipping the call saves ~1–2 s of latency per turn.
    # The regex is conservative: a false negative (pronoun detected but rewrite
    # turns out to be a no-op) is harmless; a false positive (pronoun missed)
    # would degrade retrieval — so we only skip when we are confident.
    if not _FOLLOWUP_RE.search(query):
        logger.debug("Query rewrite skipped (no contextual reference): %r", query)
        return query

    history_text = "\n".join(
        f"{'User' if h['role'] == 'user' else 'Assistant'}: {h['content'][:300]}"
        for h in recent
    )

    prompt = (
        f"Given this conversation:\n{history_text}\n\n"
        "Rewrite the follow-up question below as a standalone search query by replacing "
        "any pronouns or references (it, that, this, they, etc.) with the actual topic "
        "from the conversation. If the question is already self-contained, return it "
        "unchanged. Reply with the rewritten query only — no explanation, no quotes.\n\n"
        f"Follow-up: {query}\n"
        "Standalone query:"
    )

    try:
        client = _get_openai_client_if_available()
        if client is None:
            return query

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.0,
        )
        rewritten = response.choices[0].message.content.strip().strip('"\'')
        if rewritten:
            logger.info("Query rewrite: %r → %r", query, rewritten)
            return rewritten
    except Exception as exc:
        logger.warning("Query rewrite failed (%s) — using original query", exc)

    return query


def _get_openai_client_if_available():
    """
    Return an OpenAI client only if the API key is configured.
    Returns None (instead of raising) so callers can degrade gracefully.
    """
    try:
        from generation.generator import _get_client
        return _get_client()
    except Exception:
        return None


# ── Primary pipeline function ─────────────────────────────────────────────────

def ask(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    video_id_filter: Optional[str] = None,
    chat_history: list[dict] | None = None,
) -> PipelineResult:
    """
    Run the full RAG pipeline for a user question.

    This is the primary function for the EduVision UI's Q&A interface.
    It orchestrates Stage 8 (retrieval) and Stage 9 (generation) in
    sequence, collecting timing and diagnostics along the way.

    Args:
        query:               Natural language question from the user.
        top_k:               Number of transcript chunks to retrieve from ChromaDB
                             (default RETRIEVAL_TOP_K=10). The generator will use
                             at most MAX_LLM_EVIDENCE=5 of the above-threshold results.
        similarity_threshold: Minimum similarity for evidence to be
                             passed to the LLM.
        video_id_filter:     Restrict retrieval to one specific video
                             (optional — for "Search in Video X" UI).
        chat_history:        Optional prior conversation turns for follow-up context.
                             Each entry is a dict with "role" and "content" keys.
                             The last 3 user+assistant pairs are forwarded to the
                             generator so GPT can resolve pronouns and references.

    Returns:
        PipelineResult — always returns, never raises. The UI should
        check result.success, result.not_found, and result.error to
        decide what to display.
    """
    t_pipeline_start = time.time()
    query = query.strip() if query else ""

    # ── Input validation (before any model work) ──────────────────────────────
    validation_error = validate_query(query)
    if validation_error:
        logger.info("Pipeline validation failed: %s", validation_error)
        return PipelineResult(
            query=query,
            answer="",
            not_found=False,
            error=None,
            query_valid=False,
            validation_error=validation_error,
        )

    logger.info("Pipeline: ask(%r, top_k=%d, threshold=%.2f)", query[:60], top_k, similarity_threshold)

    # ── Follow-up query rewriting (retrieval only) ────────────────────────────
    # Resolve pronouns/references in follow-up questions so ChromaDB can
    # find the right chunks.  The ORIGINAL query is still used for the
    # conversational answer so the response reads naturally.
    retrieval_query = _rewrite_query_for_retrieval(query, chat_history)

    # ── Stage 8: Retrieval ────────────────────────────────────────────────────
    t_retrieval_start = time.time()
    try:
        retrieval_results = retrieve(
            query=retrieval_query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            video_id_filter=video_id_filter,
            collection_name=ACTIVE_COLLECTION,
        )
    except Exception as exc:
        logger.exception("Retrieval failed: %s", exc)
        return PipelineResult(
            query=query,
            answer="",
            not_found=False,
            error=f"Retrieval error: {exc}",
            query_valid=True,
        )

    retrieval_latency = time.time() - t_retrieval_start

    # Build retrieval stats
    above = [r for r in retrieval_results if not r.below_threshold]
    retrieval_stats = RetrievalStats(
        total_results=len(retrieval_results),
        above_threshold=len(above),
        below_threshold=len(retrieval_results) - len(above),
        top_similarity=retrieval_results[0].similarity if retrieval_results else 0.0,
        latency_s=retrieval_latency,
    )

    # ── Stage 9: Generation ───────────────────────────────────────────────────
    t_generation_start = time.time()
    try:
        gen_response: GeneratorResponse = generate(
            query=query,
            retrieval_results=retrieval_results,
            similarity_threshold=similarity_threshold,
            max_evidence_chunks=MAX_LLM_EVIDENCE,
            chat_history=chat_history,
        )
    except MissingAPIKeyError as exc:
        # API key not configured — retrieval succeeded, but LLM unavailable.
        # Return retrieval results so the UI can still show evidence.
        generation_latency = time.time() - t_generation_start
        logger.warning("Pipeline: API key not set — returning retrieval results only.")
        return PipelineResult(
            query=query,
            answer="",
            not_found=False,
            error=str(exc),
            retrieval_results=retrieval_results,
            sources_used=[],
            retrieval_stats=retrieval_stats,
            model=OPENAI_MODEL,
            retrieval_latency_s=retrieval_latency,
            generation_latency_s=generation_latency,
            total_latency_s=time.time() - t_pipeline_start,
        )
    except Exception as exc:
        logger.exception("Generation failed: %s", exc)
        generation_latency = time.time() - t_generation_start
        return PipelineResult(
            query=query,
            answer="",
            not_found=False,
            error=f"Generation error: {exc}",
            retrieval_results=retrieval_results,
            sources_used=[],
            retrieval_stats=retrieval_stats,
            model=OPENAI_MODEL,
            retrieval_latency_s=retrieval_latency,
            generation_latency_s=generation_latency,
            total_latency_s=time.time() - t_pipeline_start,
        )

    generation_latency = time.time() - t_generation_start
    total_latency = time.time() - t_pipeline_start

    result = PipelineResult(
        query=query,
        answer=gen_response.answer,
        not_found=gen_response.not_found,
        error=gen_response.error,
        retrieval_results=retrieval_results,
        sources_used=gen_response.sources,
        retrieval_stats=retrieval_stats,
        model=gen_response.model,
        prompt_tokens=gen_response.prompt_tokens,
        completion_tokens=gen_response.completion_tokens,
        retrieval_latency_s=retrieval_latency,
        generation_latency_s=generation_latency,
        total_latency_s=total_latency,
    )

    logger.info(
        "Pipeline complete: success=%s not_found=%s error=%s "
        "retrieval=%.2fs generation=%.2fs total=%.2fs tokens=%d",
        result.success, result.not_found, result.error is not None,
        retrieval_latency, generation_latency, total_latency,
        result.total_tokens,
    )

    return result


# ── Retrieval-only function ───────────────────────────────────────────────────

def search(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    video_id_filter: Optional[str] = None,
) -> list[RetrievalResult]:
    """
    Retrieve matching transcript chunks without generating an answer.

    Used by the UI's "Search" tab: the user can browse evidence directly
    without consuming LLM tokens. Also useful for debugging retrieval.

    Args:
        query:               Natural language search query.
        top_k:               Number of results to return.
        similarity_threshold: Used to set the below_threshold flag on results.
        video_id_filter:     Restrict to one video (optional).

    Returns:
        list[RetrievalResult] sorted by similarity descending.
        Empty list if query is invalid or retrieval fails.
    """
    if validate_query(query) is not None:
        return []

    try:
        return retrieve(
            query=query.strip(),
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            video_id_filter=video_id_filter,
            collection_name=ACTIVE_COLLECTION,
        )
    except Exception as exc:
        logger.exception("search() retrieval failed: %s", exc)
        return []


# ── Video catalogue ───────────────────────────────────────────────────────────

def list_indexed_videos() -> list[tuple[str, str]]:
    """
    Return a sorted list of (video_id, display_label) for every distinct video
    currently indexed in the active ChromaDB v2 collection.

    display_label format: "Tutorial #N — <Human Title>"
    e.g.  "Tutorial #3 — Basic Structure of an HTML Website"

    The label is derived entirely from ChromaDB metadata — no hardcoded list.
    Returns an empty list if ChromaDB is unavailable.
    """
    import re as _re
    try:
        from ingestion.indexer import get_chroma_client
        client = get_chroma_client()
        col = client.get_collection(name=ACTIVE_COLLECTION)
        # Fetch only metadatas (no embeddings / documents needed)
        all_meta = col.get(include=["metadatas"])["metadatas"]
    except Exception as exc:
        logger.warning("list_indexed_videos: ChromaDB unavailable — %s", exc)
        return []

    # Collect distinct (video_id, video_filename) pairs
    seen: dict[str, str] = {}
    for m in all_meta:
        vid_id = m.get("video_id", "")
        vid_fn = m.get("video_filename", "")
        if vid_id and vid_id not in seen:
            seen[vid_id] = vid_fn

    def _make_label(video_id: str, video_filename: str) -> str:
        # Extract leading number from video_id (e.g. "03_basic_structure..." → 3)
        num_match = _re.match(r"^(\d+)_", video_id)
        num = int(num_match.group(1)) if num_match else 0

        # Build human title from video_filename by stripping the number prefix,
        # the course suffix ("Sigma Web Development..."), and the .mp4 extension.
        title = video_filename
        # Remove .mp4 / .MP4
        title = _re.sub(r"\.mp4$", "", title, flags=_re.IGNORECASE)
        # Remove leading "NN_" or "NN " prefix
        title = _re.sub(r"^\d+[_\s]+", "", title)
        # Remove trailing course boilerplate after " | " or " ｜ "
        title = _re.split(r"\s*[|｜]\s*", title)[0].strip()
        # Collapse whitespace
        title = " ".join(title.split())
        if not title:
            title = video_id

        return f"Tutorial #{num} — {title}"

    # Sort by tutorial number
    items = sorted(seen.items(), key=lambda kv: int((_re.match(r"^(\d+)_", kv[0]) or _re.match(r"(0)", "0")).group(1)))
    return [(vid_id, _make_label(vid_id, vid_fn)) for vid_id, vid_fn in items]


# ── Health check ──────────────────────────────────────────────────────────────

def health_check() -> HealthStatus:
    """
    Verify that all pipeline dependencies are available.

    Checks:
      1. ChromaDB collection is accessible and non-empty.
      2. OPENAI_API_KEY is configured (warning only — UI still shows evidence).

    Does NOT load BGE-M3 (that happens lazily on first query).
    Does NOT make any OpenAI API calls.

    Returns:
        HealthStatus — the UI shows a banner if not ready.
    """
    errors: list[str] = []
    chroma_ok = False
    chroma_count = 0

    # Check ChromaDB (active collection)
    try:
        from ingestion.indexer import get_chroma_client
        client = get_chroma_client()
        col = client.get_collection(name=ACTIVE_COLLECTION)
        chroma_count = col.count()
        chroma_ok = chroma_count > 0
        if not chroma_ok:
            errors.append(f"ChromaDB collection '{ACTIVE_COLLECTION}' is empty.")
    except Exception as exc:
        errors.append(f"ChromaDB unavailable: {exc}")

    # Check API key
    from generation.generator import OPENAI_API_KEY, _PLACEHOLDER_KEY
    api_key_set = bool(OPENAI_API_KEY) and OPENAI_API_KEY != _PLACEHOLDER_KEY

    return HealthStatus(
        chroma_ok=chroma_ok,
        chroma_count=chroma_count,
        api_key_set=api_key_set,
        model_name=OPENAI_MODEL,
        errors=errors,
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
    print("EduVision RAG — Stage 10: End-to-End Pipeline Test")
    print("=" * 68)
    print()

    # ── Health check first ────────────────────────────────────────────────────
    print("── Health Check ─────────────────────────────────────────────────")
    status = health_check()
    print(f"  ChromaDB       : {'✅' if status.chroma_ok else '❌'}  {status.chroma_count} documents")
    print(f"  API key set    : {'✅' if status.api_key_set else '⚠️  NOT SET'}")
    print(f"  Model          : {status.model_name}")
    print(f"  System ready   : {'✅ YES' if status.ready else '❌ NO — ' + str(status.errors)}")
    print()

    # ── Test cases ────────────────────────────────────────────────────────────
    test_cases = [
        # (query, label, expected_success, expected_not_found)
        ("How do I install VS Code?",               "Relevant — VS Code install",   True,  False),
        ("How do websites work with HTML and CSS?",  "Relevant — HTML/CSS",          True,  False),
        ("What is HTML and what does it do?",        "Relevant — HTML definition",   True,  False),
        ("What is the capital of France?",           "Irrelevant — out of scope",    False, True),
        ("",                                         "Empty query validation",       False, False),
        ("x",                                       "Too-short query validation",   False, False),
    ]

    all_ok = True

    for query, label, expect_success, expect_not_found in test_cases:
        print(f"── {label} {'─'*(54-len(label))}")
        print(f"  Query: {query!r}")

        result = ask(query)

        # Print structured result
        print(f"  success         : {result.success}")
        print(f"  not_found       : {result.not_found}")
        print(f"  query_valid     : {result.query_valid}")
        print(f"  validation_error: {result.validation_error}")
        print(f"  error           : {result.error}")
        if result.retrieval_stats:
            rs = result.retrieval_stats
            print(f"  retrieval       : {rs.total_results} chunks, "
                  f"{rs.above_threshold} above threshold, "
                  f"top_sim={rs.top_similarity:.4f}, {rs.latency_s:.3f}s")
        if result.answer:
            print(f"  answer (100ch)  : {result.answer[:100]}{'...' if len(result.answer)>100 else ''}")
        print(f"  sources_used    : {len(result.sources_used)}")
        for s in result.sources_used:
            print(f"    [{s.start_time_fmt} → {s.end_time_fmt}]  sim={s.similarity:.4f}  "
                  f"{s.video_filename[:45]}")
        print(f"  latency         : retrieve={result.retrieval_latency_s:.2f}s  "
              f"generate={result.generation_latency_s:.2f}s  "
              f"total={result.total_latency_s:.2f}s")
        print(f"  tokens          : {result.total_tokens}")

        # Validate expectations
        checks = []
        checks.append(("success matches expectation", result.success == expect_success))
        checks.append(("not_found matches expectation", result.not_found == expect_not_found))

        # Timestamp integrity for successful results
        for s in result.sources_used:
            checks.append((f"  {s.chunk_id[-15:]} start<end",
                           s.start_time < s.end_time))
            checks.append((f"  {s.chunk_id[-15:]} start_time_fmt valid",
                           ":" in s.start_time_fmt and len(s.start_time_fmt) >= 4))

        # to_dict() must be JSON-serialisable
        try:
            json.dumps(result.to_dict())
            checks.append(("to_dict() JSON-serialisable", True))
        except Exception as e:
            checks.append(("to_dict() JSON-serialisable", False))

        failed = [(name, ok) for name, ok in checks if not ok]
        if failed:
            all_ok = False
            for name, _ in failed:
                print(f"  ❌ FAIL: {name}")
        else:
            print(f"  ✅ All {len(checks)} checks pass")
        print()

    # ── search() function test ────────────────────────────────────────────────
    print("── search() function ────────────────────────────────────────────")
    results = search("how to install VS Code")
    ok_search = (len(results) > 0 and
                 all(isinstance(r, RetrievalResult) for r in results) and
                 results == sorted(results, key=lambda r: -r.similarity))
    print(f"  search('how to install VS Code'): {len(results)} results, sorted by sim: {'✅' if ok_search else '❌'}")
    for r in results:
        print(f"    #{r.rank}  sim={r.similarity:.4f}  [{r.start_time_fmt}]  {r.video_filename[:45]}")
    if not ok_search:
        all_ok = False

    # Empty search
    empty_search = search("")
    print(f"  search(''): {len(empty_search)} results (expected 0): {'✅' if empty_search == [] else '❌'}")
    if empty_search != []:
        all_ok = False
    print()

    # ── Source integrity ──────────────────────────────────────────────────────
    print("── Source file integrity ────────────────────────────────────────")
    import chromadb, json as _json, pathlib
    from chromadb.config import Settings as CS

    col = chromadb.PersistentClient(
        path="data/vector_db", settings=CS(anonymized_telemetry=False)
    ).get_collection("eduvision_chunks")
    chroma_ok = col.count() == 235

    segs1 = len(_json.load(open("data/transcripts/01_installing_vs_code_how_websites_work_sigma_web_development_course_tutorial_1_cleaned.json"))["segments"])
    segs2 = len(_json.load(open("data/transcripts/02_your_first_html_website_sigma_web_development_course_tutorial_2_cleaned.json"))["segments"])
    chunks1 = len(_json.load(open("data/processed/chunks/01_installing_vs_code_how_websites_work_sigma_web_development_course_tutorial_1_chunks.json"))["chunks"])
    chunks2 = len(_json.load(open("data/processed/chunks/02_your_first_html_website_sigma_web_development_course_tutorial_2_chunks.json"))["chunks"])

    print(f"  ChromaDB count = 235      : {'✅' if chroma_ok else '❌'}  got={col.count()}")
    print(f"  Cleaned segs Video 1 = 263: {'✅' if segs1==263 else '❌'}  got={segs1}")
    print(f"  Cleaned segs Video 2 = 649: {'✅' if segs2==649 else '❌'}  got={segs2}")
    print(f"  Chunks Video 1 = 69       : {'✅' if chunks1==69 else '❌'}  got={chunks1}")
    print(f"  Chunks Video 2 = 166      : {'✅' if chunks2==166 else '❌'}  got={chunks2}")
    if not all([chroma_ok, segs1==263, segs2==649, chunks1==69, chunks2==166]):
        all_ok = False
    print()

    print("=" * 68)
    print(f"FINAL RESULT : {'✅ ALL PASS' if all_ok else '❌ SOME FAILED'}")
    print("=" * 68)
