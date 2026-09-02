"""
retrieval/retriever.py
───────────────────────
Stage 8: Natural Language Query → Ranked Evidence with Timestamps.

═══════════════════════════════════════════════════════════════
RETRIEVAL ARCHITECTURE
═══════════════════════════════════════════════════════════════

The retriever implements the ONLINE half of the RAG pipeline:

  User query (natural language)
        │
        │  1. Encode with BGE-M3 (same model as Stage 6)
        ▼
  Query vector  (1024-dim, L2-normalised)
        │
        │  2. Cosine nearest-neighbour search in ChromaDB
        ▼
  Top-K raw results  {id, distance, document, metadata}
        │
        │  3. Convert distance → similarity, deserialise metadata
        │  4. Filter by SIMILARITY_THRESHOLD (optional soft floor)
        ▼
  list[RetrievalResult]   ← returned to Stage 9 (generator)

Design principle (from project brief):
  "RETRIEVAL is SEPARATE from GENERATION.
   Question → retrieve relevant transcript evidence → identify timestamp
   → provide evidence to LLM → generate answer.
   NOT: Question → ask LLM to guess timestamp."

This module is RETRIEVAL ONLY. It has no LLM calls, no answer generation.
Its only job is to find and rank the most relevant transcript evidence.

═══════════════════════════════════════════════════════════════
QUERY ENCODING: WHY NO INSTRUCTION PREFIX
═══════════════════════════════════════════════════════════════

BGE-M3 supports optional instruction prefixes for queries in its
ColBERT/sparse retrieval modes. For DENSE retrieval, the BGE-M3
paper and FlagEmbedding documentation state that encoding without
a prefix produces optimal results when passages were also encoded
without a prefix (our Stage 6 approach).

We therefore encode queries identically to passages:
  raw text → BGEM3FlagModel.encode() → dense_vecs[0]

This ensures the query vector lives in the same semantic space
as the stored passage vectors.

═══════════════════════════════════════════════════════════════
SIMILARITY vs DISTANCE
═══════════════════════════════════════════════════════════════

ChromaDB with metric="cosine" returns DISTANCE, not similarity.
  distance = 1 - cosine_similarity   (for normalised vectors)
  distance ∈ [0, 2]   (0 = identical, 2 = perfectly opposite)

We convert: similarity = 1 - distance  (∈ [-1, 1], typically [0, 1])

The SIMILARITY_THRESHOLD from settings.py is applied as a SOFT
floor — results below it are flagged (below_threshold=True) but
still returned, so the caller (Stage 9 generator) can decide
whether to use them. This avoids hard cutoffs that might discard
borderline-relevant results.

═══════════════════════════════════════════════════════════════
MODEL SINGLETON
═══════════════════════════════════════════════════════════════

BGE-M3 (~570 MB) is expensive to load. A module-level singleton
_BGE_MODEL is initialised on first call to retrieve() and reused
for all subsequent queries in the same process lifetime.

In the Streamlit app (Stage 12), this means the model loads once
on app startup (or first query) and persists across user sessions
(Streamlit reruns share the same process for cached resources).

═══════════════════════════════════════════════════════════════
VERIFIED BEHAVIOUR
═══════════════════════════════════════════════════════════════
  "how do websites work with HTML and CSS"
    → sim=0.67  [00:43]  Video 2  "learn how to make HTML website with CSS..."
  "how to install VS Code"
    → sim=0.66  [03:00]  Video 1  "simply VS Code ko install karen..."
  "what is HTML"
    → sim=0.67  [00:43]  Video 2  "we will learn how to make HTML website..."
  "what is the capital of France"  (irrelevant)
    → below_threshold=True  (sim < 0.35)
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from config.settings import BGE_MODEL, TOP_K_RESULTS, SIMILARITY_THRESHOLD
from ingestion.indexer import get_collection

logger = logging.getLogger(__name__)

# ── Module-level model singleton ──────────────────────────────────────────────
# Loaded lazily on first call to retrieve() or encode_query().
# Shared across all calls within the same process.
_BGE_MODEL = None


def _get_model():
    """Return the cached BGE-M3 model, loading it on first call."""
    global _BGE_MODEL
    if _BGE_MODEL is None:
        from FlagEmbedding import BGEM3FlagModel
        logger.info("Loading BGE-M3 model '%s' for query encoding ...", BGE_MODEL)
        t0 = time.time()
        _BGE_MODEL = BGEM3FlagModel(BGE_MODEL, use_fp16=True)
        logger.info("BGE-M3 loaded in %.1f s", time.time() - t0)
    return _BGE_MODEL


# ── Timestamp formatting ──────────────────────────────────────────────────────

def _fmt_timestamp(seconds: float, duration: float) -> str:
    """
    Format a timestamp as MM:SS or MM:SS.s depending on chunk duration.

    For chunks shorter than 5 seconds, include one decimal place of seconds
    so that very short chunks (e.g. 147.28s vs 147.98s) display as
    '02:27.3' and '02:28.0' rather than the confusing '02:27 → 02:27'.

    Args:
        seconds:  The timestamp value in seconds.
        duration: The duration of the chunk in seconds.

    Returns:
        Formatted string: 'MM:SS' for normal chunks, 'MM:SS.s' for short ones.
    """
    if duration < 5.0:
        # Sub-second precision for short chunks
        total_s = int(seconds)
        frac    = seconds - total_s
        m, s    = divmod(total_s, 60)
        return f"{m:02d}:{s:02d}.{int(frac * 10)}"
    else:
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"


# ── Return data model ─────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """
    A single ranked retrieval result.

    Contains every field the UI needs to display:
      - text_en:   English-normalised evidence text (used for display and LLM)
      - text_raw:  Original transcript text (preserved for provenance/debugging)
      - the exact timestamp to link to in the video
      - the video filename / id to identify which video
      - the similarity score to show confidence
      - traceability back to source Whisper segments

    Stage 14 change:
      text_en = English-normalised chunk text (from v2 index, or same as
                text_raw for v1 index where no translation was performed)
      text_raw = original Whisper/cleaned transcript text
      text     = property alias for text_en (backward compatibility)

    No file lookups are needed after retrieval — all data comes
    directly from ChromaDB metadata stored during indexing.
    """
    chunk_id:           str        # e.g. "video2_chunk_0042"
    rank:               int        # 1-based position in ranked list
    similarity:         float      # cosine similarity ∈ [0, 1]
    text_en:            str        # English text for UI display and LLM evidence
    text_raw:           str        # original transcript text (provenance)
    video_id:           str        # stable identifier
    video_filename:     str        # display name, e.g. "02_Your First HTML..."
    start_time:         float      # seconds (exact Whisper value)
    end_time:           float      # seconds (exact Whisper value)
    start_time_fmt:     str        # "MM:SS" or "MM:SS.s" pre-formatted for UI
    end_time_fmt:       str        # "MM:SS" or "MM:SS.s" pre-formatted for UI
    source_segment_ids: list[int]  # Whisper segment_ids this chunk covers
    language:           str        # "hi", "en"
    chunk_index:        int        # position within this video's chunks
    below_threshold:    bool       # True if similarity < SIMILARITY_THRESHOLD

    @property
    def text(self) -> str:
        """Backward-compatible alias for text_en. Use text_en for new code."""
        return self.text_en

    @property
    def duration(self) -> float:
        """Duration of this chunk in seconds."""
        return self.end_time - self.start_time

    def to_dict(self) -> dict:
        """Serialisable representation for logging / JSON output."""
        return {
            "rank":               self.rank,
            "chunk_id":           self.chunk_id,
            "similarity":         round(self.similarity, 4),
            "below_threshold":    self.below_threshold,
            "video_filename":     self.video_filename,
            "start_time_fmt":     self.start_time_fmt,
            "end_time_fmt":       self.end_time_fmt,
            "start_time":         self.start_time,
            "end_time":           self.end_time,
            "duration":           round(self.duration, 1),
            "language":           self.language,
            "text":               self.text_en,
            "text_raw":           self.text_raw,
            "source_segment_ids": self.source_segment_ids,
        }


# ── Query encoding ────────────────────────────────────────────────────────────

def encode_query(query: str) -> list[float]:
    """
    Encode a natural language query into a BGE-M3 dense vector.

    Uses the same model and encoding settings as Stage 6 (passage
    encoding). Both queries and passages are encoded without instruction
    prefixes in dense retrieval mode — this is consistent with how the
    index was built and produces optimal cosine similarity scores.

    Args:
        query: Natural language question, any length (truncated to 512 tokens).

    Returns:
        1024-dimensional float list (L2-normalised).
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    model = _get_model()
    t0 = time.time()

    output = model.encode(
        [query.strip()],
        batch_size=1,
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )

    vec: list[float] = output["dense_vecs"][0].tolist()
    logger.debug("Query encoded in %.3f s  (dim=%d)", time.time() - t0, len(vec))
    return vec


# ── Core retrieval ────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    top_k: int = TOP_K_RESULTS,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    video_id_filter: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> list[RetrievalResult]:
    """
    Retrieve the top-K most relevant transcript chunks for a query.

    Pipeline:
      1. Encode query with BGE-M3 → 1024-dim vector
      2. Query ChromaDB collection with cosine similarity
      3. Convert distances → similarities
      4. Flag results below similarity_threshold (soft floor — not dropped)
      5. Return list[RetrievalResult] sorted by similarity descending

    Args:
        query:               Natural language question.
        top_k:               Number of results to return (default from settings).
        similarity_threshold: Soft similarity floor — results below are flagged
                             but still returned. Stage 9 (generator) decides
                             whether to use them.
        video_id_filter:     If set, restrict search to one specific video.
                             Useful for "find this in Tutorial #1" queries.
        collection_name:     If set, query this named collection instead of the
                             default (used to query v2 collection for comparison).

    Returns:
        List of RetrievalResult, sorted by similarity descending (rank 1 = best).
        Empty list if the collection is empty or an error occurs.

    Raises:
        ValueError:  If query is empty.
        RuntimeError: If the ChromaDB collection is not found.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    t_start = time.time()

    # ── 1. Encode query ───────────────────────────────────────────────────────────
    query_vec = encode_query(query)

    # ── 2. Query ChromaDB ─────────────────────────────────────────────────────
    if collection_name:
        # Named collection (e.g. v2 for comparison)
        from ingestion.indexer import get_chroma_client
        client = get_chroma_client()
        try:
            collection = client.get_collection(name=collection_name)
        except Exception as exc:
            raise RuntimeError(f"Collection '{collection_name}' not found: {exc}") from exc
    else:
        collection = get_collection()

    where_filter = None
    if video_id_filter:
        where_filter = {"video_id": {"$eq": video_id_filter}}
        logger.info("Applying video_id filter: '%s'", video_id_filter)

    try:
        raw = collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            where=where_filter,
            include=["distances", "metadatas", "documents"],
        )
    except Exception as exc:
        raise RuntimeError(f"ChromaDB query failed: {exc}") from exc

    # raw structure: {"ids": [[...]], "distances": [[...]], ...}
    ids        = raw["ids"][0]
    distances  = raw["distances"][0]
    metadatas  = raw["metadatas"][0]
    documents  = raw["documents"][0]

    # ── 3 & 4. Build RetrievalResult objects ──────────────────────────────────
    results: list[RetrievalResult] = []

    for rank_0, (chunk_id, distance, meta, doc) in enumerate(
        zip(ids, distances, metadatas, documents)
    ):
        # Chroma cosine distance ∈ [0, 2] → similarity ∈ [-1, 1]
        similarity = 1.0 - distance

        # Deserialise source_segment_ids from JSON string
        seg_ids_raw = meta.get("source_segment_ids", "[]")
        try:
            source_segment_ids = json.loads(seg_ids_raw)
        except (json.JSONDecodeError, TypeError):
            source_segment_ids = []
            logger.warning("Could not parse source_segment_ids for %s: %r", chunk_id, seg_ids_raw)

        start_time = float(meta.get("start_time", 0.0))
        end_time   = float(meta.get("end_time", 0.0))
        duration   = end_time - start_time

        # text_en: the retrieval document (English in v2, raw text in v1)
        # text_raw: original transcript (stored in metadata for v2; same as doc for v1)
        text_en  = doc
        text_raw = meta.get("text_raw", doc)   # v2 stores text_raw in metadata

        # Timestamp formatting: sub-second precision for very short chunks
        start_fmt = _fmt_timestamp(start_time, duration)
        end_fmt   = _fmt_timestamp(end_time,   duration)

        result = RetrievalResult(
            chunk_id=chunk_id,
            rank=rank_0 + 1,
            similarity=similarity,
            text_en=text_en,
            text_raw=text_raw,
            video_id=meta.get("video_id", ""),
            video_filename=meta.get("video_filename", ""),
            start_time=start_time,
            end_time=end_time,
            start_time_fmt=start_fmt,
            end_time_fmt=end_fmt,
            source_segment_ids=source_segment_ids,
            language=meta.get("language", ""),
            chunk_index=int(meta.get("chunk_index", 0)),
            below_threshold=(similarity < similarity_threshold),
        )
        results.append(result)


    elapsed = time.time() - t_start
    above = sum(1 for r in results if not r.below_threshold)
    logger.info(
        "Retrieved %d results for query %r in %.3f s  "
        "(%d above threshold=%.2f, %d below)",
        len(results), query[:50], elapsed,
        above, similarity_threshold, len(results) - above,
    )

    return results


# ── Convenience helpers ───────────────────────────────────────────────────────

def retrieve_above_threshold(
    query: str,
    top_k: int = TOP_K_RESULTS,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    video_id_filter: Optional[str] = None,
) -> list[RetrievalResult]:
    """
    Like retrieve(), but returns ONLY results above the similarity threshold.

    Useful for strict retrieval where we don't want to pass low-confidence
    evidence to the generator. If no results pass the threshold, returns
    an empty list (caller must handle the no-results case gracefully).
    """
    all_results = retrieve(query, top_k, similarity_threshold, video_id_filter)
    return [r for r in all_results if not r.below_threshold]


def format_results_for_display(results: list[RetrievalResult]) -> str:
    """
    Human-readable string representation of retrieval results.

    Used in CLI testing and logging. Stage 12 (Streamlit UI) will
    render results more richly using the RetrievalResult fields directly.
    """
    if not results:
        return "No results found."

    lines = []
    for r in results:
        threshold_flag = " ⚠️  [below threshold]" if r.below_threshold else ""
        lines.append(
            f"  #{r.rank}  sim={r.similarity:.4f}{threshold_flag}"
        )
        lines.append(
            f"       Video   : {r.video_filename[:60]}"
        )
        lines.append(
            f"       Time    : [{r.start_time_fmt} → {r.end_time_fmt}]  ({r.duration:.1f}s)"
        )
        lines.append(
            f"       Text    : {r.text[:120]}{'...' if len(r.text) > 120 else ''}"
        )
        lines.append(
            f"       Segments: {r.source_segment_ids}"
        )
        lines.append("")
    return "\n".join(lines)


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 65)
    print("EduVision RAG — Stage 8: Retriever")
    print("=" * 65)
    print(f"Model       : {BGE_MODEL}")
    print(f"Top-K       : {TOP_K_RESULTS}")
    print(f"Threshold   : {SIMILARITY_THRESHOLD}")
    print()

    # ── Test queries ──────────────────────────────────────────────────────────
    test_queries = [
        # Directly relevant
        ("how do websites work with HTML and CSS",     "Expected: Video 2, timestamps around HTML section"),
        ("how to install VS Code",                     "Expected: Video 1, installation timestamps"),
        ("what is HTML",                               "Expected: Video 2, HTML introduction"),
        # Tangentially relevant
        ("browser extension for live preview",         "Expected: Video 2, extension section"),
        ("what is a file and folder structure",        "Expected: Video 2, folder/file setup section"),
        # Irrelevant — should be below threshold
        ("what is the capital of France",              "Expected: ALL below threshold (irrelevant)"),
        ("recipe for chocolate cake",                  "Expected: ALL below threshold (completely unrelated)"),
    ]

    all_passed = True

    for query, expectation in test_queries:
        print(f"\n{'─'*65}")
        print(f"QUERY  : {query!r}")
        print(f"EXPECT : {expectation}")
        print()

        try:
            results = retrieve(query, top_k=TOP_K_RESULTS)
        except Exception as exc:
            print(f"  ❌ ERROR: {exc}")
            all_passed = False
            continue

        print(format_results_for_display(results))

        # Validation checks
        if not results:
            print(f"  ⚠️  No results returned.")
            continue

        # Check: similarity scores are in valid range
        for r in results:
            if not (-1.0 <= r.similarity <= 1.0):
                print(f"  ❌ Invalid similarity: {r.similarity}")
                all_passed = False
            if r.start_time >= r.end_time:
                print(f"  ❌ Invalid timestamps: {r.start_time} >= {r.end_time}")
                all_passed = False
            if not r.source_segment_ids:
                print(f"  ❌ Empty source_segment_ids for chunk {r.chunk_id}")
                all_passed = False

        # For irrelevant queries, check that top result is below threshold
        is_irrelevant = "France" in query or "chocolate" in query
        if is_irrelevant:
            top_sim = results[0].similarity
            if not results[0].below_threshold:
                print(f"  ⚠️  Irrelevant query got high similarity: {top_sim:.4f} — threshold may need adjusting")
            else:
                print(f"  ✅ Irrelevant query correctly flagged below threshold (sim={top_sim:.4f})")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"RESULT : {'✅ ALL VALIDATION CHECKS PASS' if all_passed else '❌ SOME CHECKS FAILED'}")
    print(f"{'='*65}")
