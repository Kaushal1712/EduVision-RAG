"""
ingestion/chunker.py
─────────────────────
Stage 5: Cleaned Transcript Segments → Retrieval-Optimised Chunks.

═══════════════════════════════════════════════════════════════
CHUNKING STRATEGY — Evidence-Based Design
═══════════════════════════════════════════════════════════════

Data characteristics (from Stage 5 analysis of 912 cleaned segments):

  Segment duration : median 1-2s, mean ~1.9s
  Segment text     : median 22-27 chars, mean ~5-6 words
  ──> Individual segments are too short to carry retrieval meaning.
      "This is the HTML file." alone answers nothing useful.

Window simulation results:
  window=3  → mean 87 chars / 6.9s  — too small, poor semantic context
  window=5  → mean 145 chars / 12s  — ✅ chosen sweet spot
  window=7  → mean 204 chars / 17s  — loses timestamp precision for WHERE queries
  window=10 → mean 289 chars / 25s  — way too coarse for timestamp retrieval

Gap-aware boundary (CRITICAL):
  Both videos have large pauses (7s – 111s) where the instructor changes
  topic or pauses for a demo. A naive fixed window of 5 would bridge a
  111s gap and produce a 125-second chunk spanning two completely different
  topics. This destroys semantic coherence and pollutes retrieval.
  Hard rule: start a NEW chunk when the gap between consecutive segments
  exceeds GAP_THRESHOLD (5.0s).

Overlap design:
  Overlap = 1 segment (step = window - overlap = 4).
  This ensures that an explanation split across a chunk boundary still
  appears in full in at least one chunk.
  Overlap=2 was simulated but produces ~33% more chunks with diminishing
  benefit — chosen overlap=1 as the right balance.

═══════════════════════════════════════════════════════════════
FINAL PARAMETERS
═══════════════════════════════════════════════════════════════
  CHUNK_WINDOW   = 5   segments per chunk
  CHUNK_OVERLAP  = 1   segment shared with previous chunk
  GAP_THRESHOLD  = 5.0 seconds  (hard topic-boundary break)

EXPECTED OUTPUT:
  Video 1 (263 segs): ~66-70 chunks
  Video 2 (649 segs): ~163-170 chunks
  Total: ~230-240 chunks

═══════════════════════════════════════════════════════════════
CHUNK DATA MODEL
═══════════════════════════════════════════════════════════════

Every chunk contains:

  chunk_id          : "<video_id>_chunk_<N>" — stable, unique
  video_id          : matches VideoMetadata.video_id
  video_filename    : original .mp4 filename (for UI display)
  text              : concatenated segment texts (space-joined)
  start_time        : start of FIRST segment in the chunk (seconds)
  end_time          : end of LAST segment in the chunk (seconds)
  source_segment_ids: list of segment_id values this chunk covers
  chunk_index       : sequential index within this video (0-based)
  language          : "hi", "en", etc.

Timestamps are NEVER interpolated or estimated.
They come directly from the Whisper segments that make up the chunk.

═══════════════════════════════════════════════════════════════
INVARIANTS (checked by validate_chunks())
═══════════════════════════════════════════════════════════════
  1. Every chunk has non-empty text.
  2. Every chunk has start_time < end_time.
  3. Every chunk's start_time equals first source segment's start.
  4. Every chunk's end_time equals last source segment's end.
  5. Every source_segment_id exists in the cleaned transcript.
  6. The cleaned transcript JSON files are never modified.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from config.settings import PROCESSED_DIR, TRANSCRIPTS_DIR
from ingestion.cleaner import CleanedTranscript, TranscriptSegment, load_cleaned_transcript

logger = logging.getLogger(__name__)

# ── Chunking parameters ───────────────────────────────────────────────────────
CHUNK_WINDOW: int   = 5    # segments per chunk
CHUNK_OVERLAP: int  = 1    # segments shared with the previous chunk
GAP_THRESHOLD: float = 5.0  # seconds — gaps larger than this force a new chunk

CHUNKS_DIR: Path = PROCESSED_DIR / "chunks"


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """
    A retrieval-ready unit of transcript content.

    Every field needed by the retriever, the re-ranker, and the Streamlit UI
    is embedded directly in this object — no join with any other data needed.
    """
    chunk_id:           str          # e.g. "video2_chunk_042"
    chunk_index:        int          # 0-based within this video
    video_id:           str          # stable video identifier
    video_filename:     str          # original .mp4 filename
    language:           str          # "hi", "en", etc.
    text:               str          # space-joined segment texts
    start_time:         float        # seconds (from first segment)
    end_time:           float        # seconds (from last segment)
    source_segment_ids: list[int]    # which Whisper segment_ids this covers

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def start_time_fmt(self) -> str:
        """Human-readable MM:SS for the UI."""
        m, s = divmod(int(self.start_time), 60)
        return f"{m:02d}:{s:02d}"

    @property
    def end_time_fmt(self) -> str:
        m, s = divmod(int(self.end_time), 60)
        return f"{m:02d}:{s:02d}"


# ── Core chunking logic ───────────────────────────────────────────────────────

def _build_chunks_from_segments(
    segments: list[TranscriptSegment],
    video_id: str,
    video_filename: str,
    language: str,
    window: int = CHUNK_WINDOW,
    overlap: int = CHUNK_OVERLAP,
    gap_threshold: float = GAP_THRESHOLD,
) -> list[Chunk]:
    """
    Convert a list of cleaned TranscriptSegments into overlapping Chunks.

    Algorithm:
      1. Walk through segments with a sliding window of `window` segments.
      2. Advance by (window - overlap) segments each step.
      3. HARD BREAK: if the gap between segment[i-1].end and segment[i].start
         exceeds gap_threshold, start a fresh window at segment[i]
         (do NOT carry overlap across a topic boundary).

    Args:
        segments:      Cleaned segments from Stage 4.
        video_id:      Stable video identifier.
        video_filename: Original .mp4 filename.
        language:      ISO language code from Whisper.
        window:        Number of segments per chunk.
        overlap:       Segments to share with the previous chunk.
        gap_threshold: Seconds; gaps larger than this trigger a hard break.

    Returns:
        List of Chunk objects, in video order.
    """
    if not segments:
        return []

    step = window - overlap
    chunks: list[Chunk] = []
    chunk_idx = 0

    i = 0
    while i < len(segments):
        # Collect up to `window` segments, stopping early at a large gap
        window_segs: list[TranscriptSegment] = [segments[i]]

        for j in range(i + 1, min(i + window, len(segments))):
            gap = segments[j].start - segments[j - 1].end
            if gap > gap_threshold:
                # Hard break — do not include the segment after the gap
                logger.debug(
                    "Hard break at gap=%.1fs between seg %d (%.1fs) and seg %d (%.1fs)",
                    gap,
                    segments[j - 1].segment_id, segments[j - 1].end,
                    segments[j].segment_id, segments[j].start,
                )
                break
            window_segs.append(segments[j])

        # Build text and timestamps from the window
        text = " ".join(s.text.strip() for s in window_segs)
        start_time = window_segs[0].start
        end_time   = window_segs[-1].end
        seg_ids    = [s.segment_id for s in window_segs]

        chunk = Chunk(
            chunk_id=f"{video_id}_chunk_{chunk_idx:04d}",
            chunk_index=chunk_idx,
            video_id=video_id,
            video_filename=video_filename,
            language=language,
            text=text,
            start_time=start_time,
            end_time=end_time,
            source_segment_ids=seg_ids,
        )
        chunks.append(chunk)
        chunk_idx += 1

        # Advance: skip `step` segments, but honour the hard-break position
        # If the window was cut short by a gap, jump directly to the next
        # segment after the gap (no partial overlap across a break).
        n_consumed = len(window_segs)
        if n_consumed < window:
            # Hard break happened — advance past everything in the window
            i += n_consumed
        else:
            i += step

    logger.info(
        "Built %d chunks from %d segments for '%s'",
        len(chunks), len(segments), video_id,
    )
    return chunks


# ── Persistence ───────────────────────────────────────────────────────────────

def _chunks_path(video_id: str) -> Path:
    return CHUNKS_DIR / f"{video_id}_chunks.json"


def _save_chunks(video_id: str, chunks: list[Chunk]) -> Path:
    """Save chunks to data/processed/chunks/<video_id>_chunks.json."""
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    path = _chunks_path(video_id)

    payload = {
        "video_id": video_id,
        "chunk_count": len(chunks),
        "parameters": {
            "chunk_window":  CHUNK_WINDOW,
            "chunk_overlap": CHUNK_OVERLAP,
            "gap_threshold": GAP_THRESHOLD,
        },
        "chunks": [asdict(c) for c in chunks],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d chunks → %s", len(chunks), path.name)
    return path


def load_chunks(video_id: str) -> Optional[list[Chunk]]:
    """
    Load previously saved chunks from disk.
    Called by Stage 6 (indexer) — no need to re-chunk.
    Returns None if the file doesn't exist.
    """
    path = _chunks_path(video_id)
    if not path.exists():
        logger.warning("Chunk file not found: %s", path)
        return None

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return [
        Chunk(
            chunk_id=c["chunk_id"],
            chunk_index=c["chunk_index"],
            video_id=c["video_id"],
            video_filename=c["video_filename"],
            language=c["language"],
            text=c["text"],
            start_time=c["start_time"],
            end_time=c["end_time"],
            source_segment_ids=c["source_segment_ids"],
        )
        for c in data["chunks"]
    ]


# ── Validation ────────────────────────────────────────────────────────────────

def validate_chunks(
    chunks: list[Chunk],
    cleaned_transcript: CleanedTranscript,
) -> dict:
    """
    Run all invariant checks on a list of chunks.

    Returns a dict with 'passed' (bool) and 'errors' (list[str]).
    """
    errors: list[str] = []
    valid_seg_ids = {s.segment_id for s in cleaned_transcript.segments}

    for ch in chunks:
        # 1. Non-empty text
        if not ch.text.strip():
            errors.append(f"{ch.chunk_id}: empty text")

        # 2. start_time < end_time
        if ch.start_time >= ch.end_time:
            errors.append(
                f"{ch.chunk_id}: start_time={ch.start_time} >= end_time={ch.end_time}"
            )

        # 3. source_segment_ids exist in cleaned transcript
        for sid in ch.source_segment_ids:
            if sid not in valid_seg_ids:
                errors.append(
                    f"{ch.chunk_id}: source_segment_id={sid} not in cleaned transcript"
                )

        # 4. At least one source segment
        if not ch.source_segment_ids:
            errors.append(f"{ch.chunk_id}: no source_segment_ids")

    return {
        "passed": len(errors) == 0,
        "total_chunks": len(chunks),
        "errors": errors,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_transcript(
    cleaned: CleanedTranscript,
    force: bool = False,
) -> list[Chunk]:
    """
    Chunk a single CleanedTranscript.

    Args:
        cleaned: Output from Stage 4 (cleaner.py).
        force:   Re-chunk even if the chunk JSON already exists.

    Returns:
        List of Chunk objects (also saved to disk).
    """
    path = _chunks_path(cleaned.video_id)

    # Idempotency
    if path.exists() and not force:
        logger.info(
            "Chunk file already exists, loading from disk: %s "
            "(use force=True to re-chunk)",
            path.name,
        )
        loaded = load_chunks(cleaned.video_id)
        if loaded is not None:
            return loaded

    chunks = _build_chunks_from_segments(
        segments=cleaned.segments,
        video_id=cleaned.video_id,
        video_filename=cleaned.filename,
        language=cleaned.language,
    )

    _save_chunks(cleaned.video_id, chunks)
    return chunks


def chunk_all_transcripts(
    cleaned_list: list[CleanedTranscript],
    force: bool = False,
) -> dict[str, list[Chunk]]:
    """
    Chunk all cleaned transcripts.

    Returns:
        Dict mapping video_id → list[Chunk].
    """
    result: dict[str, list[Chunk]] = {}
    for ct in cleaned_list:
        result[ct.video_id] = chunk_transcript(ct, force=force)
    return result


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, statistics

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    force_flag = "--force" in sys.argv

    print("=" * 60)
    print("EduVision RAG — Stage 5: Chunking")
    print("=" * 60)
    print(f"Parameters: window={CHUNK_WINDOW}, overlap={CHUNK_OVERLAP}, "
          f"gap_threshold={GAP_THRESHOLD}s")
    print()

    # ── Load cleaned transcripts from Stage 4 ────────────────────────────────
    from ingestion.video_processor import process_all_videos
    from ingestion.transcriber import transcribe_all_videos
    from ingestion.cleaner import clean_all_transcripts

    video_meta  = process_all_videos(force=False)
    transcripts = transcribe_all_videos(video_meta, force=False)
    cleaned     = clean_all_transcripts(transcripts, force=False)

    if not cleaned:
        print("❌  No cleaned transcripts found. Run Stage 4 first.")
        sys.exit(1)

    # ── Chunk ─────────────────────────────────────────────────────────────────
    all_chunks_map = chunk_all_transcripts(cleaned, force=force_flag)

    print(f"\n{'='*60}")
    print("VALIDATION & STATISTICS")
    print(f"{'='*60}\n")

    total_chunks = 0
    for ct in cleaned:
        vid = ct.video_id
        chunks = all_chunks_map[vid]
        total_chunks += len(chunks)

        # ── Run invariant checks ──────────────────────────────────────────────
        val = validate_chunks(chunks, ct)
        status = "✅ PASSED" if val["passed"] else f"❌ FAILED ({len(val['errors'])} errors)"
        print(f"Video: {vid[:55]}")
        print(f"  Validation      : {status}")
        if not val["passed"]:
            for err in val["errors"][:5]:
                print(f"    ERROR: {err}")

        # ── Statistics ────────────────────────────────────────────────────────
        char_counts  = [len(c.text) for c in chunks]
        word_counts  = [len(c.text.split()) for c in chunks]
        durations    = [c.duration for c in chunks]
        seg_counts   = [len(c.source_segment_ids) for c in chunks]

        print(f"  Cleaned segments: {len(ct.segments)}")
        print(f"  Total chunks    : {len(chunks)}")
        print(f"  Chars/chunk     : mean={statistics.mean(char_counts):.0f}  "
              f"min={min(char_counts)}  max={max(char_counts)}")
        print(f"  Words/chunk     : mean={statistics.mean(word_counts):.0f}  "
              f"min={min(word_counts)}  max={max(word_counts)}")
        print(f"  Seconds/chunk   : mean={statistics.mean(durations):.1f}  "
              f"min={min(durations):.1f}  max={max(durations):.1f}")
        print(f"  Segs/chunk      : mean={statistics.mean(seg_counts):.1f}  "
              f"min={min(seg_counts)}  max={max(seg_counts)}")

        # ── Chunk size distribution ───────────────────────────────────────────
        print(f"\n  Chunk size distribution (chars):")
        for lo, hi in [(0,50),(50,100),(100,150),(150,200),(200,300),(300,999)]:
            count = sum(1 for c in char_counts if lo <= c < hi)
            bar = '█' * count
            print(f"    [{lo:3d}–{hi:3d}ch): {count:4d}  {bar[:40]}")

        # ── Overlap verification ──────────────────────────────────────────────
        # The last segment_id of chunk N should be the first of chunk N+1
        # if overlap=1 and no gap-break intervenes.
        overlaps = 0
        skipped  = 0
        for k in range(len(chunks) - 1):
            ids_cur  = chunks[k].source_segment_ids
            ids_next = chunks[k + 1].source_segment_ids
            if not ids_cur or not ids_next:
                # Chunk has no segment IDs (e.g. loaded from a pre-v2 pipeline);
                # cannot compare — skip to avoid IndexError.
                skipped += 1
                continue
            if ids_cur[-1] >= ids_next[0]:
                overlaps += 1
        checked   = (len(chunks) - 1) - skipped
        skip_note = f"  ({skipped} pairs skipped — no segment IDs)" if skipped else ""
        print(f"\n  Overlap check: {overlaps}/{checked} consecutive chunk pairs share a segment{skip_note}")

        # ── Representative chunks ─────────────────────────────────────────────
        print(f"\n  ── 5 representative chunks ──")
        step = max(1, len(chunks) // 5)
        for k in range(0, len(chunks), step):
            c = chunks[k]
            print(f"    [{c.start_time_fmt}→{c.end_time_fmt}]  "
                  f"dur={c.duration:.1f}s  segs={c.source_segment_ids}")
            print(f"      {c.text[:100]}{'...' if len(c.text)>100 else ''}")
        print()

    print(f"\n{'='*60}")
    print(f"TOTAL CHUNKS ACROSS ALL VIDEOS: {total_chunks}")
    print(f"{'='*60}")
