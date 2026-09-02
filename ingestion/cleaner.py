"""
ingestion/cleaner.py
─────────────────────
Stage 4: Transcript Cleaning + Timestamp-Preserving Segmentation.

WHAT THIS MODULE DOES:
  1. Loads raw transcript JSON produced by Stage 3 (Whisper).
  2. Filters out noise segments using evidence-based thresholds.
  3. NEVER removes timestamps — only the segment's text content.
  4. Saves cleaned segments to data/transcripts/<video_id>_cleaned.json.
  5. Returns CleanedTranscript for Stage 5 (chunking).

FILTER DESIGN — derived from actual data analysis:
═══════════════════════════════════════════════════

  Whisper confidence signals:
    avg_logprob  : log-probability of the transcription.
                   Range: -7.0 (terrible) → 0.0 (perfect).
                   Our data: mean ≈ -0.66, most segments between -0.75 and -0.25.
    no_speech_prob: probability that the audio segment contains NO speech.
                   Range: 0.0 (definitely speech) → 1.0 (definitely silence).

  What we observed in our two videos (1019 total segments):
    • Video 1 (339 segs): high no_speech_prob (mean=0.78) — likely instructor
      talking over a screen with background noise. 76 segments are noise.
    • Video 2 (680 segs): better quality (mean nsp=0.40). 31 segments are noise.

  FOUR FILTER RULES (applied in order):

    R1 — Extreme low confidence (logprob < -2.5):
         These are Whisper "best guesses" that it is nearly certain are wrong.
         Example: 'аgeometric' at logp=-6.81. Safe to drop.

    R2 — Low confidence + silence signal (logprob < -1.5 AND nsp > 0.8):
         Transcription is uncertain AND audio is mostly non-speech.
         Likely: music, background noise, screen-recording clicks.

    R3 — Tiny high-silence segment (nsp > 0.98 AND text < 5 chars):
         Almost certainly a mis-transcribed silence or click artifact.
         Example: 'گٹھپ' at nsp=0.98.

    R4 — Repetitive hallucination (same text ≥ 3 times in last 5 segments):
         Whisper's failure mode on long silences: it repeats the same phrase.
         Example: 57 consecutive copies of 'aap we', 'بیس کوژ کیوں نہیں'.

  TECH-CONTENT RESCUE OVERRIDE (applied after all filters):
         If a segment would be dropped by any rule BUT contains a known
         technical keyword (html, css, javascript, browser, etc.),
         it is ALWAYS kept. This ensures no pedagogically important
         moment is silently discarded.

  RESULT (after word-boundary fix):
    Video 1: 339 → kept (audit pending re-run)
    Video 2: 680 → kept (audit pending re-run)

  TIMESTAMPS: NEVER MODIFIED.
    We only filter segments. The start/end times on kept segments
    are identical to what Whisper produced. No timestamp interpolation
    or adjustment is performed.

  TECH-TERM MATCHING — AUDIT FINDINGS (2026-08-28):
    Original implementation used naive substring matching (term in text.lower()).
    A full collision audit against 928 cleaned segments revealed 16 terms with
    false positives. Worst offenders:

      'p'  → 233 FPs  (matched any word containing 'p' — removed entirely)
      'li' →  52 FPs  (matched 'like', 'ability', etc. — removed entirely)
      'id' →  45 FPs  (matched 'video', Urdu text — removed entirely)
      'ol' →  31 FPs  (matched 'follow', 'roll' — removed entirely)
      'ul' →   5 FPs  (matched 'full', 'pull' — removed entirely)
      'web'→  33 FPs  (matched inside 'website' — now word-boundary)
      'script'→14 FPs (matched inside 'javascript' — now word-boundary)
      'code' → 13 FPs (matched inside 'vscode' — now word-boundary)

    Fix: switch to regex \b word-boundary matching for all terms.
    Remove terms that are ambiguous even with word boundaries.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config.settings import TRANSCRIPTS_DIR
from ingestion.transcriber import TranscriptSegment, TranscriptResult, load_transcript

logger = logging.getLogger(__name__)


# ── Technical keyword vocabulary ─────────────────────────────────────────────
# Any segment matching one of these WHOLE WORDS is ALWAYS kept, even if
# it would be discarded by the noise filters.
#
# RULES FOR ADDING TERMS:
#   - Only add terms that are unambiguous as standalone spoken words.
#   - Avoid single letters and 2-letter terms unless they are pronounced
#     as acronyms (e.g. 'js', 'css', 'h1').
#   - REMOVED: 'p', 'li', 'ol', 'ul', 'id'
#     Reason: too short / too common in non-technical speech.
#     Use 'paragraph', 'list-item', 'ordered list' etc. if needed.

_TECH_TERMS_RAW: list[str] = [
    # Web fundamentals
    "html", "css", "javascript", "js", "browser", "server", "client",
    "http", "https", "url", "website", "web", "internet",
    # HTML elements / structure  (removed: 'p', 'ol', 'ul', 'li')
    "tag", "element", "attribute", "doctype", "boilerplate",
    "head", "body", "title", "div", "span", "h1", "h2",
    "anchor", "link", "img", "input", "form", "table",
    # CSS  (removed: 'id' — too generic)
    "style", "selector", "class", "margin", "padding",
    "flexbox", "grid", "color", "font", "border",
    # JavaScript / DOM
    "script", "dom", "function", "variable", "const", "let", "var",
    "event", "onclick", "console", "alert",
    # Tools / workflow
    "vscode", "editor", "terminal", "file", "folder", "extension",
    "install", "download", "github", "code", "source",
    # Hosting / deployment
    "frontend", "backend", "host", "deploy", "domain",
]

# Pre-compiled regex: \b<term>\b for each term, combined with | for efficiency.
# re.IGNORECASE so 'HTML', 'Html', 'html' all match.
_TECH_PATTERN: re.Pattern = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in _TECH_TERMS_RAW) + r")\b",
    re.IGNORECASE,
)


def _has_tech_content(text: str) -> bool:
    """
    Return True if text contains any technical keyword as a WHOLE WORD.

    Uses regex word-boundary matching (\\b) instead of substring search.
    This prevents false positives like:
      - 'p'      matching any word containing 'p'
      - 'id'     matching 'video', 'confident'
      - 'li'     matching 'like', 'ability'
      - 'script' matching inside 'javascript', 'description'
      - 'web'    matching inside 'website'
      - 'code'   matching inside 'vscode'
    """
    return bool(_TECH_PATTERN.search(text))


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class CleaningReport:
    """Statistics about what was filtered — useful for debugging and tuning."""
    video_id: str
    input_count: int
    kept_count: int
    removed_count: int
    tech_rescued_count: int
    removal_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def retention_rate(self) -> float:
        return self.kept_count / self.input_count if self.input_count else 0.0


@dataclass
class CleanedTranscript:
    """
    Output of Stage 4.
    Contains only the cleaned segments; timestamps are untouched.
    Passed directly to Stage 5 (chunker).
    """
    video_id: str
    filename: str
    duration_seconds: float
    language: str
    segments: list[TranscriptSegment]   # cleaned subset, timestamps intact
    report: CleaningReport
    cleaned_path: Path


# ── Filter logic ──────────────────────────────────────────────────────────────

def _classify_segment(
    seg: TranscriptSegment,
    all_segments: list[TranscriptSegment],
    idx: int,
) -> tuple[bool, list[str]]:
    """
    Decide whether to discard a segment.

    Returns:
        (should_discard: bool, reasons: list[str])

    The tech-rescue override is NOT applied here — it is applied by the caller
    so we can count how many segments were rescued.
    """
    text  = seg.text.strip()
    logp  = seg.avg_logprob
    nsp   = seg.no_speech_prob
    reasons: list[str] = []

    # R1: Extreme low confidence
    if logp < -2.5:
        reasons.append("logprob<-2.5")

    # R2: Moderately low confidence + silence signal
    if logp < -1.5 and nsp > 0.8:
        reasons.append("low-conf+high-nsp")

    # R3: Tiny text with very high silence probability
    if nsp > 0.98 and len(text) < 5:
        reasons.append("nsp>0.98+tiny")

    # R4: Repetitive hallucination — same text ≥ 3 times in the last 5 segments
    if idx >= 2:
        window = [all_segments[j].text.strip() for j in range(max(0, idx - 5), idx)]
        if window.count(text) >= 2:
            reasons.append("repetitive")

    return bool(reasons), reasons


def clean_transcript(transcript: TranscriptResult) -> CleanedTranscript:
    """
    Apply noise filters to a TranscriptResult and return a CleanedTranscript.

    Rules are applied from Stage 4 design above.
    Tech-content rescue is always the final override.
    Timestamps are never modified.
    """
    segs = transcript.segments
    report = CleaningReport(
        video_id=transcript.video_id,
        input_count=len(segs),
        kept_count=0,
        removed_count=0,
        tech_rescued_count=0,
    )

    clean_segs: list[TranscriptSegment] = []

    for idx, seg in enumerate(segs):
        should_drop, reasons = _classify_segment(seg, segs, idx)

        if not should_drop:
            clean_segs.append(seg)
            continue

        # Tech-rescue override
        if _has_tech_content(seg.text):
            logger.debug(
                "Tech-rescued at %s: %r (would have been dropped for: %s)",
                _fmt_time(seg.start), seg.text[:60], reasons,
            )
            clean_segs.append(seg)
            report.tech_rescued_count += 1
            continue

        # Discard
        logger.debug(
            "Discarding [%s] %r  reasons=%s",
            _fmt_time(seg.start), seg.text[:60], reasons,
        )
        report.removed_count += 1
        for r in reasons:
            report.removal_reasons[r] = report.removal_reasons.get(r, 0) + 1

    report.kept_count = len(clean_segs)

    # ── Save cleaned transcript ───────────────────────────────────────────────
    cleaned_path = TRANSCRIPTS_DIR / f"{transcript.video_id}_cleaned.json"
    _save_cleaned(transcript, clean_segs, report, cleaned_path)

    logger.info(
        "Cleaning complete for '%s': %d → %d segments "
        "(removed %d | tech-rescued %d | retention %.1f%%)",
        transcript.video_id[:40],
        report.input_count,
        report.kept_count,
        report.removed_count,
        report.tech_rescued_count,
        report.retention_rate * 100,
    )

    return CleanedTranscript(
        video_id=transcript.video_id,
        filename=transcript.filename,
        duration_seconds=transcript.duration_seconds,
        language=transcript.language,
        segments=clean_segs,
        report=report,
        cleaned_path=cleaned_path,
    )


# ── Persistence ───────────────────────────────────────────────────────────────

def _save_cleaned(
    original: TranscriptResult,
    clean_segs: list[TranscriptSegment],
    report: CleaningReport,
    path: Path,
) -> None:
    """Save cleaned segments + cleaning report to JSON."""
    payload = {
        "video_id": original.video_id,
        "filename": original.filename,
        "duration_seconds": original.duration_seconds,
        "language": original.language,
        "whisper_model": original.whisper_model,
        "cleaning_report": {
            "input_count":        report.input_count,
            "kept_count":         report.kept_count,
            "removed_count":      report.removed_count,
            "tech_rescued_count": report.tech_rescued_count,
            "retention_rate_pct": round(report.retention_rate * 100, 1),
            "removal_reasons":    report.removal_reasons,
        },
        "segments": [
            {
                "segment_id":     s.segment_id,
                "start":          s.start,
                "end":            s.end,
                "text":           s.text,
                "avg_logprob":    s.avg_logprob,
                "no_speech_prob": s.no_speech_prob,
            }
            for s in clean_segs
        ],
    }
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Saved cleaned transcript: %s", path.name)


def load_cleaned_transcript(video_id: str) -> Optional[CleanedTranscript]:
    """
    Load a previously cleaned transcript from disk.

    Called by Stage 5 (chunker) — it does NOT need to re-clean.
    Returns None if the file doesn't exist.
    """
    path = TRANSCRIPTS_DIR / f"{video_id}_cleaned.json"
    if not path.exists():
        logger.warning("Cleaned transcript not found: %s", path)
        return None

    with open(path, encoding="utf-8") as f:
        d = json.load(f)

    r = d["cleaning_report"]
    report = CleaningReport(
        video_id=d["video_id"],
        input_count=r["input_count"],
        kept_count=r["kept_count"],
        removed_count=r["removed_count"],
        tech_rescued_count=r["tech_rescued_count"],
        removal_reasons=r.get("removal_reasons", {}),
    )

    segments = [
        TranscriptSegment(
            segment_id=s["segment_id"],
            start=s["start"],
            end=s["end"],
            text=s["text"],
            avg_logprob=s["avg_logprob"],
            no_speech_prob=s["no_speech_prob"],
        )
        for s in d["segments"]
    ]

    return CleanedTranscript(
        video_id=d["video_id"],
        filename=d["filename"],
        duration_seconds=d["duration_seconds"],
        language=d["language"],
        segments=segments,
        report=report,
        cleaned_path=path,
    )


def clean_all_transcripts(
    transcripts: list[TranscriptResult],
    force: bool = False,
) -> list[CleanedTranscript]:
    """
    Clean all transcripts.

    Args:
        transcripts: List of TranscriptResult from Stage 3.
        force:       Re-clean even if cleaned JSON already exists.

    Returns:
        List of CleanedTranscript.
    """
    results: list[CleanedTranscript] = []

    for tr in transcripts:
        cleaned_path = TRANSCRIPTS_DIR / f"{tr.video_id}_cleaned.json"

        if cleaned_path.exists() and not force:
            logger.info(
                "Cleaned transcript already exists, loading from disk: %s",
                cleaned_path.name,
            )
            ct = load_cleaned_transcript(tr.video_id)
            if ct:
                results.append(ct)
                continue

        ct = clean_transcript(tr)
        results.append(ct)

    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    """Format seconds as MM:SS for display."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    force_flag = "--force" in sys.argv

    print("=" * 60)
    print("EduVision RAG — Stage 4: Transcript Cleaning")
    print("=" * 60)

    # Load raw transcripts from Stage 3 (no need to re-run Whisper)
    from ingestion.video_processor import process_all_videos
    from ingestion.transcriber import transcribe_all_videos

    video_meta = process_all_videos(force=False)
    transcripts = transcribe_all_videos(video_meta, force=False)

    if not transcripts:
        print("❌  No transcripts found. Run Stage 3 first.")
        sys.exit(1)

    cleaned = clean_all_transcripts(transcripts, force=force_flag)

    print(f"\n✅  Cleaned {len(cleaned)} transcript(s):\n")
    total_in = total_kept = total_rescued = 0

    for ct in cleaned:
        r = ct.report
        total_in      += r.input_count
        total_kept    += r.kept_count
        total_rescued += r.tech_rescued_count

        print(f"  video_id       : {ct.video_id[:55]}")
        print(f"  input segments : {r.input_count}")
        print(f"  kept           : {r.kept_count}  ({r.retention_rate*100:.1f}%)")
        print(f"  removed        : {r.removed_count}")
        print(f"  tech-rescued   : {r.tech_rescued_count}")
        print(f"  removal reasons: {r.removal_reasons}")
        print(f"  saved to       : {ct.cleaned_path.name}")
        print()

        # Show a sample of kept segments around a key topic
        html_segs = [
            s for s in ct.segments
            if any(t in s.text.lower() for t in ["html", "css", "javascript"])
        ]
        if html_segs:
            print(f"  ── Sample kept tech segments ──")
            for s in html_segs[:5]:
                m, sec = divmod(int(s.start), 60)
                print(f"    [{m:02d}:{sec:02d}]  {s.text}")
        print()

    print(f"TOTAL: {total_in} → {total_kept} kept ({100*total_kept/total_in:.1f}%) | {total_rescued} tech-rescued")
