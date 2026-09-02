"""
ingestion/transcriber.py
─────────────────────────
Stage 3: Audio → Timestamped Transcript using OpenAI Whisper.

Responsibilities:
  1. Load the extracted WAV audio for a video.
  2. Run Whisper transcription to produce time-stamped segments.
  3. Save the raw Whisper output + clean segment list to JSON.
  4. Return a structured TranscriptResult for downstream stages.

Why Whisper?
  - Best open-source ASR model; strong accuracy on tutorial/lecture audio.
  - Its output already contains segment-level start/end timestamps, which
    is the foundation of our entire timestamp retrieval system.
  - Runs fully locally — no API cost for transcription.

Why save to JSON?
  - Transcription is the most time-consuming step (~3–10× real-time on CPU).
  - Persisting results means we NEVER re-transcribe the same video.
  - The JSON file is also readable for manual inspection/debugging.

Output format (data/transcripts/<video_id>.json):
  {
    "video_id": "...",
    "filename": "...",
    "duration_seconds": 1879.4,
    "whisper_model": "base",
    "language": "en",
    "transcribed_at": "2026-08-28T10:44:00",
    "full_text": "...",
    "segments": [
      {
        "segment_id": 0,
        "start": 0.0,
        "end": 4.72,
        "text": "Welcome to the Sigma Web Development course.",
        "avg_logprob": -0.23,
        "no_speech_prob": 0.01
      },
      ...
    ]
  }

Each segment is one Whisper sentence/phrase — typically 2–15 seconds long.
These are the atomic units that Stage 4 will clean and Stage 5 will chunk.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.settings import TRANSCRIPTS_DIR, WHISPER_MODEL

logger = logging.getLogger(__name__)


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class TranscriptSegment:
    """
    One Whisper segment — a short phrase with precise timestamps.

    This is the atomic unit of the entire pipeline.
    Every retrieved chunk maps back to one or more of these segments.
    """
    segment_id: int          # sequential index within the video (0-based)
    start: float             # start time in seconds (e.g. 19.22 → 0:19)
    end: float               # end time in seconds
    text: str                # transcribed text (already stripped of whitespace)
    avg_logprob: float       # Whisper's confidence proxy (higher = more confident)
    no_speech_prob: float    # probability this segment is silence/noise


@dataclass
class TranscriptResult:
    """
    Complete transcription result for one video.
    Passed from this module to the chunker (Stage 4/5).
    """
    video_id: str
    filename: str
    duration_seconds: float
    whisper_model: str
    language: str
    full_text: str
    segments: list[TranscriptSegment]
    transcript_path: Path    # where the JSON was saved


# ── Core functions ────────────────────────────────────────────────────────────

def _load_whisper_model(model_name: str):
    """
    Load the Whisper model.

    We load it lazily inside this function rather than at module import
    time so that importing transcriber.py doesn't immediately download
    the model — useful for testing and for the Streamlit app that imports
    this module but may not need to transcribe anything.

    The model is NOT cached as a module-level global here intentionally:
    callers that process multiple videos should pass the model object
    around themselves (see transcribe_all_videos).
    """
    import whisper  # local import — only needed at transcription time

    logger.info("Loading Whisper model: '%s' ...", model_name)
    start = time.time()
    model = whisper.load_model(model_name)
    elapsed = time.time() - start
    logger.info("Whisper model loaded in %.1f s", elapsed)
    return model


def _parse_segments(raw_segments: list[dict]) -> list[TranscriptSegment]:
    """
    Convert Whisper's raw segment dicts into clean TranscriptSegment objects.

    We only keep the fields we actually use downstream. Whisper also
    returns token IDs, compression ratios, and seek values — those are
    dropped here to keep the JSON files compact.
    """
    segments = []
    for i, seg in enumerate(raw_segments):
        text = seg.get("text", "").strip()
        if not text:          # skip empty/silence segments
            continue
        segments.append(
            TranscriptSegment(
                segment_id=i,
                start=round(seg["start"], 3),
                end=round(seg["end"], 3),
                text=text,
                avg_logprob=round(seg.get("avg_logprob", 0.0), 4),
                no_speech_prob=round(seg.get("no_speech_prob", 0.0), 4),
            )
        )
    return segments


def _save_transcript(result: TranscriptResult) -> None:
    """
    Persist the transcript as a JSON file in data/transcripts/.

    Having a human-readable JSON lets you:
      - Inspect what Whisper produced before committing to embedding.
      - Re-run chunking/indexing without re-transcribing.
      - Manually correct transcription errors if needed.
    """
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "video_id": result.video_id,
        "filename": result.filename,
        "duration_seconds": result.duration_seconds,
        "whisper_model": result.whisper_model,
        "language": result.language,
        "transcribed_at": datetime.now().isoformat(timespec="seconds"),
        "full_text": result.full_text,
        "segments": [
            {
                "segment_id": s.segment_id,
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "avg_logprob": s.avg_logprob,
                "no_speech_prob": s.no_speech_prob,
            }
            for s in result.segments
        ],
    }

    with open(result.transcript_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("Transcript saved: %s", result.transcript_path)


def load_transcript(video_id: str) -> Optional[TranscriptResult]:
    """
    Load a previously saved transcript JSON back into a TranscriptResult.

    Called by the chunker/indexer so they don't depend on Whisper being
    installed — they just read the saved JSON.

    Returns None if the transcript file doesn't exist.
    """
    path = TRANSCRIPTS_DIR / f"{video_id}.json"
    if not path.exists():
        logger.warning("Transcript not found: %s", path)
        return None

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    segments = [
        TranscriptSegment(
            segment_id=s["segment_id"],
            start=s["start"],
            end=s["end"],
            text=s["text"],
            avg_logprob=s["avg_logprob"],
            no_speech_prob=s["no_speech_prob"],
        )
        for s in data["segments"]
    ]

    return TranscriptResult(
        video_id=data["video_id"],
        filename=data["filename"],
        duration_seconds=data["duration_seconds"],
        whisper_model=data["whisper_model"],
        language=data["language"],
        full_text=data["full_text"],
        segments=segments,
        transcript_path=path,
    )


def transcribe_video(
    video_id: str,
    filename: str,
    audio_path: Path,
    duration_seconds: float,
    model=None,
    force: bool = False,
) -> Optional[TranscriptResult]:
    """
    Transcribe a single audio file using Whisper.

    Args:
        video_id:         Stable identifier (from VideoMetadata).
        filename:         Original .mp4 filename (stored in JSON for display).
        audio_path:       Path to the extracted .wav file.
        duration_seconds: Video duration (stored in JSON).
        model:            Pre-loaded Whisper model object. If None, loads it.
                          Pass the same model when transcribing multiple videos
                          to avoid loading it repeatedly.
        force:            Re-transcribe even if JSON already exists.

    Returns:
        TranscriptResult on success, None on failure.
    """
    transcript_path = TRANSCRIPTS_DIR / f"{video_id}.json"

    # ── Idempotency: skip if transcript already exists ────────────────────────
    if transcript_path.exists() and not force:
        logger.info(
            "Transcript already exists, loading from disk: %s "
            "(use force=True to re-transcribe)",
            transcript_path.name,
        )
        return load_transcript(video_id)

    if not audio_path.exists():
        logger.error("Audio file not found: %s — run Stage 2 first.", audio_path)
        return None

    # ── Load model if not provided ────────────────────────────────────────────
    if model is None:
        model = _load_whisper_model(WHISPER_MODEL)

    # ── Run Whisper ───────────────────────────────────────────────────────────
    logger.info(
        "Transcribing '%s' (%.1f min) with Whisper '%s' ...",
        filename,
        duration_seconds / 60,
        WHISPER_MODEL,
    )
    t_start = time.time()

    try:
        raw_result = model.transcribe(
            str(audio_path),
            # language="en",    # uncomment to force English (slightly faster)
            verbose=False,      # suppress per-segment stdout noise
            fp16=False,         # fp16 requires CUDA GPU; keep False for CPU
            word_timestamps=False,  # segment-level timestamps are sufficient
        )
    except Exception as e:
        logger.error("Whisper transcription failed for %s: %s", filename, e)
        return None

    elapsed = time.time() - t_start
    speed_ratio = duration_seconds / elapsed  # >1.0 means faster than real-time
    logger.info(
        "Transcription complete in %.1f s (%.1f× real-time speed)",
        elapsed,
        speed_ratio,
    )

    # ── Parse and structure ───────────────────────────────────────────────────
    segments = _parse_segments(raw_result.get("segments", []))
    language = raw_result.get("language", "unknown")

    logger.info(
        "Parsed %d segments | language detected: %s",
        len(segments),
        language,
    )

    result = TranscriptResult(
        video_id=video_id,
        filename=filename,
        duration_seconds=duration_seconds,
        whisper_model=WHISPER_MODEL,
        language=language,
        full_text=raw_result.get("text", "").strip(),
        segments=segments,
        transcript_path=transcript_path,
    )

    # ── Persist to disk ───────────────────────────────────────────────────────
    _save_transcript(result)

    return result


def transcribe_all_videos(
    video_metadata_list: list,  # list[VideoMetadata] — avoid circular import
    force: bool = False,
) -> list[TranscriptResult]:
    """
    Transcribe all videos, loading the Whisper model ONCE.

    Loading Whisper takes ~5–20 s. Loading it once and reusing it for
    every video is a significant performance win when processing many videos.

    Args:
        video_metadata_list: Output of process_all_videos() from Stage 2.
        force:               Re-transcribe even if JSON already exists.

    Returns:
        List of TranscriptResult for successfully transcribed videos.
    """
    if not video_metadata_list:
        logger.warning("No videos to transcribe.")
        return []

    # Load model once for all videos
    model = _load_whisper_model(WHISPER_MODEL)

    results: list[TranscriptResult] = []
    for vm in video_metadata_list:
        result = transcribe_video(
            video_id=vm.video_id,
            filename=vm.filename,
            audio_path=vm.audio_path,
            duration_seconds=vm.duration_seconds,
            model=model,
            force=force,
        )
        if result:
            results.append(result)
        else:
            logger.error("Transcription failed for: %s — skipping.", vm.filename)

    logger.info(
        "Transcription complete: %d/%d videos transcribed successfully.",
        len(results),
        len(video_metadata_list),
    )
    return results


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Import Stage 2 to discover & process videos first
    from ingestion.video_processor import process_all_videos

    force_flag = "--force" in sys.argv

    print("=" * 60)
    print("EduVision RAG — Stage 3: Whisper Transcription")
    print("=" * 60)
    print(f"Whisper model : {WHISPER_MODEL}")
    print(f"Force retranscribe: {force_flag}\n")

    # Stage 2: get video metadata (audio already extracted)
    video_metadata = process_all_videos(force=False)
    if not video_metadata:
        print("❌  No videos found. Run Stage 2 first.")
        sys.exit(1)

    # Stage 3: transcribe
    transcripts = transcribe_all_videos(video_metadata, force=force_flag)

    if not transcripts:
        print("\n❌  No transcripts produced. Check the logs above.")
        sys.exit(1)

    print(f"\n✅  Successfully transcribed {len(transcripts)} video(s):\n")
    for tr in transcripts:
        print(f"  video_id   : {tr.video_id}")
        print(f"  language   : {tr.language}")
        print(f"  segments   : {len(tr.segments)}")
        print(f"  json path  : {tr.transcript_path}")
        if tr.segments:
            print(f"\n  ── First 3 segments ──")
            for seg in tr.segments[:3]:
                mins, secs = divmod(int(seg.start), 60)
                print(f"    [{mins:02d}:{secs:02d}]  {seg.text}")
        print()
