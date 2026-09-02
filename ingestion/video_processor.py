"""
ingestion/video_processor.py
─────────────────────────────
Stage 2: Video → Audio extraction using FFmpeg.

Responsibilities:
  1. Discover all .mp4 files in the videos/ directory.
  2. For each video, extract a clean mono 16 kHz WAV audio file.
  3. Return structured VideoMetadata so downstream stages
     (Whisper, chunker, indexer) know the video_id, paths, and duration.

Why this design:
  - Whisper works best on mono, 16 kHz audio (that is the format it was
    trained on internally). Feeding it the raw video wastes compute on
    decoding video frames.
  - Extracting audio first also lets us re-run transcription without
    re-processing video, and vice versa.
  - video_id is derived from the filename stem (lowercased, spaces→underscores)
    so it is stable, human-readable, and does not require a database.

Why FFmpeg (not moviepy / pydub)?
  - FFmpeg is the industry standard. It handles every codec, container,
    and edge case (variable frame rate, corrupt moov atoms, etc.).
  - It is already installed on the system (confirmed in Stage 1).
  - subprocess calls keep the dependency footprint minimal — we do NOT
    need the ffmpeg-python binding for this simple use case.
"""

import subprocess
import logging
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config.settings import VIDEOS_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class VideoMetadata:
    """All information about one video file, produced by this module."""
    video_id: str          # stable identifier, e.g. "01_installing_vs_code"
    original_path: Path    # absolute path to the .mp4 file
    audio_path: Path       # absolute path to the extracted .wav file
    duration_seconds: float
    title: str             # human-readable title derived from filename
    filename: str          # just the filename, e.g. "01_installing_vs_code.mp4"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_video_id(stem: str) -> str:
    """
    Convert a filename stem to a clean, stable video_id.

    Example:
        "01_Installing VS Code & How Websites Work ｜ Sigma..." 
        → "01_installing_vs_code_how_websites_work_sigma..."

    Rules:
      - Lowercase everything.
      - Replace non-alphanumeric characters (spaces, &, |, etc.) with _.
      - Collapse runs of underscores to a single _.
      - Strip leading/trailing underscores.
    """
    vid = stem.lower()
    vid = re.sub(r"[^a-z0-9]+", "_", vid)
    vid = re.sub(r"_+", "_", vid)
    vid = vid.strip("_")
    return vid


def _probe_duration(video_path: Path) -> float:
    """
    Use ffprobe to get the video duration in seconds.

    Returns 0.0 if ffprobe fails (the video is still processed,
    duration is just unknown).
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        return float(info["format"].get("duration", 0.0))
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
        logger.warning("ffprobe failed for %s: %s", video_path.name, e)
        return 0.0


def _extract_audio(video_path: Path, audio_path: Path) -> bool:
    """
    Extract audio from video_path → audio_path using FFmpeg.

    Output spec:
      -vn          : no video stream (audio only)
      -ac 1        : mono channel (Whisper was trained on mono)
      -ar 16000    : 16 kHz sample rate (Whisper's native rate)
      -acodec pcm_s16le : uncompressed 16-bit PCM WAV
      -y           : overwrite output if it exists

    Returns True on success, False on failure.
    """
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i", str(video_path),   # input
        "-vn",                    # strip video
        "-ac", "1",               # mono
        "-ar", "16000",           # 16 kHz
        "-acodec", "pcm_s16le",   # uncompressed WAV (PCM 16-bit little-endian)
        "-y",                     # overwrite
        str(audio_path),          # output
    ]

    logger.info("Extracting audio: %s → %s", video_path.name, audio_path.name)

    try:
        subprocess.run(
            cmd,
            capture_output=True,   # swallow ffmpeg's noisy stderr
            check=True,            # raise on non-zero exit code
        )
        logger.info("Audio extracted successfully: %s", audio_path.name)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(
            "FFmpeg failed for %s:\n%s",
            video_path.name,
            e.stderr.decode(errors="replace"),
        )
        return False


# ── Public API ────────────────────────────────────────────────────────────────

def discover_videos() -> list[Path]:
    """
    Return a sorted list of all .mp4 files found in VIDEOS_DIR.

    Sorted so processing order is deterministic (helps with debugging
    and makes video_ids consistent across runs).
    """
    if not VIDEOS_DIR.exists():
        logger.warning("videos/ directory not found: %s", VIDEOS_DIR)
        return []

    videos = sorted(VIDEOS_DIR.glob("*.mp4"))
    logger.info("Discovered %d video(s) in %s", len(videos), VIDEOS_DIR)
    return videos


def process_video(video_path: Path, force: bool = False) -> Optional[VideoMetadata]:
    """
    Process a single video file: extract audio and return VideoMetadata.

    Args:
        video_path: Absolute path to the .mp4 file.
        force:      If True, re-extract audio even if the .wav already exists.
                    Default is False (skip already-processed files).

    Returns:
        VideoMetadata on success, None on failure.
    """
    stem = video_path.stem
    video_id = _make_video_id(stem)
    audio_filename = f"{video_id}.wav"
    audio_path = PROCESSED_DIR / audio_filename

    # ── Skip if already processed (incremental ingestion) ────────────────────
    if audio_path.exists() and not force:
        logger.info(
            "Audio already exists, skipping extraction: %s (use force=True to re-extract)",
            audio_path.name,
        )
        duration = _probe_duration(video_path)
        return VideoMetadata(
            video_id=video_id,
            original_path=video_path,
            audio_path=audio_path,
            duration_seconds=duration,
            title=stem,
            filename=video_path.name,
        )

    # ── Probe duration before extracting ─────────────────────────────────────
    duration = _probe_duration(video_path)
    logger.info(
        "Processing: %s (%.1f min)", video_path.name, duration / 60
    )

    # ── Extract audio ─────────────────────────────────────────────────────────
    success = _extract_audio(video_path, audio_path)
    if not success:
        return None

    return VideoMetadata(
        video_id=video_id,
        original_path=video_path,
        audio_path=audio_path,
        duration_seconds=duration,
        title=stem,
        filename=video_path.name,
    )


def process_all_videos(force: bool = False) -> list[VideoMetadata]:
    """
    Discover and process every .mp4 in videos/.

    This is the main entry point called by the ingestion pipeline.

    Args:
        force: If True, re-extract audio for ALL videos even if .wav exists.

    Returns:
        List of VideoMetadata for successfully processed videos.
        Videos that fail processing are logged and excluded.
    """
    video_paths = discover_videos()
    if not video_paths:
        logger.warning("No videos found. Place .mp4 files in: %s", VIDEOS_DIR)
        return []

    results: list[VideoMetadata] = []
    for video_path in video_paths:
        metadata = process_video(video_path, force=force)
        if metadata:
            results.append(metadata)
        else:
            logger.error("Failed to process: %s — skipping.", video_path.name)

    logger.info(
        "Video processing complete: %d/%d videos processed successfully.",
        len(results),
        len(video_paths),
    )
    return results


# ── CLI entry-point (run directly for testing) ────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    force_flag = "--force" in sys.argv
    if force_flag:
        print("⚠  --force flag set: re-extracting all audio files.\n")

    print("=" * 60)
    print("EduVision RAG — Stage 2: Video Processing")
    print("=" * 60)

    all_metadata = process_all_videos(force=force_flag)

    if not all_metadata:
        print("\n❌  No videos were processed. Check the logs above.")
        sys.exit(1)

    print(f"\n✅  Successfully processed {len(all_metadata)} video(s):\n")
    for m in all_metadata:
        wav_size_mb = m.audio_path.stat().st_size / 1024 / 1024
        print(f"  video_id   : {m.video_id}")
        print(f"  filename   : {m.filename}")
        print(f"  duration   : {m.duration_seconds / 60:.1f} min ({m.duration_seconds:.1f} s)")
        print(f"  audio_path : {m.audio_path}")
        print(f"  wav size   : {wav_size_mb:.1f} MB")
        print()
