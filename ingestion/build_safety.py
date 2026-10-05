"""
ingestion/build_safety.py
──────────────────────────
Guards for index builds and cached ingestion outputs. Offline; no ChromaDB, model or API access.

1. Index targets. Every index build must name its output directory explicitly. The runtime
   CHROMA_DB_PATH is never used as a build target, and these directories are refused outright
   (the path itself, anything inside it, and any directory that contains it):
     - data/vector_db_lv2g_translate/    shipped production index (Stage 2.6)
     - data/vector_db/                   previous production index (rollback; allow_legacy_index=True
                                         unlocks it deliberately)
     - experiments/stage2_transcription/artifacts/lv2g_translate/vector_db/   verified Stage 2.6 source
     - the directory settings.CHROMA_DB_PATH resolves to
2. Cached outputs. Ingestion stages skip work when their output file already exists. That reuse is
   only allowed when the cached output provably matches the current run:
     - a transcript must record the same Whisper model and decoding options as the current settings
       (transcripts without recorded options, e.g. the Whisper-base ones, never match);
     - a cleaned transcript, chunk file or embedding file must not be older than the file it was
       derived from, so regenerating one stage invalidates everything after it.
   Otherwise StaleOutputError is raised; pass force=True or use fresh output directories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from config import settings

PRODUCTION_INDEX_DIR: Path = settings.STAGE26_VECTOR_DB_DIR
LEGACY_INDEX_DIR: Path = settings.VECTOR_DB_DIR
STAGE26_SOURCE_INDEX_DIR: Path = (settings.ROOT_DIR / "experiments" / "stage2_transcription" / "artifacts"
                                  / "lv2g_translate" / "vector_db")


class UnsafeBuildTarget(RuntimeError):
    """An index build was pointed at a protected or unspecified location."""


class StaleOutputError(RuntimeError):
    """A cached ingestion output does not match the current run and would be reused silently."""


def _absolute(path) -> Path:
    p = Path(path).expanduser()
    return (p if p.is_absolute() else settings.ROOT_DIR / p).resolve()


def _overlaps(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def protected_index_dirs(allow_legacy_index: bool = False) -> dict[str, Path]:
    dirs = {"shipped production index": PRODUCTION_INDEX_DIR,
            "verified Stage 2.6 source index": STAGE26_SOURCE_INDEX_DIR,
            "runtime CHROMA_DB_PATH": Path(settings.CHROMA_DB_PATH)}
    if not allow_legacy_index:
        dirs["previous production index"] = LEGACY_INDEX_DIR
    return {name: _absolute(p) for name, p in dirs.items()}


def require_build_index_dir(path, *, allow_legacy_index: bool = False) -> Path:
    """
    Absolute build target, or UnsafeBuildTarget. Nothing is created or opened.
    """
    if path is None or not str(path).strip():
        raise UnsafeBuildTarget("An explicit index output directory is required for builds; "
                                "the runtime CHROMA_DB_PATH is never used as a build target.")
    target = _absolute(path)
    for name, protected in protected_index_dirs(allow_legacy_index).items():
        if _overlaps(target, protected):
            raise UnsafeBuildTarget(f"Refusing to build into {target}: it overlaps the {name} ({protected}). "
                                    "Use a separate directory, e.g. under experiments/.")
    return target


def require_transcript_matches(existing, whisper_model: str, whisper_options: dict, path: Path) -> None:
    """A cached transcript may be reused only if it was produced with the current model and options."""
    recorded_model = getattr(existing, "whisper_model", None)
    recorded_options = getattr(existing, "whisper_options", None)
    if recorded_model != whisper_model or recorded_options != whisper_options:
        raise StaleOutputError(
            f"Cached transcript {path} was produced with model={recorded_model!r}, options={recorded_options!r}; "
            f"the current settings are model={whisper_model!r}, options={whisper_options!r}. "
            "Re-transcribe with force=True or use a fresh TRANSCRIPTS_DIR.")


def require_not_older_than(output: Path, inputs: Sequence[Optional[Path]], stage: str) -> None:
    """A cached output may be reused only if no input it was derived from is newer (or missing)."""
    out_mtime = output.stat().st_mtime
    for src in inputs:
        if src is None or not Path(src).exists():
            raise StaleOutputError(f"Cached {stage} {output} cannot be checked: its input {src} is missing. "
                                   "Regenerate with force=True.")
        if Path(src).stat().st_mtime > out_mtime:
            raise StaleOutputError(f"Cached {stage} {output} is older than its input {src}. "
                                   "Regenerate with force=True.")
