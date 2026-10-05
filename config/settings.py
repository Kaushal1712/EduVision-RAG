"""
config/settings.py
──────────────────
Central configuration module for EduVision RAG.

All tuneable parameters live here, loaded once from the .env file.
The rest of the codebase imports from this module instead of reading
environment variables directly — keeping configuration in one place.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env from project root ───────────────────────────────────────────────
# This must run before any os.getenv() calls.
_ROOT = Path(__file__).resolve().parent.parent  # project root
load_dotenv(_ROOT / ".env")


# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR: Path = _ROOT
VIDEOS_DIR: Path = ROOT_DIR / "videos"
DATA_DIR: Path = ROOT_DIR / "data"
TRANSCRIPTS_DIR: Path = DATA_DIR / "transcripts"
PROCESSED_DIR: Path = DATA_DIR / "processed"
# Previous production index (Whisper base, collections v1/v2). Kept unchanged for rollback.
VECTOR_DB_DIR: Path = DATA_DIR / "vector_db"
# Production index: the Stage 2.6 locked selection (collection lv2g_translate, 1,391 chunks,
# see experiments/stage2_6_abstention/locked_selection.json).
STAGE26_VECTOR_DB_DIR: Path = DATA_DIR / "vector_db_lv2g_translate"


def _project_path(value: str) -> str:
    """
    Absolute form of a path setting. Relative values are resolved against ROOT_DIR, so the
    result does not depend on the working directory the app or a script is started from.
    """
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (ROOT_DIR / path).resolve())


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in ("1", "true", "yes", "on")


# ── Transcription ─────────────────────────────────────────────────────────────
# Stage 2.6 decoding (the settings the shipped index was transcribed with; first run by
# experiments/stage2_transcription/build_candidate.py --greedy): large-v2, task translate,
# greedy temperature 0 without fallback, no conditioning on previous text.
# Whisper's own defaults (the pre-Stage-2 production behaviour) are: task transcribe,
# WHISPER_TEMPERATURE_FALLBACK=true and WHISPER_CONDITION_ON_PREVIOUS_TEXT=true.
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "large-v2")
WHISPER_TASK: str = os.getenv("WHISPER_TASK", "translate")
WHISPER_TEMPERATURE: float = float(os.getenv("WHISPER_TEMPERATURE", "0.0"))
WHISPER_TEMPERATURE_FALLBACK: bool = _env_bool("WHISPER_TEMPERATURE_FALLBACK", False)
WHISPER_CONDITION_ON_PREVIOUS_TEXT: bool = _env_bool("WHISPER_CONDITION_ON_PREVIOUS_TEXT", False)

# ── Embeddings ────────────────────────────────────────────────────────────────
BGE_MODEL: str = os.getenv("BGE_MODEL", "BAAI/bge-m3")

# ── Vector database ───────────────────────────────────────────────────────────
CHROMA_DB_PATH: str = _project_path(os.getenv("CHROMA_DB_PATH", str(STAGE26_VECTOR_DB_DIR)))
CHROMA_COLLECTION_NAME: str = "eduvision_chunks"          # v1 (original)
CHROMA_COLLECTION_NAME_V2: str = "eduvision_chunks_v2"    # v2 (English-normalised)
CHROMA_COLLECTION_NAME_LV2G: str = "lv2g_translate"       # Stage 2.6 (Whisper large-v2 translate)

# ACTIVE_COLLECTION: which collection the pipeline uses for all queries.
# Rollback to the previous production index: CHROMA_DB_PATH=data/vector_db and
# ACTIVE_COLLECTION=eduvision_chunks_v2 (and SIMILARITY_THRESHOLD=0.50).
ACTIVE_COLLECTION: str = os.getenv("ACTIVE_COLLECTION", CHROMA_COLLECTION_NAME_LV2G)

# ── Retrieval ─────────────────────────────────────────────────────────────────
# RETRIEVAL_TOP_K: how many chunks to fetch from ChromaDB per query.
# Wider pool = better chance the correct chunk is in the candidate set,
# even when some candidates are below threshold.
RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "10"))

# MAX_LLM_EVIDENCE: maximum above-threshold chunks passed to the LLM.
# Kept at 5 — enough context for grounded answers without inflating prompt tokens.
MAX_LLM_EVIDENCE: int = int(os.getenv("MAX_LLM_EVIDENCE", "5"))

# Backward-compatible alias of RETRIEVAL_TOP_K for older code paths (retriever defaults).
# It is not read from the environment; set RETRIEVAL_TOP_K instead.
TOP_K_RESULTS: int = RETRIEVAL_TOP_K

# SIMILARITY_THRESHOLD: Stage 2.6 locked selection (0.44), chosen on DEV as the midpoint of the
# plateau with zero in-scope gate refusals and unchanged out-of-scope gate decisions
# (experiments/stage2_6_abstention/locked_selection.json). It is both the no-LLM refusal gate
# and the per-chunk evidence filter; results below it are flagged (not dropped) by the retriever.
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.44"))

# ── OpenAI / LLM ──────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1024"))
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.2"))

# ── Chunking ──────────────────────────────────────────────────────────────────
# Maximum number of Whisper segments to group into one retrieval chunk.
SEGMENTS_PER_CHUNK: int = 5
# Overlap: how many segments from the previous chunk to include at the start
# of the next chunk (prevents context from being cut at chunk boundaries).
CHUNK_OVERLAP_SEGMENTS: int = 1


def ensure_directories() -> None:
    """Create all required data directories if they do not exist."""
    for directory in (TRANSCRIPTS_DIR, PROCESSED_DIR, VECTOR_DB_DIR):
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Quick sanity-check: print all resolved settings.
    ensure_directories()
    print("=== EduVision RAG — Settings ===")
    print(f"ROOT_DIR          : {ROOT_DIR}")
    print(f"VIDEOS_DIR        : {VIDEOS_DIR}")
    print(f"TRANSCRIPTS_DIR   : {TRANSCRIPTS_DIR}")
    print(f"PROCESSED_DIR     : {PROCESSED_DIR}")
    print(f"VECTOR_DB_DIR     : {VECTOR_DB_DIR}")
    print(f"WHISPER_MODEL     : {WHISPER_MODEL}")
    print(f"WHISPER_TASK      : {WHISPER_TASK}  temperature {WHISPER_TEMPERATURE} "
          f"fallback {WHISPER_TEMPERATURE_FALLBACK}  condition_on_previous_text {WHISPER_CONDITION_ON_PREVIOUS_TEXT}")
    print(f"BGE_MODEL         : {BGE_MODEL}")
    print(f"CHROMA_DB_PATH    : {CHROMA_DB_PATH}")
    print(f"ACTIVE_COLLECTION : {ACTIVE_COLLECTION}")
    print(f"RETRIEVAL_TOP_K   : {RETRIEVAL_TOP_K}")
    print(f"SIMILARITY_THRESHOLD : {SIMILARITY_THRESHOLD}")
    print(f"OPENAI_MODEL      : {OPENAI_MODEL}")
    print(f"SEGMENTS_PER_CHUNK: {SEGMENTS_PER_CHUNK}")
    print(f"OPENAI_API_KEY set: {'YES' if OPENAI_API_KEY else 'NO — set it in .env!'}")
