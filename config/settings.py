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
VECTOR_DB_DIR: Path = DATA_DIR / "vector_db"

# ── Transcription ─────────────────────────────────────────────────────────────
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")

# ── Embeddings ────────────────────────────────────────────────────────────────
BGE_MODEL: str = os.getenv("BGE_MODEL", "BAAI/bge-m3")

# ── Vector database ───────────────────────────────────────────────────────────
CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", str(VECTOR_DB_DIR))
CHROMA_COLLECTION_NAME: str = "eduvision_chunks"          # v1 (original)
CHROMA_COLLECTION_NAME_V2: str = "eduvision_chunks_v2"    # v2 (English-normalised)

# ACTIVE_COLLECTION: which collection the pipeline uses for all queries.
# Set to v2 to enable English-normalised retrieval + quality filtering.
# Revert to CHROMA_COLLECTION_NAME to restore v1 behaviour.
ACTIVE_COLLECTION: str = os.getenv("ACTIVE_COLLECTION", CHROMA_COLLECTION_NAME_V2)

# ── Retrieval ─────────────────────────────────────────────────────────────────
# RETRIEVAL_TOP_K: how many chunks to fetch from ChromaDB per query.
# Wider pool = better chance the correct chunk is in the candidate set,
# even when some candidates are below threshold.
RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "10"))

# MAX_LLM_EVIDENCE: maximum above-threshold chunks passed to the LLM.
# Kept at 5 — enough context for grounded answers without inflating prompt tokens.
MAX_LLM_EVIDENCE: int = int(os.getenv("MAX_LLM_EVIDENCE", "5"))

# Backward-compatible alias (used in older code paths).
TOP_K_RESULTS: int = RETRIEVAL_TOP_K

# SIMILARITY_THRESHOLD: empirically calibrated against 2-video corpus (2026-08-28)
#   Relevant queries (HTML, VS Code install, etc.) → top-1 sim: 0.60 – 0.70
#   Irrelevant queries (France, chocolate, USA)    → top-1 sim: 0.38 – 0.48
#   Midpoint = 0.54  →  using 0.50 for a comfortable margin.
#   Results below threshold are flagged (not dropped) by the retriever.
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.50"))

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
    print(f"BGE_MODEL         : {BGE_MODEL}")
    print(f"CHROMA_DB_PATH    : {CHROMA_DB_PATH}")
    print(f"TOP_K_RESULTS     : {TOP_K_RESULTS}")
    print(f"SIMILARITY_THRESHOLD : {SIMILARITY_THRESHOLD}")
    print(f"OPENAI_MODEL      : {OPENAI_MODEL}")
    print(f"SEGMENTS_PER_CHUNK: {SEGMENTS_PER_CHUNK}")
    print(f"OPENAI_API_KEY set: {'YES' if OPENAI_API_KEY else 'NO — set it in .env!'}")
