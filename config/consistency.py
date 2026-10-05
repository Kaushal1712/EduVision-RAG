"""
config/consistency.py
──────────────────────
Offline check that the production runtime configuration matches the locked Stage 2.6 selection
(experiments/stage2_6_abstention/locked_selection.json).

    venv/bin/python -m config.consistency        # prints a report; exit code 1 on any mismatch

    from config.consistency import assert_production_config
    assert_production_config()                    # raises ConfigMismatchError listing every mismatch

It only reads settings and module constants and checks that the index directory exists. It does not
open ChromaDB, create collections, load models or call any API. The Whisper decoding options are
checked as the exact keyword arguments ingestion/transcriber.py passes to model.transcribe(), which
must equal those recorded for the Stage 2.6 build (experiments/stage2_transcription/build_candidate.py
--greedy, artifacts/lv2g_translate/run_log.json).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from config import settings

LOCKED_SELECTION = settings.ROOT_DIR / "experiments" / "stage2_6_abstention" / "locked_selection.json"

# The Stage 2.6 selection, as represented in the runtime configuration.
EXPECTED: dict[str, Any] = {
    "settings.WHISPER_MODEL": "large-v2",
    "settings.WHISPER_TASK": "translate",
    "settings.WHISPER_TEMPERATURE": 0.0,
    "settings.WHISPER_TEMPERATURE_FALLBACK": False,
    "settings.WHISPER_CONDITION_ON_PREVIOUS_TEXT": False,
    "ingestion.transcriber.transcription_options()": {
        "task": "translate", "verbose": False, "fp16": False, "word_timestamps": False,
        "temperature": 0.0, "condition_on_previous_text": False,
    },
    "settings.BGE_MODEL": "BAAI/bge-m3",
    "settings.CHROMA_DB_PATH": str((settings.ROOT_DIR / "data" / "vector_db_lv2g_translate").resolve()),
    "settings.ACTIVE_COLLECTION": "lv2g_translate",
    "settings.RETRIEVAL_TOP_K": 10,
    "settings.TOP_K_RESULTS": 10,
    "settings.SIMILARITY_THRESHOLD": 0.44,
    "settings.MAX_LLM_EVIDENCE": 5,
    "settings.OPENAI_MODEL": "gpt-4o-mini",
    "settings.TEMPERATURE": 0.2,
    "settings.SEGMENTS_PER_CHUNK": 5,
    "settings.CHUNK_OVERLAP_SEGMENTS": 1,
    "ingestion.chunker.CHUNK_WINDOW": 5,
    "ingestion.chunker.CHUNK_OVERLAP": 1,
    "ingestion.chunker.GAP_THRESHOLD": 5.0,
    "generation.generator.MAX_EVIDENCE_CHUNKS": 5,
    "generation.generator.SYSTEM_PROMPT.sha256": "9b294ecd05aa9bc350d5056005774e533e1700373fd5b5bd939046fbb133b4e3",
}

class ConfigMismatchError(RuntimeError):
    """The runtime configuration does not match the locked Stage 2.6 selection."""


def collect_runtime() -> dict[str, Any]:
    """Current values for every EXPECTED key (imports only; no side effects beyond module import)."""
    from generation import generator
    from ingestion import chunker, transcriber
    return {
        "settings.WHISPER_MODEL": settings.WHISPER_MODEL,
        "settings.WHISPER_TASK": settings.WHISPER_TASK,
        "settings.WHISPER_TEMPERATURE": settings.WHISPER_TEMPERATURE,
        "settings.WHISPER_TEMPERATURE_FALLBACK": settings.WHISPER_TEMPERATURE_FALLBACK,
        "settings.WHISPER_CONDITION_ON_PREVIOUS_TEXT": settings.WHISPER_CONDITION_ON_PREVIOUS_TEXT,
        "ingestion.transcriber.transcription_options()": transcriber.transcription_options(),
        "settings.BGE_MODEL": settings.BGE_MODEL,
        "settings.CHROMA_DB_PATH": str(Path(settings.CHROMA_DB_PATH).resolve()),
        "settings.ACTIVE_COLLECTION": settings.ACTIVE_COLLECTION,
        "settings.RETRIEVAL_TOP_K": settings.RETRIEVAL_TOP_K,
        "settings.TOP_K_RESULTS": settings.TOP_K_RESULTS,
        "settings.SIMILARITY_THRESHOLD": settings.SIMILARITY_THRESHOLD,
        "settings.MAX_LLM_EVIDENCE": settings.MAX_LLM_EVIDENCE,
        "settings.OPENAI_MODEL": settings.OPENAI_MODEL,
        "settings.TEMPERATURE": settings.TEMPERATURE,
        "settings.SEGMENTS_PER_CHUNK": settings.SEGMENTS_PER_CHUNK,
        "settings.CHUNK_OVERLAP_SEGMENTS": settings.CHUNK_OVERLAP_SEGMENTS,
        "ingestion.chunker.CHUNK_WINDOW": chunker.CHUNK_WINDOW,
        "ingestion.chunker.CHUNK_OVERLAP": chunker.CHUNK_OVERLAP,
        "ingestion.chunker.GAP_THRESHOLD": chunker.GAP_THRESHOLD,
        "generation.generator.MAX_EVIDENCE_CHUNKS": generator.MAX_EVIDENCE_CHUNKS,
        "generation.generator.SYSTEM_PROMPT.sha256": hashlib.sha256(generator.SYSTEM_PROMPT.encode()).hexdigest(),
    }


def compare(runtime: dict[str, Any], expected: dict[str, Any] = EXPECTED) -> list[str]:
    """One message per mismatched or missing key."""
    out = []
    for key, want in expected.items():
        got = runtime.get(key, "<missing>")
        if got != want:
            out.append(f"{key}: expected {want!r}, got {got!r}")
    return out


def check_index_present(db_path: str) -> list[str]:
    """The index directory must exist; it is not opened."""
    return [] if (Path(db_path) / "chroma.sqlite3").is_file() else [f"index not found: {db_path}/chroma.sqlite3"]


def check_locked_selection(path: Path = LOCKED_SELECTION) -> list[str]:
    """If the locked selection file is present, it must agree with EXPECTED (guards against drift)."""
    if not path.is_file():
        return []
    locked = json.loads(path.read_text())
    out = []
    if locked.get("similarity_threshold") != EXPECTED["settings.SIMILARITY_THRESHOLD"]:
        out.append(f"locked_selection.json similarity_threshold {locked.get('similarity_threshold')!r} "
                   f"!= expected {EXPECTED['settings.SIMILARITY_THRESHOLD']!r}")
    if locked.get("system_prompt_sha256") != EXPECTED["generation.generator.SYSTEM_PROMPT.sha256"]:
        out.append("locked_selection.json system_prompt_sha256 != expected prompt hash")
    return out


def check_production_config() -> list[str]:
    """All mismatches between the runtime configuration and the locked Stage 2.6 selection."""
    runtime = collect_runtime()
    return compare(runtime) + check_index_present(runtime["settings.CHROMA_DB_PATH"]) + check_locked_selection()


def assert_production_config() -> None:
    problems = check_production_config()
    if problems:
        raise ConfigMismatchError("Runtime configuration does not match the locked Stage 2.6 selection:\n  "
                                  + "\n  ".join(problems))


def main() -> int:
    runtime = collect_runtime()
    problems = check_production_config()
    print("Stage 2.6 configuration consistency check")
    for key, want in EXPECTED.items():
        print(f"  {'OK  ' if runtime[key] == want else 'FAIL'} {key} = {runtime[key]!r}")
    print(f"  {'OK  ' if not check_index_present(runtime['settings.CHROMA_DB_PATH']) else 'FAIL'} "
          f"index directory present (not opened)")
    print(f"  {'OK  ' if not check_locked_selection() else 'FAIL'} agrees with "
          f"{LOCKED_SELECTION.relative_to(settings.ROOT_DIR)}"
          f"{'' if LOCKED_SELECTION.is_file() else ' (file absent: skipped)'}")
    if problems:
        print("RESULT: MISMATCH\n  " + "\n  ".join(problems))
        return 1
    print("RESULT: OK — runtime configuration matches the locked Stage 2.6 selection")
    return 0


if __name__ == "__main__":
    sys.exit(main())
