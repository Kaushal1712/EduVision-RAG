"""
experiments/stage2_transcription/build_candidate.py
────────────────────────────────────────────────────
Stage 2 A/B experiment: build an EXPERIMENTAL index from a candidate transcription,
reusing the production pipeline code unchanged.

    candidate Whisper  →  ingestion.cleaner.clean_transcript (unchanged)
                       →  ingestion.chunker._build_chunks_from_segments (unchanged parameters)
                       →  ingestion.embedder._load_bge_model / _embed_chunks (unchanged BGE-M3 settings)
                       →  ingestion.normalizer.normalize_chunks (quality filter + English text_en)
                       →  separate ChromaDB in artifacts/<name>/vector_db (same metadata as indexer_v2)

ISOLATION — nothing is written outside artifacts/<name>/:
  * Whisper output is saved with transcriber._save_transcript to an artifacts path.
  * cleaner.clean_transcript always saves to its module-level TRANSCRIPTS_DIR, so that
    name is pointed at artifacts/<name>/transcripts for the duration of the run.
  * Chunks and embeddings are saved by this script, not by the production savers.
  * normalize_chunks runs with use_cache=False, save_cache=False: its cache is keyed by
    chunk_id, which collides with production chunk ids.
  * The production index is never opened; --fill-from-production reads a copy of the
    COMMITTED index extracted with `git archive` (opening ChromaDB rewrites its files).

Usage (from the project root):
  venv/bin/python experiments/stage2_transcription/build_candidate.py transcribe \
      --name lv2_transcribe --model large-v2 --task transcribe --videos 3 4 5 8 18
  venv/bin/python experiments/stage2_transcription/build_candidate.py index \
      --name lv2_transcribe --videos 3 4 5 8 18 --fill-from-production
  venv/bin/python experiments/stage2_transcription/build_candidate.py parity
Then evaluate with the unchanged retrieval code:
  CHROMA_DB_PATH=<abs path to artifacts/<name>/vector_db> ACTIVE_COLLECTION=<name> \
      venv/bin/python eval/run_benchmark.py --label stage2_<name>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parent.parent
ARTIFACTS = EXP_DIR / "artifacts"
sys.path.insert(0, str(ROOT))

from config.settings import BGE_MODEL, PROCESSED_DIR  # noqa: E402
from ingestion.build_safety import require_build_index_dir  # noqa: E402

MANIFEST = json.loads((ROOT / "eval/benchmark/corpus_manifest.json").read_text())
VIDEOS = {v["tutorial_number"]: v for v in MANIFEST["videos"]}
PRODUCTION_COLLECTION = "eduvision_chunks_v2"


def _art(name: str) -> Path:
    path = ARTIFACTS / name
    path.mkdir(parents=True, exist_ok=True)
    gitignore = ARTIFACTS / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("# Stage 2 experiment artifacts (transcripts, chunks, embeddings, indexes);\n"
                             "# regenerate with build_candidate.py. Not committed.\n*\n!.gitignore\n")
    return path


def _peak_rss_gb() -> float:
    # macOS reports ru_maxrss in bytes (Linux: kilobytes).
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / (1024 ** 3 if platform.system() == "Darwin" else 1024 ** 2), 2)


def _update_log(art: Path, key: str, value: dict) -> None:
    log_path = art / "run_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    log[key] = value
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n")


def _targets(numbers: list[int] | None) -> list[dict]:
    return [VIDEOS[n] for n in (numbers or sorted(VIDEOS))]


# ── transcribe: Whisper → clean → chunk ──────────────────────────────────────

def cmd_transcribe(args) -> None:
    import whisper
    from ingestion import cleaner, transcriber
    from ingestion.chunker import _build_chunks_from_segments

    require_build_index_dir(ARTIFACTS / args.name / "vector_db")   # refuses the verified lv2g_translate artifact
    art = _art(args.name)
    tdir, cdir = art / "transcripts", art / "chunks"
    tdir.mkdir(exist_ok=True)
    cdir.mkdir(exist_ok=True)
    cleaner.TRANSCRIPTS_DIR = tdir          # redirect clean_transcript's save path

    # Same call as ingestion/transcriber.py, plus the candidate's model and task.
    options = {"task": args.task, "verbose": False, "fp16": False, "word_timestamps": False}
    if args.greedy:
        # Production uses Whisper's default temperature fallback (0.0 → 1.0, best_of=5 when
        # sampling) and conditions on previous text. With large-v2 on CPU that ran below
        # 0.33x real time in the pilot, so the candidate decodes greedily without fallback
        # and without previous-text conditioning (also the usual guard against loops).
        options.update({"temperature": 0.0, "condition_on_previous_text": False})
    t0 = time.time()
    model = whisper.load_model(args.model, device=args.device)
    load_s = time.time() - t0

    for v in _targets(args.videos):
        vid = v["video_id"]
        raw_path = tdir / f"{vid}.json"
        if raw_path.exists() and not args.force:
            print(f"skip {vid[:40]} (transcript exists)")
            continue
        audio = PROCESSED_DIR / f"{vid}.wav"
        t0 = time.time()
        raw = model.transcribe(str(audio), **options)
        elapsed = time.time() - t0

        result = transcriber.TranscriptResult(
            video_id=vid, filename=v["video_filename"], duration_seconds=v["duration_s"],
            whisper_model=args.model, language=raw.get("language", "unknown"),
            full_text=raw.get("text", "").strip(),
            segments=transcriber._parse_segments(raw.get("segments", [])),
            transcript_path=raw_path,
        )
        transcriber._save_transcript(result)
        cleaned = cleaner.clean_transcript(result)
        assert cleaned.cleaned_path.parent == tdir, "cleaner wrote outside the artifacts directory"

        chunks = _build_chunks_from_segments(
            segments=cleaned.segments, video_id=vid, video_filename=v["video_filename"],
            language=cleaned.language,
        )
        (cdir / f"{vid}_chunks.json").write_text(json.dumps(
            {"video_id": vid, "chunks": [dataclasses.asdict(c) for c in chunks]},
            indent=1, ensure_ascii=False) + "\n")

        stats = {
            "audio_s": v["duration_s"], "runtime_s": round(elapsed, 1),
            "realtime_factor": round(v["duration_s"] / elapsed, 2),
            "language": result.language, "raw_segments": len(result.segments),
            "cleaned_segments": len(cleaned.segments), "chunks": len(chunks),
            "cleaning_removal_reasons": cleaned.report.removal_reasons,
            "peak_rss_gb_so_far": _peak_rss_gb(),
        }
        _update_log(art, f"transcribe/{vid}", stats)
        print(f"T{v['tutorial_number']:02d} {elapsed:6.0f}s  rtf {stats['realtime_factor']:.1f}x  "
              f"lang={result.language}  segs {len(result.segments)}→{len(cleaned.segments)}  "
              f"chunks {len(chunks)}", flush=True)

    _update_log(art, "transcribe/config", {
        "model": args.model, "task": args.task, "device": args.device,
        "whisper_options": options, "language": "auto-detect (as production)",
        "model_load_s": round(load_s, 1), "openai_whisper": _version("openai-whisper"),
        "torch": _version("torch"), "python": platform.python_version(),
        "machine": f"{platform.machine()} {platform.platform()}",
    })


# ── rechunk: existing cleaned transcripts → chunks with other window/overlap ─

def cmd_rechunk(args) -> None:
    """Stage 2.5: re-chunk a candidate's cleaned transcripts with different chunker parameters.

    Uses the production chunker function unchanged; only its window/overlap arguments differ.
    """
    from ingestion.chunker import GAP_THRESHOLD, _build_chunks_from_segments
    from ingestion.transcriber import TranscriptSegment

    src = ARTIFACTS / args.source / "transcripts"
    art = _art(args.name)
    cdir = art / "chunks"
    cdir.mkdir(exist_ok=True)
    total = 0
    for v in _targets(None):
        cleaned = json.loads((src / f"{v['video_id']}_cleaned.json").read_text())
        segs = [TranscriptSegment(**s) for s in cleaned["segments"]]
        chunks = _build_chunks_from_segments(
            segments=segs, video_id=v["video_id"], video_filename=v["video_filename"],
            language=cleaned["language"], window=args.window, overlap=args.overlap,
            gap_threshold=args.gap if args.gap is not None else GAP_THRESHOLD,
        )
        (cdir / f"{v['video_id']}_chunks.json").write_text(json.dumps(
            {"video_id": v["video_id"], "chunks": [dataclasses.asdict(c) for c in chunks]},
            indent=1, ensure_ascii=False) + "\n")
        total += len(chunks)
    _update_log(art, "rechunk", {"source_transcripts": str(src.relative_to(ROOT)), "window": args.window,
                                  "overlap": args.overlap, "gap_threshold": args.gap if args.gap is not None else GAP_THRESHOLD, "chunks": total})
    print(f"{args.name}: {total} chunks (window={args.window}, overlap={args.overlap})")


# ── index: embed → normalize → separate Chroma ───────────────────────────────

def _production_snapshot(art: Path) -> Path:
    """Extract the committed production index (read-only copy) under artifacts."""
    snap = ARTIFACTS / "_production_index_snapshot"
    if not (snap / "data/vector_db/chroma.sqlite3").exists():
        snap.mkdir(parents=True, exist_ok=True)
        archive = subprocess.run(["git", "archive", "HEAD", "data/vector_db"], cwd=ROOT,
                                 capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(snap)], input=archive, check=True)
    return snap / "data/vector_db"


def cmd_index(args) -> None:
    import chromadb
    from chromadb.config import Settings
    from ingestion.chunker import Chunk
    from ingestion.embedder import _embed_chunks, _load_bge_model
    from ingestion.normalizer import normalize_chunks

    require_build_index_dir(ARTIFACTS / args.name / "vector_db")   # refuses the verified lv2g_translate artifact
    art = _art(args.name)
    targets = _targets(args.videos)
    chunks: list[Chunk] = []
    for v in targets:
        data = json.loads((art / "chunks" / f"{v['video_id']}_chunks.json").read_text())
        chunks += [Chunk(**c) for c in data["chunks"]]

    t0 = time.time()
    bge = _load_bge_model(BGE_MODEL)
    embeddings = _embed_chunks(chunks, bge)
    embed_s = time.time() - t0
    bge_device = str(getattr(bge, "target_devices", getattr(bge, "device", "unknown")))
    del bge

    t0 = time.time()
    normalized = normalize_chunks(chunks, use_cache=False, save_cache=False)
    normalize_s = time.time() - t0
    (art / "embeddings.json").write_text(json.dumps(embeddings))

    client = chromadb.PersistentClient(path=str(art / "vector_db"), settings=Settings(anonymized_telemetry=False))
    if args.name in [c.name for c in client.list_collections()]:
        client.delete_collection(args.name)
    col = client.create_collection(name=args.name, metadata={"hnsw:space": "cosine"})

    kept = [n for n in normalized if n.quality_ok]
    for i in range(0, len(kept), 50):                        # same batching/metadata as indexer_v2.py
        batch = kept[i:i + 50]
        col.upsert(
            ids=[n.chunk_id for n in batch],
            embeddings=[embeddings[n.chunk_id] for n in batch],
            documents=[n.text_en for n in batch],
            metadatas=[{
                "video_id": n.chunk.video_id, "video_filename": n.chunk.video_filename,
                "language": n.chunk.language, "start_time": n.chunk.start_time,
                "end_time": n.chunk.end_time, "start_time_fmt": n.chunk.start_time_fmt,
                "end_time_fmt": n.chunk.end_time_fmt, "chunk_index": n.chunk.chunk_index,
                "source_segment_ids": json.dumps(n.chunk.source_segment_ids),
                "text_raw": n.text_raw, "translated": str(n.translated),
            } for n in batch],
        )

    filled = 0
    if args.fill_from_production:
        prod = chromadb.PersistentClient(path=str(_production_snapshot(art)),
                                         settings=Settings(anonymized_telemetry=False)
                                         ).get_collection(PRODUCTION_COLLECTION)
        candidate_ids = {v["video_id"] for v in targets}
        got = prod.get(include=["embeddings", "documents", "metadatas"])
        rows = [i for i, m in enumerate(got["metadatas"]) if m["video_id"] not in candidate_ids]
        for i in range(0, len(rows), 100):
            b = rows[i:i + 100]
            col.upsert(ids=[got["ids"][j] for j in b], embeddings=[got["embeddings"][j] for j in b],
                       documents=[got["documents"][j] for j in b], metadatas=[got["metadatas"][j] for j in b])
        filled = len(rows)

    docs = col.get(include=["documents"])["documents"]
    _update_log(art, "index", {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "collection": args.name, "vector_db": str((art / "vector_db").relative_to(ROOT)),
        "candidate_videos": [v["tutorial_number"] for v in targets],
        "candidate_chunks": len(chunks), "candidate_quality_ok": len(kept),
        "candidate_translated_by_normalizer": sum(n.translated for n in kept),
        "candidate_unclear_audio_chunks": sum("[unclear audio]" in n.text_en for n in kept),
        "filled_from_production": filled, "total_count": col.count(),
        "total_unclear_audio_chunks": sum("[unclear audio]" in (d or "") for d in docs),
        "embedding": {"model": BGE_MODEL, "use_fp16": True, "batch_size": 16, "max_length": 512,
                      "text": "raw chunk text (as production)", "device": bge_device,
                      "runtime_s": round(embed_s, 1)},
        "normalizer": {"use_cache": False, "save_cache": False, "runtime_s": round(normalize_s, 1)},
        "peak_rss_gb": _peak_rss_gb(),
    })
    print(f"indexed {len(kept)} candidate chunks (+{filled} production) into {args.name}: {col.count()} total")


# ── parity: are candidate embeddings made exactly like production ones? ───────

def cmd_parity(args) -> None:
    """Re-embed production chunk texts with the same code and compare to the stored vectors."""
    import numpy as np
    from ingestion.chunker import Chunk
    from ingestion.embedder import _embed_chunks, _load_bge_model

    sample: list[Chunk] = []
    stored: dict[str, list[float]] = {}
    for n in (2, 7, 13, 17):
        vid = VIDEOS[n]["video_id"]
        data = json.loads((PROCESSED_DIR / "chunks" / f"{vid}_chunks.json").read_text())
        sample += [Chunk(**c) for c in data["chunks"][:10]]
        stored.update(json.loads((PROCESSED_DIR / "embeddings" / f"{vid}_embeddings.json").read_text())["embeddings"])
    new = _embed_chunks(sample, _load_bge_model(BGE_MODEL))
    cos = [float(np.dot(new[c.chunk_id], stored[c.chunk_id])
                 / (np.linalg.norm(new[c.chunk_id]) * np.linalg.norm(stored[c.chunk_id]))) for c in sample]
    print(f"re-embedded {len(cos)} production chunks: cosine to stored vectors min {min(cos):.6f} "
          f"mean {sum(cos) / len(cos):.6f}")


def _version(pkg: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("transcribe")
    t.add_argument("--name", required=True)
    t.add_argument("--model", required=True)
    t.add_argument("--task", choices=["transcribe", "translate"], required=True)
    t.add_argument("--device", default="cpu")
    t.add_argument("--videos", type=int, nargs="*")
    t.add_argument("--force", action="store_true")
    t.add_argument("--greedy", action="store_true",
                   help="temperature=0 without fallback, condition_on_previous_text=False")
    i = sub.add_parser("index")
    i.add_argument("--name", required=True)
    i.add_argument("--videos", type=int, nargs="*")
    i.add_argument("--fill-from-production", action="store_true",
                   help="pilot: use the committed production chunks for all other videos")
    sub.add_parser("parity")
    r = sub.add_parser("rechunk")
    r.add_argument("--source", required=True, help="artifact name whose cleaned transcripts are reused")
    r.add_argument("--name", required=True)
    r.add_argument("--window", type=int, required=True)
    r.add_argument("--overlap", type=int, required=True)
    r.add_argument("--gap", type=float, help="gap threshold in seconds (default: production 5.0)")
    args = p.parse_args()
    {"transcribe": cmd_transcribe, "index": cmd_index, "parity": cmd_parity,
     "rechunk": cmd_rechunk}[args.cmd](args)


if __name__ == "__main__":
    main()
