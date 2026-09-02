"""
archive/patch_t1_stage15.py
────────────────────────────
ARCHIVE — ONE-TIME HISTORICAL REPAIR SCRIPT.  DO NOT RUN.

This script was executed once during Stage 15 of the EduVision RAG project to
repair a specific section (08:30–16:00) of Tutorial #1 whose Whisper base-model
transcription contained hallucinated text.  It has been moved here for audit
trail purposes only.

It is NOT part of the normal ingestion pipeline.  Running it again would re-patch
data that has already been correctly indexed in eduvision_chunks_v2.

Normal ingestion entry point: ingestion/indexer_v2.py
"""

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
VIDEO_ID_T1 = "01_installing_vs_code_how_websites_work_sigma_web_development_course_tutorial_1"
VIDEO_FILE  = "videos/01_Installing VS Code & How Websites Work ｜ Sigma Web Development Course - Tutorial #1.mp4"

PATCH_START_SEC = 510.0   # 08:30 — base model hallucination starts here
PATCH_END_SEC   = 960.0   # 16:00

AUDIO_CACHE = Path("/tmp/t1_patch_stage15.wav")

# Must match ingestion/chunker.py
CHUNK_WINDOW  = 5
CHUNK_OVERLAP = 1
GAP_THRESHOLD = 5.0


def extract_audio(force: bool = False) -> Path:
    if AUDIO_CACHE.exists() and not force:
        logger.info("Using cached audio: %s", AUDIO_CACHE)
        return AUDIO_CACHE
    logger.info("Extracting audio 08:30–16:00 ...")
    cmd = [
        "ffmpeg", "-y", "-i", VIDEO_FILE,
        "-ss", "00:08:30", "-to", "00:16:00",
        "-vn", "-ac", "1", "-ar", "16000", "-f", "wav",
        str(AUDIO_CACHE),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{r.stderr[-400:]}")
    logger.info("Audio ready: %.1f MB", AUDIO_CACHE.stat().st_size / 1024 / 1024)
    return AUDIO_CACHE


def transcribe(audio_path: Path) -> list[dict]:
    import whisper
    logger.info("Loading Whisper medium ...")
    model = whisper.load_model("medium")
    logger.info("Transcribing (translate to English) ...")
    result = model.transcribe(
        str(audio_path),
        language="hi",
        task="translate",
        word_timestamps=False,
        verbose=False,
        fp16=False,
        temperature=0.0,
        no_speech_threshold=0.6,
        condition_on_previous_text=True,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
    )
    noise = {"[Music]", "[Applause]", "[Laughter]"}
    segs = []
    for seg in result["segments"]:
        text = seg["text"].strip()
        if not text or text in noise or len(text) < 4:
            continue
        segs.append({
            "start": seg["start"] + PATCH_START_SEC,
            "end":   seg["end"]   + PATCH_START_SEC,
            "text":  text,
        })
    logger.info("Transcribed: %d usable segments", len(segs))
    return segs


def make_chunks(segments: list[dict], id_offset: int, video_filename: str) -> list[dict]:
    chunks, i, idx = [], 0, id_offset
    while i < len(segments):
        win = [segments[i]]
        j = i + 1
        while j < len(segments) and len(win) < CHUNK_WINDOW:
            if segments[j]["start"] - segments[j-1]["end"] > GAP_THRESHOLD:
                break
            win.append(segments[j])
            j += 1
        text = " ".join(s["text"] for s in win).strip()
        if text:
            chunks.append({
                "chunk_id":           f"{VIDEO_ID_T1}_chunk_{idx:04d}",
                "chunk_index":        idx,
                "video_id":           VIDEO_ID_T1,
                "video_filename":     video_filename,
                "language":           "en",
                "text":               text,
                "start_time":         win[0]["start"],
                "end_time":           win[-1]["end"],
                "source_segment_ids": [],
            })
            idx += 1
        i += max(1, CHUNK_WINDOW - CHUNK_OVERLAP)
    logger.info("Created %d new chunks (indices %d–%d)", len(chunks), id_offset, idx-1)
    return chunks


def embed_chunks(chunks: list[dict]) -> dict[str, list[float]]:
    from FlagEmbedding import BGEM3FlagModel
    from config.settings import BGE_MODEL
    if not chunks:
        return {}
    logger.info("Loading BGE-M3 ...")
    model = BGEM3FlagModel(BGE_MODEL, use_fp16=False)
    texts = [c["text"] for c in chunks]
    logger.info("Embedding %d chunks ...", len(texts))
    out = model.encode(texts, batch_size=12, max_length=512)
    return {c["chunk_id"]: v.tolist() for c, v in zip(chunks, out["dense_vecs"])}


def patch_chunk_file(new_chunks: list[dict]) -> tuple[int, set[str]]:
    from config.settings import PROCESSED_DIR
    path = PROCESSED_DIR / "chunks" / f"{VIDEO_ID_T1}_chunks.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    kept = [c for c in data["chunks"] if c["start_time"] < PATCH_START_SEC]
    kept_ids = {c["chunk_id"] for c in kept}
    logger.info("Chunk file: kept %d / dropped %d / adding %d",
                len(kept), len(data["chunks"])-len(kept), len(new_chunks))
    data["chunks"] = kept + new_chunks
    data["total_chunks"] = len(data["chunks"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data["total_chunks"], kept_ids


def patch_embedding_file(new_embs: dict, kept_ids: set[str]) -> None:
    from config.settings import PROCESSED_DIR
    path = PROCESSED_DIR / "embeddings" / f"{VIDEO_ID_T1}_embeddings.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    kept_embs = {k: v for k, v in data["embeddings"].items() if k in kept_ids}
    logger.info("Embedding file: kept %d / dropped %d / adding %d",
                len(kept_embs), len(data["embeddings"])-len(kept_embs), len(new_embs))
    kept_embs.update(new_embs)
    data["embeddings"] = kept_embs
    data["total_embeddings"] = len(kept_embs)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def patch_normalizer_cache(new_chunks: list[dict]) -> None:
    from config.settings import PROCESSED_DIR
    path = PROCESSED_DIR / "normalizer_cache.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    added = 0
    for c in new_chunks:
        if c["chunk_id"] not in cache:
            cache[c["chunk_id"]] = {
                "chunk_id":  c["chunk_id"],
                "text_raw":  c["text"],
                "text_en":   c["text"],
                "translated": True,
                "quality_ok": True,
                "stage":     "stage15_medium_translate",
            }
            added += 1
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Normalizer cache: +%d entries (%d total)", added, len(cache))


def rebuild_v2() -> int:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from config.settings import CHROMA_DB_PATH, PROCESSED_DIR
    from ingestion.chunker import load_chunks
    from ingestion.normalizer import normalize_chunks

    COLL = "eduvision_chunks_v2"
    VIDS = [
        VIDEO_ID_T1,
        "02_your_first_html_website_sigma_web_development_course_tutorial_2",
    ]

    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    existing = [c.name for c in client.list_collections()]
    if COLL in existing:
        logger.info("Deleting existing v2 collection ...")
        client.delete_collection(COLL)

    all_chunks, all_embs = [], {}
    for vid in VIDS:
        chunks = load_chunks(vid)
        all_chunks.extend(chunks)
        emb_path = PROCESSED_DIR / "embeddings" / f"{vid}_embeddings.json"
        emb_data = json.loads(emb_path.read_text(encoding="utf-8"))
        all_embs.update(emb_data["embeddings"])
        logger.info("  %s: %d chunks, %d embeddings", vid[:50], len(chunks), len(emb_data["embeddings"]))

    logger.info("Total: %d chunks, %d embeddings", len(all_chunks), len(all_embs))

    normalized = normalize_chunks(all_chunks, use_cache=True, save_cache=True)
    quality_ok = [n for n in normalized if n.quality_ok]
    missing = [n for n in quality_ok if n.chunk_id not in all_embs]
    if missing:
        raise RuntimeError(f"Missing embeddings for: {[m.chunk_id for m in missing[:5]]}")

    collection = client.create_collection(name=COLL, metadata={"hnsw:space": "cosine"})
    batch, BSIZE = [], 50

    def _flush(b):
        if b:
            collection.upsert(
                ids=[x["id"] for x in b],
                embeddings=[x["emb"] for x in b],
                documents=[x["doc"] for x in b],
                metadatas=[x["meta"] for x in b],
            )

    for norm in quality_ok:
        c = norm.chunk
        batch.append({
            "id":  norm.chunk_id,
            "emb": all_embs[norm.chunk_id],
            "doc": norm.text_en,
            "meta": {
                "video_id":           c.video_id,
                "video_filename":     c.video_filename,
                "language":           c.language,
                "start_time":         c.start_time,
                "end_time":           c.end_time,
                "start_time_fmt":     c.start_time_fmt,
                "end_time_fmt":       c.end_time_fmt,
                "chunk_index":        c.chunk_index,
                "source_segment_ids": json.dumps(c.source_segment_ids),
                "text_raw":           norm.text_raw,
                "translated":         str(norm.translated),
            },
        })
        if len(batch) >= BSIZE:
            _flush(batch); batch = []
    _flush(batch)

    count = collection.count()
    logger.info("v2 rebuilt: %d documents ✅", count)
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-audio", action="store_true")
    args = parser.parse_args()

    print("=" * 68)
    print("EduVision RAG — Stage 15: Patch T1 + Rebuild v2")
    print("=" * 68)

    # Step 1
    print("\n── Step 1: Audio ──────────────────────────────────────────────")
    audio = extract_audio(force=args.force_audio)

    # Step 2
    print("\n── Step 2: Whisper medium translate ───────────────────────────")
    t0 = time.time()
    segments = transcribe(audio)
    print(f"   {len(segments)} segments in {time.time()-t0:.1f}s")
    print("   Sample (first/last 2):")
    for seg in segments[:2] + segments[-2:]:
        m1,s1 = divmod(int(seg['start']),60)
        print(f"     [{m1:02d}:{s1:02d}] {seg['text'][:90]}")

    # Step 3
    print("\n── Step 3: Chunk ID offset ─────────────────────────────────────")
    from config.settings import PROCESSED_DIR
    path = PROCESSED_DIR / "chunks" / f"{VIDEO_ID_T1}_chunks.json"
    existing_data = json.loads(path.read_text(encoding="utf-8"))
    kept = [c for c in existing_data["chunks"] if c["start_time"] < PATCH_START_SEC]
    video_filename = existing_data["chunks"][0]["video_filename"]
    print(f"   {len(kept)} chunks before 08:30 → new chunks start at index {len(kept)}")

    # Step 4
    print("\n── Step 4: Build new chunks ────────────────────────────────────")
    new_chunks = make_chunks(segments, len(kept), video_filename)
    print(f"   {len(new_chunks)} new chunks created")

    # Step 5
    print("\n── Step 5: Embed with BGE-M3 ───────────────────────────────────")
    t0 = time.time()
    new_embs = embed_chunks(new_chunks)
    print(f"   {len(new_embs)} embeddings in {time.time()-t0:.1f}s")

    # Step 6
    print("\n── Step 6: Patch chunk file ─────────────────────────────────────")
    total_chunks, kept_ids = patch_chunk_file(new_chunks)
    print(f"   T1 chunks total: {total_chunks}")

    # Step 7
    print("\n── Step 7: Patch embedding file ────────────────────────────────")
    patch_embedding_file(new_embs, kept_ids)

    # Step 8
    print("\n── Step 8: Update normalizer cache ─────────────────────────────")
    patch_normalizer_cache(new_chunks)

    # Step 9
    print("\n── Step 9: Rebuild eduvision_chunks_v2 ─────────────────────────")
    t0 = time.time()
    count = rebuild_v2()
    print(f"   v2 = {count} docs in {time.time()-t0:.1f}s")

    print("\n" + "=" * 68)
    print(f"Stage 15 complete  |  new T1 chunks: {len(new_chunks)}  |  v2: {count} docs")
    print("=" * 68)


if __name__ == "__main__":
    main()
