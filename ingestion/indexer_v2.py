"""
ingestion/indexer_v2.py
────────────────────────
Stage 14: Build the v2 ChromaDB index using English-normalised chunk text.

Architecture:
  Chunks (v1) → Normalizer (quality filter + English translation)
              → BGE-M3 embedding of text_en
              → ChromaDB collection: eduvision_chunks_v2

Differences from v1:
  - text_en (English) is the ChromaDB document field (used for retrieval)
  - text_raw (original) is stored in metadata for provenance
  - Low-quality chunks (ASR noise) are skipped
  - Collection name is CHROMA_COLLECTION_NAME_V2 = "eduvision_chunks_v2"
  - v1 collection is never modified

Run:
    python ingestion/indexer_v2.py
    python ingestion/indexer_v2.py --force   # rebuild from scratch
"""

import argparse
import json
import logging
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import (
    CHROMA_DB_PATH,
    PROCESSED_DIR,
    BGE_MODEL,
)
from ingestion.chunker import load_chunks
from ingestion.normalizer import normalize_chunks, print_normalization_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── V2 collection name (v1 is never touched) ──────────────────────────────────
CHROMA_COLLECTION_NAME_V2 = "eduvision_chunks_v2"

# ── Batch size for Chroma upserts ─────────────────────────────────────────────
INDEX_BATCH_SIZE = 50


def _get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def _load_embeddings_v1(video_id: str) -> dict[str, list[float]]:
    """
    Load the existing v1 BGE-M3 embeddings (built in Stage 6).
    These are indexed by chunk_id.
    """
    emb_path = PROCESSED_DIR / "embeddings" / f"{video_id}_embeddings.json"
    if not emb_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {emb_path}")
    with open(emb_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["embeddings"]  # dict: chunk_id → embedding list


def main(force: bool = False) -> None:
    print("=" * 68)
    print("EduVision RAG — Stage 14: Build v2 Index (English Normalised)")
    print("=" * 68)
    print()

    client = _get_chroma_client()

    # Check if v2 already exists
    existing_names = [c.name for c in client.list_collections()]
    if CHROMA_COLLECTION_NAME_V2 in existing_names:
        if force:
            logger.info("--force: deleting existing v2 collection")
            client.delete_collection(CHROMA_COLLECTION_NAME_V2)
        else:
            existing = client.get_collection(CHROMA_COLLECTION_NAME_V2)
            count = existing.count()
            print(f"  v2 collection already exists with {count} docs.")
            print(f"  Use --force to rebuild. Exiting.")
            return

    # ── Step 1: Discover and load all processed chunks + embeddings ─────────
    print("── Step 1: Discovering processed videos ──────────────────────────")
    chunk_files = sorted((PROCESSED_DIR / "chunks").glob("*_chunks.json"))
    if not chunk_files:
        logger.error(
            "No chunk files found in %s. Run Stages 2–6 (video_processor, "
            "transcriber, cleaner, chunker, embedder) first.",
            PROCESSED_DIR / "chunks",
        )
        return

    # Read video_id from each chunk file's metadata header — same pattern as indexer.py
    video_ids = []
    for cf in chunk_files:
        with open(cf, encoding="utf-8") as fh:
            meta = json.load(fh)
        video_ids.append(meta["video_id"])

    print(f"  Discovered {len(video_ids)} processed video(s):")
    for vid in video_ids:
        print(f"    {vid[:70]}")
    print()

    print("── Step 1b: Loading chunks and embeddings ────────────────────────")
    all_chunks = []
    all_embeddings = {}
    for vid in video_ids:
        chunks = load_chunks(vid)
        if not chunks:
            logger.error("No chunks found for %s. Run Stage 5 first.", vid)
            return
        embs = _load_embeddings_v1(vid)
        all_chunks.extend(chunks)
        all_embeddings.update(embs)
        print(f"  {vid[:55]}: {len(chunks)} chunks, {len(embs)} embeddings")

    print(f"\n  Total: {len(all_chunks)} chunks, {len(all_embeddings)} embeddings\n")

    # ── Step 2: Normalise (quality filter + translation) ──────────────────────
    print("── Step 2: Normalising chunks (quality filter + English translation) ──")
    t0 = time.time()
    normalized = normalize_chunks(all_chunks, use_cache=True, save_cache=True)
    elapsed = time.time() - t0

    # Print audit report
    print_normalization_report(normalized)

    quality_ok   = [n for n in normalized if n.quality_ok]
    low_quality  = [n for n in normalized if not n.quality_ok]
    translated   = [n for n in normalized if n.translated]

    print(f"  Normalisation complete in {elapsed:.1f}s")
    print(f"  Quality OK:   {len(quality_ok)}")
    print(f"  Low-quality:  {len(low_quality)} (excluded from v2)")
    print(f"  Translated:   {len(translated)}")
    print()

    # ── Step 3: IMPORTANT — We do NOT re-embed. ───────────────────────────────
    # We use the EXISTING v1 BGE-M3 embeddings for ALL chunks.
    # The v2 experiment tests whether the English text DISPLAY improves
    # the user experience. The embedding vectors remain the same as v1.
    #
    # RATIONALE: Stage 13 showed that BGE-M3 already cross-linguistically
    # anchors on shared technical keywords. Re-embedding would require a
    # full ~30s GPU pass and would change the vectors, making it impossible
    # to isolate whether any eval improvement came from the text change
    # or the embedding change. We test display quality first.
    #
    # If this experiment shows text_en display improvements are valuable,
    # re-embedding with text_en can be tested as Stage 15.

    print("── Step 3: Verifying v1 embeddings cover quality-OK chunks ──────")
    missing_embs = [n for n in quality_ok if n.chunk_id not in all_embeddings]
    if missing_embs:
        logger.error("Missing embeddings for %d chunks: %s", len(missing_embs),
                     [n.chunk_id for n in missing_embs[:3]])
        return
    print(f"  All {len(quality_ok)} quality-OK chunks have embeddings ✅\n")

    # ── Step 4: Build v2 ChromaDB collection ─────────────────────────────────
    print("── Step 4: Building v2 ChromaDB collection ──────────────────────")
    collection = client.create_collection(
        name=CHROMA_COLLECTION_NAME_V2,
        metadata={"hnsw:space": "cosine"},
    )

    total_upserted = 0
    batch: list = []

    def _flush_batch(batch):
        if not batch:
            return
        collection.upsert(
            ids=[b["id"] for b in batch],
            embeddings=[b["embedding"] for b in batch],
            documents=[b["document"] for b in batch],   # text_en
            metadatas=[b["metadata"] for b in batch],
        )

    for norm in quality_ok:
        chunk = norm.chunk
        metadata = {
            "video_id":           chunk.video_id,
            "video_filename":     chunk.video_filename,
            "language":           chunk.language,
            "start_time":         chunk.start_time,
            "end_time":           chunk.end_time,
            "start_time_fmt":     chunk.start_time_fmt,
            "end_time_fmt":       chunk.end_time_fmt,
            "chunk_index":        chunk.chunk_index,
            "source_segment_ids": json.dumps(chunk.source_segment_ids),
            # Provenance fields (v2 adds these)
            "text_raw":           norm.text_raw,
            "translated":         str(norm.translated),
        }
        batch.append({
            "id":        norm.chunk_id,
            "embedding": all_embeddings[norm.chunk_id],
            "document":  norm.text_en,   # ← English text as retrieval document
            "metadata":  metadata,
        })

        if len(batch) >= INDEX_BATCH_SIZE:
            _flush_batch(batch)
            total_upserted += len(batch)
            print(f"  Upserted {total_upserted}/{len(quality_ok)}...")
            batch = []

    if batch:
        _flush_batch(batch)
        total_upserted += len(batch)

    print(f"\n  ✅ v2 collection built: {total_upserted} documents in '{CHROMA_COLLECTION_NAME_V2}'")
    print()

    # ── Step 5: Verification ──────────────────────────────────────────────────
    print("── Step 5: Verification ─────────────────────────────────────────")
    col = client.get_collection(CHROMA_COLLECTION_NAME_V2)
    count = col.count()
    expected = len(quality_ok)
    print(f"  Collection count = {count} (expected {expected}): {'✅' if count == expected else '❌'}")

    # Spot-check: retrieve one chunk by ID
    if quality_ok:
        spot = quality_ok[0]
        result = col.get(ids=[spot.chunk_id], include=["documents", "metadatas"])
        doc = result["documents"][0] if result["documents"] else ""
        meta = result["metadatas"][0] if result["metadatas"] else {}
        print(f"  Spot-check {spot.chunk_id}:")
        print(f"    document (text_en): {doc[:80]}")
        print(f"    metadata text_raw:  {meta.get('text_raw', '')[:80]}")
        print(f"    start_time:         {meta.get('start_time')}")
        print(f"    video_id:           {meta.get('video_id', '')[:50]}")

    print()
    print("=" * 68)
    print(f"v2 index complete: {total_upserted} chunks indexed")
    print(f"  Low-quality chunks excluded: {len(low_quality)}")
    print(f"  Collection name: {CHROMA_COLLECTION_NAME_V2}")
    print(f"  Existing v1 (eduvision_chunks): UNTOUCHED")
    print("=" * 68)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build v2 ChromaDB index")
    parser.add_argument("--force", action="store_true",
                        help="Delete and rebuild the v2 collection from scratch")
    args = parser.parse_args()
    main(force=args.force)
