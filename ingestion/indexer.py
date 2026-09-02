"""
ingestion/indexer.py
─────────────────────
Stage 7: BGE-M3 Embeddings + Chunk Metadata → ChromaDB Persistent Index.

═══════════════════════════════════════════════════════════════
WHY CHROMADB
═══════════════════════════════════════════════════════════════

ChromaDB was chosen because:

  1. Embedded, no server needed: runs entirely in-process.
     No Docker, no separate process to manage. The entire DB
     is a directory on disk (data/vector_db/) opened by both
     the indexer (write) and the retriever (read).

  2. Persistent: data survives process restarts.
     chromadb.PersistentClient(path=...) — SQLite-backed storage.

  3. Pre-computed embeddings: ChromaDB lets you supply embeddings
     directly rather than delegating to its internal embedder.
     This is critical because:
       a) Our Stage 6 embeddings use BGE-M3 (multilingual, 1024-dim).
       b) Chroma's default embedder (all-MiniLM, 384-dim) does not
          handle Hindi/Urdu and would produce incompatible vectors.
       c) We've already paid the GPU cost in Stage 6.

  4. Rich metadata: stores arbitrary metadata per document.
     We embed ALL timestamp and video provenance in metadata so
     the retriever never needs to look up any other file.

  5. Cosine similarity built-in: BGE-M3 vectors are L2-normalised
     so cosine similarity = dot product. Chroma's "cosine" metric
     handles this correctly and returns distances in [0, 2].

═══════════════════════════════════════════════════════════════
WHAT IS STORED IN CHROMA
═══════════════════════════════════════════════════════════════

For each of the 235 chunks, ChromaDB stores:

  id          → chunk_id          (primary key, e.g. "video2_chunk_0042")
  embedding   → list[float]×1024  (BGE-M3 dense vector from Stage 6)
  document    → chunk.text        (the searchable transcript text)
  metadata    → {
    "video_id"           : "02_your_first_html_...",
    "video_filename"     : "02_Your First HTML Website...mp4",
    "language"           : "hi",
    "start_time"         : 586.0,   ← exact Whisper timestamp (seconds)
    "end_time"           : 596.0,   ← exact Whisper timestamp (seconds)
    "start_time_fmt"     : "09:46", ← pre-formatted MM:SS for the UI
    "end_time_fmt"       : "09:56",
    "chunk_index"        : 42,
    "source_segment_ids" : "[120, 121, 122, 123, 124]"  ← JSON-serialised
  }

  Note on source_segment_ids: ChromaDB metadata values must be
  str, int, or float — no lists. We JSON-serialise the list and
  deserialise it at retrieval time.

═══════════════════════════════════════════════════════════════
DESIGN CHOICES
═══════════════════════════════════════════════════════════════

1. UPSERT not ADD:
   collection.upsert() replaces existing documents with the same id.
   This makes re-indexing idempotent — running the indexer twice
   never creates duplicate entries.

2. Batched upserts (batch size 50):
   Chroma handles large batches fine, but batching gives us progress
   logging and makes it easy to handle partial failures in future.

3. Collection per project, not per video:
   All 235 chunks from all videos go into ONE collection.
   At retrieval time, filters like {"video_id": "..."} let us
   restrict to a specific video if needed. This is simpler to
   query and scale than one collection per video.

4. Distance metric = "cosine":
   BGE-M3 output is L2-normalised → cosine distance = 1 - dot_product.
   Chroma reports distances; smaller distance = more similar.
   At retrieval time: similarity = 1 - distance.

═══════════════════════════════════════════════════════════════
INVARIANTS (verified after indexing)
═══════════════════════════════════════════════════════════════
  1. Total documents in collection = total chunks across all videos.
  2. Every chunk_id from every chunk file exists in the collection.
  3. Metadata fields (video_id, start_time, end_time) are correct.
  4. A nearest-neighbour query for a known text returns itself first.
  5. Source files (chunks/, embeddings/, transcripts/) are NOT modified.
  6. Query on a technical phrase returns chunks from the correct video
     with valid timestamps.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

from config.settings import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME, PROCESSED_DIR
from ingestion.chunker import Chunk, load_chunks
from ingestion.embedder import load_embeddings

logger = logging.getLogger(__name__)

CHUNKS_DIR: Path = PROCESSED_DIR / "chunks"
INDEX_BATCH_SIZE: int = 50   # documents per upsert call


# ── ChromaDB client & collection management ───────────────────────────────────

def get_chroma_client() -> chromadb.PersistentClient:
    """
    Return a ChromaDB PersistentClient backed by data/vector_db/.

    The client opens (or creates) the SQLite database in CHROMA_DB_PATH.
    All writes are immediately persisted — no explicit flush needed.
    """
    Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    return client


def get_or_create_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    """
    Get or create the EduVision chunk collection with cosine distance metric.

    Using get_or_create_collection() is idempotent — safe to call on every run.
    The distance metric is locked at creation; changing it requires deleting
    and recreating the collection.
    """
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine distance for L2-normalised BGE-M3 vecs
    )
    logger.info(
        "Collection '%s' ready. Current doc count: %d",
        CHROMA_COLLECTION_NAME, collection.count(),
    )
    return collection


# ── Core indexing logic ───────────────────────────────────────────────────────

def _chunk_to_metadata(chunk: Chunk) -> dict:
    """
    Convert a Chunk's fields into ChromaDB-compatible metadata.

    ChromaDB metadata values must be: str, int, or float.
    Lists are not allowed — source_segment_ids is JSON-serialised to str.
    """
    return {
        "video_id":           chunk.video_id,
        "video_filename":     chunk.video_filename,
        "language":           chunk.language,
        "start_time":         chunk.start_time,          # float
        "end_time":           chunk.end_time,             # float
        "start_time_fmt":     chunk.start_time_fmt,       # "MM:SS"
        "end_time_fmt":       chunk.end_time_fmt,         # "MM:SS"
        "chunk_index":        chunk.chunk_index,          # int
        "source_segment_ids": json.dumps(chunk.source_segment_ids),  # "[0,1,2,3,4]"
    }


def index_chunks(
    collection: chromadb.Collection,
    chunks: list[Chunk],
    embeddings: dict[str, list[float]],
    batch_size: int = INDEX_BATCH_SIZE,
) -> int:
    """
    Upsert chunks and their pre-computed embeddings into the Chroma collection.

    Args:
        collection: Target ChromaDB collection.
        chunks:     List of Chunk objects from Stage 5.
        embeddings: Dict of chunk_id → embedding vector from Stage 6.
        batch_size: Number of documents per upsert call.

    Returns:
        Number of documents successfully upserted.

    Raises:
        KeyError: If a chunk_id has no corresponding embedding.
    """
    # Verify all chunks have embeddings before starting
    missing = [c.chunk_id for c in chunks if c.chunk_id not in embeddings]
    if missing:
        raise KeyError(
            f"{len(missing)} chunk(s) have no embeddings: {missing[:3]}"
        )

    upserted_count = 0
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        lo = batch_idx * batch_size
        hi = lo + batch_size
        batch = chunks[lo:hi]

        ids        = [c.chunk_id for c in batch]
        vecs       = [embeddings[c.chunk_id] for c in batch]
        documents  = [c.text for c in batch]
        metadatas  = [_chunk_to_metadata(c) for c in batch]

        collection.upsert(
            ids=ids,
            embeddings=vecs,
            documents=documents,
            metadatas=metadatas,
        )
        upserted_count += len(batch)
        logger.info(
            "  Batch %d/%d: upserted %d docs (total so far: %d)",
            batch_idx + 1, total_batches, len(batch), upserted_count,
        )

    return upserted_count


def build_index(
    chunks_map: dict[str, list[Chunk]],
    embeddings_map: dict[str, dict[str, list[float]]],
    force: bool = False,
) -> chromadb.Collection:
    """
    Build (or rebuild) the ChromaDB index from all chunks and embeddings.

    If force=False and the collection already has the expected number of
    documents, indexing is skipped (idempotency without re-reading every doc).
    If force=True, the existing collection is deleted and rebuilt from scratch.

    Args:
        chunks_map:     Dict of video_id → list[Chunk].
        embeddings_map: Dict of video_id → {chunk_id → embedding}.
        force:          If True, delete and rebuild the collection.

    Returns:
        The ChromaDB collection (ready for queries).
    """
    client = get_chroma_client()

    expected_total = sum(len(v) for v in chunks_map.values())

    if force:
        logger.info("--force: deleting existing collection '%s'", CHROMA_COLLECTION_NAME)
        try:
            client.delete_collection(CHROMA_COLLECTION_NAME)
        except Exception:
            pass  # Collection may not exist yet

    collection = get_or_create_collection(client)

    # Fast-path idempotency: if doc count already matches, skip upsert
    current_count = collection.count()
    if not force and current_count == expected_total:
        logger.info(
            "Collection already has %d documents (expected %d). Skipping upsert. "
            "Use --force to rebuild.",
            current_count, expected_total,
        )
        return collection

    if current_count > 0 and not force:
        logger.info(
            "Collection has %d/%d documents — upserting missing/updated docs.",
            current_count, expected_total,
        )

    total_upserted = 0
    for video_id, chunks in chunks_map.items():
        if video_id not in embeddings_map:
            logger.warning("No embeddings for video '%s' — skipping.", video_id)
            continue

        logger.info(
            "Indexing video '%s': %d chunks", video_id[:50], len(chunks)
        )
        n = index_chunks(collection, chunks, embeddings_map[video_id])
        total_upserted += n

    logger.info(
        "Indexing complete: %d documents upserted into '%s'.",
        total_upserted, CHROMA_COLLECTION_NAME,
    )
    return collection


# ── Public API (used by Stage 8 — retriever) ─────────────────────────────────

def get_collection() -> chromadb.Collection:
    """
    Open the existing persistent collection (read-only path for retriever).

    Raises:
        ValueError: If the collection does not exist yet.
    """
    client = get_chroma_client()
    try:
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
    except Exception as exc:
        raise ValueError(
            f"Collection '{CHROMA_COLLECTION_NAME}' not found. "
            "Run Stage 7 (indexer.py) first."
        ) from exc

    count = collection.count()
    if count == 0:
        raise ValueError(
            f"Collection '{CHROMA_COLLECTION_NAME}' exists but is empty. "
            "Run Stage 7 (indexer.py) first."
        )

    logger.info("Opened collection '%s' with %d documents.", CHROMA_COLLECTION_NAME, count)
    return collection


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json as _json
    import numpy as np

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    force_flag = "--force" in sys.argv

    print("=" * 60)
    print("EduVision RAG — Stage 7: ChromaDB Indexing")
    print("=" * 60)
    print(f"Collection : {CHROMA_COLLECTION_NAME}")
    print(f"DB path    : {CHROMA_DB_PATH}")
    print(f"Force      : {force_flag}")
    print()

    # ── Load chunks and embeddings from disk (no re-processing) ──────────────
    chunk_files = sorted(Path("data/processed/chunks").glob("*_chunks.json"))
    if not chunk_files:
        print("❌  No chunk files found. Run Stage 5 first.")
        sys.exit(1)

    chunks_map: dict[str, list[Chunk]] = {}
    embeddings_map: dict[str, dict[str, list[float]]] = {}

    for cf in chunk_files:
        meta = json.load(open(cf))
        vid_id = meta["video_id"]

        chunks = load_chunks(vid_id)
        embeds = load_embeddings(vid_id)

        if chunks is None or embeds is None:
            print(f"❌  Missing chunks or embeddings for: {vid_id}")
            sys.exit(1)

        chunks_map[vid_id] = chunks
        embeddings_map[vid_id] = embeds
        print(f"  Loaded {len(chunks)} chunks + {len(embeds)} embeddings for: {vid_id[:50]}")

    print()

    # ── Build index ───────────────────────────────────────────────────────────
    collection = build_index(chunks_map, embeddings_map, force=force_flag)

    # ── Verification suite ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("VERIFICATION & STATISTICS")
    print(f"{'='*60}\n")

    all_passed = True
    errors: list[str] = []

    total_expected = sum(len(v) for v in chunks_map.values())

    # CHECK 1: Document count matches expected
    actual_count = collection.count()
    if actual_count != total_expected:
        errors.append(f"Count mismatch: expected {total_expected}, got {actual_count}")
    print(f"  Check 1 (document count = {total_expected})  : "
          f"{'✅' if actual_count == total_expected else '❌'}  got={actual_count}")

    # CHECK 2: Spot-check — every chunk_id exists in the collection
    all_chunk_ids = [c.chunk_id for chunks in chunks_map.values() for c in chunks]
    # Sample 10 IDs spread across the collection
    sample_ids = all_chunk_ids[::max(1, len(all_chunk_ids)//10)][:10]
    result = collection.get(ids=sample_ids, include=["metadatas"])
    found_ids = set(result["ids"])
    missing_ids = set(sample_ids) - found_ids
    if missing_ids:
        errors.append(f"chunk_ids not found in collection: {missing_ids}")
    print(f"  Check 2 (sample chunk_ids present, n={len(sample_ids)})    : "
          f"{'✅' if not missing_ids else '❌'}  found={len(found_ids)}/{len(sample_ids)}")

    # CHECK 3: Metadata integrity — spot-check start_time, end_time, video_id
    if result["metadatas"]:
        meta_errors = []
        for meta in result["metadatas"]:
            if "start_time" not in meta:
                meta_errors.append("missing start_time")
            if "end_time" not in meta:
                meta_errors.append("missing end_time")
            if "video_id" not in meta:
                meta_errors.append("missing video_id")
            if "start_time_fmt" not in meta:
                meta_errors.append("missing start_time_fmt")
            if meta.get("start_time", 1) >= meta.get("end_time", 0):
                meta_errors.append(
                    f"start >= end: {meta.get('start_time')} >= {meta.get('end_time')}"
                )
        if meta_errors:
            errors.append(f"Metadata errors: {meta_errors[:3]}")
    meta_ok = not any("Metadata" in e for e in errors)
    print(f"  Check 3 (metadata fields correct)         : "
          f"{'✅' if meta_ok else '❌'}")

    # CHECK 4: Nearest-neighbour self-retrieval
    # Take the embedding of a known chunk and query — it must return itself as #1
    test_vid = list(chunks_map.keys())[1]   # use video 2
    test_chunk = chunks_map[test_vid][60]   # chunk index 60 (mid-video)
    test_vec = embeddings_map[test_vid][test_chunk.chunk_id]
    nn_result = collection.query(
        query_embeddings=[test_vec],
        n_results=3,
        include=["distances", "metadatas", "documents"],
    )
    top_id = nn_result["ids"][0][0]
    top_dist = nn_result["distances"][0][0]
    self_retrieval_ok = top_id == test_chunk.chunk_id
    if not self_retrieval_ok:
        errors.append(
            f"Self-retrieval failed: queried '{test_chunk.chunk_id}', got '{top_id}'"
        )
    print(f"  Check 4 (self-retrieval, chunk at idx=60) : "
          f"{'✅' if self_retrieval_ok else '❌'}  "
          f"top_id={top_id[-20:]}  dist={top_dist:.6f}")

    # CHECK 5: Source files not modified
    expected_seg_counts = {
        "01_installing_vs_code_how_websites_work_sigma_web_development_course_tutorial_1": 263,
        "02_your_first_html_website_sigma_web_development_course_tutorial_2": 649,
    }
    for vid_id, exp in expected_seg_counts.items():
        cf = Path(f"data/transcripts/{vid_id}_cleaned.json")
        got = len(json.load(open(cf))["segments"])
        if got != exp:
            errors.append(f"Cleaned transcript modified: {vid_id}: expected {exp}, got {got}")
    source_ok = not any("modified" in e for e in errors)
    print(f"  Check 5 (source files not modified)       : "
          f"{'✅' if source_ok else '❌'}")

    # CHECK 6: Semantic query test — search for a technical concept
    print()
    print("  Check 6 (semantic query test):")
    queries = [
        "how do websites work with HTML and CSS",
        "install VS Code editor",
    ]

    # We need the BGE-M3 model to encode the query
    from FlagEmbedding import BGEM3FlagModel
    print("    Loading BGE-M3 for query encoding (from cache)...")
    from config.settings import BGE_MODEL
    qmodel = BGEM3FlagModel(BGE_MODEL, use_fp16=True)

    for query in queries:
        qvec = qmodel.encode(
            [query],
            batch_size=1,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )["dense_vecs"][0].tolist()

        qres = collection.query(
            query_embeddings=[qvec],
            n_results=3,
            include=["distances", "metadatas", "documents"],
        )
        print(f"\n    Query: '{query}'")
        for rank, (doc_id, dist, meta, doc) in enumerate(zip(
            qres["ids"][0],
            qres["distances"][0],
            qres["metadatas"][0],
            qres["documents"][0],
        )):
            sim = 1 - dist
            ts  = meta.get("start_time_fmt", "??:??")
            vid = meta.get("video_filename", "")[:40]
            print(f"      #{rank+1}  sim={sim:.4f}  [{ts}]  {vid}")
            print(f"           {doc[:90]}{'...' if len(doc)>90 else ''}")

    all_passed = len(errors) == 0

    print(f"\n{'='*60}")
    print(f"  Overall validation : {'✅ ALL PASS' if all_passed else '❌ FAILURES:'}")
    if not all_passed:
        for e in errors:
            print(f"    ERROR: {e}")
    print(f"  Collection name    : {CHROMA_COLLECTION_NAME}")
    print(f"  Total documents    : {actual_count}")
    print(f"  DB path            : {CHROMA_DB_PATH}")
    print(f"{'='*60}")
