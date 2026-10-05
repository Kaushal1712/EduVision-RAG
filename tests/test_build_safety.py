"""
Offline tests for ingestion/build_safety.py and the guards wired into the ingestion stages.
No ChromaDB, Whisper, BGE-M3 or API access; every file written lives in a temporary directory.

Run:  venv/bin/python -m unittest discover -s tests -v
"""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from ingestion import build_safety as bs  # noqa: E402
from ingestion.build_safety import StaleOutputError, UnsafeBuildTarget  # noqa: E402


def _no_chroma(*args, **kwargs):
    raise AssertionError("chromadb.PersistentClient must not be called")


class TestIndexTargets(unittest.TestCase):

    def test_production_index_rejected_with_and_without_legacy_flag(self):
        for p in (settings.STAGE26_VECTOR_DB_DIR, "data/vector_db_lv2g_translate",
                  settings.STAGE26_VECTOR_DB_DIR / "sub", settings.DATA_DIR, ROOT):
            with self.assertRaises(UnsafeBuildTarget):
                bs.require_build_index_dir(p)
            with self.assertRaises(UnsafeBuildTarget):
                bs.require_build_index_dir(p, allow_legacy_index=True)

    def test_runtime_chroma_db_path_rejected(self):
        with self.assertRaises(UnsafeBuildTarget):
            bs.require_build_index_dir(settings.CHROMA_DB_PATH)
        with tempfile.TemporaryDirectory() as d:
            saved = settings.CHROMA_DB_PATH
            try:
                settings.CHROMA_DB_PATH = d             # e.g. an overridden runtime path
                with self.assertRaises(UnsafeBuildTarget):
                    bs.require_build_index_dir(d)
            finally:
                settings.CHROMA_DB_PATH = saved

    def test_legacy_index_rejected_unless_explicit(self):
        with self.assertRaises(UnsafeBuildTarget):
            bs.require_build_index_dir(settings.VECTOR_DB_DIR)
        self.assertEqual(bs.require_build_index_dir(settings.VECTOR_DB_DIR, allow_legacy_index=True),
                         settings.VECTOR_DB_DIR.resolve())

    def test_stage26_source_artifact_rejected(self):
        with self.assertRaises(UnsafeBuildTarget):
            bs.require_build_index_dir(bs.STAGE26_SOURCE_INDEX_DIR)

    def test_missing_target_rejected(self):
        for p in (None, "", "   "):
            with self.assertRaises(UnsafeBuildTarget):
                bs.require_build_index_dir(p)

    def test_non_production_target_accepted_and_not_created(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "rebuild" / "vector_db"
            self.assertEqual(bs.require_build_index_dir(target), target.resolve())
            self.assertFalse(target.exists())
        rel = "experiments/rebuild_check_never_created/vector_db"
        self.assertEqual(bs.require_build_index_dir(rel), (ROOT / rel).resolve())
        self.assertFalse((ROOT / rel).exists())


class TestIndexersRejectBeforeOpening(unittest.TestCase):

    def test_indexer_v2_rejects_production_and_missing_collection(self):
        from ingestion import indexer_v2
        saved = indexer_v2.chromadb.PersistentClient
        indexer_v2.chromadb.PersistentClient = _no_chroma
        try:
            with self.assertRaises(UnsafeBuildTarget):
                indexer_v2.main(db_path=str(settings.STAGE26_VECTOR_DB_DIR), collection_name="x")
            with self.assertRaises(UnsafeBuildTarget):
                indexer_v2.main(db_path=None, collection_name="x")
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaises(UnsafeBuildTarget):
                    indexer_v2.main(db_path=d, collection_name=None)
        finally:
            indexer_v2.chromadb.PersistentClient = saved

    def test_indexer_v1_build_index_rejects_production(self):
        from ingestion import indexer
        saved = indexer.chromadb.PersistentClient
        indexer.chromadb.PersistentClient = _no_chroma
        try:
            for p in (settings.STAGE26_VECTOR_DB_DIR, settings.VECTOR_DB_DIR, settings.CHROMA_DB_PATH):
                with self.assertRaises(UnsafeBuildTarget):
                    indexer.build_index({}, {}, db_path=str(p))
        finally:
            indexer.chromadb.PersistentClient = saved


class FakeWhisper:
    def __init__(self):
        self.calls = 0

    def transcribe(self, audio, **kwargs):
        self.calls += 1
        return {"segments": [], "language": "en", "text": ""}


class TestStaleOutputs(unittest.TestCase):

    def setUp(self):
        from ingestion import transcriber
        self.tr = transcriber
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.saved_dir = transcriber.TRANSCRIPTS_DIR
        transcriber.TRANSCRIPTS_DIR = self.dir
        self.audio = self.dir / "a.wav"
        self.audio.write_bytes(b"")

    def tearDown(self):
        self.tr.TRANSCRIPTS_DIR = self.saved_dir
        self.tmp.cleanup()

    def _write_transcript(self, model, options):
        (self.dir / "v1.json").write_text(json.dumps({
            "video_id": "v1", "filename": "v1.mp4", "duration_seconds": 1.0, "whisper_model": model,
            "whisper_options": options, "language": "en", "full_text": "", "segments": []}))

    def test_whisper_base_transcript_is_not_reused(self):
        self._write_transcript("base", None)
        fake = FakeWhisper()
        with self.assertRaises(StaleOutputError):
            self.tr.transcribe_video("v1", "v1.mp4", self.audio, 1.0, model=fake)
        self.assertEqual(fake.calls, 0)

    def test_matching_transcript_is_reused(self):
        self._write_transcript(settings.WHISPER_MODEL, self.tr.transcription_options())
        fake = FakeWhisper()
        result = self.tr.transcribe_video("v1", "v1.mp4", self.audio, 1.0, model=fake)
        self.assertEqual((fake.calls, result.whisper_model), (0, settings.WHISPER_MODEL))

    def test_force_retranscribes_and_records_options(self):
        self._write_transcript("base", None)
        fake = FakeWhisper()
        self.tr.transcribe_video("v1", "v1.mp4", self.audio, 1.0, model=fake, force=True)
        saved = json.loads((self.dir / "v1.json").read_text())
        self.assertEqual((fake.calls, saved["whisper_model"], saved["whisper_options"]),
                         (1, settings.WHISPER_MODEL, self.tr.transcription_options()))

    def test_require_not_older_than(self):
        src, out = self.dir / "src.json", self.dir / "out.json"
        out.write_text("{}")
        src.write_text("{}")
        old = time.time() - 100
        os.utime(out, (old, old))
        with self.assertRaises(StaleOutputError):
            bs.require_not_older_than(out, [src], "x")
        os.utime(src, (old - 10, old - 10))
        bs.require_not_older_than(out, [src], "x")
        with self.assertRaises(StaleOutputError):
            bs.require_not_older_than(out, [self.dir / "missing.json"], "x")

    def _age(self, path, seconds):
        t = time.time() - seconds
        os.utime(path, (t, t))

    def test_cleaner_chunker_embedder_refuse_outputs_older_than_inputs(self):
        from ingestion import chunker, cleaner, embedder
        saved = (cleaner.TRANSCRIPTS_DIR, chunker.CHUNKS_DIR, embedder.EMBEDDINGS_DIR)
        try:
            cleaner.TRANSCRIPTS_DIR = self.dir
            chunker.CHUNKS_DIR = self.dir / "chunks"
            embedder.EMBEDDINGS_DIR = self.dir / "emb"
            chunker.CHUNKS_DIR.mkdir()
            embedder.EMBEDDINGS_DIR.mkdir()
            transcript = self.dir / "v1.json"
            cleaned = self.dir / "v1_cleaned.json"
            chunks = chunker.CHUNKS_DIR / "v1_chunks.json"
            embeddings = embedder.EMBEDDINGS_DIR / "v1_embeddings.json"
            for p in (transcript, cleaned, chunks, embeddings):
                p.write_text("{}")
            self._age(cleaned, 300)                        # cleaned older than its transcript
            self._age(chunks, 400)                         # chunks older than cleaned
            self._age(embeddings, 500)                     # embeddings older than chunks
            tr = SimpleNamespace(video_id="v1", transcript_path=transcript)
            with self.assertRaises(StaleOutputError):
                cleaner.clean_all_transcripts([tr])
            with self.assertRaises(StaleOutputError):
                chunker.chunk_transcript(SimpleNamespace(video_id="v1", cleaned_path=cleaned))
            with self.assertRaises(StaleOutputError):
                embedder.embed_video_chunks("v1", [], model=object())
        finally:
            cleaner.TRANSCRIPTS_DIR, chunker.CHUNKS_DIR, embedder.EMBEDDINGS_DIR = saved


class TestProductionSettingsUnchanged(unittest.TestCase):

    def test_runtime_still_matches_stage26(self):
        from config.consistency import check_production_config
        self.assertEqual(check_production_config(), [])
        self.assertEqual(Path(settings.CHROMA_DB_PATH), settings.STAGE26_VECTOR_DB_DIR.resolve())
        self.assertEqual(settings.ACTIVE_COLLECTION, "lv2g_translate")


if __name__ == "__main__":
    unittest.main()
