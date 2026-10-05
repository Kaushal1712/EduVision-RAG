"""
Offline tests for config/consistency.py — the runtime configuration must match the locked Stage 2.6
selection, and any deviation must be reported by name. No ChromaDB, model or API access.

Run:  venv/bin/python -m unittest discover -s tests -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import consistency  # noqa: E402


class TestProductionConfig(unittest.TestCase):

    def test_current_runtime_matches_locked_stage26(self):
        self.assertEqual(consistency.check_production_config(), [])

    def test_resolved_index_path_and_collection(self):
        runtime = consistency.collect_runtime()
        root = consistency.settings.ROOT_DIR.resolve()
        self.assertEqual(Path(runtime["settings.CHROMA_DB_PATH"]), root / "data" / "vector_db_lv2g_translate")
        self.assertEqual(runtime["settings.ACTIVE_COLLECTION"], "lv2g_translate")

    def test_each_mismatch_is_named(self):
        runtime = consistency.collect_runtime()
        runtime["settings.SIMILARITY_THRESHOLD"] = 0.5
        runtime["generation.generator.SYSTEM_PROMPT.sha256"] = "0" * 64
        problems = consistency.compare(runtime)
        self.assertEqual(len(problems), 2)
        self.assertTrue(problems[0].startswith("settings.SIMILARITY_THRESHOLD: expected 0.44, got 0.5"))
        self.assertTrue(problems[1].startswith("generation.generator.SYSTEM_PROMPT.sha256"))

    def test_assert_raises_with_details(self):
        original = consistency.collect_runtime
        try:
            consistency.collect_runtime = lambda: {**original(), "settings.ACTIVE_COLLECTION": "eduvision_chunks_v2"}
            with self.assertRaises(consistency.ConfigMismatchError) as ctx:
                consistency.assert_production_config()
            self.assertIn("settings.ACTIVE_COLLECTION: expected 'lv2g_translate'", str(ctx.exception))
        finally:
            consistency.collect_runtime = original

    def test_transcriber_options_are_stage26_greedy_translate(self):
        from ingestion import transcriber
        self.assertEqual(transcriber.transcription_options(),
                         {"task": "translate", "verbose": False, "fp16": False, "word_timestamps": False,
                          "temperature": 0.0, "condition_on_previous_text": False})

    def test_fallback_and_conditioning_map_to_whisper_defaults(self):
        from ingestion import transcriber
        saved = (transcriber.WHISPER_TASK, transcriber.WHISPER_TEMPERATURE_FALLBACK,
                 transcriber.WHISPER_CONDITION_ON_PREVIOUS_TEXT)
        try:
            transcriber.WHISPER_TASK = "transcribe"
            transcriber.WHISPER_TEMPERATURE_FALLBACK = True
            transcriber.WHISPER_CONDITION_ON_PREVIOUS_TEXT = True
            opts = transcriber.transcription_options()
            self.assertEqual(opts["temperature"], (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))   # Whisper's own default
            self.assertEqual((opts["task"], opts["condition_on_previous_text"]), ("transcribe", True))
        finally:
            (transcriber.WHISPER_TASK, transcriber.WHISPER_TEMPERATURE_FALLBACK,
             transcriber.WHISPER_CONDITION_ON_PREVIOUS_TEXT) = saved
        self.assertEqual(transcriber.transcription_options()["temperature"], 0.0)

    def test_missing_index_and_locked_selection_drift(self):
        self.assertEqual(consistency.check_index_present("/nonexistent/index"),
                         ["index not found: /nonexistent/index/chroma.sqlite3"])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "locked.json"
            p.write_text(json.dumps({"similarity_threshold": 0.5, "system_prompt_sha256": "x"}))
            self.assertEqual(len(consistency.check_locked_selection(p)), 2)
            self.assertEqual(consistency.check_locked_selection(Path(d) / "absent.json"), [])


if __name__ == "__main__":
    unittest.main()
