"""
Offline regression tests for UI text safety (app/text_safety.py, app/ui.py, Streamlit config).
No Streamlit server, ChromaDB or API access.

Run:  venv/bin/python -m unittest discover -s tests -v
"""

import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.text_safety import GENERIC_ERROR, escape, user_facing_error  # noqa: E402

UI_SOURCE = (ROOT / "app" / "ui.py").read_text()


class TestEscape(unittest.TestCase):

    def test_html_is_neutralised(self):
        self.assertEqual(escape('<script>alert("x")</script>'),
                         "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;")
        self.assertEqual(escape("Tom & Jerry's <title>"), "Tom &amp; Jerry&#x27;s &lt;title&gt;")
        self.assertEqual(escape("[02:50 → 03:05]"), "[02:50 → 03:05]")   # ordinary text unchanged
        self.assertEqual(escape(0.44), "0.44")


class TestUserFacingError(unittest.TestCase):

    def test_known_errors_map_to_fixed_messages_without_details(self):
        cases = {
            "OPENAI_API_KEY is not set. Add your key to the .env file:\n  OPENAI_API_KEY=sk-...": "no OpenAI API key",
            "Invalid API key. Check OPENAI_API_KEY in .env. (Incorrect API key provided: sk-proj-****abcd)": "not configured",
            "OpenAI rate limit exceeded. Please wait and retry. (insufficient_quota: org-123)": "busy",
            "Could not connect to OpenAI. Check your internet connection. (proxy 10.0.0.1)": "Could not reach",
            "OpenAI API error (HTTP 500): upstream": "returned an error",
            "Retrieval error: Collection [lv2g_translate] at /Users/x/data/vector_db does not exist": "could not be searched",
        }
        for raw, expected in cases.items():
            with self.assertLogs("app.text_safety", level="ERROR"):
                msg = user_facing_error(raw)
            self.assertIn(expected, msg)
            for secret_bit in ("sk-", "/Users/", "org-123", "10.0.0.1", "lv2g_translate", ".env"):
                self.assertNotIn(secret_bit, msg)

    def test_unknown_errors_get_generic_message(self):
        with self.assertLogs("app.text_safety", level="ERROR"):
            self.assertEqual(user_facing_error("Generation error: Traceback ... /opt/app/pipeline.py"), GENERIC_ERROR)


class TestUiSource(unittest.TestCase):

    def test_dynamic_values_in_raw_html_are_escaped(self):
        # The unsafe_allow_html snippets that interpolated these values before the fix.
        # (vid_label is still used unescaped in a markdown tooltip and in the stored player label,
        # which is escaped when the player banner renders it.)
        for raw in ('\\u23f1 {ts}</span>', '\\U0001f4f9 {vid_label}</div>', '"{text_prev}"</div>',
                    '<strong>{vp["label"]}</strong>', '{key_icon} {status.model_name}</div>'):
            self.assertNotIn(raw, UI_SOURCE, f"unescaped {raw} in app/ui.py")
        for escaped in ("{escape(ts)}", "{escape(vid_label)}", "{escape(text_prev)}",
                        '{escape(vp["label"])}', "{escape(status.model_name)}"):
            self.assertIn(escaped, UI_SOURCE)

    def test_raw_errors_not_shown_or_stored(self):
        self.assertNotIn("System error: {result.error}", UI_SOURCE)
        self.assertNotIn("result.validation_error or result.error", UI_SOURCE)
        self.assertIn("user_facing_error(result.error)", UI_SOURCE)

    def test_no_stale_rebuild_instruction(self):
        self.assertNotIn("python ingestion/indexer_v2.py", UI_SOURCE)


class TestStreamlitConfig(unittest.TestCase):

    def test_cors_protection_not_disabled(self):
        cfg = tomllib.loads((ROOT / "app" / ".streamlit" / "config.toml").read_text())
        self.assertNotEqual(cfg.get("server", {}).get("enableCORS", True), False)
        self.assertNotEqual(cfg.get("server", {}).get("enableXsrfProtection", True), False)


if __name__ == "__main__":
    unittest.main()
