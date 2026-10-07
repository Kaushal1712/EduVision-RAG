"""
Regression tests for the selected-video scope and Go-to-Timestamp (app/ui.py).

The real app (streamlit.testing AppTest), the real pipeline.ask/search, retriever.retrieve and
generator.generate run against a small throwaway ChromaDB index in a temp directory. Only the query
embedding (BGE-M3) and the OpenAI client are replaced by offline fakes, so there is no model
download, no API call and no access to the production index.

Run:  venv/bin/python -m unittest tests.test_video_scope -v
"""

import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chromadb  # noqa: E402
import streamlit as st  # noqa: E402
from chromadb.config import Settings  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import pipeline  # noqa: E402
from config import settings  # noqa: E402
from generation import generator  # noqa: E402
from ingestion import indexer  # noqa: E402
from retrieval import retriever  # noqa: E402

T1_ID, T1_FILE = "01_installing_vs_code", "01_Installing VS Code.mp4"
T2_ID, T2_FILE = "02_your_first_html_website", "02_Your First HTML Website.mp4"
T1_LABEL = "Tutorial #1 — Installing VS Code"
T2_LABEL = "Tutorial #2 — Your First HTML Website"

# Query vector and chunk vectors: every chunk is well above the 0.44 threshold, and Tutorial #1
# chunks rank above Tutorial #2 chunks, as in the reported bug ("what is CSS" → Tutorial #1).
QUERY_VEC = [1.0, 0.0, 0.0]
CHUNKS = [  # (video_id, video_filename, start_time_s, embedding)
    (T1_ID, T1_FILE, 10.0, [1.0, 0.05, 0.0]),
    (T1_ID, T1_FILE, 70.0, [1.0, 0.10, 0.0]),
    (T1_ID, T1_FILE, 130.0, [1.0, 0.15, 0.0]),
    (T2_ID, T2_FILE, 30.0, [1.0, 0.50, 0.0]),
    (T2_ID, T2_FILE, 90.0, [1.0, 0.55, 0.0]),
    (T2_ID, T2_FILE, 150.0, [1.0, 0.60, 0.0]),
]

_INDEX_DIR: Path
_CLIENT = None


def setUpModule():
    global _INDEX_DIR, _CLIENT
    _INDEX_DIR = Path(tempfile.mkdtemp(prefix="eduvision_scope_idx_"))
    _CLIENT = chromadb.PersistentClient(path=str(_INDEX_DIR), settings=Settings(anonymized_telemetry=False))
    col = _CLIENT.create_collection(name=pipeline.ACTIVE_COLLECTION, metadata={"hnsw:space": "cosine"})
    per_video: dict[str, int] = {}
    for vid, fn, start, emb in CHUNKS:
        n = per_video.get(vid, 0)
        per_video[vid] = n + 1
        col.add(
            ids=[f"{vid}_chunk_{n:04d}"],
            embeddings=[emb],
            documents=[f"CSS is explained in {fn} at {int(start)} seconds."],
            metadatas=[{
                "video_id": vid, "video_filename": fn, "language": "en",
                "start_time": start, "end_time": start + 20.0, "chunk_index": n,
                "source_segment_ids": "[]", "text_raw": "", "translated": "False",
            }],
        )


def tearDownModule():
    shutil.rmtree(_INDEX_DIR, ignore_errors=True)


class FakeLLM:
    """OpenAI-client stand-in that cites every video named anywhere in its prompt.

    This is the worst case for scope leaks: it reuses citations from Prior context and history
    (allowed by system prompt rule 7), so any other-video text reaching the model shows up in
    the answer.
    """

    def __init__(self):
        self.generation_calls: list[list[dict]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, messages, max_tokens, temperature):
        if messages[-1]["content"].rstrip().endswith("Standalone query:"):   # follow-up rewrite
            content = "why is CSS useful"
        else:
            self.generation_calls.append(messages)
            text = "\n".join(m["content"] for m in messages)
            names = list(dict.fromkeys(re.findall(r'Video: "([^"]+)"', text)))
            content = "CSS styles web pages " + " ".join(f'[Video: "{n}" @ 00:10]' for n in names) + "."
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


class _ScopeTestBase(unittest.TestCase):

    def setUp(self):
        st.cache_resource.clear()
        self.videos_dir = Path(tempfile.mkdtemp(prefix="eduvision_videos_"))
        self.addCleanup(shutil.rmtree, self.videos_dir, True)
        self.llm = FakeLLM()
        self.ask_calls: list[dict] = []
        self.retrieve_calls: list[dict] = []
        self.search_calls: list[dict] = []

        real_ask, real_search, real_retrieve = pipeline.ask, pipeline.search, pipeline.retrieve

        def spy_ask(query, **kw):
            self.ask_calls.append({"query": query, **kw})
            return real_ask(query, **kw)

        def spy_search(query, **kw):
            self.search_calls.append({"query": query, **kw})
            return real_search(query, **kw)

        def spy_retrieve(**kw):
            results = real_retrieve(**kw)
            self.retrieve_calls.append({**kw, "video_ids": {r.video_id for r in results}})
            return results

        for target, attr, value in (
            (indexer, "get_chroma_client", lambda: _CLIENT),
            (retriever, "encode_query", lambda query: QUERY_VEC),
            (generator, "_get_client", lambda: self.llm),
            (pipeline, "_get_openai_client_if_available", lambda: self.llm),
            (pipeline, "ask", spy_ask),
            (pipeline, "search", spy_search),
            (pipeline, "retrieve", spy_retrieve),
            (settings, "VIDEOS_DIR", self.videos_dir),
        ):
            patcher = mock.patch.object(target, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.at = AppTest.from_file(str(ROOT / "app" / "ui.py"), default_timeout=120)
        self.at.run()
        self.assertFalse(self.at.exception, self.at.exception)

    # ── UI actions ───────────────────────────────────────────────────────────

    def select_video(self, label: str):
        self.at.sidebar.radio[0].set_value(label).run()
        self.assertFalse(self.at.exception, self.at.exception)

    def ask(self, question: str):
        self.at.text_input(key="chat_composer").input(question)
        self.at.button(key="chat_send").click().run()
        self.assertFalse(self.at.exception, self.at.exception)
        return self.at.session_state.chat_history[-1]

    def search(self, query: str):
        self.at.text_input(key="search_input").input(query)
        next(b for b in self.at.button if b.label == "Search").click().run()
        self.assertFalse(self.at.exception, self.at.exception)

    def go_to_buttons(self):
        # Existing Go-to-Timestamp buttons: key "play_<group>_<chunk_id>", label "\u25b6 MM:SS".
        return [b for b in self.at.button if (b.key or "").startswith("play_")]


class TestRetrieverVideoFilter(unittest.TestCase):
    """retriever.retrieve against a real ChromaDB collection (tests 1 and 2)."""

    def setUp(self):
        for target, attr, value in ((indexer, "get_chroma_client", lambda: _CLIENT),
                                    (retriever, "encode_query", lambda query: QUERY_VEC)):
            patcher = mock.patch.object(target, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_all_videos_searches_the_whole_index(self):
        results = retriever.retrieve("what is CSS", top_k=10, video_id_filter=None,
                                     collection_name=pipeline.ACTIVE_COLLECTION)
        self.assertEqual(len(results), len(CHUNKS))
        self.assertEqual({r.video_id for r in results}, {T1_ID, T2_ID})

    def test_selected_video_returns_only_that_video(self):
        # Tutorial #1 chunks are the closest overall; none may appear under a Tutorial #2 filter,
        # even though top_k (10) is larger than Tutorial #2's chunk count (3).
        results = retriever.retrieve("what is CSS", top_k=10, video_id_filter=T2_ID,
                                     collection_name=pipeline.ACTIVE_COLLECTION)
        self.assertEqual(len(results), 3)
        self.assertEqual({r.video_id for r in results}, {T2_ID})
        self.assertEqual({r.video_filename for r in results}, {T2_FILE})


class TestChatVideoScope(_ScopeTestBase):

    def test_catalogue_labels(self):
        self.assertEqual(self.at.sidebar.radio[0].options, ["All Videos", T1_LABEL, T2_LABEL])

    def test_all_videos_answer_may_use_every_video(self):
        turn = self.ask("what is CSS")
        self.assertIsNone(self.ask_calls[-1]["video_id_filter"])
        self.assertEqual(self.retrieve_calls[-1]["video_ids"], {T1_ID, T2_ID})
        self.assertEqual({s.video_id for s in turn["result"].sources_used}, {T1_ID, T2_ID})

    def test_selected_video_answer_cannot_cite_another_video(self):
        # Reported sequence: an All Videos answer cites Tutorial #1, then Tutorial #2 is selected.
        first = self.ask("what is CSS")
        self.assertIn(T1_FILE, first["content"])

        self.select_video(T2_LABEL)
        turn = self.ask("what is CSS")

        self.assertEqual(self.ask_calls[-1]["video_id_filter"], T2_ID)
        self.assertEqual(self.retrieve_calls[-1]["video_ids"], {T2_ID})
        self.assertEqual({s.video_id for s in turn["result"].sources_used}, {T2_ID})
        prompt_text = "\n".join(m["content"] for m in self.llm.generation_calls[-1])
        self.assertNotIn(T1_FILE, prompt_text, "another video's text reached the generator")
        self.assertNotIn(T1_FILE, turn["content"])
        self.assertIn(T2_FILE, turn["content"])

    def test_selected_video_follow_up_keeps_scope_and_context(self):
        self.select_video(T2_LABEL)
        self.ask("what is CSS")
        turn = self.ask("why is it useful?")

        call = self.ask_calls[-1]
        self.assertEqual(call["video_id_filter"], T2_ID)
        self.assertEqual([h["content"] for h in call["chat_history"] if h["role"] == "user"], ["what is CSS"])
        self.assertEqual(self.retrieve_calls[-1]["video_id_filter"], T2_ID)
        self.assertEqual(self.retrieve_calls[-1]["video_ids"], {T2_ID})
        self.assertEqual({s.video_id for s in turn["result"].sources_used}, {T2_ID})
        prompt_text = "\n".join(m["content"] for m in self.llm.generation_calls[-1])
        self.assertIn("Prior context from this conversation:", prompt_text)
        self.assertNotIn(T1_FILE, prompt_text)

    def test_all_videos_keeps_history_from_single_video_turns(self):
        self.select_video(T2_LABEL)
        self.ask("what is CSS")
        self.select_video("All Videos")
        self.ask("why is it useful?")
        call = self.ask_calls[-1]
        self.assertIsNone(call["video_id_filter"])
        self.assertEqual([h["content"] for h in call["chat_history"] if h["role"] == "user"], ["what is CSS"])


class TestChatComposer(_ScopeTestBase):
    """Each question is typed and submitted once (Enter or the send button).

    The input is emptied by the browser after each submit (st.form clear_on_submit), which AppTest
    does not emulate; that part is checked by asserting the form setup. What AppTest does check is
    the server side of the bug: the next question's text is never overwritten before it is read.
    """

    def composer_form(self):
        def walk(node):
            proto = getattr(node, "proto", None)
            if proto is not None and getattr(proto, "WhichOneof", None) and \
                    "form" in proto.DESCRIPTOR.fields_by_name and proto.WhichOneof("type") == "form":
                return proto.form
            children = getattr(node, "children", None) or {}
            for child in (children.values() if isinstance(children, dict) else children):
                found = walk(child)
                if found is not None:
                    return found
            return None
        return walk(self.at._tree)

    def test_composer_is_a_clear_on_submit_form(self):
        form = self.composer_form()
        self.assertIsNotNone(form)
        self.assertEqual(form.form_id, "chat_composer_form")
        self.assertTrue(form.clear_on_submit)
        self.assertTrue(form.enter_to_submit)
        self.assertEqual(self.at.text_input(key="chat_composer").form_id, "chat_composer_form")
        self.assertEqual(self.at.button(key="chat_send").form_id, "chat_composer_form")

    def test_consecutive_questions_each_submitted_once(self):
        questions = ["what is html?", "what is css?", "why is it useful?"]
        for i, q in enumerate(questions, start=1):
            turn = self.ask(q)
            self.assertEqual([c["query"] for c in self.ask_calls], questions[:i], f"question {i} not processed")
            self.assertEqual(turn["role"], "assistant")
            self.assertTrue(turn["result"].answer)
            self.assertNotIn("chat_composer_clear", self.at.session_state)   # no deferred clear
        history = self.at.session_state.chat_history
        self.assertEqual([h["content"] for h in history if h["role"] == "user"], questions)
        self.assertEqual(len(history), 6)
        # The follow-up still got the earlier turns as context.
        self.assertEqual([h["content"] for h in self.ask_calls[-1]["chat_history"] if h["role"] == "user"],
                         questions[:2])

    def test_empty_submit_does_nothing(self):
        self.at.button(key="chat_send").click().run()
        self.assertFalse(self.at.exception, self.at.exception)
        self.assertEqual(self.ask_calls, [])
        self.assertEqual(self.at.session_state.chat_history, [])

    def test_example_button_then_typed_question(self):
        next(b for b in self.at.button if b.key == "chat_ex_0").click().run()
        self.at.run()   # the example sets chat_prefill and calls st.rerun()
        self.assertFalse(self.at.exception, self.at.exception)
        self.assertEqual([c["query"] for c in self.ask_calls], ["What is HTML and what is it used for?"])
        self.assertEqual(self.at.text_input(key="chat_composer").value, "")   # example text not put in the input
        self.ask("what is css?")
        self.assertEqual([c["query"] for c in self.ask_calls],
                         ["What is HTML and what is it used for?", "what is css?"])


class TestSearchVideoScope(_ScopeTestBase):

    def test_search_respects_selected_video(self):
        self.select_video(T2_LABEL)
        self.search("what is CSS")
        self.assertEqual(self.search_calls[-1]["video_id_filter"], T2_ID)
        self.assertEqual({r.video_id for r in self.at.session_state.search_results_cache}, {T2_ID})

    def test_scope_change_does_not_show_stale_results(self):
        self.search("what is CSS")
        self.assertEqual({r.video_id for r in self.at.session_state.search_results_cache}, {T1_ID, T2_ID})

        self.select_video(T2_LABEL)
        self.assertEqual(self.search_calls[-1]["video_id_filter"], T2_ID)
        self.assertEqual({r.video_id for r in self.at.session_state.search_results_cache}, {T2_ID})
        cards = "\n".join(m.value for m in self.at.markdown)
        self.assertIn("Tutorial #2", cards)
        self.assertNotIn("Tutorial #1</div>", cards)


class TestGoToTimestamp(_ScopeTestBase):
    """The existing Go-to-Timestamp behaviour, unchanged by the video-scope fix."""

    def test_button_rendered_when_local_video_exists(self):
        (self.videos_dir / T2_FILE).write_bytes(b"\x00" * 16)
        self.select_video(T2_LABEL)
        self.search("what is CSS")
        self.assertEqual(sorted(b.label for b in self.go_to_buttons()),
                         ["\u25b6 00:30", "\u25b6 01:30", "\u25b6 02:30"])

    def test_button_seeks_correct_local_video_and_time(self):
        (self.videos_dir / T1_FILE).write_bytes(b"\x00" * 16)
        (self.videos_dir / T2_FILE).write_bytes(b"\x00" * 16)
        self.select_video(T2_LABEL)
        self.search("what is CSS")

        next(b for b in self.go_to_buttons() if b.label == "\u25b6 01:30").click().run()
        self.assertFalse(self.at.exception, self.at.exception)

        player = self.at.session_state.video_player
        self.assertEqual(player["path"], str(self.videos_dir / T2_FILE))
        self.assertEqual(player["start_time"], 90)
        videos = self.at.get("video")
        self.assertEqual(len(videos), 1)
        self.assertEqual(videos[0].proto.start_time, 90)
        # The search results stay visible next to the player.
        self.assertEqual(len(self.go_to_buttons()), 3)

    def test_chat_source_button_seeks_cited_chunk(self):
        (self.videos_dir / T2_FILE).write_bytes(b"\x00" * 16)
        self.select_video(T2_LABEL)
        self.ask("what is CSS")
        next(b for b in self.go_to_buttons() if b.label == "\u25b6 02:30").click().run()
        self.assertFalse(self.at.exception, self.at.exception)
        self.assertEqual(self.at.session_state.video_player["path"], str(self.videos_dir / T2_FILE))
        self.assertEqual(self.at.session_state.video_player["start_time"], 150)

    def test_without_local_videos_app_still_works(self):
        # Deployed app: no videos/ directory. Existing behaviour: no Go-to buttons, no player,
        # no upload workflow; search and answers still work.
        shutil.rmtree(self.videos_dir)
        self.search("what is CSS")
        self.ask("what is CSS")
        self.assertEqual(self.go_to_buttons(), [])
        self.assertIsNone(self.at.session_state.video_player)
        self.assertEqual(self.at.get("video"), [])
        self.assertEqual(self.at.get("file_uploader"), [], "no upload workflow")
        self.assertTrue(self.at.session_state.search_results_cache)
        self.assertTrue(self.at.session_state.chat_history[-1]["result"].sources_used)

    def test_stale_player_with_missing_file_shows_notice(self):
        self.at.session_state.video_player = {"path": str(self.videos_dir / "gone.mp4"),
                                              "start_time": 30, "label": "Tutorial #2 \u2014 00:30"}
        self.at.run()
        self.assertFalse(self.at.exception, self.at.exception)
        self.assertEqual(self.at.get("video"), [])
        self.assertTrue(any("not available in this environment" in i.value for i in self.at.info))


if __name__ == "__main__":
    unittest.main()
