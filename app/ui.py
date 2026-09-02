"""
app/ui.py
──────────
Stage 11: EduVision RAG — Streamlit UI.

Entry point:
    cd <project_root>
    streamlit run app/ui.py --server.port 8501

Architecture:
    This module only imports from pipeline.py (Stage 10).
    It does NOT import directly from retriever, generator, indexer,
    embedder, or any other pipeline stage module.
    All RAG logic is accessed through pipeline.ask(), pipeline.search(),
    and pipeline.health_check().
"""

import sys
import os
import time
from pathlib import Path

from config.settings import (
    VIDEOS_DIR,
    SIMILARITY_THRESHOLD,
    RETRIEVAL_TOP_K,
    MAX_LLM_EVIDENCE,
)

# ── Ensure project root is on sys.path so pipeline.py is importable ──────────
# When running from app/, the project root is one level up.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

# ── Page config — must be the FIRST Streamlit call ───────────────────────────
st.set_page_config(
    page_title="EduVision RAG",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "EduVision RAG — AI Teaching Assistant powered by BGE-M3 + GPT-4o-mini",
    },
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ────────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide default Streamlit chrome ─────────────────────────────────────────── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── Sidebar ───────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12152A 0%, #0E1117 100%);
    border-right: 1px solid #2A2D45;
}

/* ── Source cards ──────────────────────────────────────────────────────────── */
.source-card {
    background: #1A1D2E;
    border: 1px solid #2A2D45;
    border-left: 4px solid #6C63FF;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
}
.source-card:hover { border-left-color: #9C94FF; }

.source-card.below-threshold {
    border-left-color: #444;
    opacity: 0.7;
}

.source-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}

.timestamp-badge {
    background: #6C63FF22;
    border: 1px solid #6C63FF55;
    color: #9C94FF;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.8em;
    font-weight: 600;
    font-family: 'Courier New', monospace;
    white-space: nowrap;
}

.similarity-badge {
    background: #1E3A2F;
    border: 1px solid #2D5A42;
    color: #4CAF50;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.78em;
    font-weight: 600;
}

.similarity-badge.low {
    background: #2A1E1E;
    border-color: #5A2D2D;
    color: #EF9A9A;
}

.video-label {
    font-size: 0.78em;
    color: #7B7FA0;
    margin-bottom: 4px;
    font-weight: 500;
}

.chunk-text {
    font-size: 0.88em;
    color: #C5C7D4;
    line-height: 1.5;
    font-style: italic;
    border-top: 1px solid #2A2D45;
    padding-top: 8px;
    margin-top: 4px;
}

/* ── Answer box ────────────────────────────────────────────────────────────── */
.answer-box {
    background: linear-gradient(135deg, #1A1D2E 0%, #12152A 100%);
    border: 1px solid #2A2D45;
    border-top: 3px solid #6C63FF;
    border-radius: 8px;
    padding: 18px 20px;
    margin: 8px 0 16px 0;
}

/* ── Status bar ────────────────────────────────────────────────────────────── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1A1D2E;
    border: 1px solid #2A2D45;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.8em;
    color: #9C94FF;
    margin-bottom: 4px;
}

/* ── Not-found state ───────────────────────────────────────────────────────── */
.not-found-box {
    background: #1E1A2E;
    border: 1px solid #3A2D5A;
    border-left: 4px solid #9575CD;
    border-radius: 8px;
    padding: 14px 18px;
    color: #B39DDB;
    margin: 8px 0;
}

/* ── Metric chips ──────────────────────────────────────────────────────────── */
.metric-chip {
    display: inline-block;
    background: #1A1D2E;
    border: 1px solid #2A2D45;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.78em;
    color: #7B7FA0;
    margin: 2px 4px 2px 0;
}

.metric-chip span {
    color: #E8EAF6;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ── Pipeline import + cached resource loading ─────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_pipeline():
    """
    Load the pipeline once per server process.

    st.cache_resource caches across all user sessions and Streamlit
    reruns — equivalent to a module-level singleton. BGE-M3 (~570MB)
    is expensive to load; this ensures it loads exactly once.

    Returns (ask_fn, search_fn, health_check_fn, status)
    """
    from pipeline import ask, search, health_check
    status = health_check()
    return ask, search, health_check, status


# ── Helpers ───────────────────────────────────────────────────────────────────

def _short_video_name(filename: str) -> str:
    """Convert full video filename to a short display label."""
    if "Tutorial #1" in filename or "tutorial_1" in filename.lower() or "01_" in filename:
        return "Tutorial #1"
    if "Tutorial #2" in filename or "tutorial_2" in filename.lower() or "02_" in filename:
        return "Tutorial #2"
    return filename.split("|")[0].strip()[:40]


def _sim_color_class(similarity: float) -> str:
    return "low" if similarity < 0.55 else ""


def _video_path(video_filename: str):
    """Return the absolute Path to the local video file, or None if missing."""
    p = VIDEOS_DIR / video_filename
    return p if p.exists() else None


def _render_source_card(
    result,
    idx: int,
    show_below_threshold: bool = True,
    group_id: str = "",
):
    """Render a single source evidence card with a Go-to-Timestamp play button."""
    if result.below_threshold and not show_below_threshold:
        return

    card_class = "source-card below-threshold" if result.below_threshold else "source-card"
    sim_class  = _sim_color_class(result.similarity)
    vid_label  = _short_video_name(result.video_filename)
    ts         = f"[{result.start_time_fmt} \u2192 {result.end_time_fmt}]"
    sim_pct    = f"{result.similarity * 100:.1f}%"

    # Always display text_en (English) in the source card
    display_text = result.text_en if hasattr(result, 'text_en') else result.text
    text_prev  = display_text[:180] + ("..." if len(display_text) > 180 else "")

    st.markdown(f"""
<div class="{card_class}">
    <div class="source-header">
        <span class="timestamp-badge">\u23f1 {ts}</span>
        <span class="similarity-badge {sim_class}">sim {sim_pct}</span>
    </div>
    <div class="video-label">\U0001f4f9 {vid_label}</div>
    <div class="chunk-text">"{text_prev}"</div>
</div>
""", unsafe_allow_html=True)

    # ── Go to Timestamp button ────────────────────────────────────────────────
    # text_raw is kept in ChromaDB metadata for provenance only — never displayed.
    vpath = _video_path(result.video_filename)
    if vpath:
        btn_key = f"play_{group_id}_{result.chunk_id}"
        col_btn, _ = st.columns([1, 4])
        with col_btn:
            if st.button(
                f"\u25b6 {result.start_time_fmt}",
                key=btn_key,
                help=f"Jump to {result.start_time_fmt} in {vid_label}",
                use_container_width=True,
            ):
                st.session_state.video_player = {
                    "path": str(vpath),
                    "start_time": int(result.start_time),
                    "label": f"{vid_label} \u2014 {result.start_time_fmt}",
                }
                st.rerun()


def _render_diagnostics(result):
    """Render small metric chips for latency and token info."""
    chips = []
    if result.retrieval_stats:
        chips.append(f'<span class="metric-chip">retrieved <span>{result.retrieval_stats.total_results}</span></span>')
        chips.append(f'<span class="metric-chip">above threshold <span>{result.retrieval_stats.above_threshold}</span></span>')
        chips.append(f'<span class="metric-chip">top sim <span>{result.retrieval_stats.top_similarity:.2f}</span></span>')
    chips.append(f'<span class="metric-chip">⏱ <span>{result.total_latency_s:.1f}s</span></span>')
    if result.total_tokens:
        chips.append(f'<span class="metric-chip">🪙 <span>{result.total_tokens} tokens</span></span>')
    st.markdown("".join(chips), unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────

def _init_session():
    if "chat_history" not in st.session_state:
        # Each entry: {"role": "user"|"assistant", "content": str,
        #              "result": PipelineResult|None}
        st.session_state.chat_history = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "video_filter" not in st.session_state:
        st.session_state.video_filter = "All Videos"
    if "video_player" not in st.session_state:
        # Holds {"path": str, "start_time": int, "label": str} when a
        # timestamp has been clicked, or None when no video is playing.
        st.session_state.video_player = None
    if "search_input" not in st.session_state:
        # Populated by search example buttons so the text_input is pre-filled.
        st.session_state.search_input = ""
    if "search_auto_submit" not in st.session_state:
        # Set to True by search example buttons to trigger an immediate search.
        st.session_state.search_auto_submit = False
    if "chat_prefill" not in st.session_state:
        # Populated by chat example buttons to submit a pre-set query.
        st.session_state.chat_prefill = ""


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar(status):
    with st.sidebar:
        # Branding
        st.markdown("""
<div style="text-align:center; padding: 12px 0 20px 0;">
    <div style="font-size:2.2em;">🎓</div>
    <div style="font-size:1.3em; font-weight:700; color:#E8EAF6; letter-spacing:0.5px;">EduVision RAG</div>
    <div style="font-size:0.78em; color:#7B7FA0; margin-top:2px;">Video Teaching Assistant</div>
</div>
""", unsafe_allow_html=True)

        st.divider()

        # System status
        st.markdown("**System Status**")
        chroma_icon = "✅" if status.chroma_ok else "❌"
        key_icon    = "✅" if status.api_key_set else "⚠️"

        st.markdown(f"""
<div class="status-pill">{chroma_icon} {status.chroma_count} chunks indexed</div><br>
<div class="status-pill">{key_icon} {status.model_name}</div>
""", unsafe_allow_html=True)

        if not status.api_key_set:
            st.warning("⚠️ OPENAI_API_KEY not set. Search tab still works.", icon="🔑")

        if not status.chroma_ok:
            st.error("ChromaDB unavailable. Run Stage 7 (indexer.py) first.")

        st.divider()

        # Video filter
        st.markdown("**Filter by Video**")
        video_filter = st.radio(
            label="video_filter_radio",
            options=["All Videos", "Tutorial #1 — Installing VS Code", "Tutorial #2 — Your First HTML Website"],
            label_visibility="collapsed",
        )
        st.session_state.video_filter = video_filter

        st.divider()

        # About — values are derived from runtime state and settings, not hardcoded
        with st.expander("ℹ️ About this app", expanded=False):
            st.markdown(f"""
**EduVision RAG** answers questions about video course material using a
retrieval-augmented generation pipeline:

1. **BGE-M3** encodes your question as a 1024-dim vector
2. **ChromaDB** finds the most relevant transcript chunks
3. **GPT-4o-mini** generates a grounded answer from the evidence

All answers cite exact timestamps — click the timestamp to know exactly
where in the video the explanation appears.

---
- Chunks: **{status.chroma_count}** (v2 index — English normalised, noise filtered)
- Model: **BGE-M3** + **gpt-4o-mini**
- Threshold: **{SIMILARITY_THRESHOLD}** cosine similarity
- Retrieval pool: **{RETRIEVAL_TOP_K}** candidates → top **{MAX_LLM_EVIDENCE}** to LLM
""")

        # Clear chat
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.last_result = None
            st.rerun()


# ── Chat tab ──────────────────────────────────────────────────────────────────

# Queries shown as example buttons when the chat tab is empty.
_CHAT_EXAMPLES = [
    "What is HTML and what is it used for?",
    "How do I install VS Code?",
    "Where is the Live Preview extension discussed?",
    "How do websites work with HTML, CSS and JavaScript?",
]


def _render_chat_tab(ask_fn, video_filter: str):
    """Render the main Q&A chat interface."""

    # ── Example queries (only when conversation is empty) ─────────────────────
    if not st.session_state.chat_history:
        st.markdown(
            "<div style='color:#7B7FA0; font-size:0.85em; margin-bottom:8px;'>"
            "Try an example question:"
            "</div>",
            unsafe_allow_html=True,
        )
        ex_cols = st.columns(2)
        for i, ex in enumerate(_CHAT_EXAMPLES):
            with ex_cols[i % 2]:
                if st.button(ex, key=f"chat_ex_{i}", use_container_width=True):
                    st.session_state.chat_prefill = ex
                    st.rerun()

    # Display existing chat history
    for j, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"], avatar="\U0001f9d1\u200d\U0001f393" if msg["role"] == "user" else "\U0001f916"):
            st.markdown(msg["content"])

            # If this is an assistant message with a result, show sources
            if msg["role"] == "assistant" and msg.get("result"):
                result = msg["result"]

                # Diagnostics row
                _render_diagnostics(result)

                # Sources
                if result.sources_used:
                    with st.expander(f"\U0001f4cc {len(result.sources_used)} Source(s) Used", expanded=False):
                        for i, s in enumerate(result.sources_used):
                            _render_source_card(s, i + 1, group_id=f"h{j}")

                elif result.not_found and result.retrieval_results:
                    with st.expander("\U0001f50d Closest Matches Found (below threshold)", expanded=False):
                        st.markdown(
                            "<small style='color:#7B7FA0;'>These chunks were retrieved but scored "
                            "below the similarity threshold \u2014 the evidence was too weak to generate "
                            "a confident answer.</small>",
                            unsafe_allow_html=True,
                        )
                        for i, r in enumerate(result.retrieval_results[:3]):
                            _render_source_card(r, i + 1, show_below_threshold=True, group_id=f"h{j}b")

    # ── Chat input ────────────────────────────────────────────────────────────
    query = st.chat_input(
        placeholder="Ask about the course... e.g. 'How do I install VS Code?'",
    )

    # If an example button was clicked this rerun, use it as the query.
    # chat_prefill is set by the example buttons above and cleared here.
    if not query and st.session_state.get("chat_prefill"):
        query = st.session_state.chat_prefill
        st.session_state.chat_prefill = ""

    if query:
        # Determine video_id_filter from sidebar selection
        video_id_filter = None
        if "Tutorial #1" in video_filter:
            video_id_filter = "01_installing_vs_code_how_websites_work_sigma_web_development_course_tutorial_1"
        elif "Tutorial #2" in video_filter:
            video_id_filter = "02_your_first_html_website_sigma_web_development_course_tutorial_2"

        # Snapshot history BEFORE appending the current user turn —
        # these are the prior turns the generator uses for follow-up context.
        prior_history = list(st.session_state.chat_history)

        # Show user message immediately
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(query)
        st.session_state.chat_history.append({"role": "user", "content": query, "result": None})

        # Generate answer with spinner
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Retrieving evidence and generating answer..."):
                result = ask_fn(
                    query,
                    video_id_filter=video_id_filter,
                    chat_history=prior_history,
                )

            # ── Render answer ──────────────────────────────────────────────────
            if not result.query_valid:
                answer_text = f"⚠️ {result.validation_error}"
                st.warning(answer_text)

            elif result.error:
                answer_text = f"❌ System error: {result.error}"
                st.error(answer_text)

            elif result.not_found:
                answer_text = "I could not find this topic in the provided course material."
                st.markdown(f"""
<div class="not-found-box">
🔍 <strong>Not found in course material</strong><br>
<span style="font-size:0.9em;">{answer_text}</span>
</div>
""", unsafe_allow_html=True)

            else:
                answer_text = result.answer
                st.markdown(answer_text)

            # Diagnostics + sources
            _render_diagnostics(result)

            if result.sources_used:
                with st.expander(f"\U0001f4cc {len(result.sources_used)} Source(s) Used", expanded=True):
                    for i, s in enumerate(result.sources_used):
                        _render_source_card(s, i + 1, group_id="cur")

            elif result.not_found and result.retrieval_results:
                with st.expander("\U0001f50d Closest Matches Found (below threshold)", expanded=True):
                    st.markdown(
                        "<small style='color:#7B7FA0;'>Retrieved but below similarity threshold.</small>",
                        unsafe_allow_html=True,
                    )
                    for i, r in enumerate(result.retrieval_results[:3]):
                        _render_source_card(r, i + 1, show_below_threshold=True, group_id="curb")

        # Save to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer_text if result.query_valid and not result.error else (result.validation_error or result.error or ""),
            "result": result,
        })
        st.session_state.last_result = result


# ── Search tab ────────────────────────────────────────────────────────────────

def _render_search_tab(search_fn, video_filter: str):
    """Render the evidence-search tab (retrieval only, no LLM)."""

    st.markdown("""
<div style="color:#7B7FA0; font-size:0.88em; margin-bottom:16px;">
Search for relevant transcript evidence without generating an LLM answer.
Useful for exploring what the videos cover and verifying timestamps.
</div>
""", unsafe_allow_html=True)

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        # key="search_input" ties this widget to st.session_state.search_input,
        # so example buttons can pre-populate it by writing to that key.
        search_query = st.text_input(
            label="search_query",
            placeholder="Search transcript evidence... e.g. 'HTML structure'",
            label_visibility="collapsed",
            key="search_input",
        )
    with col_btn:
        search_clicked = st.button("Search", use_container_width=True, type="primary")

    # search_auto_submit is set by example buttons so they trigger an immediate search.
    auto_submit = st.session_state.get("search_auto_submit", False)
    if auto_submit:
        st.session_state.search_auto_submit = False

    if (search_clicked or auto_submit) and search_query:
        video_id_filter = None
        if "Tutorial #1" in video_filter:
            video_id_filter = "01_installing_vs_code_how_websites_work_sigma_web_development_course_tutorial_1"
        elif "Tutorial #2" in video_filter:
            video_id_filter = "02_your_first_html_website_sigma_web_development_course_tutorial_2"

        with st.spinner("Searching..."):
            results = search_fn(search_query, video_id_filter=video_id_filter)

        if not results:
            st.info("No results found. Try a different query.")
        else:
            above = [r for r in results if not r.below_threshold]
            below = [r for r in results if r.below_threshold]

            st.markdown(
                f"**{len(results)} results** — "
                f"{len(above)} above threshold · {len(below)} below threshold",
            )

            if above:
                st.markdown("##### \u2705 Above Threshold")
                for i, r in enumerate(above):
                    _render_source_card(r, i + 1, group_id="srch")

            if below:
                st.markdown("##### \u26a0\ufe0f Below Threshold (weak match)")
                for i, r in enumerate(below):
                    _render_source_card(r, i + 1, show_below_threshold=True, group_id="srchb")

    elif not search_query:
        # ── Example queries (functional buttons, shown when search field is empty) ──
        st.markdown("**Try searching for:**")
        example_cols = st.columns(3)
        examples = [
            "HTML structure",
            "install VS Code",
            "CSS styling",
            "live preview extension",
            "website files folder",
            "JavaScript basics",
        ]
        for i, ex in enumerate(examples):
            with example_cols[i % 3]:
                if st.button(f"🔍 {ex}", key=f"srch_ex_{i}", use_container_width=True):
                    # Populate the search field and trigger an auto-submit on next rerun
                    st.session_state.search_input = ex
                    st.session_state.search_auto_submit = True
                    st.rerun()


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    _init_session()

    # Load pipeline (cached after first call)
    with st.spinner("Loading EduVision RAG pipeline..."):
        ask_fn, search_fn, health_check_fn, status = _load_pipeline()

    # Sidebar
    _render_sidebar(status)

    # Main content
    st.markdown("""
<div style="padding: 8px 0 20px 0;">
    <h1 style="font-size:1.8em; font-weight:700; margin:0; color:#E8EAF6;">
        \U0001f393 EduVision RAG
    </h1>
    <p style="color:#7B7FA0; margin:4px 0 0 0; font-size:0.92em;">
        Ask questions about your video course \u2014 get grounded answers with exact timestamps.
    </p>
</div>
""", unsafe_allow_html=True)

    # Stop early if system not ready
    if not status.chroma_ok:
        st.error("\U0001f534 ChromaDB is not available. Run `python ingestion/indexer.py` first.")
        st.stop()

    # ── Video Player (appears when a timestamp button is clicked) ─────────────
    vp = st.session_state.get("video_player")
    if vp:
        with st.expander(f"\U0001f4fa Now Playing \u2014 {vp['label']}", expanded=True):
            col_vid, col_close = st.columns([12, 1])
            with col_vid:
                st.video(vp["path"], start_time=vp["start_time"])
            with col_close:
                st.markdown("<br>", unsafe_allow_html=True)  # vertical nudge
                if st.button("\u2715", key="close_player", help="Close player"):
                    st.session_state.video_player = None
                    st.rerun()

    # Tabs
    tab_chat, tab_search = st.tabs(["💬 Ask a Question", "🔍 Search Evidence"])

    with tab_chat:
        _render_chat_tab(ask_fn, st.session_state.video_filter)

    with tab_search:
        _render_search_tab(search_fn, st.session_state.video_filter)


if __name__ == "__main__":
    main()
