"""Streamlit front end. Talks to the FastAPI backend over HTTP only.

Run:  streamlit run app/ui/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run` puts this file's folder on sys.path, not the repo root; add the root for `app.*` imports.
_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st  # noqa: E402

from app.ui.api_client import ApiClient, ApiError  # noqa: E402
from app.ui.chat_page import active_conversation, init_state, page_chat, start_new_chat  # noqa: E402
from app.ui.pages import page_library, page_search, page_timeline, sidebar_filters  # noqa: E402

st.set_page_config(page_title="Bharat Political History AI", page_icon="📜", layout="wide",
                   initial_sidebar_state="expanded")

CSS = """
<style>
:root { --accent:#2F5D62; --accent-soft:#E6EFEF; --ink:#1F2328; --muted:#5B6168; --line:#E3DFD6; --paper:#FFFFFF; }
.block-container { max-width: 920px; padding-top: 1.2rem; padding-bottom: 7rem; }
h1, h2, h3, h4 { font-family: Georgia, "Iowan Old Style", "Times New Roman", serif; letter-spacing: -0.01em; }
.app-title { font-family: Georgia, serif; font-size: 1.55rem; font-weight: 700; color: var(--ink); margin: 0; }
.app-sub { color: var(--muted); font-size: 0.92rem; margin: 0.1rem 0 0.8rem; }

/* Chat bubbles */
[data-testid="stChatMessage"] { background: transparent; padding: 0.35rem 0; gap: 0.6rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { flex-direction: row-reverse; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
  background: var(--accent); color: #fff; border-radius: 18px 18px 4px 18px; padding: 0.65rem 1rem;
  max-width: 78%; margin-left: auto; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] p { color:#fff; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
  background: var(--paper); border: 1px solid var(--line); border-radius: 18px 18px 18px 4px;
  padding: 0.9rem 1.15rem; box-shadow: 0 1px 2px rgba(0,0,0,0.04); max-width: 100%; }
.interpreted { font-size: 0.8rem; opacity: 0.85; margin-top: 0.35rem; font-style: italic; }

/* Answer content */
.answer p { font-size: 1.03rem; line-height: 1.65; margin: 0 0 0.6rem; }
.section { margin-top: 0.7rem; }
.section h4 { font-size: 0.98rem; margin: 0.4rem 0 0.25rem; color: var(--ink); }
.section ul { margin: 0 0 0.3rem 1.1rem; padding: 0; }
.section li { margin: 0.2rem 0; line-height: 1.55; }
.attr { color: var(--muted); font-style: italic; }
a.cite, span.cite { display: inline-block; font-size: 0.7em; font-weight: 700; padding: 0 0.38em; margin: 0 0.08em;
  border-radius: 0.45em; background: var(--accent); color: #fff !important; vertical-align: super; line-height: 1.45;
  text-decoration: none; }
a.cite:hover { background: #1E4246; }
span.cite-muted { background: #9AA0A6; }
.notice { border-left: 3px solid #B7791F; background: #FFF8E6; padding: 0.45rem 0.75rem; border-radius: 4px;
  margin: 0 0 0.6rem; font-size: 0.88rem; color: var(--ink); }
.banner { border-radius: 8px; padding: 0.6rem 0.85rem; margin-bottom: 0.6rem; }
.banner-hint { font-size: 0.9rem; margin-top: 0.15rem; }
.banner-warn { background: #FFF8E6; border: 1px solid #F0D9A8; }
.banner-info { background: #EEF4FB; border: 1px solid #C9DCF2; }
.banner-error { background: #FDECEC; border: 1px solid #F3C2C2; }

/* Sources */
.sources-title { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted);
  margin: 0.9rem 0 0.35rem; font-weight: 700; }
.source-card { border: 1px solid var(--line); border-radius: 10px; padding: 0.55rem 0.75rem; margin: 0.4rem 0;
  background: #FCFBF8; scroll-margin-top: 80px; }
.source-card:target { outline: 2px solid var(--accent); }
.source-head { display: flex; gap: 0.5rem; align-items: baseline; }
.sid { font-size: 0.72rem; font-weight: 700; border-radius: 6px; padding: 0.05rem 0.4rem; white-space: nowrap; }
.sid-cited { background: var(--accent); color: #fff; }
.sid-retrieved { background: #E7E5E0; color: var(--ink); }
.stitle { font-weight: 600; color: var(--ink); }
.smeta { font-size: 0.82rem; color: var(--muted); margin-top: 0.15rem; }
.source-card details, details.others { margin-top: 0.3rem; }
.source-card summary, details.others > summary { cursor: pointer; font-size: 0.85rem; color: var(--accent); }
.source-card blockquote { border-left: 3px solid var(--line); margin: 0.4rem 0; padding: 0.1rem 0.7rem;
  color: #3B4046; font-size: 0.9rem; }
.kv { font-size: 0.85rem; margin-top: 0.3rem; } .kv span { color: var(--muted); margin-right: 0.4rem; }
.muted { color: var(--muted); font-size: 0.8rem; }

.welcome { text-align: center; margin: 1.5rem 0 1rem; }
.welcome h2 { font-size: 1.6rem; margin-bottom: 0.3rem; }
.welcome p { color: var(--muted); }
.disclaimer { font-size: 0.78rem; color: var(--muted); border-top: 1px solid var(--line); margin-top: 1.4rem;
  padding-top: 0.6rem; }
[data-testid="stSidebar"] .conv-meta { font-size: 0.78rem; color: var(--muted); }

@media (max-width: 640px) {
  .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] { max-width: 92%; }
  .answer p { font-size: 0.98rem; }
}
</style>
"""

VIEWS = ["💬 Research chat", "🔎 Search sources", "🗓️ Timeline", "📚 Library & ingestion"]


@st.cache_data(ttl=30, show_spinner=False)
def cached_health(base_url: str) -> dict:
    return ApiClient(base_url).health()


@st.cache_data(ttl=60, show_spinner=False)
def cached_filter_options(base_url: str) -> dict:
    return ApiClient(base_url).filter_options()


@st.cache_data(ttl=60, show_spinner=False)
def cached_documents(base_url: str) -> list[dict]:
    return ApiClient(base_url).documents()


def render_conversation_list(container) -> None:
    """Filled at the end of the run so a conversation started this turn already shows its title."""
    with container:
        st.markdown("**Conversations (this session)**")
        for c in list(st.session_state.conversations.values())[::-1]:
            active = c["id"] == st.session_state.active_id
            label = ("▸ " if active else "") + c["title"]
            if st.button(label, key=f"conv-{c['id']}", width="stretch", disabled=active):
                st.session_state.active_id = c["id"]
                st.rerun()
        st.markdown('<div class="conv-meta">Chats are kept only for this browser session.</div>',
                    unsafe_allow_html=True)


def sidebar(client: ApiClient, health: dict, opts: dict) -> tuple[str, dict, bool, bool, object]:
    with st.sidebar:
        st.markdown("### 📜 Bharat Political History AI")
        view = st.radio("View", VIEWS, label_visibility="collapsed")
        use_context = reveal = True
        conv_container = None
        if view == VIEWS[0]:
            init_state()
            if st.button("＋ New chat", width="stretch", type="primary"):
                start_new_chat()
                st.rerun()
            conv_container = st.container()
            with st.expander("Answer settings"):
                use_context = st.toggle("Use my previous question for follow-ups", value=True,
                                        help="Short follow-ups (e.g. “what about his views on railways?”) are sent "
                                             "together with your previous question so retrieval can resolve the "
                                             "reference. Previous answers are never sent; the exact text sent is "
                                             "shown under your message.")
                reveal = st.toggle("Typewriter reveal", value=True,
                                   help="Animates the completed answer after it is returned and validated. "
                                        "This is a visual effect, not live token streaming.")
        with st.expander("Filters"):
            filters = sidebar_filters(opts)
        with st.expander("Indexed documents", expanded=False):
            try:
                docs = [d for d in cached_documents(client.base_url) if d.get("status") == "indexed"]
            except ApiError:
                docs = []
            st.caption(f"{len(docs)} documents · {health.get('indexed_chunks', 0)} passages")
            for d in docs:
                year = d.get("publication_date") or "n.d."
                st.markdown(f"- {d.get('title')} <span class='conv-meta'>({year})</span>", unsafe_allow_html=True)
        llm = health["llm_model"] if health.get("llm_configured") else "not configured"
        st.caption(f"Model: {llm} (Gemini free tier) · Embeddings: {health.get('embedding_model')}")
    return view, filters, use_context, reveal, conv_container


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    client = ApiClient()
    try:
        health = cached_health(client.base_url)
        opts = cached_filter_options(client.base_url)
    except ApiError as exc:
        st.markdown('<p class="app-title">📜 Bharat Political History AI</p>', unsafe_allow_html=True)
        st.error(exc.user_message)
        if st.button("Retry"):
            st.cache_data.clear()
            st.rerun()
        st.stop()
    if health.get("status") != "ok":
        st.error(health.get("detail", "The backend reported an error."))
        st.stop()

    view, filters, use_context, reveal, conv_container = sidebar(client, health, opts)
    header_title = active_conversation()["title"] if view == VIEWS[0] and "conversations" in st.session_state else ""
    st.markdown('<p class="app-title">Bharat Political History AI</p>'
                '<p class="app-sub">Evidence-grounded research assistant for Indian political history'
                + (f" · {header_title}" if header_title and header_title != "New chat" else "") + "</p>",
                unsafe_allow_html=True)
    if health.get("indexed_chunks", 0) == 0:
        st.warning("The knowledge base is empty. Add documents to `data/sources/` with metadata sidecars and run "
                   "`python -m scripts.ingest` (see README → Adding documents).")
    if filters:
        st.caption("Filters active: " + ", ".join(f"{k}={v}" for k, v in filters.items()))

    if view == VIEWS[0]:
        page_chat(client, filters, health, use_context, reveal)
        render_conversation_list(conv_container)
    elif view == VIEWS[1]:
        page_search(client, filters)
    elif view == VIEWS[2]:
        page_timeline(client)
    else:
        page_library(client)


main()
