"""Chat view: ChatGPT-style conversation backed by POST /query."""

from __future__ import annotations

import time
import uuid

import streamlit as st

from app.ui import render
from app.ui.api_client import ApiClient, ApiError
from app.ui.conversation import build_query, new_conversation, title_from

ERROR_TITLES = {
    "connection": "Backend not reachable",
    "timeout": "Request timed out",
    "validation": "Question not accepted",
    "server": "Backend error",
    "invalid_response": "Unexpected response",
    "http": "Request rejected",
}

# Manual Retry is offered only for transient failures: Gemini HTTP 5xx and per-minute rate limits
# (never for daily-quota exhaustion, auth/config problems or insufficient evidence).
RETRYABLE_ERROR_CATEGORIES = {"http_5xx", "rate_limit"}
SPINNER_TEXT = ("Searching the indexed sources and drafting a cited answer… "
                "(if Gemini is busy, this can take up to about a minute)")

EXAMPLE_PROMPTS = [
    "What does Article 17 of the Constitution say about untouchability?",
    "What did the 1858 Proclamation say about religious toleration?",
    "How does the Arthashastra say a king should choose his ministers?",
    "How does Gandhi define passive resistance in Indian Home Rule?",
]


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_state() -> None:
    ss = st.session_state
    if "conversations" not in ss:
        first = new_conversation()
        ss.conversations = {first["id"]: first}
        ss.active_id = first["id"]
    ss.setdefault("pending_prompt", None)
    ss.setdefault("pending_retry", None)


def active_conversation() -> dict:
    return st.session_state.conversations[st.session_state.active_id]


def start_new_chat() -> None:
    conv = active_conversation()
    if not conv["messages"]:  # reuse an empty chat instead of piling up blanks
        return
    c = new_conversation()
    st.session_state.conversations[c["id"]] = c
    st.session_state.active_id = c["id"]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_user(msg: dict) -> None:
    with st.chat_message("user"):
        st.markdown(render.esc(msg["content"]), unsafe_allow_html=True)
        if msg.get("context_added"):
            st.markdown(f'<div class="interpreted">Sent with context from your earlier question: '
                        f'“{render.esc(msg["sent"])}”</div>', unsafe_allow_html=True)


def assistant_body(msg: dict) -> str:
    if msg.get("error"):
        err = msg["error"]
        return (f'<div class="banner banner-error"><strong>{render.esc(ERROR_TITLES.get(err["kind"], "Error"))}'
                f'</strong><div class="banner-hint">{render.esc(err["message"])}</div></div>')
    res = msg["response"]
    body = render.answer_html(res, msg["id"]) if res["status"] == "answered" else render.status_html(res, msg["id"])
    return render.notices_html(res.get("notices", [])) + body


def is_retryable(msg: dict) -> bool:
    if msg.get("error"):  # the backend itself returned HTTP 5xx
        return msg["error"]["kind"] == "server"
    res = msg.get("response") or {}
    return res.get("status") == "error" and (
        bool(res.get("retryable")) or res.get("error_category") in RETRYABLE_ERROR_CATEGORIES)


def request_retry(conv_id: str, msg_id: str) -> None:
    """Retry button callback: queue exactly one retry; repeated clicks are ignored."""
    conv = st.session_state.conversations.get(conv_id)
    msg = next((m for m in (conv or {}).get("messages", []) if m.get("id") == msg_id), None)
    if msg is None or msg.get("retry_requested") or st.session_state.get("pending_retry"):
        return
    msg["retry_requested"] = True
    st.session_state.pending_retry = {"conv_id": conv_id, "msg_id": msg_id}


def render_retry_button(msg: dict, conv_id: str) -> None:
    st.button("↻ Retry this question", key=f"retry-{msg['id']}", on_click=request_retry,
              args=(conv_id, msg["id"]), disabled=bool(msg.get("retry_requested")),
              help="Sends the same question (with the same filters) again as one new request.")


def render_assistant(msg: dict, reveal: bool = False, conv_id: str | None = None,
                     allow_retry: bool = False) -> None:
    with st.chat_message("assistant"):
        body = assistant_body(msg)
        res = msg.get("response")
        if reveal and res and res["status"] == "answered":
            # Typewriter reveal of the already-complete, validated answer (not model token streaming).
            placeholder = st.empty()
            chunks = render.reveal_chunks(render.plain_answer_text(res))
            delay = min(0.03, 1.5 / max(len(chunks), 1))
            for chunk in chunks:
                placeholder.markdown(f'<div class="answer"><p>{render.esc(chunk)}▍</p></div>', unsafe_allow_html=True)
                time.sleep(delay)
            placeholder.markdown(body, unsafe_allow_html=True)
        else:
            st.markdown(body, unsafe_allow_html=True)
        if res:
            sources = render.sources_html(res, msg["id"])
            if sources:
                st.markdown(sources, unsafe_allow_html=True)
            footer = []
            if res.get("model") and res["status"] == "answered":
                footer.append(f"Generated by {res['model']} from {len(res.get('retrieved', []))} retrieved passages")
            if res.get("removed_citations"):
                footer.append("citations removed in validation: " + ", ".join(res["removed_citations"]))
            if footer:
                st.caption(" · ".join(footer))
        if allow_retry and conv_id and is_retryable(msg):
            render_retry_button(msg, conv_id)


def render_history(conv: dict, offer_retry: bool = True) -> None:
    last = len(conv["messages"]) - 1
    for i, msg in enumerate(conv["messages"]):
        if msg["role"] == "user":
            render_user(msg)
        else:  # Retry is offered only on the latest reply, and not while a new turn is being processed
            render_assistant(msg, conv_id=conv["id"], allow_retry=offer_retry and i == last)


def render_welcome() -> None:
    st.markdown('<div class="welcome"><h2>What would you like to research?</h2>'
                '<p>Ask about Indian political history. Answers come only from the indexed documents, '
                'with every claim linked to its source passage.</p></div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, prompt in enumerate(EXAMPLE_PROMPTS):
        if cols[i % 2].button(prompt, key=f"example-{i}", width="stretch"):
            st.session_state.pending_prompt = prompt
            st.rerun()


# ---------------------------------------------------------------------------
# Turn handling
# ---------------------------------------------------------------------------

def answer_turn(client: ApiClient, conv: dict, user_msg: dict, reveal: bool) -> None:
    """Send the user's stored question (same text and filters) to /query and render the reply."""
    assistant_msg = {"role": "assistant", "id": uuid.uuid4().hex[:8], "response": None, "error": None}
    with st.spinner(SPINNER_TEXT):
        try:
            assistant_msg["response"] = client.query(user_msg["sent"], user_msg.get("filters") or {})
        except ApiError as exc:
            assistant_msg["error"] = {"kind": exc.kind, "message": exc.user_message}
    conv["messages"].append(assistant_msg)
    render_assistant(assistant_msg, reveal=reveal, conv_id=conv["id"], allow_retry=True)


def ask(client: ApiClient, conv: dict, question: str, filters: dict, use_context: bool, reveal: bool) -> None:
    sent, context_added = build_query(question, conv["messages"], use_context)
    user_msg = {"role": "user", "content": question, "sent": sent, "context_added": context_added,
                "filters": dict(filters)}
    conv["messages"].append(user_msg)
    if conv["title"] == "New chat":
        conv["title"] = title_from(question)
    render_user(user_msg)
    answer_turn(client, conv, user_msg, reveal)


def take_pending_retry(conv: dict) -> dict | None:
    """Consume a queued retry for this conversation; return the user message to resend, if valid.

    The failed reply is removed and replaced by the new one; the original question is reused verbatim.
    """
    retry = st.session_state.pending_retry
    if not retry:
        return None
    st.session_state.pending_retry = None
    msgs = conv["messages"]
    if (retry["conv_id"] != conv["id"] or len(msgs) < 2 or msgs[-1].get("id") != retry["msg_id"]
            or msgs[-2]["role"] != "user"):
        return None  # stale (e.g. user switched chats or asked something else first)
    msgs.pop()
    user_msg = msgs[-1]
    user_msg["retries"] = user_msg.get("retries", 0) + 1
    return user_msg


def page_chat(client: ApiClient, filters: dict, health: dict, use_context: bool, reveal: bool) -> None:
    init_state()
    conv = active_conversation()

    if not health.get("llm_configured"):
        st.info("No Gemini API key is configured, so replies will list the retrieved evidence without a "
                "generated answer. Add `GEMINI_API_KEY` to `.env` and restart the API.")
    elif health.get("llm_model_free_tier_allowlisted") is False:
        st.error(f"LLM_MODEL '{health.get('llm_model')}' is not on the verified free-tier allowlist; "
                 "no generation requests will be sent.")

    # chat_input is pinned to the bottom of the page wherever it is called; read it first so a
    # newly submitted question replaces the welcome screen in the same run.
    prompt = st.chat_input("Ask a question about Indian political history…", max_chars=1000)
    if st.session_state.pending_prompt and not prompt:
        prompt, st.session_state.pending_prompt = st.session_state.pending_prompt, None

    retry_msg = take_pending_retry(conv)

    if not conv["messages"] and not prompt:
        render_welcome()
    new_turn = retry_msg is not None or bool(prompt and prompt.strip())
    render_history(conv, offer_retry=not new_turn)
    if retry_msg is not None:
        answer_turn(client, conv, retry_msg, reveal)
    if prompt and prompt.strip():
        ask(client, conv, prompt.strip(), filters, use_context, reveal)

    st.markdown('<div class="disclaimer">AI-generated answers can be wrong or incomplete — check each claim '
                'against its cited passage. Answers use only the indexed documents (no live web search). '
                'Questions and retrieved passages are sent to the Gemini API free tier, where Google may use '
                'them to improve its products; avoid sensitive or personal information.</div>',
                unsafe_allow_html=True)
