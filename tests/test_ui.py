"""Chat UI tests: rendering, follow-up context, UI client ↔ FastAPI integration, and headless AppTest runs.

No live Gemini calls: the backend uses the scripted FakeProvider; AppTest runs stub the HTTP client.
"""

import re
from types import SimpleNamespace

import pytest
import requests
from fastapi.testclient import TestClient

from app.api import main as api
from app.generation.answer import AnswerGenerator
from app.llm.fake import FakeProvider
from app.services import set_services
from app.ui import render
from app.ui.api_client import ApiClient, ApiError
from app.ui.conversation import MAX_QUESTION_CHARS, build_query, is_follow_up, previous_context_question
from tests.conftest import CHARTER_MD, CHARTER_META, CHRONICLE_HTML, CHRONICLE_META, write_doc

CIT = {"source_id": "S1", "chunk_id": "doc::c00001", "document_id": "doc", "title": "Charter of Testland (fixture)",
       "author_or_institution": "Testland Assembly", "source_type": "primary_legal", "publication_date": "1230",
       "source_url": "https://example.org/charter", "page_start": 40, "page_end": 40,
       "section_heading": "PART III", "excerpt": "The Assembly shall consist of forty members."}

ANSWERED = {
    "status": "answered", "question": "q", "answer": "The Assembly has forty members [S1]. Unknown [S9].",
    "documented_facts": [{"statement": "Forty members.", "citations": ["S1"], "attributed_to": None}],
    "interpretations": [{"statement": "A balanced body.", "citations": ["S1"], "attributed_to": "Ammara"}],
    "disputed_or_uncertain": [], "conflicts": [], "evidence_gaps": ["Election procedure"],
    "citations": [CIT], "retrieved": [CIT, dict(CIT, source_id="S2", chunk_id="doc::c00002", page_start=None,
                                                page_end=None, source_url=None, excerpt="Other passage")],
    "notices": [], "removed_citations": [], "model": "gemini-3.8-flash",
}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_citation_markers_become_links_to_source_cards():
    out = render.link_markers("Fact [S1]. Two [S1, S2]. Bogus [S9].", "m1", {"S1", "S2"})
    assert out.count('href="#src-m1-S1"') == 2 and 'href="#src-m1-S2"' in out
    assert 'class="cite cite-muted">S9<' in out and 'href="#src-m1-S9"' not in out


def test_text_is_escaped_and_latex_neutralised():
    out = render.link_markers('<script>alert(1)</script> costs $5 [S1]', "m", {"S1"})
    assert "<script>" not in out and "&lt;script&gt;" in out and "&#36;5" in out


def test_answer_html_has_headings_bullets_and_attribution():
    h = render.answer_html(ANSWERED, "m1")
    assert "<h4>📌 Documented facts</h4>" in h and "<li>" in h
    assert "— Ammara" in h and "Not covered by the indexed sources" in h and "Election procedure" in h
    assert "None" not in h and '{"' not in h


def test_source_card_fields_anchor_and_expandable_details():
    card = render.source_card_html(CIT, "m1", cited=True)
    assert 'id="src-m1-S1"' in card
    for expected in ["Charter of Testland (fixture)", "Testland Assembly", "p. 40", "PART III",
                     "forty members", "published 1230", "<details>", "Primary source · legal"]:
        assert expected in card, expected
    assert 'href="https://example.org/charter"' in card and 'rel="noopener noreferrer"' in card


def test_source_card_omits_missing_fields_and_unsafe_urls():
    c = dict(CIT, author_or_institution=None, page_start=None, page_end=None, section_heading=None,
             publication_date=None, source_url="javascript:alert(1)")
    card = render.source_card_html(c, "m1", cited=False)
    assert "None" not in card and "javascript:" not in card and "publication date unknown" in card
    assert "p. " not in card


def test_sources_html_orders_cited_first_and_groups_others():
    h = render.sources_html(ANSWERED, "m1")
    assert h.index("Sources cited") < h.index('id="src-m1-S1"') < h.index("Other retrieved passages (1)")
    assert 'id="src-m1-S2"' in h


@pytest.mark.parametrize("status,title", [("insufficient_evidence", "Not enough evidence"),
                                          ("declined", "Outside this tool"), ("error", "could not be generated")])
def test_status_banners(status, title):
    h = render.status_html(dict(ANSWERED, status=status, answer="Backend explanation."), "m1")
    assert title in h and "Backend explanation." in h


def test_reveal_chunks_are_progressive_and_end_with_full_text():
    text = " ".join(f"w{i}" for i in range(200))
    chunks = render.reveal_chunks(text, max_steps=20)
    assert chunks[-1] == text and len(chunks) <= 22
    assert all(text.startswith(c) for c in chunks)
    assert render.reveal_chunks("") == []


# ---------------------------------------------------------------------------
# Follow-up context strategy
# ---------------------------------------------------------------------------

def msgs(*pairs):
    out = []
    for q, status in pairs:
        out.append({"role": "user", "content": q})
        out.append({"role": "assistant", "response": {"status": status}})
    return out


def test_follow_up_detection():
    assert is_follow_up("What about his views on railways?")
    assert is_follow_up("And the Partition of Bengal?")
    assert not is_follow_up("What does Article 17 of the Constitution say about untouchability?")
    assert not is_follow_up(" ".join(["word"] * 30) + " it")


def test_build_query_adds_previous_question_only():
    history = msgs(("How does Gandhi define swaraj in Indian Home Rule?", "answered"))
    sent, added = build_query("What about his views on railways?", history)
    assert added and sent.startswith("What about his views on railways?")
    assert '"How does Gandhi define swaraj in Indian Home Rule?"' in sent
    assert build_query("What about his views on railways?", history, use_context=False) == (
        "What about his views on railways?", False)
    assert build_query("What did the 1858 Proclamation say about princes?", history)[1] is False


def test_declined_questions_are_not_carried_over_and_length_is_bounded():
    history = msgs(("Explain the Preamble", "answered"), ("Which party will win next?", "declined"))
    assert previous_context_question(history) == "Explain the Preamble"
    long_prev = msgs(("x" * 2000, "answered"))
    sent, added = build_query("And what about it?", long_prev)
    assert added and len(sent) <= MAX_QUESTION_CHARS


# ---------------------------------------------------------------------------
# UI client ↔ real FastAPI app (in-process, fake LLM)
# ---------------------------------------------------------------------------

@pytest.fixture
def ui_client(settings, services):
    write_doc(settings.sources_dir, "charter.md", CHARTER_MD, CHARTER_META)
    write_doc(settings.sources_dir, "chronicle.html", CHRONICLE_HTML, CHRONICLE_META)
    services.pipeline().run()
    set_services(services)

    def responder(system, user):
        ids = re.findall(r'<source id="(S\d+)"', user)
        return {"sufficient_evidence": True, "answer": f"The Council abolished the salt levy [{ids[0]}] [S99].",
                "documented_facts": [{"statement": "The levy was abolished.", "citations": [ids[0]]}],
                "interpretations": [], "disputed_or_uncertain": [], "conflicts": [], "evidence_gaps": []}

    api.app.dependency_overrides[api.answerer_dep] = lambda: AnswerGenerator(
        services.retriever, FakeProvider(responder), top_k=4)
    yield ApiClient("http://testserver", session=TestClient(api.app))
    api.app.dependency_overrides.clear()
    set_services(None)


def test_ui_client_query_roundtrip_and_citation_rendering(ui_client):
    res = ui_client.query("Who abolished the salt levy?")
    assert res["status"] == "answered" and res["citations"][0]["source_id"] == "S1"
    assert "S99" in res["removed_citations"]  # backend validation preserved
    h = render.answer_html(res, "m7") + render.sources_html(res, "m7")
    assert 'href="#src-m7-S1"' in h and 'id="src-m7-S1"' in h  # link target exists
    assert "S99" not in render.answer_html(res, "m7")
    assert "Chronicle of the Testland Reforms (fixture)" in h


def test_ui_client_filters_and_health(ui_client):
    res = ui_client.query("Who abolished the salt levy?", {"periods": ["ancient"]})
    assert {c["document_id"] for c in res["retrieved"]} == {"testland_charter"}
    assert ui_client.health()["llm_provider"] == "gemini"
    assert {d["document_id"] for d in ui_client.documents()} == {"testland_charter", "testland_chronicle"}


def test_ui_client_validation_error_is_friendly(ui_client):
    with pytest.raises(ApiError) as info:
        ui_client.query("?????")
    assert info.value.kind == "validation" and "Query contains no searchable words" in info.value.user_message


def test_ui_client_maps_transport_failures():
    with pytest.raises(ApiError) as info:
        ApiClient("http://127.0.0.1:9", timeout=2).health()  # nothing listens on port 9
    assert info.value.kind == "connection" and "uvicorn app.api.main:app" in info.value.user_message

    def session_raising(exc):
        return SimpleNamespace(get=lambda *a, **k: (_ for _ in ()).throw(exc),
                               post=lambda *a, **k: (_ for _ in ()).throw(exc))
    with pytest.raises(ApiError) as info:
        ApiClient("http://x", session=session_raising(requests.Timeout())).query("valid question")
    assert info.value.kind == "timeout"


class _Resp:
    def __init__(self, code, body=None, bad_json=False):
        self.status_code, self._body, self._bad = code, body, bad_json
        self.text = "<html>oops</html>"

    def json(self):
        if self._bad:
            raise ValueError("not json")
        return self._body


@pytest.mark.parametrize("resp,kind", [(_Resp(500, {"detail": "Internal server error."}), "server"),
                                       (_Resp(200, bad_json=True), "invalid_response"),
                                       (_Resp(200, {"status": "answered"}), "invalid_response"),
                                       (_Resp(200, {**ANSWERED, "status": "bogus"}), "invalid_response")])
def test_ui_client_rejects_bad_responses_without_leaking_body(resp, kind):
    session = SimpleNamespace(post=lambda *a, **k: resp, get=lambda *a, **k: resp)
    with pytest.raises(ApiError) as info:
        ApiClient("http://x", session=session).query("valid question")
    assert info.value.kind == kind and "<html>" not in info.value.user_message and '{"' not in info.value.user_message


# ---------------------------------------------------------------------------
# Headless Streamlit runs (HTTP client stubbed)
# ---------------------------------------------------------------------------

HEALTH = {"status": "ok", "indexed_chunks": 1682, "embedding_model": "intfloat/multilingual-e5-base",
          "llm_provider": "gemini", "llm_configured": True, "llm_model": "gemini-3.8-flash",
          "llm_model_free_tier_allowlisted": True}


@pytest.fixture
def app_test(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    sent = []
    filters_seen = []
    behaviour = {"query": lambda q, f: dict(ANSWERED, question=q)}
    monkeypatch.setattr(ApiClient, "health", lambda self: HEALTH)
    monkeypatch.setattr(ApiClient, "filter_options", lambda self: {"periods": {"ancient": "Ancient India"},
                                                                    "source_types": [], "topics": [], "languages": []})
    monkeypatch.setattr(ApiClient, "documents", lambda self: [{"document_id": "doc", "status": "indexed",
                                                               "title": "Charter of Testland (fixture)",
                                                               "publication_date": "1230"}])

    def fake_query(self, question, filters=None):
        sent.append(question)
        filters_seen.append(filters)
        return behaviour["query"](question, filters)

    monkeypatch.setattr(ApiClient, "query", fake_query)
    st.cache_data.clear()
    from pathlib import Path
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "ui" / "streamlit_app.py"),
                           default_timeout=60)
    return SimpleNamespace(at=at, sent=sent, filters=filters_seen, behaviour=behaviour)


def all_markdown(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_chat_initial_render(app_test):
    at = app_test.at.run()
    assert not at.exception
    assert "What would you like to research?" in all_markdown(at)
    assert any(b.label == "＋ New chat" for b in at.sidebar.button)
    assert len(at.chat_input) == 1


def test_chat_question_renders_bubbles_answer_and_sources(app_test):
    at = app_test.at.run()
    at.chat_input[0].set_value("How many members does the Assembly have?").run()
    assert not at.exception
    assert len(at.chat_message) == 2
    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    md = all_markdown(at)
    assert "The Assembly has forty members" in md and 'href="#src-' in md and "Sources cited" in md
    assert "Charter of Testland (fixture)" in md and "p. 40" in md
    assert '"status"' not in md and '"documented_facts"' not in md  # no raw JSON
    assert "What would you like to research?" not in md


def test_chat_follow_up_is_sent_with_context_and_shown(app_test):
    at = app_test.at.run()
    at.chat_input[0].set_value("What does the Charter say about the Assembly?").run()
    at.chat_input[0].set_value("What about its term?").run()
    assert not at.exception
    assert app_test.sent[1].startswith("What about its term?")
    assert "What does the Charter say about the Assembly?" in app_test.sent[1]
    assert "Sent with context from your earlier question" in all_markdown(at)
    assert len(at.chat_message) == 4


def test_chat_backend_errors_are_friendly(app_test):
    def boom(q, f):
        raise ApiError("connection", "Can't reach the research backend at http://127.0.0.1:8000.")
    app_test.behaviour["query"] = boom
    at = app_test.at.run()
    at.chat_input[0].set_value("Any question here?").run()
    assert not at.exception
    md = all_markdown(at)
    assert "Backend not reachable" in md and "Can&#x27;t reach the research backend" in md


def test_chat_gemini_rate_limit_and_insufficient_evidence(app_test):
    app_test.behaviour["query"] = lambda q, f: dict(
        ANSWERED, status="error", citations=[],
        answer="The Gemini free-tier per-minute rate limit was reached. Please wait about a minute and try again.")
    at = app_test.at.run()
    at.chat_input[0].set_value("First question?").run()
    md = all_markdown(at)
    assert "could not be generated" in md and "per-minute rate limit" in md and "Retrieved passages" in md
    app_test.behaviour["query"] = lambda q, f: dict(ANSWERED, status="insufficient_evidence", citations=[],
                                                     answer="The indexed sources do not contain evidence.")
    at.chat_input[0].set_value("Second, unrelated research question here?").run()
    assert "Not enough evidence" in all_markdown(at) and not at.exception


def test_new_chat_and_switching_conversations(app_test):
    at = app_test.at.run()
    at.chat_input[0].set_value("What does the Charter say about courts?").run()
    next(b for b in at.sidebar.button if b.label == "＋ New chat").click().run()
    assert not at.exception
    assert "What would you like to research?" in all_markdown(at)
    labels = [b.label for b in at.sidebar.button]
    assert any("What does the Charter say about courts?" in lbl for lbl in labels)
    next(b for b in at.sidebar.button if "What does the Charter say about courts?" in b.label).click().run()
    assert len(at.chat_message) == 2 and "The Assembly has forty members" in all_markdown(at)


def test_new_conversation_title_appears_in_sidebar_same_run(app_test):
    at = app_test.at.run()
    at.chat_input[0].set_value("What does the Charter say about courts?").run()
    assert any("What does the Charter say about courts?" in b.label for b in at.sidebar.button)


# ---------------------------------------------------------------------------
# Manual Retry (HTTP 5xx and per-minute rate limits only)
# ---------------------------------------------------------------------------

RETRY_LABEL = "↻ Retry this question"


def error_response(category, retryable, answer="Gemini is experiencing high demand (HTTP 503 UNAVAILABLE)."):
    return dict(ANSWERED, status="error", citations=[], answer=answer, error_category=category,
                retryable=retryable)


def retry_buttons(at):
    return [b for b in at.button if b.label == RETRY_LABEL]


def test_retry_button_shown_for_5xx_and_never_auto_retries(app_test):
    app_test.behaviour["query"] = lambda q, f: error_response("http_5xx", True)
    at = app_test.at.run()
    at.chat_input[0].set_value("What is Article 14?").run()
    assert len(retry_buttons(at)) == 1 and len(app_test.sent) == 1
    at.run()  # plain rerun while the button is displayed: no request
    at.run()
    assert len(app_test.sent) == 1 and len(retry_buttons(at)) == 1


def test_retry_resends_identical_question_and_filters_and_replaces_failed_reply(app_test):
    app_test.behaviour["query"] = lambda q, f: error_response("rate_limit", True,
                                                               "The Gemini free-tier per-minute rate limit was reached.")
    at = app_test.at.run()
    at.sidebar.multiselect[0].set_value(["ancient"]).run()
    at.chat_input[0].set_value("What is Article 14?").run()
    at.sidebar.multiselect[0].set_value([]).run()  # filters changed after the failure
    assert len(app_test.sent) == 1
    app_test.behaviour["query"] = lambda q, f: dict(ANSWERED, question=q)
    retry_buttons(at)[0].click().run()
    assert not at.exception
    assert app_test.sent == ["What is Article 14?", "What is Article 14?"]
    assert app_test.filters[0] == app_test.filters[1] == {"periods": ["ancient"]}
    assert len(at.chat_message) == 2  # failed reply replaced, question not duplicated
    md = all_markdown(at)
    assert "The Assembly has forty members" in md and "per-minute rate limit" not in md
    assert retry_buttons(at) == []


def test_repeated_failure_offers_one_retry_per_click(app_test):
    app_test.behaviour["query"] = lambda q, f: error_response("http_5xx", True)
    at = app_test.at.run()
    at.chat_input[0].set_value("What is Article 14?").run()
    retry_buttons(at)[0].click().run()
    assert len(app_test.sent) == 2 and len(retry_buttons(at)) == 1 and len(at.chat_message) == 2
    at.run()
    assert len(app_test.sent) == 2


@pytest.mark.parametrize("response", [
    error_response("quota_daily", False, "The Gemini free-tier daily quota for this model is used up."),
    error_response("auth", False, "The Gemini API rejected the API key."),
    error_response("not_configured", False, "No GEMINI_API_KEY is configured."),
    dict(ANSWERED, status="insufficient_evidence", citations=[], answer="Not covered."),
    dict(ANSWERED, status="declined", citations=[], answer="Out of scope."),
    ANSWERED,
])
def test_no_retry_button_for_non_transient_outcomes(app_test, response):
    app_test.behaviour["query"] = lambda q, f: dict(response)
    at = app_test.at.run()
    at.chat_input[0].set_value("Some research question here?").run()
    assert retry_buttons(at) == []


def test_retry_for_backend_5xx_but_not_connection_errors(app_test):
    def server_error(q, f):
        raise ApiError("server", "The research backend hit an internal error.")
    app_test.behaviour["query"] = server_error
    at = app_test.at.run()
    at.chat_input[0].set_value("First question here?").run()
    assert len(retry_buttons(at)) == 1

    def conn_error(q, f):
        raise ApiError("connection", "Can't reach the research backend.")
    app_test.behaviour["query"] = conn_error
    at.chat_input[0].set_value("Second question here?").run()
    assert retry_buttons(at) == []


def test_retry_button_only_on_latest_reply(app_test):
    app_test.behaviour["query"] = lambda q, f: error_response("http_5xx", True)
    at = app_test.at.run()
    at.chat_input[0].set_value("First question here?").run()
    app_test.behaviour["query"] = lambda q, f: dict(ANSWERED, question=q)
    at.chat_input[0].set_value("Second, different question?").run()
    assert retry_buttons(at) == [] and len(at.chat_message) == 4


class _State(dict):
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v


def test_request_retry_ignores_duplicate_clicks(monkeypatch):
    from app.ui import chat_page
    conv = {"id": "c1", "messages": [{"role": "user", "content": "q", "sent": "q"},
                                     {"role": "assistant", "id": "m1", "response": error_response("http_5xx", True)}]}
    state = _State(conversations={"c1": conv}, pending_retry=None)
    monkeypatch.setattr(chat_page.st, "session_state", state)
    chat_page.request_retry("c1", "m1")
    first = dict(state["pending_retry"])
    chat_page.request_retry("c1", "m1")  # second click before the rerun processed the first
    assert state["pending_retry"] == first == {"conv_id": "c1", "msg_id": "m1"}
    assert conv["messages"][1]["retry_requested"] is True
    retry_msg = chat_page.take_pending_retry(conv)
    assert retry_msg["sent"] == "q" and retry_msg["retries"] == 1 and len(conv["messages"]) == 1
    assert chat_page.take_pending_retry(conv) is None  # consumed exactly once
    chat_page.request_retry("c1", "m1")  # stale button (message already removed): ignored
    assert state["pending_retry"] is None


def test_error_category_and_retryable_flow_through_api(settings, services):
    from app.llm.base import LLMQuotaExhaustedError, LLMServerError
    write_doc(settings.sources_dir, "charter.md", CHARTER_MD, CHARTER_META)
    services.pipeline().run()
    set_services(services)
    try:
        for exc, category, retryable in [(LLMServerError("503"), "http_5xx", True),
                                         (LLMQuotaExhaustedError("429"), "quota_daily", False)]:
            api.app.dependency_overrides[api.answerer_dep] = lambda e=exc: AnswerGenerator(
                services.retriever, FakeProvider(e), top_k=3)
            res = ApiClient("http://testserver", session=TestClient(api.app)).query("What does the Assembly do?")
            assert res["status"] == "error" and res["error_category"] == category
            assert res["retryable"] is retryable and res["retrieved"]
    finally:
        api.app.dependency_overrides.clear()
        set_services(None)
