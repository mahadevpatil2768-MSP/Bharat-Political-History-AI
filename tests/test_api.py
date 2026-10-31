import json

import pytest
from fastapi.testclient import TestClient

from app.api import main as api
from app.generation.answer import AnswerGenerator
from app.llm.fake import FakeProvider
from app.services import set_services
from tests.conftest import CHARTER_MD, CHARTER_META, CHRONICLE_HTML, CHRONICLE_META, make_pdf, write_doc

GOOD_ANSWER = {"sufficient_evidence": True, "answer": "The levy was abolished [S1].",
               "documented_facts": [{"statement": "The levy was abolished.", "citations": ["S1"]}],
               "interpretations": [], "disputed_or_uncertain": [], "conflicts": [], "evidence_gaps": []}


@pytest.fixture
def client(settings, services):
    write_doc(settings.sources_dir, "charter.md", CHARTER_MD, CHARTER_META)
    write_doc(settings.sources_dir, "chronicle.html", CHRONICLE_HTML, CHRONICLE_META)
    services.pipeline().run()
    set_services(services)
    api.app.dependency_overrides[api.answerer_dep] = lambda: AnswerGenerator(
        services.retriever, FakeProvider(GOOD_ANSWER), top_k=4)
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()
    set_services(None)


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["indexed_chunks"] > 0
    assert r["llm_configured"] is False and r["web_search"] is False
    assert "api_key" not in json.dumps(r).lower()


def test_query_returns_grounded_answer(client):
    r = client.post("/query", json={"question": "Who abolished the salt levy?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered" and body["citations"][0]["source_id"] == "S1"


def test_query_validation_errors(client):
    assert client.post("/query", json={"question": "x"}).status_code == 422
    assert client.post("/query", json={"question": "??????"}).status_code == 422
    assert client.post("/query", json={"question": "valid question", "bogus": 1}).status_code == 422
    assert client.post("/query", json={"question": "valid q", "filters": {"periods": ["jurassic"]}}).status_code == 422
    assert client.post("/query", json={"question": "valid q", "filters": {"published_from": "Jan 1950"}}).status_code == 422


def test_query_with_filters(client):
    r = client.post("/query", json={"question": "Who abolished the salt levy?",
                                    "filters": {"periods": ["ancient"]}}).json()
    assert all(c["document_id"] == "testland_charter" for c in r["retrieved"])


def test_search_endpoint(client):
    r = client.post("/search", json={"query": "High Court", "top_k": 3, "mode": "keyword"}).json()
    assert r["count"] >= 1 and r["hits"][0]["citation"]["document_id"] == "testland_charter"
    assert "not probabilities" in r["score_note"]


def test_documents_and_status(client):
    docs = client.get("/documents").json()["documents"]
    assert {d["document_id"] for d in docs} == {"testland_charter", "testland_chronicle"}
    status = client.get("/ingest/status").json()
    assert status["summary"]["indexed_documents"] == 2
    opts = client.get("/meta/filters").json()
    assert "ancient" in opts["periods"] and "taxation" in opts["topics"]


def test_upload_valid_document(client, settings):
    meta = {"title": "Uploaded fixture", "source_type": "other", "historical_periods": ["colonial"]}
    r = client.post("/documents/upload", data={"metadata": json.dumps(meta)},
                    files={"file": ("notes.md", b"# Notes\n\nThe Testland lighthouse act of 1301.", "text/markdown")})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "indexed"
    assert (settings.uploads_dir / "notes.md").exists()
    assert (settings.uploads_dir / "notes.md.meta.json").exists()
    hits = client.post("/search", json={"query": "lighthouse act"}).json()["hits"]
    assert hits and hits[0]["citation"]["title"] == "Uploaded fixture"


def test_upload_rejections(client, settings):
    meta = json.dumps({"title": "x"})
    bad_type = client.post("/documents/upload", data={"metadata": meta},
                           files={"file": ("evil.exe", b"MZ...", "application/octet-stream")})
    assert bad_type.status_code == 415
    fake_pdf = client.post("/documents/upload", data={"metadata": meta},
                           files={"file": ("doc.pdf", b"not a pdf", "application/pdf")})
    assert fake_pdf.status_code == 415
    bad_meta = client.post("/documents/upload", data={"metadata": "{not json"},
                           files={"file": ("a.txt", b"hello world", "text/plain")})
    assert bad_meta.status_code == 422
    empty = client.post("/documents/upload", data={"metadata": meta},
                        files={"file": ("e.txt", b"", "text/plain")})
    assert empty.status_code == 400
    settings.max_upload_mb = 1
    big = client.post("/documents/upload", data={"metadata": meta},
                      files={"file": ("big.txt", b"a" * (1024 * 1024 + 10), "text/plain")})
    assert big.status_code == 413


def test_upload_path_traversal_neutralised(client, settings):
    meta = json.dumps({"title": "Traversal attempt"})
    r = client.post("/documents/upload", data={"metadata": meta},
                    files={"file": ("../../../outside.txt", b"some text here", "text/plain")})
    assert r.status_code == 200
    assert (settings.uploads_dir / "outside.txt").exists()
    assert not (settings.data_dir / "outside.txt").exists()
    assert not (settings.data_dir.parent / "outside.txt").exists()
    dup = client.post("/documents/upload", data={"metadata": meta},
                      files={"file": ("outside.txt", b"other text", "text/plain")})
    assert dup.status_code == 409


def test_ingest_run_endpoint(client, settings):
    write_doc(settings.sources_dir, "extra.md", "# Extra\n\nThe Testland canal edict of 1310.", {"title": "Extra"})
    r = client.post("/ingest/run", json={}).json()
    assert r["counts"]["indexed"] == 1 and r["counts"]["unchanged"] == 2
    # BM25 sees the new document immediately.
    hits = client.post("/search", json={"query": "canal edict", "mode": "keyword"}).json()["hits"]
    assert hits and hits[0]["citation"]["document_id"] == "extra"


def test_timeline_only_shows_verified_events(client, settings):
    settings.timeline_path.parent.mkdir(parents=True, exist_ok=True)
    settings.timeline_path.write_text(json.dumps({"events": [
        {"id": "ok", "date": "1234", "title": "Salt levy abolished", "document_id": "testland_chronicle",
         "evidence_quote": "the Council of Testland abolished the salt levy"},
        {"id": "fake", "date": "1235", "title": "Invented event", "document_id": "testland_chronicle",
         "evidence_quote": "this sentence does not appear anywhere"},
        {"id": "nodoc", "date": "1236", "title": "Missing doc", "document_id": "nope",
         "evidence_quote": "whatever text is here"},
    ]}), encoding="utf-8")
    r = client.get("/timeline").json()
    assert [e["id"] for e in r["events"]] == ["ok"]
    assert len(r["unverified"]) == 2
    assert {s["document_id"] for s in r["document_spans"]} == {"testland_charter", "testland_chronicle"}


def test_unhandled_errors_do_not_leak(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret internal detail sk-xyz")
    monkeypatch.setattr(api.get_services().retriever, "retrieve", boom)
    r = TestClient(api.app, raise_server_exceptions=False).post("/search", json={"query": "anything"})
    assert r.status_code == 500 and "sk-xyz" not in r.text
