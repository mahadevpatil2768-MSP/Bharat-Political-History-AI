"""Shared fixtures.

Test documents describe the fictional "Kingdom of Testland" so tests never
assert real historical claims.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# Tests must not write to the project's logs/backend.log (set before any app module reads settings).
os.environ["LOG_FILE"] = ""

from app.config import Settings  # noqa: E402

FICTION_NOTE = "Synthetic test fixture — fictional content, not a historical source."


def write_doc(sources: Path, name: str, content: str | bytes, meta: dict | None) -> Path:
    path = sources / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    if meta is not None:
        (sources / f"{name}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return path


def make_pdf(path: Path, pages: list[list[tuple[str, float]]], fontfile: str | None = None) -> Path:
    """pages: list of [(text, fontsize), ...]. An empty list makes a blank page.

    ``fontfile`` embeds a TrueType font so non-Latin-1 characters (e.g. the em dash "—") survive;
    the default base-14 font renders them as "?".
    """
    import pymupdf as fitz

    doc = fitz.open()
    for items in pages:
        page = doc.new_page()
        font = {}
        if fontfile:
            page.insert_font(fontname="F0", fontfile=fontfile)
            font = {"fontname": "F0"}
        y = 72
        for text, size in items:
            rect = fitz.Rect(72, y, 540, y + 400)
            page.insert_textbox(rect, text, fontsize=size, **font)
            lines = max(1, len(text) // int(900 / size) + 1)
            y += lines * size * 1.5 + 14
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        embedding_model="hashing-test-embedder",
        gemini_api_key=None,
        chunk_size=400,
        chunk_overlap=60,
        min_chunk_chars=80,
        retrieval_candidate_k=20,
    )
    s.sources_dir.mkdir(parents=True, exist_ok=True)
    return s


@pytest.fixture
def services(settings):
    from app.services import build_services

    return build_services(settings)


CHARTER_MD = f"""# Charter of Testland

{FICTION_NOTE}

## Part I: The Assembly

The Assembly of Testland shall consist of forty members elected by the provinces.
Each member shall serve a term of five years unless the Assembly is dissolved sooner.

The Speaker of the Assembly presides over its sittings and maintains order.

## Part II: The Courts

The High Court of Testland may review any law passed by the Assembly.
Judges of the High Court are appointed by the Monarch on the advice of the Council.

| Court | Seat | Judges |
|---|---|---|
| High Court | Capital City | 7 |
| Provincial Court | Each province | 3 |
"""

CHRONICLE_HTML = f"""<html><head><title>Chronicle</title><script>var x=1;</script></head>
<body><nav>Menu Home About</nav><main>
<h1>Chronicle of the Testland Reforms</h1>
<p>{FICTION_NOTE}</p>
<h2>The Salt Reform</h2>
<p>In year 1234 the Council of Testland abolished the salt levy after petitions from farmers.</p>
<p>The chronicler Ammara of Riverton wrote that the reform was driven by famine; the court poet Belan
attributed it instead to the Monarch's personal generosity.</p>
<h2>Provincial Revenue</h2>
<table><tr><th>Province</th><th>Revenue</th></tr><tr><td>North</td><td>120</td></tr>
<tr><td>South</td><td>95</td></tr></table>
</main><footer>Footer text</footer></body></html>"""

CHARTER_META = {
    "document_id": "testland_charter", "title": "Charter of Testland (fixture)",
    "author_or_institution": "Testland Assembly", "source_type": "primary_legal",
    "publication_date": "1230", "historical_periods": ["ancient"],
    "event_year_start": 1230, "event_year_end": 1230, "language": "en",
    "topics": ["constitution", "courts"], "institutions": ["Assembly", "High Court"],
}

CHRONICLE_META = {
    "document_id": "testland_chronicle", "title": "Chronicle of the Testland Reforms (fixture)",
    "author_or_institution": "Ammara of Riverton", "source_type": "scholarship",
    "publication_date": "1240-05", "historical_periods": ["medieval"],
    "event_year_start": 1234, "event_year_end": 1236, "language": "en",
    "topics": ["taxation", "reform"], "persons": ["Ammara of Riverton", "Belan"],
}


@pytest.fixture(autouse=True)
def _no_live_gemini(monkeypatch):
    """Fail loudly if any test tries to build a real Gemini client (no live API calls in tests)."""
    import app.llm.gemini_client as gc

    def _blocked(*args, **kwargs):
        raise AssertionError("Tests must not construct a real google.genai.Client")

    monkeypatch.setattr(gc.genai, "Client", _blocked)
