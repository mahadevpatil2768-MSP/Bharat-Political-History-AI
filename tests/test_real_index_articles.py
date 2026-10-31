"""Article 14 / 17 / 21 / 32 retrieval on the REAL index (Constitution of India, 2024 edition).

Skipped automatically unless the index is present and has been annotated
(`python -m scripts.annotate_legal_structure`). Uses the real multilingual-E5 embedder; no Gemini calls.
Expected pages were read from the indexed PDF text, not from general knowledge.
"""

import os

import pytest

pytestmark = pytest.mark.real_index

EXPECTED = {
    "14": ("Equality before law.—The State shall not deny to any person equality before the law", 37),
    "17": ("Abolition of Untouchability.—", 40),
    "21": ("Protection of life and personal liberty.—No person shall be deprived of his life", 42),
    "32": ("Remedies for enforcement of rights conferred by this Part.—(1) The right to move the Supreme Court", 50),
}


@pytest.fixture(scope="module")
def retriever():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from app.config import Settings
    from app.services import build_services

    s = Settings()  # real .env / DATA_DIR
    if not (s.index_dir / "chroma.sqlite3").exists():
        pytest.skip("real index not present")
    svc = build_services(s)
    probe = svc.index.collection.get(where={"document_id": "constitution_of_india_2024"}, limit=1,
                                     include=["metadatas"])
    if not probe["ids"]:
        pytest.skip("Constitution of India not indexed")
    if "legal_structure_version" not in probe["metadatas"][0]:
        pytest.skip("index not annotated yet: run python -m scripts.annotate_legal_structure")
    return svc.retriever


@pytest.mark.parametrize("article", sorted(EXPECTED, key=int))
@pytest.mark.parametrize("template", ["What is Article {a}?", "Explain Art. {a}"])
def test_article_text_ranked_first_with_correct_page(retriever, article, template):
    needle, page = EXPECTED[article]
    hits = retriever.retrieve(template.format(a=article), top_k=8)
    top = hits[0]
    assert top.match_reason == "article_heading"
    assert needle in top.text.replace("\n", " ")
    assert top.metadata["document_id"] == "constitution_of_india_2024"
    assert top.metadata["page_start"] <= page <= top.metadata["page_end"]


@pytest.mark.parametrize("article", sorted(EXPECTED, key=int))
def test_no_contents_pages_or_mention_only_passages(retriever, article):
    from app.retrieval.legal_structure import has
    hits = retriever.retrieve(f"What is Article {article}?", top_k=8)
    for h in hits:
        m = h.metadata
        assert not m.get("is_contents")
        assert h.match_reason in ("article_heading", "article_text", "related_provision")
        if h.match_reason == "related_provision":
            assert has(m, "article_refs_substantive", article)


def test_article_21_ignores_sixth_schedule_paragraph_21(retriever):
    hits = retriever.retrieve("What does Article 21 say?", top_k=8)
    assert all("SCHEDULE" not in (h.metadata.get("section_heading") or "").upper() for h in hits)


def test_article_14_keeps_substantive_exceptions_only(retriever):
    hits = retriever.retrieve("What is Article 14?", top_k=8)
    related = [h for h in hits if h.match_reason == "related_provision"]
    assert len(related) <= 2
    for h in related:
        assert "inconsistent with" in h.text and "article 14" in h.text
