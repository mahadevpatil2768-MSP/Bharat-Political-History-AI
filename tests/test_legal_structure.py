"""Article-aware retrieval: structure detection, query parsing, ranking tiers and in-place annotation.

Fixtures describe the fictional "Charter of Testland" (no real historical claims). Real-index checks for
Articles 14, 17, 21 and 32 of the Constitution of India live in test_real_index_articles.py.
"""

from pathlib import Path

import pytest

from app.retrieval.legal_structure import (
    annotate_document,
    article_refs,
    content_terms,
    contents_like,
    has,
    inline_headings,
    query_articles,
)
from tests.conftest import make_pdf

# ---------------------------------------------------------------------------
# Detection units
# ---------------------------------------------------------------------------


def test_inline_heading_formats():
    text = ("Right to Equality 14. Equality before law.—The State shall not deny. "
            "2[21A. Right to education.—The State shall provide. "
            "368. 1[Power of Parliament to amend the Constitution and procedure therefor].— 2[(1) Text. "
            "31D. [Saving of laws in respect of anti-national activities.].—Omitted by the Act. "
            "2. Subs. by the Constitution (Forty-fourth Amendment) Act, 1978, s. 2, for something—x")
    found = {num: title for _, num, title in inline_headings(text)}
    assert found["14"] == "Equality before law" and found["21A"] == "Right to education"
    assert found["368"].startswith("Power of Parliament to amend the Constitution")
    assert "31D" in found and "2" not in found  # amendment footnotes are not headings


def test_contents_run_is_not_a_heading():
    toc = "32. Remedies for enforcement of rights conferred by this Part. 33. Power of Parliament to modify."
    assert [n for _, n, _ in inline_headings(toc)] == []
    assert contents_like(toc) and contents_like("40 . | Organisationofvillagepanchayats . 41 . | Righttowork . 42 . | X")
    assert not contents_like("WE, THE PEOPLE OF TESTLAND, having solemnly resolved to constitute the Realm into "
                             "a free commonwealth and to secure to all its residents justice, liberty and "
                             "equality of status and of opportunity; and to promote among them all fraternity "
                             "assuring the dignity of the individual and the unity of the Realm, do hereby "
                             "adopt, enact and give to ourselves this Charter. " * 2)


def test_reference_classification():
    refs, sub = article_refs("no law shall be deemed to be void on the ground that it is inconsistent with, "
                             "or takes away or abridges any of the rights conferred by article 14 or article 19")
    assert refs == {"14", "19"} and sub == {"14", "19"}
    refs, sub = article_refs("matters which under clause (3) of article 16, clause (3) of article 32, article 33 "
                             "and article 34 may be provided for by law made by Parliament")
    assert {"16", "32", "33", "34"} <= refs and sub == set()
    refs, _ = article_refs("as provided in articles 14 and 19 and articles 21A to 22")
    assert {"14", "19", "21A", "22"} <= refs


@pytest.mark.parametrize("q,expected", [
    ("What is Article 14?", ["14"]), ("Explain Art. 21a", ["21A"]), ("what does article 32 say", ["32"]),
    ("Compare Article 14 and Article 21", ["14", "21"]), ("What is the Arthashastra?", []),
    ("Describe Part 14 of the text", []), ("Art 17 meaning", ["17"]),
])
def test_query_articles(q, expected):
    assert query_articles(q) == expected


def test_content_terms_separate_lookups_from_topical_questions():
    assert content_terms("What is Article 14?") == []
    assert content_terms("Explain Art. 21 of the Indian Constitution") == []
    assert content_terms("How does Article 21 apply to prisoners?") == ["apply", "prisoners"]


def test_annotate_document_spans_sections_schedules_and_front_matter():
    chunks = [
        {"chunk_id": "c0", "chunk_index": 0, "page_start": 1, "section_heading": "PART I",
         "text": "13. Laws void. 14. Equal standing before the Assembly. 17. Abolition of serfdom."},
        {"chunk_id": "c1", "chunk_index": 1, "page_start": 2, "section_heading": "PART I",
         "text": "Right to Equality 14. Equal standing before the Assembly.—The Realm shall not deny to any "
                 "resident equal standing before the Assembly of the Realm in any province."},
        {"chunk_id": "c2", "chunk_index": 2, "page_start": 2, "section_heading": "PART I",
         "text": "This continues the equal standing clause with a further sentence about provinces and courts. "
                 "15. Ports.—Harbour rules apply."},
        {"chunk_id": "c3", "chunk_index": 3, "page_start": 3, "section_heading": "PART I > 21. Protection of life",
         "text": "No resident of Testland shall be deprived of life except by the law of the Realm, says the text."},
        {"chunk_id": "c4", "chunk_index": 4, "page_start": 4, "section_heading": "FIRST SCHEDULE",
         "text": "21. Grazing rights.—Paragraph on grazing in the hills, which cites article 14 in passing."},
    ]
    ann = annotate_document(chunks)
    assert ann["c0"]["is_contents"] and "article_headings" not in ann["c0"]
    assert has(ann["c1"], "article_headings", "14") and not ann["c1"]["is_contents"]
    assert has(ann["c2"], "article_spans", "14") and has(ann["c2"], "article_headings", "15")
    assert has(ann["c3"], "article_headings", "21")  # title promoted to the section heading
    assert "article_headings" not in ann["c4"]       # Schedule paragraph "21." is not Article 21
    assert has(ann["c4"], "article_refs", "14") and not has(ann["c4"], "article_refs_substantive", "14")


# ---------------------------------------------------------------------------
# Retrieval on a fictional PDF (hashing embedder, full pipeline)
# ---------------------------------------------------------------------------

PAD = " This sentence is fictional filler describing the procedures of the Realm of Testland in some detail."


def _article(num, title, body):
    return (f"{num}. {title}.—{body}{PAD}", 10)


UNICODE_FONT = Path("C:/Windows/Fonts/arial.ttf")


@pytest.fixture
def charter(settings, services):
    if not UNICODE_FONT.exists():
        pytest.skip("needs a Unicode TrueType font to render em dashes in the fixture PDF")
    src = settings.sources_dir
    make_pdf(src / "charter.pdf", fontfile=str(UNICODE_FONT), pages=[
        [("CONTENTS", 20), ("13. Laws void. 14. Equal standing before the Assembly. 17. Abolition of serfdom. "
                            "21. Protection of life. 32. Remedies before the High Court.", 10)],
        [("PART I", 20),
         _article(13, "Laws inconsistent with this Charter", "Any law of Testland contrary to this Part is void."),
         _article(14, "Equal standing before the Assembly",
                  "The Realm shall not deny to any resident equal standing before the Assembly.")],
        [_article(17, "Abolition of serfdom", "Serfdom in Testland is abolished and its practice forbidden."),
         _article(21, "Protection of life", "No resident shall be deprived of life except by the law of the Realm.")],
        [("PART II", 20),
         _article(31, "Saving of harbour laws", "Notwithstanding anything in article 14, no harbour law shall be "
                  "void on the ground that it is inconsistent with article 14."),
         _article(32, "Remedies before the High Court",
                  "The right to petition the High Court for enforcement of Part I is guaranteed.")],
        [_article(33, "Duties of guards", "Matters under clause (3) of article 32 may be provided for by law."),
         _article(34, "Lighthouse tolls", "Tolls listed beside article 14 are payable each spring.")],
        [("FIRST SCHEDULE", 20), _article(21, "Grazing rights", "Grazing in the hills of Testland is regulated.")],
    ])
    (src / "charter.pdf.meta.json").write_text('{"document_id": "testland_charter_pdf", '
                                               '"title": "Charter of Testland (fixture)", "source_type": "primary_legal"}')
    report = services.pipeline().run()
    assert report.count("indexed") == 1
    return services


def texts(hits):
    return [h.text for h in hits]


def test_article_heading_ranked_first_with_page(charter):
    hits = charter.retriever.retrieve("What is Article 14?", top_k=8)
    assert hits[0].match_reason == "article_heading"
    assert "Equal standing before the Assembly.—" in hits[0].text and hits[0].metadata["page_start"] == 2


def test_mention_only_and_contents_excluded_substantive_kept(charter):
    hits = charter.retriever.retrieve("What is Article 14?", top_k=8)
    joined = " ".join(texts(hits))
    assert "Lighthouse tolls" not in joined          # mention-only
    assert not any(h.metadata.get("is_contents") for h in hits)
    related = [h for h in hits if h.match_reason == "related_provision"]
    assert related and "Saving of harbour laws" in related[0].text   # substantive exception kept


@pytest.mark.parametrize("article,needle,page", [
    ("17", "Serfdom in Testland is abolished", 3),
    ("21", "No resident shall be deprived of life", 3),
    ("32", "The right to petition the High Court", 4),
])
def test_articles_17_21_32_fixture(charter, article, needle, page):
    hits = charter.retriever.retrieve(f"What is Article {article}?", top_k=8)
    assert hits[0].match_reason == "article_heading" and needle in hits[0].text
    m = hits[0].metadata
    assert m["page_start"] <= page <= m["page_end"]  # citation page range covers the article's page
    assert all("Grazing rights" not in h.text for h in hits)  # Schedule paragraph 21 excluded


def test_article_32_excludes_mere_mentions(charter):
    hits = charter.retriever.retrieve("What does Article 32 say?", top_k=8)
    assert all("Duties of guards" not in h.text or h.match_reason in ("article_heading", "article_text")
               for h in hits)


def test_unknown_article_falls_back_to_hybrid(charter):
    hits = charter.retriever.retrieve("What is Article 999?", top_k=5)
    assert all(h.match_reason is None for h in hits)


def test_topical_article_query_adds_hybrid_context(charter):
    hits = charter.retriever.retrieve("How does Article 14 affect harbour tolls and courts?", top_k=6)
    assert hits[0].match_reason == "article_heading"
    assert any(h.match_reason == "hybrid" for h in hits) or len(hits) <= 3


def test_contents_demoted_for_ordinary_queries(charter):
    hits = charter.retriever.retrieve("Equal standing before the Assembly and abolition of serfdom", top_k=5)
    flags = [bool(h.metadata.get("is_contents")) for h in hits]
    assert flags == sorted(flags)  # all non-contents chunks come before any contents chunk


def test_filters_still_apply_in_article_mode(charter):
    from app.schemas import SearchFilters
    assert charter.retriever.retrieve("What is Article 14?", SearchFilters(document_ids=["other"])) == []


# ---------------------------------------------------------------------------
# In-place annotation script: metadata only, embeddings/text/ids unchanged
# ---------------------------------------------------------------------------

def test_annotation_script_updates_metadata_in_place(charter, monkeypatch):
    import scripts.annotate_legal_structure as script
    col = charter.index.collection
    ids = col.get()["ids"]
    stripped = []
    for cid, meta in zip(ids, col.get(ids=ids, include=["metadatas"])["metadatas"]):
        stripped.append({k: v for k, v in meta.items() if not k.startswith(("article_", "is_contents", "legal_"))})
    col.update(ids=ids, metadatas=stripped)  # simulate an index built before this feature
    before = col.get(ids=ids, include=["embeddings", "documents"])
    monkeypatch.setattr(script, "get_settings", lambda: charter.settings)
    assert script.main([]) == 0
    after = col.get(ids=ids, include=["embeddings", "documents", "metadatas"])
    assert after["documents"] == before["documents"]
    assert all((a == b).all() for a, b in zip(before["embeddings"], after["embeddings"]))
    assert any(has(m, "article_headings", "14") for m in after["metadatas"])
    assert all(m.get("legal_structure_version") == 1 for m in after["metadatas"])
