import pytest

from app.ingestion.loaders import (
    ExtractionError,
    load_document,
    load_html,
    load_markdown,
    load_plain_text,
    normalize_text,
)
from app.schemas import BlockType
from tests.conftest import CHARTER_MD, CHRONICLE_HTML, make_pdf


def test_markdown_headings_paragraphs_and_tables():
    blocks = load_markdown(CHARTER_MD)
    headings = [b for b in blocks if b.block_type == BlockType.heading]
    assert [h.text for h in headings] == ["Charter of Testland", "Part I: The Assembly", "Part II: The Courts"]
    para = next(b for b in blocks if "forty members" in b.text)
    assert para.section_heading == "Charter of Testland > Part I: The Assembly"
    # Wrapped lines of one paragraph are joined into one block.
    assert "term of five years" in para.text
    table = next(b for b in blocks if b.block_type == BlockType.table)
    assert table.table_header.startswith("| Court")
    assert "Provincial Court" in table.text


def test_html_strips_boilerplate_and_keeps_structure():
    blocks = load_html(CHRONICLE_HTML.encode())
    texts = " ".join(b.text for b in blocks)
    assert "Menu Home" not in texts and "Footer text" not in texts and "var x" not in texts
    reform = next(b for b in blocks if "salt levy" in b.text)
    assert reform.section_heading.endswith("The Salt Reform")
    table = next(b for b in blocks if b.block_type == BlockType.table)
    assert table.text.splitlines()[0] == "Province | Revenue"
    assert "North | 120" in table.text


def test_plain_text_headings():
    text = "PART I\n\nThe first paragraph\nwraps here.\n\nCHAPTER 2 THE COUNCIL\n\nSecond para."
    blocks = load_plain_text(text)
    assert blocks[0].block_type == BlockType.heading
    assert blocks[1].text == "The first paragraph wraps here."
    assert blocks[-1].section_heading == "CHAPTER 2 THE COUNCIL"


def test_unicode_normalisation_indic():
    # NA + nukta (decomposed) normalises to the precomposed NFC form NNNA (U+0929).
    decomposed = "ऩ"
    assert normalize_text(decomposed) == "ऩ"
    blocks = load_plain_text("भारत का संविधान\n\nयह एक परीक्षण अनुच्छेद है।")
    assert any("परीक्षण" in b.text for b in blocks)


def test_pdf_pages_headings_and_empty_page_warning(tmp_path):
    pdf = make_pdf(tmp_path / "t.pdf", [
        [("Article One", 20), ("The Assembly of Testland shall meet every spring in the capital.", 10)],
        [],  # blank page — simulates a scanned page with no text layer
        [("Article Two", 20), ("The High Court of Testland may review laws.", 10)],
    ])
    blocks, pages, warnings = load_document(pdf, extract_tables=False)
    assert pages == 3
    body = next(b for b in blocks if "spring" in b.text)
    assert body.page_number == 1 and body.section_heading == "Article One"
    court = next(b for b in blocks if "High Court" in b.text and b.block_type == BlockType.paragraph)
    assert court.page_number == 3 and court.section_heading == "Article Two"
    assert any("1 page(s) without extractable text: 2" in w for w in warnings)


def test_pdf_with_no_text_raises(tmp_path):
    pdf = make_pdf(tmp_path / "blank.pdf", [[], []])
    with pytest.raises(ExtractionError, match="no extractable text"):
        load_document(pdf)


def test_corrupt_pdf_raises(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf at all")
    with pytest.raises(ExtractionError):
        load_document(bad)


def test_unsupported_extension(tmp_path):
    f = tmp_path / "x.docx"
    f.write_bytes(b"x")
    with pytest.raises(ExtractionError, match="Unsupported"):
        load_document(f)


def test_legacy_font_mojibake_dropped_with_warning(tmp_path):
    from app.ingestion.loaders import looks_like_legacy_font_text

    assert looks_like_legacy_font_text("£ÉÉ®iÉ BÉEÉ ºÉÆÉÊ´ÉvÉÉxÉ")
    assert not looks_like_legacy_font_text("The Constitution of India")
    assert not looks_like_legacy_font_text("भारत का संविधान")
    f = tmp_path / "mixed.txt"
    f.write_text("£ÉÉ®iÉ BÉEÉ ºÉÆÉÊ´ÉvÉÉxÉ\n\nTHE CHARTER\n\nA normal English paragraph.", encoding="utf-8")
    blocks, _, warnings = load_document(f)
    assert all("£" not in b.text for b in blocks)
    assert any("legacy non-Unicode" in w for w in warnings)


def test_pdf_running_headers_removed_and_inline_part_headings_detected(tmp_path):
    pages = []
    topics = ["harbour tolls", "river ferries", "granary audits", "road repairs", "lamp oil levies"]
    for i, topic in enumerate(topics, start=1):
        pages.append([("TESTLAND STATUTE BOOK", 12), (str(i), 10),
                      (f"This page regulates {topic}. The Assembly shall review the {topic} rules each year.", 10),
                      (f"Officers responsible for {topic} report to the Council.", 10)])
    pages[2].insert(2, ("PART II", 10))
    import pymupdf

    path = make_pdf(tmp_path / "running.pdf", pages)
    # Make "PART II" bold so it is styled like a structural heading.
    doc = pymupdf.open(path)
    page = doc[2]
    hits = page.search_for("PART II")
    page.add_redact_annot(hits[0])
    page.apply_redactions()
    page.insert_text((hits[0].x0, hits[0].y1 - 2), "PART II", fontsize=10, fontname="hebo")
    doc.save(tmp_path / "running2.pdf")
    doc.close()

    blocks, _, warnings = load_document(tmp_path / "running2.pdf", extract_tables=False)
    texts = [b.text for b in blocks]
    assert not any("TESTLAND STATUTE BOOK" in t for t in texts)
    assert not any(t.strip() in {"1", "2", "3"} for t in texts)
    assert any("running header/footer" in w for w in warnings)
    # The re-inserted bold "PART II" is appended to page 3's content stream, so it governs pages 4+.
    after = [b for b in blocks if b.page_number and b.page_number >= 4 and b.block_type == BlockType.paragraph]
    before = [b for b in blocks if b.page_number and b.page_number <= 2]
    assert after and all(b.section_heading == "PART II" for b in after)
    assert all(b.section_heading is None for b in before)
    assert any(b.block_type == BlockType.heading and b.text == "PART II" and b.heading_level == 1 for b in blocks)


def test_html_multiple_parser_outputs_and_inline_chapter_headings():
    html = b"""<html><body><main><h1>Work</h1>
    <h2>Part One</h2><div class="mw-parser-output"><p>CHAPTER I. THE COUNCIL.</p><p>First chapter text.</p></div>
    <h2>Part Two</h2><div class="mw-parser-output"><p>CHAPTER II. THE COURTS.</p><p>Second chapter text.</p>
    <p>Chapter two continues in ordinary sentence case and is not a heading.</p></div></main></body></html>"""
    blocks = load_html(html)
    second = next(b for b in blocks if b.text == "Second chapter text.")
    assert second.section_heading == "Work > Part Two > CHAPTER II. THE COURTS."
    assert any(b.text.startswith("Chapter two continues") and b.block_type == BlockType.paragraph for b in blocks)
