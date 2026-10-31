from app.ingestion.chunker import chunk_blocks
from app.schemas import BlockType, TextBlock


def P(text, page=1, section="S"):
    return TextBlock(text=text, page_number=page, section_heading=section)


def test_short_paragraphs_are_packed_not_split():
    blocks = [P("Alpha sentence one."), P("Beta sentence two."), P("Gamma sentence three.")]
    chunks = chunk_blocks(blocks, "d", chunk_size=400, chunk_overlap=50, min_chunk_chars=0)
    assert len(chunks) == 1
    assert "Alpha" in chunks[0].text and "Gamma" in chunks[0].text


def test_chunks_respect_size_and_section_boundaries():
    para = "Word " * 60  # ~300 chars
    blocks = [P(para.strip(), 1, "A"), P(para.strip(), 2, "A"), P(para.strip(), 3, "B")]
    chunks = chunk_blocks(blocks, "d", chunk_size=400, chunk_overlap=50, min_chunk_chars=0)
    assert all(len(c.text) <= 400 + 50 for c in chunks)
    assert {c.section_heading for c in chunks} == {"A", "B"}
    for c in chunks:
        assert c.section_heading in {"A", "B"}
    b_chunks = [c for c in chunks if c.section_heading == "B"]
    assert b_chunks and all(c.page_start == 3 for c in b_chunks)


def test_overlap_carried_within_section():
    s1 = "First block sentence ends here. " * 8
    s2 = "Second block starts now. " * 8
    chunks = chunk_blocks([P(s1.strip(), 1), P(s2.strip(), 2)], "d",
                          chunk_size=300, chunk_overlap=60, min_chunk_chars=0)
    assert len(chunks) == 2
    assert "First block sentence ends here." in chunks[1].text  # overlap tail
    assert chunks[1].page_start == 1 and chunks[1].page_end == 2


def test_long_paragraph_split_on_sentences_with_pages():
    text = " ".join(f"Sentence number {i} is here." for i in range(80))
    chunks = chunk_blocks([P(text, 7)], "d", chunk_size=300, chunk_overlap=50, min_chunk_chars=0)
    assert len(chunks) > 3
    assert all(c.page_start == 7 == c.page_end for c in chunks)
    assert all(len(c.text) <= 300 for c in chunks)
    # Sentence-aligned: no chunk starts mid-word.
    assert all(c.text[0].isupper() or c.text[0].isdigit() for c in chunks)


def test_devanagari_sentence_split():
    text = "यह पहला वाक्य है। " * 40
    chunks = chunk_blocks([P(text.strip())], "d", chunk_size=200, chunk_overlap=30, min_chunk_chars=0)
    assert len(chunks) > 1
    assert all(c.text.endswith("।") for c in chunks[:-1])


def test_table_split_repeats_header():
    header = "Province | Revenue"
    rows = [f"Province{i} | {i * 10}" for i in range(60)]
    table = TextBlock(text="\n".join([header] + rows), block_type=BlockType.table,
                      page_number=4, section_heading="T", table_header=header)
    chunks = chunk_blocks([table], "d", chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(c.text.splitlines()[0] == header for c in chunks)
    assert all(c.contains_table and c.page_start == 4 for c in chunks)
    joined = "\n".join(c.text for c in chunks)
    assert all(r in joined for r in rows)  # no row lost


def test_tiny_trailing_chunk_merged():
    blocks = [P("x " * 180), P("Tiny tail.")]
    chunks = chunk_blocks(blocks, "d", chunk_size=380, chunk_overlap=0, min_chunk_chars=50)
    assert len(chunks) == 1 and chunks[0].text.endswith("Tiny tail.")


def test_chunk_ids_stable_and_ordered():
    blocks = [P("a " * 150, 1), P("b " * 150, 2), P("c " * 150, 3)]
    c1 = chunk_blocks(blocks, "doc_x", chunk_size=320, chunk_overlap=0, min_chunk_chars=0)
    c2 = chunk_blocks(blocks, "doc_x", chunk_size=320, chunk_overlap=0, min_chunk_chars=0)
    assert [c.chunk_id for c in c1] == [c.chunk_id for c in c2]
    assert c1[0].chunk_id == "doc_x::c00000"
    assert [c.chunk_index for c in c1] == list(range(len(c1)))
