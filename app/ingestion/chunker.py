"""Structure-aware chunking.

Strategy (character-based, configurable):
* Chunks never cross a section-heading boundary.
* Consecutive paragraphs are packed together up to ``chunk_size`` characters,
  so short paragraphs are not split unnecessarily.
* When a new chunk starts inside the same section, the tail of the previous
  chunk (up to ``chunk_overlap`` characters, aligned to a sentence or word
  boundary) is carried over for context.
* Paragraphs longer than ``chunk_size`` are split on sentence boundaries
  (including the Devanagari danda) with the same overlap rule.
* Tables are kept as units; oversized tables are split by rows with the header
  row repeated in every part.
* Every chunk keeps the page range and section heading of its contents.

These defaults are a starting point, not a claim of optimality — evaluate with
``python -m app.evaluation.run_eval`` before changing them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.schemas import BlockType, Chunk, TextBlock

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?।॥])\s+")


@dataclass
class _Part:
    text: str
    page: int | None


@dataclass
class _Buffer:
    section: str | None = None
    parts: list[_Part] = field(default_factory=list)
    carry_only: bool = False  # holds only overlap text carried from the previous chunk

    @property
    def length(self) -> int:
        return sum(len(p.text) for p in self.parts) + max(0, len(self.parts) - 1)

    def text(self) -> str:
        return "\n".join(p.text for p in self.parts)


def _split_sentences(text: str) -> list[str]:
    return [s for s in _SENT_SPLIT_RE.split(text) if s.strip()]


def _tail(text: str, max_chars: int) -> str:
    """Last <= max_chars of text, starting at a sentence boundary if possible, else a word boundary."""
    if max_chars <= 0 or not text:
        return ""
    if len(text) <= max_chars:
        return text
    window = text[-max_chars:]
    m = re.search(r"(?<=[.!?।॥])\s+", window)
    if m and m.end() < len(window):
        return window[m.end():].strip()
    space = window.find(" ")
    return window[space + 1:].strip() if 0 <= space < len(window) - 1 else window.strip()


def _hard_wrap(sentence: str, size: int) -> list[str]:
    """Split a single over-long sentence on word boundaries."""
    out, cur = [], ""
    for word in sentence.split(" "):
        if cur and len(cur) + 1 + len(word) > size:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    if cur:
        out.append(cur)
    # A single "word" longer than size (e.g. no spaces) — slice it.
    final = []
    for piece in out:
        final.extend(piece[i:i + size] for i in range(0, len(piece), size))
    return final


def _split_long_paragraph(text: str, size: int, overlap: int) -> list[str]:
    units: list[str] = []
    for s in _split_sentences(text):
        units.extend(_hard_wrap(s, size) if len(s) > size else [s])
    pieces: list[str] = []
    cur = ""
    for u in units:
        if cur and len(cur) + 1 + len(u) > size:
            pieces.append(cur)
            carry = _tail(cur, overlap)
            cur = f"{carry} {u}".strip() if carry and len(carry) + 1 + len(u) <= size else u
        else:
            cur = f"{cur} {u}".strip()
    if cur:
        pieces.append(cur)
    return pieces


def _split_table(block: TextBlock, size: int) -> list[str]:
    if len(block.text) <= size:
        return [block.text]
    rows = block.text.split("\n")
    header = block.table_header or rows[0]
    body = rows[1:] if rows and rows[0] == header else rows
    pieces, cur = [], [header]
    for row in body:
        if len(cur) > 1 and sum(len(r) + 1 for r in cur) + len(row) > size:
            pieces.append("\n".join(cur))
            cur = [header]
        cur.append(row)
    if len(cur) > 1:
        pieces.append("\n".join(cur))
    return pieces


def chunk_blocks(blocks: list[TextBlock], document_id: str, *, chunk_size: int = 1000,
                 chunk_overlap: int = 150, min_chunk_chars: int = 200) -> list[Chunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    raw: list[dict] = []  # {text, pages, section, table}
    buf = _Buffer()

    def emit(text: str, pages: list[int | None], section: str | None, table: bool = False) -> None:
        if text.strip():
            raw.append({"text": text.strip(), "pages": [p for p in pages if p is not None],
                        "section": section, "table": table})

    def flush(carry_overlap: bool) -> None:
        nonlocal buf
        if not buf.parts:
            return
        emit(buf.text(), [p.page for p in buf.parts], buf.section)
        section = buf.section
        carry = _tail(buf.parts[-1].text, chunk_overlap) if carry_overlap else ""
        last_page = buf.parts[-1].page
        buf = _Buffer(section=section)
        if carry:
            buf.parts.append(_Part(carry, last_page))
            buf.carry_only = True

    def buffer_has_new_content() -> bool:
        return bool(buf.parts) and not buf.carry_only

    for block in blocks:
        if block.block_type == BlockType.heading or block.section_heading != buf.section:
            if buffer_has_new_content():
                flush(carry_overlap=False)
            buf = _Buffer(section=block.section_heading)
            if block.block_type == BlockType.heading:
                continue

        if block.block_type == BlockType.table:
            if buffer_has_new_content():
                flush(carry_overlap=False)
            buf = _Buffer(section=block.section_heading)
            for piece in _split_table(block, chunk_size):
                emit(piece, [block.page_number], block.section_heading, table=True)
            continue

        text = block.text.strip()
        if not text:
            continue

        if len(text) > chunk_size:
            if buffer_has_new_content():
                flush(carry_overlap=False)
            buf = _Buffer(section=block.section_heading)
            for piece in _split_long_paragraph(text, chunk_size, chunk_overlap):
                emit(piece, [block.page_number], block.section_heading)
            continue

        if buf.parts and buf.length + 1 + len(text) > chunk_size:
            if buffer_has_new_content():
                flush(carry_overlap=True)
            else:
                buf = _Buffer(section=buf.section)
            if buf.parts and buf.length + 1 + len(text) > chunk_size:
                buf = _Buffer(section=block.section_heading)
        buf.parts.append(_Part(text, block.page_number))
        buf.carry_only = False

    if buffer_has_new_content():
        flush(carry_overlap=False)

    # Merge undersized non-table chunks into the previous chunk of the same section.
    merged: list[dict] = []
    for item in raw:
        prev = merged[-1] if merged else None
        if (prev and not item["table"] and not prev["table"] and item["section"] == prev["section"]
                and len(item["text"]) < min_chunk_chars
                and len(prev["text"]) + 1 + len(item["text"]) <= int(chunk_size * 1.5)):
            prev["text"] = f"{prev['text']}\n{item['text']}"
            prev["pages"].extend(item["pages"])
        else:
            merged.append(item)

    return [
        Chunk(
            chunk_id=f"{document_id}::c{i:05d}",
            document_id=document_id,
            chunk_index=i,
            text=item["text"],
            page_start=min(item["pages"]) if item["pages"] else None,
            page_end=max(item["pages"]) if item["pages"] else None,
            section_heading=item["section"],
            contains_table=item["table"],
        )
        for i, item in enumerate(merged)
    ]
