"""Legal-structure annotation for numbered provisions (Articles) and article-aware query parsing.

Statute text (e.g. the Constitution of India) writes a provision as ``14. Equality before law.—…``
without the word "Article", so neither BM25 nor embeddings link "Article 14" to it reliably. This
module derives, per chunk and without any LLM:

* ``article_headings`` — articles whose heading starts in this chunk (inline ``14. Title.—``, or the
  chunk's section heading when the PDF loader promoted the title line to a heading);
* ``article_spans``    — articles whose text appears in this chunk (headings plus continuation of the
  article open at the end of the previous chunk);
* ``article_refs``     — articles merely referred to ("article 14", "articles 14 and 19");
* ``article_refs_substantive`` — references in a derogating/qualifying context
  ("notwithstanding …", "inconsistent with … article 14", "shall not apply", "suspended" …);
* ``is_contents``      — table-of-contents / front-matter chunks.

Only provisions inside a PART count as articles; Schedule and Appendix paragraphs (which reuse the
same numbering style) and amendment footnotes (``2. Subs. by …``) are excluded.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

LEGAL_STRUCTURE_VERSION = 1

_NUM = r"\d{1,3}[A-Z]{0,3}"
# Inline heading: "14. Equality before law.—", "2[21A. Right to education.—", "368. 1[Power … therefor].—".
# The title may not run across another numbered entry (". 33. …"), which keeps contents lists out.
_HEADING_RE = re.compile(
    rf"(?:(?<=\s)|^|(?<=\[))(?P<num>{_NUM})\.\s+(?:\d+\[)?\[?"
    rf"(?P<title>[A-Z](?:(?!\.\s+\[?\d+\[?{_NUM}\.\s)(?!\.\s+{_NUM}\.\s)[^—\n]){{1,220}}?)"
    r"\s*\.?\s*\]?\s*\.?\s*—(?P<after>.{0,12})"
)
_SECTION_ARTICLE_RE = re.compile(rf"^(?:\d+\[)?(?P<num>{_NUM})\.\s+\[?(?P<title>[A-Z].{{2,200}})$")
_FOOTNOTE_TITLE_RE = re.compile(r"^(Subs|Sub|Ins|Inserted|Omitted|Added|Rep|Repealed|Substituted|Renumbered|"
                                r"The words|Cl\.|Clause|Vide|See|w\.e\.f)\b", re.I)
_REF_RE = re.compile(rf"\barticles?\s+(?P<list>{_NUM}(?:\s*(?:,|and|or|to)\s*(?:article\s+)?{_NUM})*)", re.I)
_REF_NUM_RE = re.compile(rf"\b{_NUM}\b")
_SUBSTANTIVE_RE = re.compile(
    r"notwithstanding|inconsistent with|takes away or abridges|deemed to be void|void on the ground|"
    r"shall not apply|shall not be called in question|in derogation|suspen(?:d|sion)|nothing in (?:this )?article|"
    r"by way of exception|except as otherwise provided", re.I)
_CONTENTS_ENTRY_RE = re.compile(rf"(?:^|\s)\[?{_NUM}\.\s+[A-Z][^.—]{{3,140}}\.(?=\s|\]|$)")
_CONTENTS_TABLE_RE = re.compile(rf"(?:^|\s|\[){_NUM}\s*\.\s*\|")
_QUERY_ARTICLE_RE = re.compile(rf"\b(?:article|art\.?|arts?\.?)\s*(?P<num>\d{{1,3}}[A-Za-z]{{0,3}})\b", re.I)
_LOOKUP_FILLER = frozenset(
    "what is are does do say says said mean means meaning explain explanation describe tell me about the a an "
    "of in on under constitution indian india's india provide provides provision provisions text content "
    "contents read reads state states article art arts please give how why when where who which can".split())


def norm_article(num: str) -> str:
    return num.upper()


def _body_section(section_heading: str | None) -> bool:
    """True when a chunk sits under a PART (articles), False under a Schedule/Appendix/unknown top level."""
    top = (section_heading or "").split(" > ")[0].strip().upper()
    return top.startswith("PART")


def _section_article(section_heading: str | None) -> str | None:
    last = (section_heading or "").split(" > ")[-1].strip()
    m = _SECTION_ARTICLE_RE.match(last)
    if m and not _FOOTNOTE_TITLE_RE.match(m.group("title")):
        return norm_article(m.group("num"))
    return None


def inline_headings(text: str) -> list[tuple[int, str, str]]:
    """[(position, article number, title)] of inline article headings in chunk text."""
    out = []
    for m in _HEADING_RE.finditer(text.replace("\n", " ")):
        title = m.group("title").strip()
        if _FOOTNOTE_TITLE_RE.match(title):
            continue
        out.append((m.start("num"), norm_article(m.group("num")), title.rstrip(".] ")))
    return out


def article_refs(text: str) -> tuple[set[str], set[str]]:
    """(all referenced articles, those referenced in a substantive/derogating context)."""
    flat = text.replace("\n", " ")
    refs, substantive = set(), set()
    for m in _REF_RE.finditer(flat):
        nums = {norm_article(n) for n in _REF_NUM_RE.findall(m.group("list"))}
        refs |= nums
        window = flat[max(0, m.start() - 220): m.end() + 120]
        if _SUBSTANTIVE_RE.search(window):
            substantive |= nums
    return refs, substantive


def contents_like(text: str) -> bool:
    """A front-matter chunk that lists provisions rather than stating them: mostly numbered title
    entries ("14. Equality before law."), or a short title line with little lowercase prose."""
    flat = text.replace("\n", " ").strip()
    if not flat:
        return True
    coverage = sum(len(m.group(0)) for m in _CONTENTS_ENTRY_RE.finditer(flat)) / len(flat)
    lowercase_ratio = len(re.findall(r"\b[a-z]{3,}\b", flat)) / max(len(flat.split()), 1)
    # Contents pages extracted as tables by PDF table detection: "40 . | Organisationof… . 41 . | …"
    table_entries = len(_CONTENTS_TABLE_RE.findall(flat))
    return (coverage >= 0.4 or (len(flat) < 450 and lowercase_ratio < 0.3) or table_entries >= 3
            or bool(re.match(r"^\(?[ivxlc]*\)?\s*\|?\s*Contents\b", flat)))


def annotate_document(chunks: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Annotate one document's chunks.

    ``chunks``: dicts with chunk_id, chunk_index, text, page_start, section_heading.
    Returns chunk_id -> metadata additions (Chroma-compatible scalars; empty lists omitted).
    """
    ordered = sorted(chunks, key=lambda c: c["chunk_index"])
    # Front matter ends where the first real (non-"Omitted") article heading with body text appears.
    body_start_page = None
    for c in ordered:
        if not _body_section(c.get("section_heading")):
            continue
        for m in _HEADING_RE.finditer(c["text"].replace("\n", " ")):
            if not m.group("after").lstrip(" ]").lower().startswith("omitted") \
                    and not _FOOTNOTE_TITLE_RE.match(m.group("title")):
                body_start_page = c.get("page_start")
                break
        if body_start_page is not None:
            break

    out: dict[str, dict[str, Any]] = {}
    open_article: str | None = None
    prev_section: str | None = None
    for c in ordered:
        text, page, section = c["text"], c.get("page_start"), c.get("section_heading")
        is_contents = bool(body_start_page and page is not None and page < body_start_page
                           and contents_like(text))
        refs, substantive = article_refs(text)
        headings: list[str] = []
        spans: list[str] = []
        if not is_contents and _body_section(section):
            events: list[tuple[int, str]] = []
            sec_article = _section_article(section)
            if sec_article and section != prev_section:
                events.append((0, sec_article))  # title promoted to the section heading by the loader
            events += [(pos, num) for pos, num, _ in inline_headings(text)]
            events.sort()
            first_pos = events[0][0] if events else len(text)
            if open_article and len(text[:first_pos].strip()) >= 40:
                spans.append(open_article)  # continuation of the article open at the previous chunk's end
            for _, num in events:
                if num not in headings:
                    headings.append(num)
                if num not in spans:
                    spans.append(num)
            if events:
                open_article = events[-1][1]
        elif not _body_section(section):
            open_article = None
        prev_section = section

        meta: dict[str, Any] = {"legal_structure_version": LEGAL_STRUCTURE_VERSION, "is_contents": is_contents}
        for key, vals in (("article_headings", headings), ("article_spans", spans),
                          ("article_refs", sorted(refs)), ("article_refs_substantive", sorted(substantive))):
            if vals:
                meta[key] = "|" + "|".join(vals) + "|"
        out[c["chunk_id"]] = meta
    return out


def has(meta: dict[str, Any], key: str, article: str) -> bool:
    return f"|{article}|" in (meta.get(key) or "")


# ---------------------------------------------------------------------------
# Query side
# ---------------------------------------------------------------------------

def query_articles(query: str) -> list[str]:
    """Article numbers named in a query: 'Article 14', 'Art. 21A', 'article 32'."""
    seen: list[str] = []
    for m in _QUERY_ARTICLE_RE.finditer(query):
        n = norm_article(m.group("num"))
        if n not in seen:
            seen.append(n)
    return seen


def content_terms(query: str) -> list[str]:
    """Words beyond the article reference and question boilerplate (e.g. 'prisoners' in
    'How does Article 21 apply to prisoners?'). Empty for a pure lookup like 'What is Article 14?'."""
    q = _QUERY_ARTICLE_RE.sub(" ", query)
    return [w for w in re.findall(r"[A-Za-zऀ-෿]+", q.lower()) if w not in _LOOKUP_FILLER and len(w) > 2]
