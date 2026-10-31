"""Format-specific text extraction into structural TextBlocks.

Each loader returns (blocks, page_count, warnings). Loaders never silently drop
content: pages without extractable text produce warnings, and an entirely empty
document raises ExtractionError.
"""

from __future__ import annotations

import logging
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

from app.schemas import BlockType, TextBlock

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".html", ".htm"}

# Bump when extraction behaviour changes so the next ingestion re-processes files.
LOADER_VERSION = 2


class ExtractionError(RuntimeError):
    pass


def normalize_text(text: str) -> str:
    """NFC-normalise (important for Indic scripts) and tidy whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


# Legacy (pre-Unicode) Hindi fonts such as Kruti Dev map Devanagari glyphs onto
# Latin-1 code points, so extraction yields mojibake like "£ÉÉ®iÉ BÉEÉ". Such lines
# are dropped with a warning rather than indexed as if they were real text.
_LEGACY_FONT_CHARS = re.compile(r"[¡-ÿŒœŠšŸŽžƒˆ˜‘-„†-•…‰‹›€™]")


def looks_like_legacy_font_text(text: str) -> bool:
    letters = [c for c in text if not c.isspace() and not c.isdigit()]
    if len(letters) < 4:
        return False
    return len(_LEGACY_FONT_CHARS.findall(text)) / len(letters) > 0.25


def drop_legacy_font_blocks(blocks: list[TextBlock]) -> tuple[list[TextBlock], int]:
    kept, dropped = [], 0
    for b in blocks:
        if looks_like_legacy_font_text(b.text):
            dropped += 1
        else:
            kept.append(b)
    return kept, dropped


def load_document(path: Path, *, ocr_enabled: bool = False, tesseract_cmd: str | None = None,
                  ocr_lang: str = "eng", extract_tables: bool = True
                  ) -> tuple[list[TextBlock], int | None, list[str]]:
    blocks, pages, warnings = _load_by_type(path, ocr_enabled=ocr_enabled, tesseract_cmd=tesseract_cmd,
                                            ocr_lang=ocr_lang, extract_tables=extract_tables)
    blocks, dropped = drop_legacy_font_blocks(blocks)
    if dropped:
        warnings = warnings + [f"{dropped} block(s) dropped: text appears to use a legacy non-Unicode "
                               "Indic font encoding and could not be decoded reliably"]
    if not blocks:
        raise ExtractionError(f"{path.name}: no usable text after extraction")
    return blocks, pages, warnings


def _load_by_type(path: Path, *, ocr_enabled: bool, tesseract_cmd: str | None, ocr_lang: str,
                  extract_tables: bool) -> tuple[list[TextBlock], int | None, list[str]]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path, ocr_enabled=ocr_enabled, tesseract_cmd=tesseract_cmd,
                        ocr_lang=ocr_lang, extract_tables=extract_tables)
    if ext in {".html", ".htm"}:
        return load_html(path.read_bytes()), None, []
    if ext in {".md", ".markdown"}:
        return load_markdown(_read_text(path)), None, []
    if ext == ".txt":
        return load_plain_text(_read_text(path)), None, []
    raise ExtractionError(f"Unsupported file type: {ext}")


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ExtractionError(f"{path.name}: not valid UTF-8/UTF-16 text")


# ---------------------------------------------------------------------------
# Plain text and Markdown
# ---------------------------------------------------------------------------

_TXT_HEADING_RE = re.compile(
    r"^(PART|CHAPTER|BOOK|SCHEDULE|ARTICLE|SECTION|APPENDIX)\b[\w .,:;\-–—()]*$", re.IGNORECASE
)


def _looks_like_txt_heading(para: str) -> bool:
    if "\n" in para or len(para) > 90 or para.endswith((".", ",", ";")):
        return False
    letters = [c for c in para if c.isalpha()]
    upper = bool(letters) and all(c.isupper() for c in letters if c.isascii())
    return bool(_TXT_HEADING_RE.match(para)) or (upper and len(letters) >= 3)


def load_plain_text(text: str) -> list[TextBlock]:
    text = normalize_text(text)
    blocks: list[TextBlock] = []
    heading: str | None = None
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if _looks_like_txt_heading(para):
            heading = para
            blocks.append(TextBlock(text=para, block_type=BlockType.heading,
                                    section_heading=heading, heading_level=1))
        else:
            blocks.append(TextBlock(text=re.sub(r"\s*\n\s*", " ", para),
                                    section_heading=heading))
    return blocks


_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_MD_TABLE_SEP_RE = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?$")


def load_markdown(text: str) -> list[TextBlock]:
    text = normalize_text(text)
    blocks: list[TextBlock] = []
    heading_stack: dict[int, str] = {}
    current_heading: str | None = None
    para_lines: list[str] = []
    lines = text.split("\n")

    def flush_para() -> None:
        if para_lines:
            blocks.append(TextBlock(text=" ".join(l.strip() for l in para_lines),
                                    section_heading=current_heading))
            para_lines.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        m = _MD_HEADING_RE.match(line)
        if m:
            flush_para()
            level = len(m.group(1))
            heading_stack = {k: v for k, v in heading_stack.items() if k < level}
            heading_stack[level] = m.group(2)
            current_heading = " > ".join(heading_stack[k] for k in sorted(heading_stack))
            blocks.append(TextBlock(text=m.group(2), block_type=BlockType.heading,
                                    section_heading=current_heading, heading_level=level))
            i += 1
            continue
        # Markdown table: header row followed by separator row
        if line.lstrip().startswith("|") and i + 1 < len(lines) and _MD_TABLE_SEP_RE.match(lines[i + 1].strip()):
            flush_para()
            header = line.strip()
            rows = []
            i += 2
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            blocks.append(TextBlock(text="\n".join([header] + rows), block_type=BlockType.table,
                                    section_heading=current_heading, table_header=header))
            continue
        if not line.strip():
            flush_para()
        else:
            para_lines.append(line)
        i += 1
    flush_para()
    return blocks


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_HTML_DROP = ["script", "style", "noscript", "nav", "footer", "header", "form", "aside", "sup.reference"]
_HTML_DROP_CLASSES = ["mw-editsection", "navbox", "reference", "references", "noprint",
                      "ws-noexport", "licenseContainer", "mw-jump-link", "catlinks"]


_INLINE_HEADING_LEVEL = 7  # below h1–h6
_INLINE_HEADING_RE = re.compile(r"^(CHAPTER|PART|BOOK|SCHEDULE|SECTION|APPENDIX)\b", re.IGNORECASE)


def _looks_like_inline_heading(text: str) -> bool:
    """A short, mostly upper-case paragraph starting with CHAPTER/PART/... is a heading."""
    if len(text) > 100 or not _INLINE_HEADING_RE.match(text):
        return False
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.8


def load_html(raw: bytes) -> list[TextBlock]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw, "lxml")
    for sel in _HTML_DROP:
        for el in soup.select(sel):
            el.decompose()
    for cls in _HTML_DROP_CLASSES:
        for el in soup.find_all(class_=cls):
            el.decompose()

    # Prefer <main>/<article>; a page may hold several mw-parser-output divs (e.g. a
    # multi-chapter Wikisource export), so that selector is only a fallback.
    root = (soup.find("main") or soup.find("article") or soup.select_one("div.mw-parser-output")
            or soup.body or soup)

    blocks: list[TextBlock] = []
    heading_stack: dict[int, str] = {}
    current_heading: str | None = None
    tags = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "table", "dd", "pre"]

    containers = ["table", "blockquote", "li", "dd", "p", "pre"]
    for el in root.find_all(tags):
        # Avoid double-counting text nested inside an already-captured block.
        if el.find_parent(containers) is not None:
            continue
        if el.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            text = normalize_text(el.get_text(" ", strip=True))
            if not text:
                continue
            level = int(el.name[1])
            heading_stack = {k: v for k, v in heading_stack.items() if k < level}
            heading_stack[level] = text
            current_heading = " > ".join(heading_stack[k] for k in sorted(heading_stack))
            blocks.append(TextBlock(text=text, block_type=BlockType.heading,
                                    section_heading=current_heading, heading_level=level))
        elif el.name == "table":
            rows = []
            for tr in el.find_all("tr"):
                cells = [normalize_text(c.get_text(" ", strip=True)) for c in tr.find_all(["th", "td"])]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                blocks.append(TextBlock(text="\n".join(rows), block_type=BlockType.table,
                                        section_heading=current_heading, table_header=rows[0]))
        else:
            text = normalize_text(el.get_text(" ", strip=True))
            if text and el.name == "p" and _looks_like_inline_heading(text):
                # e.g. Wikisource "<p>CHAPTER VIII. CREATION OF MINISTERS.</p>": nest below <h*> levels.
                heading_stack = {k: v for k, v in heading_stack.items() if k < _INLINE_HEADING_LEVEL}
                heading_stack[_INLINE_HEADING_LEVEL] = text
                current_heading = " > ".join(heading_stack[k] for k in sorted(heading_stack))
                blocks.append(TextBlock(text=text, block_type=BlockType.heading,
                                        section_heading=current_heading, heading_level=_INLINE_HEADING_LEVEL))
            elif text:
                blocks.append(TextBlock(text=text, section_heading=current_heading))
    if not blocks:
        text = normalize_text(root.get_text("\n", strip=True))
        if text:
            blocks = load_plain_text(text)
    return blocks


# ---------------------------------------------------------------------------
# PDF (PyMuPDF)
# ---------------------------------------------------------------------------

# Short lines repeated on many pages (running headers/footers) are page furniture,
# not content or section headings.
_FURNITURE_MAX_CHARS = 80
_FURNITURE_PAGE_SHARE = 0.04
_PAGE_NUMBER_RE = re.compile(r"^[\(\[]?(\d{1,4}|[ivxlcdm]{1,7})[\)\]]?$", re.IGNORECASE)


def _furniture_key(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower()).strip()


def _text_extent(blocks: list[dict]) -> tuple[float, float]:
    ys = [(b["bbox"][1], b["bbox"][3]) for b in blocks if b.get("type") == 0]
    return (min(y[0] for y in ys), max(y[1] for y in ys)) if ys else (0.0, 0.0)


def _in_margin(bbox, extent: tuple[float, float], share: float = 0.07) -> bool:
    """True when a block sits in the top or bottom band of the page's text area
    (measured against the text extent, since printed pages often have wide margins)."""
    top, bottom = extent
    band = (bottom - top) * share
    return bbox[1] <= top + band or bbox[3] >= bottom - band


_STRUCTURAL_L1_RE = re.compile(r"^(PART\s+[IVXLC]+[A-Z]?|[A-Z]+\s+SCHEDULE|SCHEDULE|APPENDIX|BOOK)\b", re.I)
_STRUCTURAL_L2_RE = re.compile(r"^(CHAPTER|SECTION|TITLE)\b", re.I)


def _heading_level(line: dict, text: str, body_size: float) -> int | None:
    """Heading level for a PDF line (1 = PART/SCHEDULE, 2 = CHAPTER, 3 = other), or None for body text."""
    if len(text) > 100 or text.endswith((",", ";")):
        return None
    spans = [s for s in line.get("spans", []) if s["text"].strip()]
    if not spans:
        return None
    max_size = max(s["size"] for s in spans)
    bold = all("bold" in s.get("font", "").lower() or (s.get("flags", 0) & 16) for s in spans)
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 2:
        return None
    upper = sum(c.isupper() for c in letters) / len(letters) > 0.8
    if _STRUCTURAL_L1_RE.match(text) and (bold or upper or max_size >= body_size * 1.1):
        return 1
    if _STRUCTURAL_L2_RE.match(text) and (bold or upper):
        return 2
    if max_size >= body_size * 1.2:
        return 1
    if bold and len(text) <= 80 and len(letters) >= 3:
        return 3
    return None


def _is_page_noise(text: str) -> bool:
    t = text.strip()
    return not re.search(r"[^\W_]", t) or bool(_PAGE_NUMBER_RE.match(t))


def load_pdf(path: Path, *, ocr_enabled: bool = False, tesseract_cmd: str | None = None,
             ocr_lang: str = "eng", extract_tables: bool = True
             ) -> tuple[list[TextBlock], int, list[str]]:
    import pymupdf as fitz

    warnings: list[str] = []
    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise ExtractionError(f"{path.name}: cannot open PDF ({exc})") from exc
    if doc.needs_pass:
        raise ExtractionError(f"{path.name}: PDF is encrypted/password-protected")

    # First pass: body font size (for heading detection) and running headers/footers.
    sizes: list[float] = []
    short_text_pages: Counter[str] = Counter()
    for page in doc:
        seen_on_page: set[str] = set()
        page_blocks = page.get_text("dict")["blocks"]
        extent = _text_extent(page_blocks)
        for b in page_blocks:
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    if span["text"].strip():
                        sizes.append(round(span["size"], 1))
            text = normalize_text(" ".join("".join(s["text"] for s in line.get("spans", []))
                                           for line in b.get("lines", [])))
            if text and len(text) <= _FURNITURE_MAX_CHARS and _in_margin(b["bbox"], extent):
                seen_on_page.add(_furniture_key(text))
        short_text_pages.update(seen_on_page)
    body_size = statistics.median(sizes) if sizes else 10.0
    min_pages = max(3, int(doc.page_count * _FURNITURE_PAGE_SHARE))
    furniture = {k for k, n in short_text_pages.items() if n >= min_pages and k}
    dropped_furniture = 0

    blocks: list[TextBlock] = []
    current_heading: str | None = None
    heading_stack: dict[int, str] = {}
    empty_pages: list[int] = []

    for page_index, page in enumerate(doc):
        page_no = page_index + 1
        table_rects = []
        if extract_tables:
            try:
                for table in page.find_tables().tables:
                    rows = table.extract()
                    lines = [" | ".join(normalize_text(c or "") for c in row) for row in rows]
                    lines = [l for l in lines if l.strip(" |")]
                    if len(lines) >= 2:
                        table_rects.append(fitz.Rect(table.bbox))
                        blocks.append(TextBlock(text="\n".join(lines), block_type=BlockType.table,
                                                page_number=page_no, section_heading=current_heading,
                                                table_header=lines[0]))
            except Exception as exc:  # table detection is best-effort
                warnings.append(f"page {page_no}: table detection failed ({exc}); text kept as paragraphs")
                table_rects = []

        page_text_found = False
        page_blocks = page.get_text("dict")["blocks"]
        extent = _text_extent(page_blocks)
        for b in page_blocks:
            if b.get("type") != 0:
                continue
            rect = fitz.Rect(b["bbox"])
            if any(rect.intersects(t) and (rect & t).get_area() > 0.5 * rect.get_area() for t in table_rects):
                continue
            text = normalize_text("\n".join(
                "".join(s["text"] for s in line.get("spans", [])) for line in b.get("lines", [])
            ))
            if not text:
                continue
            if (_is_page_noise(text)
                    or (len(text) <= _FURNITURE_MAX_CHARS and _in_margin(b["bbox"], extent)
                        and _furniture_key(text.replace("\n", " ")) in furniture)):
                dropped_furniture += 1
                continue
            page_text_found = True
            # Classify line by line: headings often share a block with body text.
            para_lines: list[str] = []
            heading_lines: list[tuple[str, int]] = []

            def flush_para() -> None:
                if para_lines:
                    t = re.sub(r"(\w)-\n(\w)", r"\1\2", "\n".join(para_lines)).replace("\n", " ")
                    blocks.append(TextBlock(text=normalize_text(t), page_number=page_no,
                                            section_heading=current_heading))
                    para_lines.clear()

            def flush_heading() -> None:
                nonlocal heading_stack, current_heading
                if heading_lines:
                    level = min(lv for _, lv in heading_lines)
                    htext = normalize_text(" ".join(t for t, _ in heading_lines))
                    heading_stack = {k: v for k, v in heading_stack.items() if k < level}
                    heading_stack[level] = htext
                    current_heading = " > ".join(heading_stack[k] for k in sorted(heading_stack))
                    blocks.append(TextBlock(text=htext, block_type=BlockType.heading, page_number=page_no,
                                            section_heading=current_heading, heading_level=level))
                    heading_lines.clear()

            for line in b.get("lines", []):
                line_text = normalize_text("".join(s["text"] for s in line.get("spans", [])))
                if not line_text or _is_page_noise(line_text):
                    continue  # empty lines, separator rules, stray page numbers
                level = _heading_level(line, line_text, body_size)
                if level:
                    flush_para()
                    heading_lines.append((line_text, level))
                else:
                    flush_heading()
                    para_lines.append(line_text)
            flush_heading()
            flush_para()

        if not page_text_found and not any(bl.page_number == page_no for bl in blocks):
            if ocr_enabled:
                from app.ingestion.ocr import ocr_pdf_page
                ocr_text = normalize_text(ocr_pdf_page(page, tesseract_cmd, lang=ocr_lang))
                if ocr_text:
                    for para in re.split(r"\n\s*\n", ocr_text):
                        if para.strip():
                            blocks.append(TextBlock(text=para.replace("\n", " "), page_number=page_no,
                                                    section_heading=current_heading))
                    warnings.append(f"page {page_no}: text obtained via OCR (verify accuracy)")
                else:
                    empty_pages.append(page_no)
            else:
                empty_pages.append(page_no)

    page_count = doc.page_count
    doc.close()
    if furniture:
        warnings.append(f"{dropped_furniture} running header/footer or page-number block(s) removed "
                        f"({len(furniture)} distinct repeated line(s), e.g. {sorted(furniture)[0]!r})")
    if empty_pages:
        hint = "" if ocr_enabled else " (scanned? set OCR_ENABLED=true to try OCR)"
        shown = ", ".join(map(str, empty_pages[:20])) + (" ..." if len(empty_pages) > 20 else "")
        warnings.append(f"{len(empty_pages)} page(s) without extractable text: {shown}{hint}")
    if not blocks:
        raise ExtractionError(f"{path.name}: no extractable text"
                              + ("" if ocr_enabled else " — the PDF may be scanned; enable OCR"))
    return blocks, page_count, warnings
