"""Pure rendering helpers: AnswerResponse dict → safe HTML fragments for the chat UI.

No Streamlit imports here, so everything is unit-testable. All model- and source-provided
text is HTML-escaped before markup is added; raw JSON is never emitted.
"""

from __future__ import annotations

import html
import re
from typing import Any

_MARKER_RE = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")

SOURCE_TYPE_LABELS = {
    "primary_legal": "Primary source · legal", "primary_government": "Primary source · government",
    "primary_speech": "Primary source · speech/writing", "primary_inscription": "Primary source · inscription",
    "judicial": "Judicial", "scholarship": "Scholarship", "news": "News", "reference": "Reference",
    "other": "Other",
}


def esc(value: Any) -> str:
    """HTML-escape; also neutralise '$' so Streamlit markdown never treats text as LaTeX."""
    if value is None:
        return ""
    return html.escape(str(value)).replace("$", "&#36;")


def page_ref(c: dict) -> str:
    ps, pe = c.get("page_start"), c.get("page_end")
    if ps is None:
        return ""
    return f"p. {ps}" if pe in (None, ps) else f"pp. {ps}–{pe}"


def anchor_id(msg_id: str, source_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]", "", f"{msg_id}-{source_id}")
    return f"src-{safe}"


def link_markers(text: str, msg_id: str, linkable: set[str]) -> str:
    """Escape text and turn [S1, S2] markers into citation links to that message's source cards."""
    def repl(m: re.Match) -> str:
        out = []
        for sid in (s.strip() for s in m.group(1).split(",")):
            if sid in linkable:
                out.append(f'<a class="cite" href="#{anchor_id(msg_id, sid)}" title="Jump to source {sid}">{sid}</a>')
            else:
                out.append(f'<span class="cite cite-muted">{esc(sid)}</span>')
        return "".join(out)

    parts, last = [], 0
    for m in _MARKER_RE.finditer(text):
        parts.append(esc(text[last:m.start()]))
        parts.append(repl(m))
        last = m.end()
    parts.append(esc(text[last:]))
    return "".join(parts)


def _paragraphs(text: str, msg_id: str, linkable: set[str]) -> str:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    return "".join(f"<p>{link_markers(p, msg_id, linkable).replace(chr(10), '<br>')}</p>" for p in paras)


def _claims(title: str, items: list[dict], msg_id: str, linkable: set[str], icon: str) -> str:
    if not items:
        return ""
    lis = []
    for it in items:
        who = f' <span class="attr">— {esc(it["attributed_to"])}</span>' if it.get("attributed_to") else ""
        cites = "".join(link_markers(f"[{c}]", msg_id, linkable) for c in it.get("citations", []))
        lis.append(f"<li>{link_markers(it.get('statement', ''), msg_id, linkable)}{who} {cites}</li>")
    return f'<div class="section"><h4>{icon} {esc(title)}</h4><ul>{"".join(lis)}</ul></div>'


def answer_html(res: dict, msg_id: str) -> str:
    """Main assistant bubble body for an `answered` response."""
    linkable = {c["source_id"] for c in res.get("retrieved", [])} | {c["source_id"] for c in res.get("citations", [])}
    parts = [f'<div class="answer">{_paragraphs(res.get("answer", ""), msg_id, linkable)}</div>']
    parts.append(_claims("Documented facts", res.get("documented_facts", []), msg_id, linkable, "📌"))
    parts.append(_claims("Interpretations (attributed)", res.get("interpretations", []), msg_id, linkable, "💬"))
    parts.append(_claims("Disputed or uncertain", res.get("disputed_or_uncertain", []), msg_id, linkable, "❓"))
    parts.append(_claims("Where sources disagree", res.get("conflicts", []), msg_id, linkable, "⚖️"))
    gaps = res.get("evidence_gaps") or []
    if gaps:
        lis = "".join(f"<li>{link_markers(g, msg_id, linkable)}</li>" for g in gaps)
        parts.append(f'<div class="section"><h4>🔍 Not covered by the indexed sources</h4><ul>{lis}</ul></div>')
    return "".join(p for p in parts if p)


STATUS_BANNERS = {
    "insufficient_evidence": ("warn", "Not enough evidence",
                              "The indexed sources don't establish a reliable answer. The closest passages are "
                              "listed below so you can judge them yourself."),
    "declined": ("info", "Outside this tool's scope",
                 "This research assistant stays neutral: it doesn't predict elections or endorse parties, "
                 "candidates or ideologies."),
    "error": ("error", "The answer could not be generated",
              "The evidence was retrieved, but the language model step failed. You can still review the "
              "passages below."),
}


def status_html(res: dict, msg_id: str) -> str:
    """Bubble body for non-answered statuses: a banner plus the backend's own explanation."""
    kind, title, hint = STATUS_BANNERS.get(res.get("status"), ("info", "Notice", ""))
    linkable = {c["source_id"] for c in res.get("retrieved", [])}
    body = _paragraphs(res.get("answer", ""), msg_id, linkable)
    gaps = res.get("evidence_gaps") or []
    gaps_html = ("<ul>" + "".join(f"<li>{esc(g)}</li>" for g in gaps) + "</ul>") if gaps else ""
    return (f'<div class="banner banner-{kind}"><strong>{esc(title)}</strong>'
            f'<div class="banner-hint">{esc(hint)}</div></div>{body}{gaps_html}')


def notices_html(notices: list[str]) -> str:
    return "".join(f'<div class="notice">ℹ️ {esc(n)}</div>' for n in notices or [])


def _safe_url(url: str | None) -> str | None:
    return url if url and re.match(r"^https?://", url) else None


def source_card_html(c: dict, msg_id: str, cited: bool) -> str:
    """Compact source card with an anchor target and an expandable <details> block."""
    meta = [esc(c.get("author_or_institution")) if c.get("author_or_institution") else "",
            esc(SOURCE_TYPE_LABELS.get(c.get("source_type"), c.get("source_type") or "")),
            f"published {esc(c['publication_date'])}" if c.get("publication_date") else "publication date unknown",
            esc(page_ref(c))]
    meta_line = " · ".join(m for m in meta if m)
    details = []
    if c.get("section_heading"):
        details.append(f'<div class="kv"><span>Section</span>{esc(c["section_heading"])}</div>')
    if c.get("excerpt"):
        details.append(f'<blockquote>{esc(c["excerpt"])}</blockquote>')
    url = _safe_url(c.get("source_url"))
    if url:
        details.append(f'<a href="{esc(url)}" target="_blank" rel="noopener noreferrer">Open source document ↗</a>'
                       f' <span class="muted">(link from ingestion metadata)</span>')
    badge = "cited" if cited else "retrieved"
    return (f'<div class="source-card" id="{anchor_id(msg_id, c["source_id"])}">'
            f'<div class="source-head"><span class="sid sid-{badge}">{esc(c["source_id"])}</span>'
            f'<span class="stitle">{esc(c.get("title") or c.get("document_id"))}</span></div>'
            f'<div class="smeta">{meta_line}</div>'
            f'<details><summary>Source details</summary>{"".join(details)}</details></div>')


def sources_html(res: dict, msg_id: str) -> str:
    """Cited sources first, then other retrieved passages (collapsed group)."""
    cited_ids = [c["source_id"] for c in res.get("citations", [])]
    by_id = {c["source_id"]: c for c in res.get("retrieved", [])}
    for c in res.get("citations", []):
        by_id.setdefault(c["source_id"], c)
    out = []
    if cited_ids:
        out.append('<div class="sources-title">Sources cited</div>')
        out.extend(source_card_html(by_id[s], msg_id, True) for s in cited_ids if s in by_id)
    others = [c for sid, c in by_id.items() if sid not in set(cited_ids)]
    if others:
        label = "Other retrieved passages" if cited_ids else "Retrieved passages"
        cards = "".join(source_card_html(c, msg_id, False) for c in others)
        out.append(f'<details class="others"><summary>{label} ({len(others)})</summary>{cards}</details>')
    return "".join(out)


def plain_answer_text(res: dict) -> str:
    """The answer prose without markup — used for the typewriter reveal before the final render."""
    return re.sub(r"\s+", " ", res.get("answer", "")).strip()


def reveal_chunks(text: str, max_steps: int = 60) -> list[str]:
    """Progressive prefixes of `text` (word boundaries) for a typewriter-style reveal.

    This only animates an answer that has already been fully generated and validated by the
    backend; it is not token streaming from the model.
    """
    words = text.split(" ")
    if not text:
        return []
    step = max(1, len(words) // max_steps)
    return [" ".join(words[:i]) for i in range(step, len(words), step)] + [text]
