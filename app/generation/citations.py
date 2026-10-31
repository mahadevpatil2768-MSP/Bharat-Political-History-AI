"""Citation construction and validation.

Only IDs that map to chunks actually retrieved for this question survive.
Claims left without any valid citation are removed (and reported), because an
uncited statement cannot be traced to evidence.
"""

from __future__ import annotations

import re
from typing import Any

from app.schemas import Citation, Claim, RetrievedChunk

_MARKER_RE = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")


def to_citation(source_id: str, chunk: RetrievedChunk, excerpt_chars: int = 400) -> Citation:
    m = chunk.metadata
    text = chunk.text.strip()
    return Citation(
        source_id=source_id, chunk_id=chunk.chunk_id, document_id=chunk.document_id,
        title=m.get("title", chunk.document_id), author_or_institution=m.get("author_or_institution"),
        source_type=m.get("source_type"), publication_date=m.get("publication_date"),
        source_url=m.get("source_url"), page_start=m.get("page_start"), page_end=m.get("page_end"),
        section_heading=m.get("section_heading"),
        excerpt=text[:excerpt_chars] + ("…" if len(text) > excerpt_chars else ""),
    )


def _norm_id(s: str) -> str:
    return s.strip().strip("[]").upper()


def clean_text_markers(text: str, valid: set[str], removed: set[str]) -> str:
    def repl(m: re.Match) -> str:
        ids = [_norm_id(x) for x in m.group(1).split(",")]
        keep = [i for i in ids if i in valid]
        removed.update(i for i in ids if i not in valid)
        return f"[{', '.join(keep)}]" if keep else ""
    return re.sub(r"\s+([.,;:])", r"\1", _MARKER_RE.sub(repl, text)).strip()


def validate_claims(items: list[dict[str, Any]] | None, valid: set[str], removed: set[str],
                    dropped: list[str]) -> list[Claim]:
    out: list[Claim] = []
    for item in items or []:
        cits: list[str] = []
        for c in item.get("citations", []) or []:
            cid = _norm_id(str(c))
            if cid in valid:
                cits.append(cid)
            else:
                removed.add(cid)
        cits = list(dict.fromkeys(cits))
        statement = clean_text_markers(str(item.get("statement", "")), valid, removed)
        if not cits:
            # Fall back to markers embedded in the statement itself.
            cits = list(dict.fromkeys(i for m in _MARKER_RE.finditer(statement)
                                      for i in (_norm_id(x) for x in m.group(1).split(",")) if i in valid))
        if not cits or not statement:
            if statement:
                dropped.append(statement)
            continue
        out.append(Claim(statement=statement, citations=cits,
                         attributed_to=(item.get("attributed_to") or None)))
    return out


def cited_ids(text: str, claims: list[Claim]) -> list[str]:
    ids = [_norm_id(x) for m in _MARKER_RE.finditer(text) for x in m.group(1).split(",")]
    for c in claims:
        ids.extend(c.citations)
    return list(dict.fromkeys(ids))
