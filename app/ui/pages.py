"""Secondary views: source search, timeline, library & ingestion (unchanged behaviour, shared client)."""

from __future__ import annotations

import json

import requests
import streamlit as st

from app.ui import render
from app.ui.api_client import ApiClient, ApiError, _detail

DOC_FIELDS = [("author_or_institution", "Author / institution"), ("source_type", "Source type"),
              ("publication_date", "Published"), ("date_accessed", "Accessed"),
              ("historical_periods", "Periods"), ("event_year_start", "Covers from"),
              ("event_year_end", "Covers to"), ("language", "Language"), ("page_count", "Pages"),
              ("chunk_count", "Indexed chunks"), ("document_version", "Version"),
              ("rights_statement", "Rights basis"), ("topics", "Topics"), ("file_path", "File")]


def sidebar_filters(opts: dict) -> dict:
    """Filter widgets; call inside a sidebar container."""
    periods = opts.get("periods", {})
    sel_periods = st.multiselect("Historical period", list(periods), format_func=lambda k: periods.get(k, k))
    use_years = st.checkbox("Limit by event years")
    year_from = year_to = None
    if use_years:
        c1, c2 = st.columns(2)
        year_from = c1.number_input("From", value=1947, min_value=-3000, max_value=2100, step=1)
        year_to = c2.number_input("To", value=1977, min_value=-3000, max_value=2100, step=1)
        st.caption("Documents without event-year metadata are excluded when this is on.")
    source_types = st.multiselect("Source type", opts.get("source_types", []),
                                  format_func=lambda s: render.SOURCE_TYPE_LABELS.get(s, s))
    topics = opts.get("topics", [])
    topic = st.selectbox("Topic", [""] + topics, format_func=lambda t: t or "Any") if topics else ""
    person = st.text_input("Person (metadata match)")
    institution = st.text_input("Institution / party (metadata match)")
    languages = opts.get("languages", [])
    langs = st.multiselect("Language", languages) if languages else []
    filters = {"periods": sel_periods, "source_types": source_types, "languages": langs,
               "topic": topic or None, "person": person or None, "institution": institution or None,
               "year_from": int(year_from) if year_from is not None else None,
               "year_to": int(year_to) if year_to is not None else None}
    return {k: v for k, v in filters.items() if v not in (None, [], "")}


def page_search(client: ApiClient, filters: dict) -> None:
    st.markdown("Search the indexed documents directly — retrieval only, no AI generation.")
    with st.form("search"):
        c1, c2, c3 = st.columns([5, 2, 1])
        q = c1.text_input("Search terms")
        mode = c2.selectbox("Mode", ["hybrid", "keyword", "vector"])
        k = c3.number_input("Results", 1, 50, 10)
        go = st.form_submit_button("Search")
    if not go or not q.strip():
        return
    try:
        res = client.post("/search", {"query": q, "filters": filters, "top_k": int(k), "mode": mode})
    except ApiError as exc:
        st.error(exc.user_message)
        return
    if not res.get("hits"):
        st.info("No matching passages in the indexed documents (with the current filters).")
        return
    st.caption(f"{res['count']} passages. {res['score_note']}")
    for i, h in enumerate(res["hits"]):
        st.markdown(render.source_card_html(h["citation"], f"search{i}", cited=False), unsafe_allow_html=True)
        st.caption(f"RRF {h['fused_score']} · vector rank {h['vector_rank']} (cos {h['vector_similarity']}) · "
                   f"BM25 rank {h['bm25_rank']} ({h['bm25_score']})")


def page_timeline(client: ApiClient) -> None:
    try:
        tl = client.get("/timeline")
    except ApiError as exc:
        st.error(exc.user_message)
        return
    st.markdown("Events appear here only when their evidence quote is found in an indexed document. "
                "Document spans show the years each indexed source covers (from its metadata).")
    if tl["events"]:
        st.markdown("#### Source-verified events")
        for e in tl["events"]:
            with st.expander(f"{e['date']} — {e['title']}"):
                if e.get("description"):
                    st.write(e["description"])
                st.markdown(f"<blockquote>{render.esc(e['evidence_quote'])}</blockquote>", unsafe_allow_html=True)
                st.markdown(f'<div class="smeta">{render.esc(e["source_title"])} · {render.page_ref(e)}</div>',
                            unsafe_allow_html=True)
                if e.get("source_url"):
                    st.markdown(f"[Open source document]({e['source_url']})")
    else:
        st.info("No curated events yet. Add entries to data/timeline/events.json (see docs/SOURCE_INGESTION.md).")
    if tl["document_spans"]:
        st.markdown("#### Coverage of indexed documents")
        st.dataframe([{"From": s["year"], "To": s["year_end"], "Document": s["title"],
                       "Type": s["source_type"], "Periods": ", ".join(s["periods"])}
                      for s in tl["document_spans"]], hide_index=True, width="stretch")
    if tl["unverified"]:
        with st.expander(f"{len(tl['unverified'])} timeline entr(ies) hidden — could not be verified"):
            for u in tl["unverified"]:
                st.write(f"- {u}")


def _fmt(v) -> str:
    return ", ".join(map(str, v)) if isinstance(v, list) else str(v)


def page_library(client: ApiClient) -> None:
    try:
        docs = client.documents()
        status = client.get("/ingest/status")
    except ApiError as exc:
        st.error(exc.user_message)
        return
    s = status["summary"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Indexed documents", s["indexed_documents"])
    c2.metric("Indexed chunks", status["index_chunks"])
    c3.metric("Failed", s["status_counts"].get("failed", 0))
    if not docs:
        st.info("No documents yet. Put PDF / TXT / MD / HTML files with `.meta.json` sidecars in "
                "`data/sources/` and run `python -m scripts.ingest`, or upload below.")
    for d in docs:
        icon = {"indexed": "✅", "failed": "❌", "missing_source": "⚠️"}.get(d["status"], "•")
        with st.expander(f"{icon} {d.get('title') or d.get('file_path')}"):
            rows = [{"Field": label, "Value": _fmt(d[key])} for key, label in DOC_FIELDS
                    if d.get(key) not in (None, [], "")]
            if rows:
                st.dataframe(rows, hide_index=True, width="stretch")
            if d.get("source_url"):
                st.markdown(f"[Open source document]({d['source_url']})")
            for w in d.get("warnings") or []:
                st.caption(f"⚠️ {w}")
            if d.get("error"):
                st.error(d["error"])
    with st.expander("Upload a document"):
        f = st.file_uploader("File", type=["pdf", "txt", "md", "html", "htm"])
        meta = st.text_area("Metadata (JSON — see docs/METADATA_SCHEMA.md)",
                            value=json.dumps({"title": "", "author_or_institution": None, "source_url": None,
                                              "source_type": "other", "publication_date": None,
                                              "historical_periods": [], "language": "en",
                                              "rights_statement": ""}, indent=2), height=240)
        if st.button("Upload and ingest", disabled=f is None):
            try:
                r = requests.post(f"{client.base_url}/documents/upload", data={"metadata": meta},
                                  files={"file": (f.name, f.getvalue())}, timeout=900)
                if r.status_code >= 400:
                    st.error(_detail(r))
                else:
                    out = r.json()
                    st.success(f"Ingested “{out.get('document_id')}”: {out.get('status')}, "
                               f"{out.get('chunk_count')} chunks.")
                    for w in out.get("warnings") or []:
                        st.caption(f"⚠️ {w}")
                    st.cache_data.clear()
            except requests.ConnectionError:
                st.error("Cannot reach the API.")
    if s.get("last_run"):
        lr = s["last_run"]
        st.caption(f"Last ingestion run finished {lr.get('finished_at')}: {_fmt(lr.get('counts'))}")
