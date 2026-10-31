"""FastAPI backend.

Run:  uvicorn app.api.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from app import __version__
from app.generation.answer import AnswerGenerator, build_answer_generator
from app.generation.citations import to_citation
from app.ingestion.loaders import SUPPORTED_EXTENSIONS
from app.llm.gemini_client import FREE_TIER_MODELS
from app.ingestion.metadata import SIDECAR_SUFFIX
from app.logging_setup import setup_logging
from app.retrieval.hybrid import QueryValidationError
from app.retrieval.vector_store import EmbeddingModelMismatchError
from app.schemas import (
    PERIOD_LABELS,
    AnswerResponse,
    Citation,
    DocumentMetadata,
    QueryRequest,
    SearchFilters,
    SourceType,
)
from app.services import Services, get_services
from app.timeline import document_spans, load_events, verify_events

setup_logging()
log = logging.getLogger(__name__)

app = FastAPI(title="Bharat Political History AI", version=__version__,
              description="Evidence-grounded research API for Indian political history.")

_ingest_lock = threading.Lock()
_answerer: AnswerGenerator | None = None
_answerer_lock = threading.Lock()


def services_dep() -> Services:
    try:
        return get_services()
    except EmbeddingModelMismatchError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def answerer_dep(services: Services = Depends(services_dep)) -> AnswerGenerator:
    global _answerer
    with _answerer_lock:
        if _answerer is None or _answerer.retriever is not services.retriever:
            _answerer = build_answer_generator(services)
        return _answerer


@app.exception_handler(QueryValidationError)
async def _query_error(_: Request, exc: QueryValidationError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def _unhandled(_: Request, exc: Exception):
    log.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"detail": "Internal server error. See server logs."})


# ---------------------------------------------------------------------------
# Health / status
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    try:
        services = get_services()
    except EmbeddingModelMismatchError as exc:
        return {"status": "error", "detail": str(exc)}
    s = services.settings
    return {
        "status": "ok", "version": __version__,
        "indexed_chunks": services.index.count(),
        "embedding_model": services.index.embedding_model_name,
        "llm_provider": "gemini", "llm_configured": s.llm_configured, "llm_model": s.llm_model,
        "llm_model_free_tier_allowlisted": s.llm_model in FREE_TIER_MODELS,
        "reranker": s.reranker, "ocr_enabled": s.ocr_enabled,
        "web_search": False,
    }


@app.get("/ingest/status")
def ingest_status(services: Services = Depends(services_dep)):
    m = services.manifest()
    docs = [{k: v for k, v in e.items() if k not in ("metadata", "fingerprint")} | {"key": k}
            for k, e in m.documents.items()]
    return {"summary": m.summary(), "documents": docs, "index_chunks": services.index.count()}


class IngestRunRequest(BaseModel):
    force: bool = False
    prune: bool = False


@app.post("/ingest/run")
def ingest_run(req: IngestRunRequest, services: Services = Depends(services_dep)):
    if not _ingest_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="An ingestion run is already in progress.")
    try:
        report = services.pipeline().run(force=req.force, prune=req.prune)
        services.retriever.invalidate()
        return report.as_dict()
    finally:
        _ingest_lock.release()


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(name: str) -> str:
    base = re.split(r"[\\/]", name or "")[-1]
    base = _SAFE_NAME_RE.sub("_", base).lstrip(".")
    if not base or base in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid file name.")
    return base[:120]


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...), metadata: str = Form(...),
                          services: Services = Depends(services_dep)):
    s = services.settings
    name = _safe_filename(file.filename or "")
    ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported file type '{ext}'. "
                                                    f"Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    try:
        meta = DocumentMetadata.model_validate(json.loads(metadata))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid metadata: {exc}") from exc

    limit = s.max_upload_mb * 1024 * 1024
    data = bytearray()
    while chunk := await file.read(1024 * 1024):
        data.extend(chunk)
        if len(data) > limit:
            raise HTTPException(status_code=413, detail=f"File exceeds {s.max_upload_mb} MB limit.")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if ext == ".pdf" and not bytes(data[:5]).startswith(b"%PDF"):
        raise HTTPException(status_code=415, detail="File does not look like a PDF.")

    upload_dir = s.uploads_dir.resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = (upload_dir / name).resolve()
    if upload_dir not in target.parents:  # path-traversal guard
        raise HTTPException(status_code=400, detail="Invalid file path.")
    if target.exists():
        raise HTTPException(status_code=409, detail=f"A file named '{name}' already exists.")

    if not _ingest_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="An ingestion run is in progress; try again shortly.")
    try:
        target.write_bytes(bytes(data))
        target.with_name(target.name + SIDECAR_SUFFIX).write_text(
            meta.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")
        report = services.pipeline().run(paths=[target])
        services.retriever.invalidate()
    finally:
        _ingest_lock.release()
    result = report.results[0]
    if result.status == "failed":
        raise HTTPException(status_code=422, detail=f"Ingestion failed: {result.error}")
    return {"file": name, "document_id": result.document_id, "status": result.status,
            "chunk_count": result.chunk_count, "warnings": result.warnings}


@app.get("/documents")
def list_documents(services: Services = Depends(services_dep)):
    docs = []
    for key, e in services.manifest().documents.items():
        meta = e.get("metadata") or {}
        docs.append({
            "document_id": e.get("document_id"), "status": e.get("status"), "title": e.get("title"),
            "file_path": e.get("file_path"), "chunk_count": e.get("chunk_count", 0),
            "page_count": e.get("page_count"), "warnings": e.get("warnings", []), "error": e.get("error"),
            **{k: meta.get(k) for k in ("author_or_institution", "source_url", "source_type",
                                        "publication_date", "date_accessed", "historical_periods",
                                        "event_year_start", "event_year_end", "language",
                                        "document_version", "rights_statement", "topics")},
        })
    return {"documents": docs}


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    top_k: int = Field(10, ge=1, le=50)
    mode: Literal["hybrid", "vector", "keyword"] = "hybrid"


class SearchHit(BaseModel):
    citation: Citation
    fused_score: float
    vector_rank: int | None
    vector_similarity: float | None
    bm25_rank: int | None
    bm25_score: float | None
    rerank_score: float | None


@app.post("/search")
def search(req: SearchRequest, services: Services = Depends(services_dep)) -> dict:
    hits = services.retriever.retrieve(req.query, req.filters, top_k=req.top_k, mode=req.mode)
    return {"query": req.query, "count": len(hits),
            "score_note": "Scores are ranking signals (RRF / cosine / BM25), not probabilities.",
            "hits": [SearchHit(citation=to_citation(f"R{i}", h, excerpt_chars=700),
                               fused_score=round(h.fused_score, 5), vector_rank=h.vector_rank,
                               vector_similarity=h.vector_similarity, bm25_rank=h.bm25_rank,
                               bm25_score=h.bm25_score, rerank_score=h.rerank_score).model_dump()
                     for i, h in enumerate(hits, start=1)]}


@app.post("/query", response_model=AnswerResponse)
def query(req: QueryRequest, answerer: AnswerGenerator = Depends(answerer_dep)) -> AnswerResponse:
    return answerer.answer(req.question, req.filters, top_k=req.top_k)


@app.get("/timeline")
def timeline(services: Services = Depends(services_dep)):
    events, errors = load_events(services.settings.timeline_path)
    verified, problems = verify_events(events, services.index.all_chunks()) if events else ([], [])
    return {"events": sorted(verified, key=lambda e: (e["year"], e["date"])),
            "document_spans": document_spans(services.manifest().documents),
            "unverified": errors + problems}


@app.get("/meta/filters")
def filter_options(services: Services = Depends(services_dep)):
    topics: set[str] = set()
    languages: set[str] = set()
    for e in services.manifest().documents.values():
        meta = e.get("metadata") or {}
        topics.update(meta.get("topics") or [])
        if meta.get("language"):
            languages.add(meta["language"])
    return {"periods": PERIOD_LABELS, "source_types": [s.value for s in SourceType],
            "topics": sorted(topics), "languages": sorted(languages)}
