"""Repeatable ingestion: discover → hash → metadata → extract → chunk → index → manifest.

* Unchanged files (same content hash, metadata and chunking settings) are skipped.
* Changed files are re-extracted and their old chunks replaced.
* Identical content under a different path is reported as a duplicate, not re-indexed.
* Every failure is recorded in the manifest and the run report — never silently dropped.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from app.config import Settings
from app.ingestion.chunker import chunk_blocks
from app.ingestion.loaders import LOADER_VERSION, SUPPORTED_EXTENSIONS, ExtractionError, load_document
from app.ingestion.manifest import Manifest, utc_now
from app.ingestion.metadata import (
    SIDECAR_SUFFIX,
    MetadataError,
    derive_document_id,
    file_sha256,
    load_sidecar,
    metadata_fingerprint,
)
from app.schemas import Chunk, IngestedDocument

log = logging.getLogger(__name__)


class Indexer(Protocol):
    embedding_model_name: str

    def replace_document(self, doc: IngestedDocument, chunks: list[Chunk]) -> None: ...

    def delete_document(self, document_id: str) -> None: ...


@dataclass
class FileResult:
    file_path: str
    document_id: str | None
    status: str  # indexed | unchanged | duplicate | failed | pruned
    chunk_count: int = 0
    page_count: int | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class IngestReport:
    started_at: str
    finished_at: str = ""
    duration_seconds: float = 0.0
    files_seen: int = 0
    results: list[FileResult] = field(default_factory=list)

    def count(self, status: str) -> int:
        return sum(1 for r in self.results if r.status == status)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["counts"] = {s: self.count(s) for s in ("indexed", "unchanged", "duplicate", "failed", "pruned")}
        d["chunks_indexed"] = sum(r.chunk_count for r in self.results if r.status == "indexed")
        return d


def discover_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        and not p.name.endswith(SIDECAR_SUFFIX) and not p.name.startswith(".")
    )


class IngestionPipeline:
    def __init__(self, settings: Settings, indexer: Indexer, manifest: Manifest | None = None):
        self.settings = settings
        self.indexer = indexer
        self.manifest = manifest or Manifest(settings.manifest_path)

    def _settings_fingerprint(self) -> dict:
        s = self.settings
        return {"loader_version": LOADER_VERSION,
                "embedding_model": self.indexer.embedding_model_name, "chunk_size": s.chunk_size,
                "chunk_overlap": s.chunk_overlap, "min_chunk_chars": s.min_chunk_chars,
                "ocr_enabled": s.ocr_enabled, "pdf_extract_tables": s.pdf_extract_tables}

    def run(self, paths: list[Path] | None = None, *, force: bool = False, prune: bool = False) -> IngestReport:
        root = self.settings.sources_dir
        files = paths if paths is not None else discover_files(root)
        report = IngestReport(started_at=utc_now(), files_seen=len(files))
        t0 = time.perf_counter()
        seen_ids: dict[str, str] = {}

        for path in files:
            result = self._process(path, root, force=force, seen_ids=seen_ids)
            report.results.append(result)
            lvl = logging.WARNING if result.status == "failed" else logging.INFO
            log.log(lvl, "%s: %s (chunks=%d)%s", result.file_path, result.status, result.chunk_count,
                    f" error={result.error}" if result.error else "")

        if paths is None:
            present = {self._rel(p, root) for p in files}
            for doc_id, entry in list(self.manifest.documents.items()):
                if entry.get("file_path") not in present:
                    if prune:
                        self.indexer.delete_document(doc_id)
                        del self.manifest.documents[doc_id]
                        report.results.append(FileResult(entry.get("file_path", "?"), doc_id, "pruned"))
                    elif entry.get("status") != "missing_source":
                        entry["status_before_missing"] = entry.get("status")
                        entry["status"] = "missing_source"
                        log.warning("%s: source file missing (chunks remain indexed; use --prune to remove)",
                                    entry.get("file_path"))

        report.finished_at = utc_now()
        report.duration_seconds = round(time.perf_counter() - t0, 2)
        summary = report.as_dict()
        self.manifest.runs.append({k: summary[k] for k in
                                   ("started_at", "finished_at", "duration_seconds", "files_seen",
                                    "counts", "chunks_indexed")})
        self.manifest.save()
        log.info("Ingestion finished: %s", summary["counts"] | {"chunks_indexed": summary["chunks_indexed"]})
        return report

    @staticmethod
    def _rel(path: Path, root: Path) -> str:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    def _process(self, path: Path, root: Path, *, force: bool, seen_ids: dict[str, str]) -> FileResult:
        rel = self._rel(path, root)
        try:
            meta = load_sidecar(path)
        except MetadataError as exc:
            return self._fail(rel, None, f"metadata error: {exc}")
        if meta is None and self.settings.require_metadata_sidecar:
            return self._fail(rel, None, f"missing metadata sidecar {path.name}{SIDECAR_SUFFIX} "
                                         "(see docs/METADATA_SCHEMA.md)")

        doc_id = (meta.document_id if meta and meta.document_id else derive_document_id(Path(rel)))
        if doc_id in seen_ids and seen_ids[doc_id] != rel:
            return self._fail(rel, doc_id, f"document_id '{doc_id}' already used by {seen_ids[doc_id]}")
        seen_ids[doc_id] = rel

        content_hash = file_sha256(path)
        dup_of = self.manifest.find_by_hash(content_hash, exclude_id=doc_id)
        if dup_of:
            return FileResult(rel, doc_id, "duplicate",
                              warnings=[f"identical content already indexed as '{dup_of}'"])

        fingerprint = self._settings_fingerprint() | {"metadata": metadata_fingerprint(meta)}
        prev = self.manifest.documents.get(doc_id)
        if (not force and prev and prev.get("status") in {"indexed", "missing_source"}
                and prev.get("content_hash") == content_hash and prev.get("fingerprint") == fingerprint
                and prev.get("file_path") == rel):
            prev["status"] = "indexed"
            return FileResult(rel, doc_id, "unchanged", chunk_count=prev.get("chunk_count", 0),
                              page_count=prev.get("page_count"))

        try:
            blocks, page_count, warnings = load_document(
                path, ocr_enabled=self.settings.ocr_enabled, tesseract_cmd=self.settings.tesseract_cmd,
                extract_tables=self.settings.pdf_extract_tables,
            )
            chunks = chunk_blocks(blocks, doc_id, chunk_size=self.settings.chunk_size,
                                  chunk_overlap=self.settings.chunk_overlap,
                                  min_chunk_chars=self.settings.min_chunk_chars)
            if not chunks:
                raise ExtractionError("extraction produced no text chunks")
            base = meta.model_dump() if meta else {"title": Path(rel).stem}
            if meta is None:
                warnings.append("no metadata sidecar: title taken from file name; other fields unknown")
            base["document_id"] = doc_id
            doc = IngestedDocument(**base, file_path=rel, file_type=path.suffix.lower().lstrip("."),
                                   content_hash=content_hash, ingestion_timestamp=utc_now(),
                                   page_count=page_count)
            self.indexer.replace_document(doc, chunks)
        except Exception as exc:  # record every failure; never crash the whole run
            log.exception("Ingestion failed for %s", rel)
            return self._fail(rel, doc_id, f"{type(exc).__name__}: {exc}")

        self.manifest.documents.pop(f"failed::{rel}", None)
        self.manifest.documents[doc_id] = {
            "document_id": doc_id, "title": doc.title, "file_path": rel, "content_hash": content_hash,
            "status": "indexed", "chunk_count": len(chunks), "block_count": len(blocks),
            "page_count": page_count, "warnings": warnings, "error": None,
            "ingested_at": doc.ingestion_timestamp, "fingerprint": fingerprint,
            "metadata": doc.model_dump(),
        }
        return FileResult(rel, doc_id, "indexed", chunk_count=len(chunks), page_count=page_count,
                          warnings=warnings)

    def _fail(self, rel: str, doc_id: str | None, error: str) -> FileResult:
        key = doc_id or f"failed::{rel}"
        prev = self.manifest.documents.get(key, {})
        self.manifest.documents[key] = {
            "document_id": doc_id, "title": prev.get("title"), "file_path": rel,
            "content_hash": prev.get("content_hash"), "status": "failed", "error": error,
            "chunk_count": prev.get("chunk_count", 0), "warnings": [], "ingested_at": utc_now(),
            "previous_status": prev.get("status"),
        }
        return FileResult(rel, doc_id, "failed", error=error)
