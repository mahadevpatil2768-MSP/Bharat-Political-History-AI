"""Metadata sidecars, document IDs and content hashing."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from pydantic import ValidationError

from app.schemas import DocumentMetadata

SIDECAR_SUFFIX = ".meta.json"


class MetadataError(ValueError):
    pass


def sidecar_path(doc_path: Path) -> Path:
    return doc_path.with_name(doc_path.name + SIDECAR_SUFFIX)


def load_sidecar(doc_path: Path) -> DocumentMetadata | None:
    meta_path = sidecar_path(doc_path)
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise MetadataError(f"{meta_path.name}: invalid JSON ({exc})") from exc
    try:
        return DocumentMetadata.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()
        )
        raise MetadataError(f"{meta_path.name}: {problems}") from exc


def derive_document_id(rel_path: Path) -> str:
    stem = rel_path.with_suffix("").as_posix().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", stem).strip("_") or "doc"
    if len(slug) > 80:
        slug = slug[:70] + "_" + hashlib.sha1(stem.encode()).hexdigest()[:8]
    return slug


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def metadata_fingerprint(meta: DocumentMetadata | None) -> str:
    payload = meta.model_dump_json() if meta else ""
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
