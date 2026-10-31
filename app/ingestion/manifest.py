"""Ingestion manifest: a JSON record of every source file and its ingestion status."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_RUNS_KEPT = 50


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Manifest:
    def __init__(self, path: Path):
        self.path = path
        self.documents: dict[str, dict[str, Any]] = {}
        self.runs: list[dict[str, Any]] = []
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.documents = data.get("documents", {})
            self.runs = data.get("runs", [])

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"documents": self.documents, "runs": self.runs[-MAX_RUNS_KEPT:]},
            ensure_ascii=False, indent=2,
        )
        # Atomic write so a crash never leaves a half-written manifest.
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, self.path)

    def find_by_hash(self, content_hash: str, exclude_id: str | None = None) -> str | None:
        for doc_id, entry in self.documents.items():
            if (doc_id != exclude_id and entry.get("content_hash") == content_hash
                    and entry.get("status") == "indexed"):
                return doc_id
        return None

    def find_by_path(self, rel_path: str) -> str | None:
        for doc_id, entry in self.documents.items():
            if entry.get("file_path") == rel_path:
                return doc_id
        return None

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for entry in self.documents.values():
            counts[entry.get("status", "unknown")] = counts.get(entry.get("status", "unknown"), 0) + 1
        return {
            "document_count": len(self.documents),
            "indexed_documents": counts.get("indexed", 0),
            "status_counts": counts,
            "chunk_count": sum(e.get("chunk_count", 0) for e in self.documents.values()
                               if e.get("status") == "indexed"),
            "last_run": self.runs[-1] if self.runs else None,
        }
