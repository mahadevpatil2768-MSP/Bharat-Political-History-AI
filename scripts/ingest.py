"""Repeatable ingestion command.

    python -m scripts.ingest                 # ingest new/changed files under data/sources
    python -m scripts.ingest --force         # re-process everything
    python -m scripts.ingest --prune         # also remove index entries whose source file is gone
    python -m scripts.ingest --rebuild       # wipe the vector index and manifest, then ingest
    python -m scripts.ingest --status        # print the manifest summary only
"""

from __future__ import annotations

import argparse
import json
import sys

from app.config import get_settings
from app.logging_setup import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest documents from data/sources")
    parser.add_argument("--force", action="store_true", help="re-process unchanged files")
    parser.add_argument("--prune", action="store_true", help="remove index entries for deleted files")
    parser.add_argument("--rebuild", action="store_true", help="wipe index + manifest first")
    parser.add_argument("--status", action="store_true", help="show manifest summary and exit")
    args = parser.parse_args(argv)

    setup_logging()
    settings = get_settings()

    if args.status:
        from app.ingestion.manifest import Manifest
        print(json.dumps(Manifest(settings.manifest_path).summary(), indent=2, ensure_ascii=False))
        return 0

    settings.sources_dir.mkdir(parents=True, exist_ok=True)

    if args.rebuild:
        import shutil
        for path in (settings.index_dir, settings.manifest_path.parent):
            if path.exists():
                shutil.rmtree(path)
        print("Index and manifest removed; rebuilding.")

    from app.retrieval.vector_store import EmbeddingModelMismatchError
    from app.services import build_services
    try:
        services = build_services(settings)
    except EmbeddingModelMismatchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    report = services.pipeline().run(force=args.force, prune=args.prune)
    summary = report.as_dict()

    print("\n=== Ingestion report ===")
    print(f"Files seen: {summary['files_seen']}   Duration: {summary['duration_seconds']}s")
    for status, n in summary["counts"].items():
        print(f"  {status:<10} {n}")
    print(f"Chunks indexed this run: {summary['chunks_indexed']}")
    print(f"Total chunks in index:   {services.index.count()}")
    for r in report.results:
        if r.status in {"failed", "duplicate"} or r.warnings:
            print(f"\n- {r.file_path} [{r.status}]")
            if r.error:
                print(f"    error: {r.error}")
            for w in r.warnings:
                print(f"    warning: {w}")
    if summary["files_seen"] == 0:
        print(f"\nNo supported files found in {settings.sources_dir}. "
              "Add PDF/TXT/MD/HTML files with .meta.json sidecars (see docs/SOURCE_INGESTION.md).")
    return 1 if summary["counts"]["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
