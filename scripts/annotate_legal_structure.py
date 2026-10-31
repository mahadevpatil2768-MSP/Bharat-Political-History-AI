"""Add legal-structure metadata to the EXISTING index in place (no re-embedding, no re-ingestion).

    python -m scripts.annotate_legal_structure --dry-run   # report what would change
    python -m scripts.annotate_legal_structure             # apply

Only metadata is updated (article_headings, article_spans, article_refs, article_refs_substantive,
is_contents, legal_structure_version). Chunk IDs, text, embeddings, page numbers and the ingestion
manifest are untouched; the script verifies this after applying.
"""

from __future__ import annotations

import argparse
import sys

from app.config import get_settings
from app.logging_setup import setup_logging
from app.retrieval.legal_structure import LEGAL_STRUCTURE_VERSION, annotate_document

LEGAL_KEYS = ("article_headings", "article_spans", "article_refs", "article_refs_substantive", "is_contents",
              "legal_structure_version")
BATCH = 200


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    setup_logging("WARNING")

    import chromadb
    from chromadb.config import Settings as ChromaSettings

    s = get_settings()
    client = chromadb.PersistentClient(path=str(s.index_dir), settings=ChromaSettings(anonymized_telemetry=False))
    col = client.get_collection(s.collection_name)
    total = col.count()
    res = col.get(include=["documents", "metadatas"], limit=total)
    by_doc: dict[str, list[dict]] = {}
    for cid, text, meta in zip(res["ids"], res["documents"], res["metadatas"]):
        by_doc.setdefault(meta["document_id"], []).append(
            {"chunk_id": cid, "chunk_index": meta["chunk_index"], "text": text,
             "page_start": meta.get("page_start"), "section_heading": meta.get("section_heading"), "meta": meta})

    sample_ids = res["ids"][:: max(1, total // 25)][:25]
    before = col.get(ids=sample_ids, include=["embeddings", "documents"])

    updates_ids, updates_meta = [], []
    for doc_id, chunks in sorted(by_doc.items()):
        ann = annotate_document(chunks)
        n_head = sum("article_headings" in a for a in ann.values())
        n_contents = sum(a["is_contents"] for a in ann.values())
        print(f"{doc_id}: {len(chunks)} chunks | {n_head} with article headings | {n_contents} contents chunks")
        for c in chunks:
            base = {k: v for k, v in c["meta"].items() if k not in LEGAL_KEYS}  # drop stale legal keys
            updates_ids.append(c["chunk_id"])
            updates_meta.append(base | ann[c["chunk_id"]])

    if args.dry_run:
        print(f"\nDry run: would update metadata of {len(updates_ids)} chunks (version {LEGAL_STRUCTURE_VERSION}).")
        return 0

    for i in range(0, len(updates_ids), BATCH):
        col.update(ids=updates_ids[i:i + BATCH], metadatas=updates_meta[i:i + BATCH])

    after = col.get(ids=sample_ids, include=["embeddings", "documents"])
    same_text = before["documents"] == after["documents"]
    same_vec = all((a == b).all() if hasattr(a, "all") else a == b
                   for a, b in zip(before["embeddings"], after["embeddings"]))
    check = col.get(ids=updates_ids[:1], include=["metadatas"])["metadatas"][0]
    print(f"\nUpdated {len(updates_ids)} chunks. Count before/after: {total}/{col.count()}. "
          f"Sample text unchanged: {same_text}. Sample embeddings unchanged: {same_vec}. "
          f"legal_structure_version={check.get('legal_structure_version')}")
    ok = same_text and same_vec and col.count() == total
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
