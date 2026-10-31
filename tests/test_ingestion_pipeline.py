import json

from tests.conftest import (
    CHARTER_MD,
    CHARTER_META,
    CHRONICLE_HTML,
    CHRONICLE_META,
    make_pdf,
    write_doc,
)


def test_ingest_preserves_metadata_and_page_refs(settings, services):
    src = settings.sources_dir
    write_doc(src, "charter.md", CHARTER_MD, CHARTER_META)
    write_doc(src, "chronicle.html", CHRONICLE_HTML, CHRONICLE_META)
    pdf = make_pdf(src / "acts.pdf", [[("Act One", 20), ("The Testland Act establishes a census.", 10)],
                                      [("Act Two", 20), ("The census is held every ten years.", 10)]])
    (src / "acts.pdf.meta.json").write_text(json.dumps({"title": "Testland Acts (fixture)",
                                                        "source_type": "primary_legal"}))

    report = services.pipeline().run()
    counts = report.as_dict()["counts"]
    assert counts["indexed"] == 3 and counts["failed"] == 0

    chunks = services.index.all_chunks()
    by_doc = {}
    for c in chunks:
        by_doc.setdefault(c["metadata"]["document_id"], []).append(c)
    assert set(by_doc) == {"testland_charter", "testland_chronicle", "acts"}

    m = by_doc["testland_chronicle"][0]["metadata"]
    assert m["title"] == CHRONICLE_META["title"]
    assert m["author_or_institution"] == "Ammara of Riverton"
    assert m["period_medieval"] is True and m["event_year_start"] == 1234
    assert m["content_hash"] and m["ingestion_timestamp"]
    # Missing metadata is not invented.
    assert "source_url" not in m

    pdf_pages = sorted(c["metadata"]["page_start"] for c in by_doc["acts"])
    assert pdf_pages[0] == 1 and pdf_pages[-1] == 2
    census = next(c for c in by_doc["acts"] if "ten years" in c["text"])
    assert census["metadata"]["page_start"] == 2 and census["metadata"]["section_heading"] == "Act Two"

    manifest = services.manifest()
    entry = manifest.documents["testland_charter"]
    assert entry["status"] == "indexed" and entry["chunk_count"] == len(by_doc["testland_charter"])
    assert entry["metadata"]["source_type"] == "primary_legal"


def test_unchanged_files_skipped_and_changed_files_reindexed(settings, services):
    src = settings.sources_dir
    path = write_doc(src, "charter.md", CHARTER_MD, CHARTER_META)
    assert services.pipeline().run().count("indexed") == 1

    second = services.pipeline().run()
    assert second.count("unchanged") == 1 and second.count("indexed") == 0

    path.write_text(CHARTER_MD + "\n\nA new amendment about lighthouses was appended.\n", encoding="utf-8")
    third = services.pipeline().run()
    assert third.count("indexed") == 1
    texts = " ".join(c["text"] for c in services.index.all_chunks())
    assert "lighthouses" in texts
    # Old chunks were replaced, not duplicated.
    ids = [c["chunk_id"] for c in services.index.all_chunks()]
    assert len(ids) == len(set(ids))


def test_metadata_change_triggers_reprocessing(settings, services):
    src = settings.sources_dir
    write_doc(src, "charter.md", CHARTER_MD, CHARTER_META)
    services.pipeline().run()
    write_doc(src, "charter.md", CHARTER_MD, CHARTER_META | {"title": "Renamed Charter (fixture)"})
    assert services.pipeline().run().count("indexed") == 1
    assert services.index.all_chunks()[0]["metadata"]["title"] == "Renamed Charter (fixture)"


def test_duplicate_content_detected(settings, services):
    src = settings.sources_dir
    write_doc(src, "a/charter.md", CHARTER_MD, CHARTER_META)
    write_doc(src, "b/copy.md", CHARTER_MD, {"title": "Copy"})
    report = services.pipeline().run()
    statuses = {r.file_path: r.status for r in report.results}
    assert statuses == {"a/charter.md": "indexed", "b/copy.md": "duplicate"}


def test_failures_are_reported_not_silent(settings, services):
    src = settings.sources_dir
    write_doc(src, "nometa.md", "# Heading\n\nSome text here.", None)
    write_doc(src, "badmeta.md", "# H\n\ntext", None)
    (src / "badmeta.md.meta.json").write_text('{"title": "x", "publication_date": "26 Jan 1950"}')
    write_doc(src, "broken.pdf", b"%PDF-garbage", {"title": "Broken"})
    report = services.pipeline().run()
    failed = {r.file_path: r.error for r in report.results if r.status == "failed"}
    assert "missing metadata sidecar" in failed["nometa.md"]
    assert "publication_date" in failed["badmeta.md"]
    assert "broken.pdf" in failed
    manifest = services.manifest()
    assert manifest.summary()["status_counts"]["failed"] == 3
    assert manifest.runs[-1]["counts"]["failed"] == 3


def test_missing_source_and_prune(settings, services):
    src = settings.sources_dir
    path = write_doc(src, "charter.md", CHARTER_MD, CHARTER_META)
    services.pipeline().run()
    path.unlink()
    services.pipeline().run()
    assert services.manifest().documents["testland_charter"]["status"] == "missing_source"
    assert services.index.count() > 0
    report = services.pipeline().run(prune=True)
    assert report.count("pruned") == 1 and services.index.count() == 0
