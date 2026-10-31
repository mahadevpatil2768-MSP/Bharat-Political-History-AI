# Source ingestion guide

## 1. Choose sources you may lawfully process

* Prefer primary sources (constitutions, acts, official reports, debates, judgments,
  speeches, inscriptions) and reputable scholarship.
* Check the reuse terms of **each** document. A hosting site is not proof of rights —
  e.g. not every Wikisource page is public domain; check the licence template on the page.
* Do not bypass logins, paywalls, CAPTCHAs, robots rules or tokens embedded in websites.
* Record the basis in `rights_statement`. Keep copyrighted material local: `data/sources/`
  is git-ignored and must not be pushed to a public repository.
* Treat downloaded files as untrusted data. The pipeline only parses them; never execute them.

## 2. Add files

Put files anywhere under `data/sources/` (sub-folders are fine):

| Format | Extraction | Page refs | Structure kept |
|---|---|---|---|
| PDF (text layer) | PyMuPDF | yes | headings (font-size/bold heuristic), paragraphs, tables (`PDF_EXTRACT_TABLES`) |
| Scanned PDF | OCR via Tesseract (optional, off by default) | yes | paragraphs |
| HTML | BeautifulSoup (boilerplate, nav, scripts removed) | no | h1–h6, paragraphs, lists, tables |
| Markdown | built-in | no | `#` headings, paragraphs, pipe tables |
| Plain text | built-in | no | blank-line paragraphs, PART/CHAPTER/ALL-CAPS headings |

Then add a metadata sidecar — see [METADATA_SCHEMA.md](METADATA_SCHEMA.md).

## 3. Ingest

```powershell
.\venv\Scripts\Activate.ps1
python -m scripts.ingest            # new + changed files only
python -m scripts.ingest --status   # manifest summary
python -m scripts.ingest --force    # re-process everything (e.g. after changing chunk settings)
python -m scripts.ingest --prune    # remove index entries whose file was deleted
python -m scripts.ingest --rebuild  # wipe index + manifest (needed after changing EMBEDDING_MODEL)
```

Or upload a single file from the UI (Library & ingestion tab) or `POST /documents/upload`.

What the pipeline does:

1. Hashes each file (SHA-256). Unchanged files with unchanged metadata and settings are skipped.
2. Identical content under another path is reported as `duplicate` and not re-indexed.
3. Extracts structural blocks, NFC-normalises Unicode, drops text in legacy non-Unicode
   Indic font encodings (with a warning), and warns about pages without a text layer.
4. Chunks by section (default 1000 chars, 150 overlap — configurable, not tuned).
5. Embeds with the configured Sentence Transformers model and stores chunks + metadata in Chroma.
6. Records status, chunk/page counts, warnings and errors in `data/manifest/manifest.json`.

Failures never stop the run; they are listed in the report and the manifest. The command
exits with code 1 if any file failed.

**Performance note.** Embedding runs on CPU by default. On the 4-thread development machine,
`multilingual-e5-base` embedded roughly 2 chunks/second, so the 402-page Constitution
(~1,250 chunks) took on the order of 10–15 minutes. Set `EMBEDDING_DEVICE=cuda` if you have a GPU.

## 4. OCR (optional)

```powershell
pip install -r requirements-ocr.txt
# Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki  (add Hindi data if needed)
```

In `.env`: `OCR_ENABLED=true` and, if needed, `TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe`.
OCR'd pages get a warning in the manifest; OCR text can contain errors — verify quotations.

## 5. Sample corpus

`python -m scripts.fetch_samples` downloads the small validation corpus (rights checked 2026-10-29):

| Document | Source | Rights basis |
|---|---|---|
| The Constitution of India [As on 1st May, 2024] | Legislative Department, GoI (S3WaaS CDN) | Official public download; site's copyright-policy page could not be retrieved automatically — verify before redistribution. Kept local. |
| Arthashastra, Book I (tr. R. Shamasastry, 1915) | English Wikisource | PD-old + PD-US templates. Wikisource marks the transcription incomplete/non-standardised. |
| Queen's Proclamation of 1 Nov 1858 | English Wikisource | PD-old template |
| Indian Home Rule (Hind Swaraj), 5th ed. 1922 | English Wikisource | PD/US template (author d. 1948) |

The script re-checks the public-domain licence template on each Wikisource page before saving.
