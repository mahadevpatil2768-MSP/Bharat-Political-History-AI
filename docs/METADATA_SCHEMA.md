# Document metadata schema

Every source file in `data/sources/` needs a sidecar named `<file name>.meta.json`
(e.g. `constitution.pdf` → `constitution.pdf.meta.json`). It is validated against
`DocumentMetadata` in [app/schemas.py](../app/schemas.py); invalid sidecars make ingestion
fail for that file with a clear error. Unknown keys are rejected.

**Rule: leave a field out (or `null`) when you do not know it. Never guess.**

| Field | Type | Required | Notes |
|---|---|---|---|
| `title` | string | **yes** | Title as it appears on the source. |
| `document_id` | string | no | Stable ID, `^[a-z0-9][a-z0-9_-]{1,120}$`. Derived from the file path if omitted. Keep it stable — chunk IDs and evaluation labels depend on it. |
| `author_or_institution` | string | no | Author, issuing body, or "X; translated by Y". |
| `source_url` | http(s) URL | no | Where the file was obtained. Shown as a clickable link in the UI. |
| `source_type` | enum | no (default `other`) | `primary_legal`, `primary_government`, `primary_speech`, `primary_inscription`, `judicial`, `scholarship`, `news`, `reference`, `other` |
| `publication_date` | `YYYY`, `YYYY-MM` or `YYYY-MM-DD` | no | Publication date of *this edition*. Drives the freshness policy and `published_from/to` filters. |
| `date_accessed` | partial date | no | When you downloaded it. |
| `historical_periods` | list of enum | no | `ancient`, `medieval`, `colonial`, `independence_movement`, `1947_1977`, `1977_1991`, `1991_2013`, `2014_present`. These are search categories, not claims of a single interpretation. |
| `event_year_start` / `event_year_end` | int (−3000…2100) | no | Years the content covers. Documents without these are **excluded** when a year-range filter is applied. |
| `language` | ISO 639-1 | no | e.g. `en`, `hi`. |
| `document_version` | string | no | Edition / "as on" date / amendment status. |
| `rights_statement` | string | strongly recommended | Why you may process this document (licence, public-domain basis, permission). |
| `topics`, `persons`, `institutions` | list of strings | no | Used by filters (case-insensitive substring match) and for suggested questions. |
| `notes` | string | no | Caveats (OCR quality, incomplete transcription, disputed dating…). |

Computed at ingestion (do not put these in the sidecar): `content_hash` (SHA-256 of the
file), `ingestion_timestamp`, `file_path`, `file_type`, `page_count`. Every chunk also
records `chunk_id`, `chunk_index`, `page_start`, `page_end`, `section_heading` and
`contains_table`.

## Example

```json
{
  "document_id": "queens_proclamation_1858",
  "title": "Proclamation by the Queen in Council, to the Princes, Chiefs, and People of India (1858)",
  "author_or_institution": "Victoria of the United Kingdom (Queen in Council)",
  "source_url": "https://en.wikisource.org/wiki/Proclamation_by_the_Queen_in_Council,_to_the_princes,_chiefs,_and_people_of_India",
  "source_type": "primary_government",
  "publication_date": "1858-11-01",
  "date_accessed": "2026-10-29",
  "historical_periods": ["colonial"],
  "event_year_start": 1858,
  "event_year_end": 1858,
  "language": "en",
  "rights_statement": "Public domain: English Wikisource page carries the PD-old licence template.",
  "topics": ["crown rule", "east india company"]
}
```

## Timeline events (`data/timeline/events.json`)

```json
{"events": [
  {"id": "unique-id", "date": "1858-11-01", "title": "Short title",
   "description": "Optional neutral description",
   "document_id": "queens_proclamation_1858", "page": null,
   "evidence_quote": "an exact phrase copied from the indexed document",
   "periods": ["colonial"]}
]}
```

An event is displayed only if `evidence_quote` is found (whitespace/case-insensitive) in
the indexed text of `document_id` (and on `page`, when given). Everything else is listed
as "unverified" and hidden.
