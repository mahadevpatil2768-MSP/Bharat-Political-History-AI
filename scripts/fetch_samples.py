"""Download the small, rights-checked sample corpus used to validate the pipeline.

    python -m scripts.fetch_samples            # fetch anything missing
    python -m scripts.fetch_samples --force    # re-download

Sources (provenance and rights verified 2026-10-29; see docs/SOURCE_INGESTION.md):
* The Constitution of India [As on 1st May, 2024] — Legislative Department, Government of India.
* Three public-domain works from English Wikisource. Before saving, the script re-checks that each
  page still carries a public-domain licence template and refuses to save it otherwise.

Files land in data/sources/ (git-ignored) with .meta.json sidecars. Requests are sequential and
rate-limited; the script honours HTTP 429 Retry-After.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from app.config import get_settings

USER_AGENT = ("BharatPoliticalHistoryAI/0.1 (offline research prototype; low-volume, sequential requests) "
              "python-requests")
WS_API = "https://en.wikisource.org/w/api.php"
WS_DELAY_S = 4.0
TODAY = date.today().isoformat()
PD_TEMPLATES = ("Template:PD-old", "Template:PD-US", "Template:PD-India", "Template:PD-GoI")

CONSTITUTION = {
    "url": "https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788ccfcdf1b99b5d8f/uploads/2024/07/20240716890312078.pdf",
    "path": "constitution/constitution_of_india_2024.pdf",
    "meta": {
        "document_id": "constitution_of_india_2024",
        "title": "The Constitution of India [As on 1st May, 2024]",
        "author_or_institution": "Government of India, Ministry of Law and Justice, Legislative Department (Official Languages Wing)",
        "source_type": "primary_legal",
        "publication_date": "2024-05-01",
        "historical_periods": ["1947_1977", "1977_1991", "1991_2013", "2014_present"],
        "event_year_start": 1949,
        "event_year_end": 2023,
        "language": "en",
        "document_version": "Pocket edition 'As on 1st May, 2024'; per its preface, incorporates amendments up to the "
                            "Constitution (One Hundred and Sixth Amendment) Act, 2023. Later amendments, if any, are not included.",
        "rights_statement": "Official text published by the Legislative Department, Government of India, for free public "
                            "download (file hosted on the government S3WaaS CDN used by legislative.gov.in). The site's own "
                            "copyright-policy page could not be retrieved by automated fetch (client-rendered / HTTP 403), so "
                            "its exact reuse terms are unverified. Processed locally for research only; not redistributed "
                            "and excluded from Git.",
        "topics": ["constitution", "fundamental rights", "directive principles", "parliament", "federalism",
                   "constitutional amendments", "elections", "emergency provisions"],
        "institutions": ["Parliament of India", "President of India", "Supreme Court of India",
                         "Election Commission of India"],
        "notes": "Cover page carries a Hindi title in a legacy (non-Unicode) font; such text is dropped at ingestion with a warning.",
    },
}

WIKISOURCE = [
    {
        "pages": ["Arthashastra/Book I"],
        "license_page": "Arthashastra",
        "path": "wikisource/arthashastra_book1_shamasastry_1915.html",
        "meta": {
            "document_id": "arthashastra_book1_shamasastry_1915",
            "title": "Arthashastra, Book I: Concerning Discipline (tr. R. Shamasastry, 1915)",
            "author_or_institution": "Kautilya (attributed); translated by R. Shamasastry",
            "source_type": "primary_government",
            "publication_date": "1915",
            "historical_periods": ["ancient"],
            "language": "en",
            "document_version": "English Wikisource transcription of the 1915 Bangalore Government Press translation. "
                                "Wikisource flags this text as incomplete and not yet standardised against scans.",
            "rights_statement": "Public domain: English Wikisource page carries PD-old and PD-US licence templates "
                                "(translation published 1915; translator died 1944).",
            "topics": ["statecraft", "kingship", "administration", "ministers", "espionage"],
            "notes": "Dating of the original Sanskrit text is debated by scholars; no composition date is recorded "
                     "in this metadata. The translation reflects the translator's 1915 choices.",
        },
    },
    {
        "pages": ["Proclamation by the Queen in Council, to the princes, chiefs, and people of India"],
        "license_page": "Proclamation by the Queen in Council, to the princes, chiefs, and people of India",
        "path": "wikisource/queens_proclamation_1858.html",
        "meta": {
            "document_id": "queens_proclamation_1858",
            "title": "Proclamation by the Queen in Council, to the Princes, Chiefs, and People of India (1858)",
            "author_or_institution": "Victoria of the United Kingdom (Queen in Council)",
            "source_type": "primary_government",
            "publication_date": "1858-11-01",
            "historical_periods": ["colonial"],
            "event_year_start": 1858,
            "event_year_end": 1858,
            "language": "en",
            "document_version": "English Wikisource transcription; page notes 'Delivered on November 1, 1858'.",
            "rights_statement": "Public domain: English Wikisource page carries the PD-old licence template (1858 document).",
            "topics": ["crown rule", "east india company", "princely states", "religious toleration"],
            "institutions": ["British Crown", "East India Company"],
            "persons": ["Queen Victoria"],
        },
    },
    {
        "pages": "follow_next:Indian Home Rule (5th edition)",
        "license_page": "Indian Home Rule (5th edition)",
        "path": "wikisource/indian_home_rule_gandhi_1922.html",
        "meta": {
            "document_id": "indian_home_rule_gandhi_1922",
            "title": "Indian Home Rule (Hind Swaraj), 5th edition (1922)",
            "author_or_institution": "Mohandas K. Gandhi",
            "source_type": "primary_speech",
            "publication_date": "1922",
            "historical_periods": ["independence_movement"],
            "language": "en",
            "document_version": "English Wikisource transcription of the 5th edition (1922), transcluded from the scan index 'Indian Home Rule by Mohandas K. Gandhi.djvu'.",
            "rights_statement": "Public domain: English Wikisource page carries PD/US (author died 1948; published 1922); "
                                "public domain in India (life + 60 years).",
            "topics": ["swaraj", "indian national congress", "civilisation", "passive resistance", "partition of bengal"],
            "institutions": ["Indian National Congress"],
            "persons": ["Mohandas K. Gandhi"],
        },
    },
]


class Fetcher:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = USER_AGENT
        self._last = 0.0

    def get(self, url: str, *, params=None, delay: float = 0.0, timeout: float = 120) -> requests.Response:
        for attempt in range(5):
            wait = self._last + delay - time.time()
            if wait > 0:
                time.sleep(wait)
            r = self.s.get(url, params=params, timeout=timeout)
            self._last = time.time()
            if r.status_code == 429:
                retry = int(r.headers.get("retry-after", "30"))
                print(f"  rate limited; waiting {retry}s", flush=True)
                time.sleep(retry)
                continue
            r.raise_for_status()
            return r
        raise RuntimeError(f"giving up on {url} after repeated rate limiting")

    def ws(self, **params) -> dict:
        params |= {"format": "json", "formatversion": 2}
        return self.get(WS_API, params=params, delay=WS_DELAY_S).json()


def ws_license_ok(f: Fetcher, page: str) -> list[str]:
    data = f.ws(action="query", titles=page, prop="templates", tllimit="max")
    tpls = [t["title"] for p in data["query"]["pages"] for t in p.get("templates", [])]
    return [t for t in tpls if t.startswith(PD_TEMPLATES)]


def ws_parse(f: Fetcher, page: str) -> tuple[str, str]:
    data = f.ws(action="parse", page=page, prop="text|wikitext", disableeditsection=1)
    return data["parse"]["text"], data["parse"]["wikitext"]


def next_from_header(wikitext: str, root: str, current: str) -> str | None:
    """Resolve the header's `next` link, including relative forms ([[/X/]], [[../X]])."""
    m = re.search(r"\|\s*next\s*=\s*'*\[\[([^\]|]+)", wikitext)
    if not m:
        return None
    target = m.group(1).strip()
    if target.startswith("../"):
        parent = current.rsplit("/", 1)[0] if "/" in current else current
        target = parent + "/" + target[3:]
    elif target.startswith("/"):
        target = current + target
    return target.rstrip("/").replace("_", " ")


def clean_ws_html(fragment: str) -> str:
    soup = BeautifulSoup(fragment, "lxml")
    for sel in ["#headertemplate", ".wst-header", ".ws-header", ".ws-noexport", ".licenseContainer",
                ".pagenum", ".ws-pagenum", ".mw-editsection", "style", "link", ".noprint"]:
        for el in soup.select(sel):
            el.decompose()
    body = soup.body or soup
    return "".join(str(c) for c in body.children)


def fetch_wikisource(f: Fetcher, item: dict, out: Path) -> None:
    lic = ws_license_ok(f, item["license_page"])
    if not lic:
        raise RuntimeError(f"{item['license_page']}: no public-domain licence template found — not saving")
    print(f"  licence templates: {', '.join(lic)}")

    pages = item["pages"]
    if isinstance(pages, str) and pages.startswith("follow_next:"):
        root = pages.split(":", 1)[1]
        order, cur, seen = [], root, set()
        sections = []
        while cur and cur not in seen and len(order) < 60:
            seen.add(cur)
            text, wt = ws_parse(f, cur)
            order.append(cur)
            title = cur.split("/", 1)[1] if "/" in cur else item["meta"]["title"]
            sections.append((title, clean_ws_html(text)))
            nxt = next_from_header(wt, root, cur)
            cur = nxt if nxt and nxt.startswith(root) else None
        print(f"  fetched {len(order)} pages by following 'next' links")
        listed = f.ws(action="query", list="allpages", apprefix=root + "/", aplimit=200)["query"]["allpages"]
        missing = sorted({p["title"] for p in listed} - set(order))
        if missing:
            print(f"  WARNING: {len(missing)} subpage(s) not reached via 'next' links: {missing}")
    else:
        sections = []
        for p in pages:
            text, _ = ws_parse(f, p)
            sections.append((p.split("/", 1)[1] if "/" in p else p, clean_ws_html(text)))

    title = html.escape(item["meta"]["title"])
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title></head><body><main>",
             f"<h1>{title}</h1>"]
    for sec_title, frag in sections:
        parts.append(f"<h2>{html.escape(sec_title)}</h2>\n{frag}")
    parts.append("</main></body></html>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")


def write_meta(path: Path, meta: dict, source_url: str) -> None:
    meta = dict(meta, source_url=source_url, date_accessed=TODAY)
    path.with_name(path.name + ".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                                        encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    src = get_settings().sources_dir
    f = Fetcher()
    failures = 0

    out = src / CONSTITUTION["path"]
    print(f"[constitution] {out}")
    if out.exists() and not args.force:
        print("  exists; skipping download")
    else:
        try:
            r = f.get(CONSTITUTION["url"], timeout=300)
            if not r.content.startswith(b"%PDF"):
                raise RuntimeError("response is not a PDF")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(r.content)
            print(f"  saved {len(r.content):,} bytes")
        except Exception as exc:
            failures += 1
            print(f"  FAILED: {exc}", file=sys.stderr)
    if out.exists():
        write_meta(out, CONSTITUTION["meta"], CONSTITUTION["url"])

    for item in WIKISOURCE:
        out = src / item["path"]
        url = "https://en.wikisource.org/wiki/" + item["license_page"].replace(" ", "_")
        print(f"[wikisource] {item['license_page']} -> {out}")
        if out.exists() and not args.force:
            print("  exists; skipping download")
        else:
            try:
                fetch_wikisource(f, item, out)
                print(f"  saved {out.stat().st_size:,} bytes")
            except Exception as exc:
                failures += 1
                print(f"  FAILED: {exc}", file=sys.stderr)
                continue
        write_meta(out, item["meta"], url)

    print("\nDone." + (f" {failures} failure(s)." if failures else "") +
          " Next: python -m scripts.ingest")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
