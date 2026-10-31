# Bharat Political History AI

An evidence-grounded research assistant for Indian political history — from ancient polities to
the present. It answers **only** from documents you index, cites every claim back to a specific
passage (with page/section where available), separates documented facts from attributed
interpretations and disputed claims, surfaces conflicting sources, and says so when the sources do
not establish an answer.

> AI-generated answers can be wrong. Always check them against the cited passages.
> There is no live web or news search: answers are limited to the indexed documents.

Stack: Python 3.11 · FastAPI · Chroma · Sentence Transformers (`intfloat/multilingual-e5-base`) ·
BM25 + reciprocal-rank fusion · Google Gemini API, free tier (`gemini-3.8-flash`) · Streamlit.

---

## Quick start (Windows PowerShell)

```powershell
cd "C:\Users\Admin\Documents\Bharat Political History AI"

# 1. Virtual environment (Python 3.11 — newer Pythons may lack wheels for torch/chromadb)
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# 2. Configuration (see "Gemini free tier" below before creating a key)
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env                      # set GEMINI_API_KEY=...  (never commit .env)

# 3. Sample documents + ingestion (first run downloads the ~1.1 GB embedding model)
python -m scripts.fetch_samples
python -m scripts.ingest

# 4. Start the API (terminal 1)
uvicorn app.api.main:app --host 127.0.0.1 --port 8000

# 5. Start the UI (terminal 2)
.\venv\Scripts\Activate.ps1
streamlit run app/ui/streamlit_app.py
```

Open http://localhost:8501. API docs: http://127.0.0.1:8000/docs.

If PowerShell blocks `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
once, or call `.\venv\Scripts\python.exe` directly.

Without an API key the app still works for search, browsing and the timeline; the Ask tab then
shows the retrieved evidence with a message explaining that generation is not configured.

## Gemini free tier (₹0 budget)

Generated answers use the **Google Gemini API free tier** with `gemini-3.8-flash` — listed as
"Free of charge" for input and output on the
[Gemini API pricing page](https://ai.google.dev/gemini-api/docs/pricing) (verified 2026-10-28) and as a
Stable model with structured-output support on its
[model page](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash).

To stay at zero cost:

* Create the key in [Google AI Studio](https://aistudio.google.com/apikey) in a project with **no billing
  account linked**. Free-tier pricing applies only while no billing is linked; the app cannot detect this,
  so check the project shows *Free tier* in AI Studio. Never add a payment method to that project.
* The app refuses any `LLM_MODEL` not on its verified free-tier allowlist, sends no tools (no Google
  Search grounding), and never falls back to another model or provider.
* HTTP 429: per-minute limits get at most `LLM_MAX_RETRIES` retries (wait ≤ 60 s); a daily-quota error
  stops immediately and no further requests are sent until the reset (midnight Pacific time).
* Free-tier rate limits are per project and are shown in AI Studio (Google does not publish fixed
  numbers on the [rate-limits page](https://ai.google.dev/gemini-api/docs/rate-limits)). One question = one request.

**Data use.** Under the [Gemini API terms](https://ai.google.dev/gemini-api/terms) for unpaid services,
Google uses submitted content and responses to provide, improve and develop its products, and human
reviewers may read API input and output (after disconnecting it from your account). Each question sends
the question plus ~8 retrieved passages to Google. That is acceptable for the public-domain and
government texts in the sample corpus — **do not index or ask about sensitive, confidential or personal
information while using the free tier.** You must be 18+ and in an
[available region](https://ai.google.dev/gemini-api/docs/available-regions) (India is listed).

## Adding historical documents

1. Make sure you may lawfully process the document (see [docs/SOURCE_INGESTION.md](docs/SOURCE_INGESTION.md)).
2. Copy it (PDF, TXT, MD or HTML) anywhere under `data/sources/`.
3. Create `<file name>.meta.json` next to it — at minimum `{"title": "..."}`; add every field you
   actually know (schema: [docs/METADATA_SCHEMA.md](docs/METADATA_SCHEMA.md)). Do not guess values.
4. Run `python -m scripts.ingest` and read the report (failures and warnings are listed).

You can also upload one file at a time in the UI (*Library & ingestion*). Scanned PDFs need the
optional OCR setup (`requirements-ocr.txt` + Tesseract + `OCR_ENABLED=true`).

## Using the app

* **💬 Research chat** (default) — ChatGPT-style conversation. Type at the bottom and press Enter (or the send
  arrow). Each reply shows the answer with clickable `[S1]` citation badges that jump to that source's card,
  plus sections for documented facts, attributed interpretations, disputed claims, disagreements between
  sources and gaps. Source cards show title, author/institution, type, publication date and page, with
  expandable details (section, excerpt, link). Insufficient evidence, out-of-scope questions, Gemini rate
  limits/quota and backend errors each get a clear message instead of raw data.
  * **Follow-ups:** the backend answers one question at a time. For short follow-ups that refer back
    ("what about his views on railways?"), the UI sends your *previous question* along with the new one so
    retrieval can resolve the reference — earlier answers are never sent, and the exact text sent is shown
    under your message. Turn this off under *Answer settings*.
  * **Typewriter reveal:** the finished, validated answer is revealed word by word. This is a visual effect
    after the backend responds — it is **not** live token streaming (the API returns one complete response).
  * **Sidebar:** New chat, this session's conversations (kept only while the browser tab is open), answer
    settings, filters (period, event years, source type, topic, person, institution/party, language) and the
    list of indexed documents.
* **🔎 Search sources** — retrieval only (hybrid, keyword or vector), with raw ranking scores.
* **🗓️ Timeline** — curated events verified against indexed text, plus coverage spans of documents.
* **📚 Library & ingestion** — indexed documents, statuses, warnings, and upload.

## Article-aware retrieval

Questions that name a provision ("What is Article 14?", "Explain Art. 21A") are answered from that article's
own text first: the passage containing its heading (`14. Equality before law.—`), then its continuation, then
at most `ARTICLE_MAX_RELATED` provisions that substantively qualify it (e.g. exceptions that say a law is not
void as "inconsistent with … article 14"). Passages that only mention the article, Schedule paragraphs with the
same number, and table-of-contents pages are excluded. If the article is not in the index, normal hybrid
retrieval is used and the usual abstention applies.

This relies on structure metadata stored with each chunk. New ingestions add it automatically; for an index
built earlier, add it once **in place** (no re-embedding, chunk IDs/text/pages unchanged) — stop the API first:

```powershell
python -m scripts.annotate_legal_structure --dry-run   # report only
python -m scripts.annotate_legal_structure             # apply, then restart the API
```

## Tests and evaluation

```powershell
python -m pytest                                  # automated tests (no key, no network)
python -m app.evaluation.run_eval --check-labels  # verify eval labels exist in the index
python -m app.evaluation.run_eval                 # retrieval metrics (Recall@k, Precision@k, MRR)
python -m app.evaluation.run_eval --dataset eval/article_lookup.jsonl   # Articles 14/17/21/32 lookup set
python -m scripts.smoke_test                      # end-to-end checks, no Gemini calls
python -m scripts.smoke_test --live               # + exactly one Gemini request (asks first)
python -m app.evaluation.run_eval --answers       # answer behaviour (~26 Gemini requests; asks first)
```

See [docs/TESTING_AND_EVALUATION.md](docs/TESTING_AND_EVALUATION.md).

## Configuration

All settings live in `.env` (see `.env.example`). Common ones:

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | Required for generated answers (free-tier project, no billing) |
| `LLM_MODEL` | `gemini-3.8-flash` | must be in the free-tier allowlist (`app/llm/gemini_client.py`) |
| `LLM_THINKING_LEVEL` | `low` | `low` / `medium` / `high`; lower uses less free quota |
| `LLM_MAX_RETRIES` | `2` | per-minute 429 only; daily quota is never retried |
| `LLM_OVERLOAD_MAX_RETRIES` | `3` | HTTP 503 (model overloaded) only: waits ~3, 8, 15 s within `LLM_OVERLOAD_MAX_TOTAL_WAIT_SECONDS` (30) |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-base` | change ⇒ `python -m scripts.ingest --rebuild` |
| `RETRIEVAL_TOP_K` | `8` | passages sent to the model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | characters; change ⇒ `--force` re-ingest |
| `RERANKER` | `none` | `cross-encoder` enables optional reranking |
| `FRESHNESS_MAX_AGE_DAYS` | `365` | staleness warning for time-sensitive questions |
| `OCR_ENABLED` | `false` | OCR for scanned PDFs |

## Project layout

```
app/        config, schemas, ingestion/, retrieval/, llm/, generation/, evaluation/, api/, ui/
scripts/    ingest.py, fetch_samples.py
eval/       dataset.jsonl
docs/       ARCHITECTURE, SOURCE_INGESTION, METADATA_SCHEMA, TESTING_AND_EVALUATION
data/       sources/ (git-ignored), index/, manifest/, eval_results/ (generated), timeline/events.json
tests/
```

## Scope and neutrality

The tool is a neutral research aid. It does not predict elections, estimate probabilities of
political events, or endorse parties, candidates, ideologies or voting choices. Historical periods
are search categories, not claims of a single interpretation. Where sources disagree, both sides
are shown with their dates and source types.

## Troubleshooting

| Symptom | Fix |
|---|---|
| UI: "Cannot reach the API" | Start uvicorn (step 4) and check `API_BASE_URL` |
| Any generation error | Check `logs/backend.log` — lines `gemini_api_error` / `gemini_transport_error` show the category (http_5xx, network, protocol, timeout, quota_daily, rate_limit, auth…), HTTP status, Google's message and tracebacks; keys are redacted |
| "No GEMINI_API_KEY is configured" | Add the key to `.env`, restart the API |
| "rejected the API key" | Key invalid, revoked, or reported as leaked — create a new one in AI Studio |
| "per-minute rate limit" | Wait ~1 minute, then press **↻ Retry this question**; free-tier RPM is low |
| "Gemini is experiencing high demand (HTTP 503)" | Google-side overload, already retried up to 3 times; press **↻ Retry this question** after a minute or two |
| "daily quota … used up" | Wait for the reset (midnight Pacific ≈ 12:30–1:30 pm IST); nothing is sent until then |
| "not on the verified free-tier allowlist" | Use a model from `FREE_TIER_MODELS` |
| "model is not available" | The model ID was retired or is not enabled for your key; re-check Google's models page |
| `EmbeddingModelMismatchError` | You changed `EMBEDDING_MODEL`; run `python -m scripts.ingest --rebuild` |
| File `failed: missing metadata sidecar` | Add `<file>.meta.json` |
| "pages without extractable text" | Scanned PDF — enable OCR |
| Ingestion slow | CPU embedding is ~2 chunks/s on a 4-thread CPU (the 4 sample documents, 1,682 chunks, took ~16 min); use a GPU (`EMBEDDING_DEVICE=cuda`) or a smaller model |
| "unauthenticated requests to the HF Hub" warning | Harmless. After the model is downloaded once, `$env:HF_HUB_OFFLINE="1"` skips the online check |
