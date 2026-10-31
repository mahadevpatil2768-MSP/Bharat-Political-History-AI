# Architecture

```
 Streamlit UI (app/ui) ──HTTP──► FastAPI (app/api/main.py)
                                   │
   ┌───────────────────────────────┼─────────────────────────────────────┐
   │ Ingestion (app/ingestion)     │ Retrieval (app/retrieval)            │ Generation (app/generation)
   │  loaders.py   PDF/HTML/MD/TXT │  embeddings.py  Sentence Transformers│  policy.py    scope + freshness
   │  ocr.py       optional        │  vector_store.py Chroma (persistent) │  prompts.py   system prompt + JSON schema
   │  metadata.py  sidecars, hash  │  bm25.py        keyword (rank-bm25)  │  answer.py    orchestration + abstention
   │  chunker.py   structure-aware │  filters.py     metadata filters     │  citations.py validation
   │  manifest.py  JSON manifest   │  hybrid.py      RRF fusion + dedupe  │
   │  pipeline.py  orchestration   │  rerank.py      optional reranker    │ LLM (app/llm): base.py interface,
   └───────────────────────────────┴──────────────────────────────────────┘  gemini_client.py, fake.py
   config.py (pydantic-settings, .env) · schemas.py (Pydantic) · services.py (wiring) · timeline.py · evaluation/
```

## User interface (`app/ui`)

`streamlit_app.py` (entry point, layout, sidebar) → `chat_page.py` (chat turns) / `pages.py` (search, timeline,
library). `api_client.py` is the only network code: it calls the FastAPI endpoints, maps failures to friendly
`ApiError`s and validates `/query` responses against `AnswerResponse`. `render.py` turns responses into escaped
HTML (citation links → source-card anchors). `conversation.py` holds session-only chats and the follow-up
strategy (previous *question* only, shown to the user, never previous answers). The UI never calls Gemini,
Chroma or the index directly.

## Article-aware retrieval (`app/retrieval/legal_structure.py`)

Statutes write provisions as `14. Equality before law.—…` (no word "Article"), so BM25 and embeddings alone
rank cross-references (`… inconsistent with article 14 …`), dates (`17-10-1973`), Schedule paragraphs and
contents pages above the article itself. At ingestion (or in place via `scripts/annotate_legal_structure.py`)
each chunk gets: `article_headings`, `article_spans` (incl. continuation of the article open at the previous
chunk's end), `article_refs`, `article_refs_substantive` (references in a derogating context such as
"notwithstanding", "inconsistent with", "shall not apply", "suspended") and `is_contents` (front matter before
the first body heading that lists titles). Only provisions under a PART count; Schedule/Appendix paragraphs
and amendment footnotes do not.

When a query names an article, `HybridRetriever` returns tiers: heading chunk(s) → continuation (≤
`ARTICLE_MAX_CONTINUATION`) → substantive related provisions (≤ `ARTICLE_MAX_RELATED`, ordered by hybrid score)
→ topical hybrid results only if the question has words beyond the article reference. Mention-only chunks and
contents pages are excluded; the query is expanded with the article's indexed title for the hybrid pass. If no
heading is found, plain hybrid retrieval is used (generation then abstains as usual). For other queries,
contents chunks are ranked after content chunks. Chunk text, IDs and page numbers are never changed, so
citations are unaffected.

## Request flow (`POST /query`)

1. **Validate** — NFC-normalise, strip control chars, 3–1000 chars (`preprocess_query`).
2. **Scope** — prediction / endorsement questions are declined without retrieval (`policy.py`).
3. **Retrieve** — Chroma vector search (exact-match filters pushed into the `where` clause) and
   BM25 over the same chunks; both pass the same Python filter predicate. Results are fused with
   reciprocal-rank fusion (k=60), duplicate texts removed, optionally reranked, top-k kept.
   Scores are logged; they are ranking signals, not probabilities.
4. **No evidence** → `insufficient_evidence` without calling the LLM.
5. **Evidence context** — chunks become `[S1]…[Sn]` blocks with title, institution, type, date,
   pages and section. Delimiter-like text inside passages is neutralised; the system prompt
   tells the model that passages are untrusted data.
6. **Generate** — Gemini returns JSON constrained by `response_json_schema` (structured output): `sufficient_evidence`,
   `answer`, `documented_facts`, `interpretations` (with `attributed_to`), `disputed_or_uncertain`,
   `conflicts`, `evidence_gaps`.
7. **Validate citations** — IDs not in the retrieved set are stripped and reported; claims left
   without a valid citation are dropped.
8. **Abstain** — if the model says evidence is insufficient, or no valid citation remains.
9. **Freshness** — time-sensitive questions get a notice that there is no live search, and a
   warning when the newest source is older than `FRESHNESS_MAX_AGE_DAYS`.

## Key decisions

* **Provider isolation.** Ingestion and retrieval never import the LLM. `LLMProvider` is a
  one-method protocol; `GeminiProvider` is the only network implementation. There is no fallback provider.
* **Embedding consistency.** The collection stores its embedding model name; opening it with a
  different model raises `EmbeddingModelMismatchError` (rebuild with `--rebuild`).
* **E5 prefixes.** `query:` / `passage:` prefixes are applied automatically for E5 models.
* **BM25 freshness.** The in-memory BM25 index is rebuilt when the chunk count or the manifest
  mtime changes, so ingestion in another process is picked up.
* **Labels independent of chunking.** Evaluation labels use document ID + evidence phrase.
* **No extra infrastructure.** No database server, auth or cloud services: Chroma (SQLite) and a
  JSON manifest suffice for a single-user research prototype. Add auth before exposing the API
  beyond localhost.
* **Timeline integrity.** Curated events are shown only when their quote is found in the cited
  indexed document.

## Gemini configuration (free tier only)

`GEMINI_API_KEY`, `LLM_MODEL` (default `gemini-3.8-flash`), `LLM_THINKING_LEVEL`, `LLM_MAX_TOKENS`,
`LLM_TIMEOUT_SECONDS`, `LLM_MAX_RETRIES`, `LLM_RETRY_MAX_WAIT_SECONDS`.

Zero-cost safeguards in `app/llm/gemini_client.py`:

* `FREE_TIER_MODELS` allowlist (verified against the Gemini pricing page on 2026-10-28); any other model
  raises `LLMModelNotAllowedError` before a request is sent.
* No tools in the request (no Google Search grounding or other billable features); no fallback model/provider.
* SDK retries are disabled. The provider retries per-minute 429s (honouring `RetryInfo.retryDelay`, capped)
  at most `LLM_MAX_RETRIES` times, and HTTP 503 UNAVAILABLE (model overloaded) at most
  `LLM_OVERLOAD_MAX_RETRIES` (3) times after ~3, 8, 15 s — waiting at least Google's `retryDelay`, and giving
  up rather than exceed `LLM_OVERLOAD_MAX_TOTAL_WAIT_SECONDS` (30). Other 5xx are never retried automatically.
  Errors carry a `category` and `retryable` flag; `/query` returns them as `error_category` / `retryable`, and
  the chat offers a manual **Retry** (same question and filters, one click = one request) only for retryable
  errors (`http_5xx`, `rate_limit`) on the latest reply. A 429 whose `QuotaFailure.quotaId` is a per-day quota
  raises `LLMQuotaExhaustedError` immediately and trips a circuit breaker: no request is sent until the next
  midnight Pacific reset.
* 400 `API_KEY_INVALID` / 401 / 403 → `LLMAuthError`; 404 → `LLMModelUnavailableError`; safety blocks →
  `LLMRefusalError`; truncated or non-JSON output → `LLMOutputError`. The key is redacted from error text and
  never logged.
