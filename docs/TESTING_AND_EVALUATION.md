# Testing and evaluation

## Automated tests

```powershell
.\venv\Scripts\Activate.ps1
python -m pytest                 # full suite; no API key or network needed
python -m pytest tests/test_api.py -v
```

Tests use fictional "Kingdom of Testland" fixtures (never real historical claims), a
deterministic hashing embedder (no model download) and a scripted fake LLM / stubbed Gemini SDK.
An autouse fixture makes constructing a real `google.genai.Client` fail, so no test can call the live API.

| File | Covers |
|---|---|
| `test_loaders.py` | PDF/HTML/MD/TXT parsing, page numbers, headings, tables, Unicode/Indic normalisation, legacy-font text, corrupt/blank files |
| `test_chunker.py` | packing, size limits, section boundaries, overlap, sentence splitting (incl. `।`), table header repetition, stable IDs |
| `test_ingestion_pipeline.py` | metadata preservation, page refs, unchanged/changed/duplicate detection, failure reporting, prune |
| `test_retrieval.py` | hybrid/vector/keyword relevance, period/year/source/person/topic/date filters, dedupe, query validation, model mismatch |
| `test_generation.py` | citation integrity, fabricated-citation removal, abstention, conflicting sources, LLM failure fallback, scope policy, prompt-injection neutralisation, freshness |
| `test_gemini_client.py` | Gemini provider with SDK stubbed: missing/placeholder key, free-tier allowlist, invalid key, per-minute vs daily 429 (bounded retries, circuit breaker), 404 model unavailable, 5xx, timeouts, safety blocks, truncation, bad JSON, no tools in request |
| `test_api.py` | endpoints, validation errors, upload type/size/path-traversal checks, timeline verification, error non-leakage |
| `test_legal_structure.py` | article heading/reference/contents detection, query parsing, tiered article retrieval on a fictional PDF (heading first, mention-only and contents excluded, substantive exceptions kept, Schedule paragraph ignored, fallback, filters), in-place annotation keeps text/embeddings |
| `test_real_index_articles.py` | (`real_index`) Articles 14/17/21/32 on the real annotated index: own text ranked first on the correct page; skipped if the index is absent or not annotated |
| `test_ui.py` | chat UI: citation links/anchors, escaping, source cards, status banners, typewriter chunks, follow-up context strategy, UI client ↔ real FastAPI app (fake LLM), transport/invalid-response errors, headless `AppTest` runs of the chat (bubbles, follow-ups, errors, new chat, conversation switching) |

## Evaluation dataset

`eval/dataset.jsonl` — one JSON object per line:

```json
{"id": "...", "type": "factual|multi_document|timeline|comparison|constitutional|disputed|insufficient|misleading_premise|recent_no_source|out_of_scope",
 "question": "...", "filters": {}, "relevant": [{"document_id": "...", "contains": ["exact phrase"]}],
 "expected_behavior": "answer|abstain|decline|answer_or_abstain", "review": "..."}
```

Relevance labels are phrases that must occur in the relevant chunk, so they stay valid when
chunk settings change. Check that every phrase still exists in the index:

```powershell
python -m app.evaluation.run_eval --check-labels
```

**Review status:** the dataset was drafted by the coding assistant. Each label phrase was
checked automatically against the indexed text, but the questions, expected behaviours and
relevance judgements still need review by a human historian before scores are relied on.

`eval/article_lookup.jsonl` — 12 fixed article-lookup questions (Articles 14, 17, 21, 32; three phrasings each).
Relevant = chunks containing the article's own text. Because most of these articles fit in one chunk,
Precision@5 is capped at 0.2 (0.4 for Article 32); `precision_of_returned@5` (relevant / passages actually
returned) and MRR show precision more directly.

## Running the evaluation

```powershell
python -m app.evaluation.run_eval                       # retrieval only: Recall@k, Precision@k, MRR (no API calls)
python -m app.evaluation.run_eval --mode keyword        # compare retrieval modes
python -m app.evaluation.run_eval --answers             # + behaviour accuracy, citation relevance (~26 Gemini free-tier requests; asks first)
python -m app.evaluation.run_eval --answers --judge     # + LLM-judged faithfulness (up to ~2x requests; asks first)
```

Results: `data/eval_results/eval_<timestamp>.json`. Answer evaluation stops early if the free-tier daily
quota is exhausted and records that in the summary.

Metric definitions (see `app/evaluation/metrics.py`):

* **Recall@k** — share of labels satisfied by at least one of the top-k chunks.
* **Precision@k** — share of the top-k chunks that satisfy any label (bounded by how many relevant chunks exist).
* **MRR@8** — mean of 1/rank of the first relevant chunk within the top 8 retrieved.
* **Behaviour accuracy** — answered / abstained / declined as expected.
* **Citation relevance** — share of cited chunks containing a labelled phrase (a lower bound).
* **Faithfulness** (`--judge`) — share of documented facts and interpretations the judge model
  considers supported by their cited passages. LLM judgements are themselves fallible; spot-check them.

Report only numbers produced by these commands. With a small corpus and a small dataset the
numbers have high variance and should not be generalised.
