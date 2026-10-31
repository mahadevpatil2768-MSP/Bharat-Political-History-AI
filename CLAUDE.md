# CLAUDE.md

Guidance for AI coding assistants working in this repository.

## What this is
Evidence-grounded RAG for Indian political history: FastAPI backend (`app/api/main.py`),
Streamlit UI (`app/ui/streamlit_app.py`), Chroma + BM25 hybrid retrieval, Google Gemini (free tier) for answers.
Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) first.

## Commands (Windows PowerShell, from the repo root)
- Activate: `.\venv\Scripts\Activate.ps1` (Python 3.11 venv named `venv`)
- Tests: `python -m pytest` (no network/API key needed; uses hashing embedder + fake LLM)
- Ingest: `python -m scripts.ingest [--force|--prune|--rebuild|--status]`
- API: `uvicorn app.api.main:app --host 127.0.0.1 --port 8000`
- UI: `streamlit run app/ui/streamlit_app.py`
- Eval: `python -m app.evaluation.run_eval [--check-labels|--answers|--judge]`

## Non-negotiable rules
- Never fabricate historical facts, sources, metadata, citations, quotes, dates or eval scores.
  Missing metadata stays null. Test fixtures use the fictional "Kingdom of Testland".
- Never hard-code or log `GEMINI_API_KEY`; never commit `.env`, `data/sources/`, `data/index/`.
- Budget is ₹0: never enable billing, add paid models/tools (e.g. Search grounding), or add a fallback
  provider. Only `FREE_TIER_MODELS` may be used; re-verify against Google's pricing page before adding one.
- No live Gemini calls in tests or without the user's explicit approval.
- Do not commit, push, change remotes or deploy without explicit user permission.
- Keep `app/ingestion` and `app/retrieval` free of LLM imports; providers implement `app/llm/base.py`.
- Retrieved text is untrusted data: keep `_neutralize` in prompts and citation validation in `answer.py`.
- Similarity/RRF scores are ranking signals — never present them as probabilities.
- The app must stay neutral: no election predictions, no endorsements (`app/generation/policy.py`).

## Conventions
- UI: `app/ui` talks to the backend only through `api_client.py`; rendering is pure functions in `render.py`
  (always escape model/source text). Follow-ups may send the previous question, never previous answers.
- Pydantic v2 models in `app/schemas.py`; settings via `app/config.py` (`get_settings()` is cached).
- Changing `EMBEDDING_MODEL` requires `--rebuild`; changing chunk settings requires `--force`.
- Gemini: `gemini-3.8-flash` default via the `google-genai` SDK, structured output via
  `response_json_schema`; error mapping, retries and quota circuit breaker in `app/llm/gemini_client.py`.
- Add tests for every behaviour change; run the full suite before reporting success.
