"""Evidence-grounded answer generation.

validate → scope check → retrieve (with filters) → assemble evidence [S1..Sn]
→ LLM (structured JSON) → validate citations → abstain if unsupported.
"""

from __future__ import annotations

import logging

from app.generation.citations import cited_ids, clean_text_markers, to_citation, validate_claims
from app.generation.policy import DECLINE_MESSAGE, freshness_notices, is_out_of_scope, is_time_sensitive
from app.generation.prompts import ANSWER_SCHEMA, SYSTEM_PROMPT, build_user_prompt
from app.llm.base import LLMError, LLMNotConfiguredError, LLMProvider
from app.retrieval.hybrid import HybridRetriever, preprocess_query
from app.schemas import AnswerResponse, AnswerStatus, SearchFilters

log = logging.getLogger(__name__)

NO_EVIDENCE_MESSAGE = ("The indexed sources do not contain evidence relevant to this question, so no "
                       "reliable answer can be given.")
EMPTY_INDEX_MESSAGE = ("No documents have been indexed yet. Add source files with metadata sidecars to "
                       "data/sources and run `python -m scripts.ingest` (see docs/SOURCE_INGESTION.md).")
UNSUPPORTED_MESSAGE = ("The available sources do not establish a reliable answer to this question. "
                       "Review the retrieved passages below directly.")


class AnswerGenerator:
    def __init__(self, retriever: HybridRetriever, llm: LLMProvider | None, *,
                 top_k: int = 8, freshness_max_age_days: int = 365,
                 llm_unavailable_reason: LLMError | None = None):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.freshness_max_age_days = freshness_max_age_days
        self.llm_unavailable_reason = llm_unavailable_reason or LLMNotConfiguredError()

    def answer(self, question: str, filters: SearchFilters | None = None,
               top_k: int | None = None) -> AnswerResponse:
        q = preprocess_query(question)  # raises QueryValidationError
        filters = filters or SearchFilters()

        if is_out_of_scope(q):
            return AnswerResponse(status=AnswerStatus.declined, question=q, answer=DECLINE_MESSAGE)

        if self.retriever.index.count() == 0:
            return AnswerResponse(status=AnswerStatus.insufficient_evidence, question=q,
                                  answer=EMPTY_INDEX_MESSAGE)

        chunks = self.retriever.retrieve(q, filters, top_k=top_k or self.top_k)
        notices: list[str] = []
        if not filters.is_empty():
            notices.append("Results were restricted by the selected filters.")
        if not chunks:
            return AnswerResponse(status=AnswerStatus.insufficient_evidence, question=q,
                                  answer=NO_EVIDENCE_MESSAGE, notices=notices)

        sources = [(f"S{i}", c) for i, c in enumerate(chunks, start=1)]
        by_id = {sid: c for sid, c in sources}
        retrieved = [to_citation(sid, c) for sid, c in sources]

        if is_time_sensitive(q):
            notices.extend(freshness_notices([c.metadata.get("publication_date") for c in chunks],
                                             self.freshness_max_age_days))

        if self.llm is None:
            err = self.llm_unavailable_reason
            return AnswerResponse(status=AnswerStatus.error, question=q,
                                  answer=f"{err.user_message} The retrieved evidence is listed below.",
                                  retrieved=retrieved, notices=notices,
                                  error_category=err.category, retryable=False)

        try:
            result = self.llm.generate_json(system=SYSTEM_PROMPT,
                                            user=build_user_prompt(q, sources, notices),
                                            schema=ANSWER_SCHEMA)
        except LLMError as exc:
            # Details (status, Google message, traceback) are logged by the provider; this is the summary line.
            log.warning("LLM generation failed category=%s error=%s: %s",
                        exc.category, type(exc).__name__, exc)
            return AnswerResponse(status=AnswerStatus.error, question=q,
                                  answer=f"{exc.user_message} The retrieved evidence is listed below "
                                         "so you can review the sources directly.",
                                  retrieved=retrieved, notices=notices,
                                  error_category=exc.category, retryable=exc.retryable)

        data = result.data or {}
        valid = set(by_id)
        removed: set[str] = set()
        dropped: list[str] = []
        facts = validate_claims(data.get("documented_facts"), valid, removed, dropped)
        interps = validate_claims(data.get("interpretations"), valid, removed, dropped)
        disputed = validate_claims(data.get("disputed_or_uncertain"), valid, removed, dropped)
        conflicts = validate_claims(data.get("conflicts"), valid, removed, dropped)
        answer_text = clean_text_markers(str(data.get("answer", "")), valid, removed)
        gaps = [clean_text_markers(str(g), valid, removed) for g in data.get("evidence_gaps") or []]

        if removed:
            notices.append(f"Removed citation(s) not present in the retrieved evidence: {', '.join(sorted(removed))}.")
        if dropped:
            notices.append(f"{len(dropped)} statement(s) without valid citations were removed.")

        used = cited_ids(answer_text, facts + interps + disputed + conflicts)
        sufficient = bool(data.get("sufficient_evidence"))
        if not sufficient or not used:
            if sufficient and not used:
                log.warning("Model claimed sufficient evidence but produced no valid citations; abstaining.")
            explanation = answer_text if (answer_text and not sufficient) else ""
            return AnswerResponse(
                status=AnswerStatus.insufficient_evidence, question=q,
                answer=(UNSUPPORTED_MESSAGE + (f"\n\n{explanation}" if explanation else "")),
                evidence_gaps=gaps, retrieved=retrieved, notices=notices,
                removed_citations=sorted(removed), model=result.model,
            )

        return AnswerResponse(
            status=AnswerStatus.answered, question=q, answer=answer_text,
            documented_facts=facts, interpretations=interps, disputed_or_uncertain=disputed,
            conflicts=conflicts, evidence_gaps=gaps,
            citations=[to_citation(sid, by_id[sid]) for sid in used],
            retrieved=retrieved, notices=notices, removed_citations=sorted(removed), model=result.model,
        )


def build_answer_generator(services) -> AnswerGenerator:
    from app.llm.gemini_client import GeminiProvider

    s = services.settings
    llm, reason = None, None
    try:
        llm = GeminiProvider(s)
    except LLMError as exc:
        reason = exc
    return AnswerGenerator(services.retriever, llm, top_k=s.retrieval_top_k,
                           freshness_max_age_days=s.freshness_max_age_days, llm_unavailable_reason=reason)
