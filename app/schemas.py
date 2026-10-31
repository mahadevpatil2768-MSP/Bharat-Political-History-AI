"""Pydantic schemas shared across ingestion, retrieval, generation and the API."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

class HistoricalPeriod(str, Enum):
    ancient = "ancient"
    medieval = "medieval"
    colonial = "colonial"
    independence_movement = "independence_movement"
    p1947_1977 = "1947_1977"
    p1977_1991 = "1977_1991"
    p1991_2013 = "1991_2013"
    p2014_present = "2014_present"


PERIOD_LABELS: dict[str, str] = {
    "ancient": "Ancient India",
    "medieval": "Medieval India",
    "colonial": "Colonial India",
    "independence_movement": "Independence movement",
    "1947_1977": "1947–1977",
    "1977_1991": "1977–1991",
    "1991_2013": "1991–2013",
    "2014_present": "2014–present",
}


class SourceType(str, Enum):
    primary_legal = "primary_legal"            # constitutions, acts, treaties
    primary_government = "primary_government"  # official reports, gazettes, debates
    primary_speech = "primary_speech"          # speeches, letters, manifestos
    primary_inscription = "primary_inscription"  # edicts, inscriptions (incl. translations)
    judicial = "judicial"                      # court judgments
    scholarship = "scholarship"                # academic books/articles
    news = "news"                              # news reports
    reference = "reference"                    # encyclopedias, reference works
    other = "other"


PARTIAL_DATE_RE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")


def _check_partial_date(v: str | None) -> str | None:
    if v is None or v == "":
        return None
    if not PARTIAL_DATE_RE.match(v):
        raise ValueError("dates must be YYYY, YYYY-MM or YYYY-MM-DD")
    return v


# ---------------------------------------------------------------------------
# Document metadata (sidecar <file>.meta.json)
# ---------------------------------------------------------------------------

class DocumentMetadata(BaseModel):
    """Source-level metadata. Unknown values stay null — never guess them."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    document_id: str | None = Field(
        None, pattern=r"^[a-z0-9][a-z0-9_\-]{1,120}$",
        description="Stable ID. Derived from the file path when omitted.",
    )
    title: str = Field(..., min_length=1, max_length=500)
    author_or_institution: str | None = None
    source_url: str | None = None
    source_type: SourceType = SourceType.other
    publication_date: str | None = None
    date_accessed: str | None = None
    historical_periods: list[HistoricalPeriod] = Field(default_factory=list)
    event_year_start: int | None = Field(None, ge=-3000, le=2100)
    event_year_end: int | None = Field(None, ge=-3000, le=2100)
    language: str | None = Field(None, description="ISO 639-1 code, e.g. en, hi")
    document_version: str | None = None
    rights_statement: str | None = Field(
        None, description="Why this document may be processed (licence / PD basis)."
    )
    topics: list[str] = Field(default_factory=list)
    persons: list[str] = Field(default_factory=list)
    institutions: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("publication_date", "date_accessed")
    @classmethod
    def _partial_dates(cls, v: str | None) -> str | None:
        return _check_partial_date(v)

    @field_validator("source_url")
    @classmethod
    def _url_scheme(cls, v: str | None) -> str | None:
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("source_url must be http(s)")
        return v or None


class IngestedDocument(DocumentMetadata):
    """Metadata plus values computed at ingestion time."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    document_id: str  # required once ingested
    file_path: str
    file_type: str
    content_hash: str
    ingestion_timestamp: str
    page_count: int | None = None


# ---------------------------------------------------------------------------
# Extraction / chunking
# ---------------------------------------------------------------------------

class BlockType(str, Enum):
    heading = "heading"
    paragraph = "paragraph"
    table = "table"


class TextBlock(BaseModel):
    """A structural unit produced by a loader before chunking."""

    text: str
    block_type: BlockType = BlockType.paragraph
    page_number: int | None = None
    section_heading: str | None = None
    heading_level: int | None = None
    table_header: str | None = None  # for tables: header row, repeated on split


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    page_start: int | None = None
    page_end: int | None = None
    section_heading: str | None = None
    contains_table: bool = False


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class SearchFilters(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    periods: list[HistoricalPeriod] = Field(default_factory=list)
    year_from: int | None = Field(None, ge=-3000, le=2100)
    year_to: int | None = Field(None, ge=-3000, le=2100)
    source_types: list[SourceType] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    topic: str | None = Field(None, max_length=200)
    person: str | None = Field(None, max_length=200)
    institution: str | None = Field(None, max_length=200)
    author_or_institution: str | None = Field(None, max_length=200)
    published_from: str | None = None
    published_to: str | None = None

    @field_validator("published_from", "published_to")
    @classmethod
    def _partial_dates(cls, v: str | None) -> str | None:
        return _check_partial_date(v)

    def is_empty(self) -> bool:
        return not any(v for v in self.model_dump().values())


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any]
    vector_rank: int | None = None
    vector_similarity: float | None = None  # cosine similarity; NOT a probability
    bm25_rank: int | None = None
    bm25_score: float | None = None
    fused_score: float = 0.0  # reciprocal-rank-fusion score; only meaningful as an ordering
    rerank_score: float | None = None
    # Why an article-aware query selected this chunk: article_heading | article_text | related_provision | hybrid
    match_reason: str | None = None


# ---------------------------------------------------------------------------
# API: query / answer
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=3, max_length=1000)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    top_k: int | None = Field(None, ge=1, le=30)


class Citation(BaseModel):
    source_id: str  # "S1", "S2", ... as shown to the LLM
    chunk_id: str
    document_id: str
    title: str
    author_or_institution: str | None = None
    source_type: str | None = None
    publication_date: str | None = None
    source_url: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    section_heading: str | None = None
    excerpt: str


class Claim(BaseModel):
    statement: str
    citations: list[str] = Field(default_factory=list)
    attributed_to: str | None = None


class AnswerStatus(str, Enum):
    answered = "answered"
    insufficient_evidence = "insufficient_evidence"
    declined = "declined"            # out of scope (predictions, endorsements)
    error = "error"                  # generation failed; evidence still returned


class AnswerResponse(BaseModel):
    status: AnswerStatus
    question: str
    answer: str
    documented_facts: list[Claim] = Field(default_factory=list)
    interpretations: list[Claim] = Field(default_factory=list)
    disputed_or_uncertain: list[Claim] = Field(default_factory=list)
    conflicts: list[Claim] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    retrieved: list[Citation] = Field(default_factory=list)
    notices: list[str] = Field(default_factory=list)
    removed_citations: list[str] = Field(default_factory=list)
    model: str | None = None
    # Set only when status == "error" because the LLM step failed (e.g. "http_5xx", "rate_limit").
    error_category: str | None = None
    retryable: bool = False  # the UI may offer a manual Retry for this error
