"""Application configuration, loaded from environment variables and a local .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM — Google Gemini API, free tier only (see app/llm/gemini_client.py FREE_TIER_MODELS)
    gemini_api_key: SecretStr | None = None
    llm_model: str = "gemini-3.8-flash"
    llm_max_tokens: int = Field(8192, ge=256, le=65536)  # includes thinking tokens
    llm_thinking_level: Literal["low", "medium", "high"] = "low"
    llm_timeout_seconds: float = Field(120.0, gt=0)
    llm_max_retries: int = Field(2, ge=0, le=3)  # bounded retries for per-minute 429 only
    llm_retry_max_wait_seconds: float = Field(60.0, gt=0, le=120)
    # HTTP 503 UNAVAILABLE (model overloaded): up to 3 retries after ~3, 8, 15 s, within a total wait budget.
    # Other 5xx errors are never retried automatically.
    llm_overload_max_retries: int = Field(3, ge=0, le=3)
    llm_overload_max_total_wait_seconds: float = Field(30.0, gt=0, le=120)

    # Embeddings / retrieval
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 16
    retrieval_top_k: int = Field(8, ge=1, le=50)
    retrieval_candidate_k: int = Field(30, ge=1, le=200)
    rrf_k: int = Field(60, ge=1)
    reranker: Literal["none", "cross-encoder"] = "none"
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    min_vector_similarity: float = Field(0.0, ge=0.0, le=1.0)
    # Article-aware retrieval ("What is Article 14?"): continuation chunks of the article, and provisions that
    # substantively qualify it (exceptions, "notwithstanding …"). Mere mentions are never included.
    article_max_continuation: int = Field(3, ge=0, le=6)
    article_max_related: int = Field(2, ge=0, le=5)

    # Chunking
    chunk_size: int = Field(1000, ge=200, le=8000)
    chunk_overlap: int = Field(150, ge=0, le=2000)
    min_chunk_chars: int = Field(200, ge=0)

    # Ingestion
    ocr_enabled: bool = False
    tesseract_cmd: str | None = None
    pdf_extract_tables: bool = True
    max_upload_mb: int = Field(25, ge=1, le=500)
    require_metadata_sidecar: bool = True

    # Freshness policy for time-sensitive questions
    freshness_max_age_days: int = Field(365, ge=1)

    # Paths
    data_dir: Path = PROJECT_ROOT / "data"
    collection_name: str = "bharat_history"

    # UI
    api_base_url: str = "http://127.0.0.1:8000"

    # Logging: rotating file relative to the project root; empty string disables file logging.
    log_file: str = "logs/backend.log"

    @property
    def log_file_path(self) -> Path | None:
        if not self.log_file.strip():
            return None
        p = Path(self.log_file)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def sources_dir(self) -> Path:
        return self.data_dir / "sources"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "sources" / "uploads"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest" / "manifest.json"

    @property
    def timeline_path(self) -> Path:
        return self.data_dir / "timeline" / "events.json"

    @property
    def llm_configured(self) -> bool:
        key = self.gemini_api_key.get_secret_value().strip() if self.gemini_api_key else ""
        return bool(key) and key != "your-gemini-api-key-here"


@lru_cache
def get_settings() -> Settings:
    return Settings()
