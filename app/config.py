from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

AppEnv = Literal["development", "test", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
EmbeddingProviderName = Literal["gemini", "fake"]


class Settings(BaseSettings):
    """Runtime configuration. All values come from the environment (or `.env`)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Core
    app_env: AppEnv = "development"
    log_level: LogLevel = "INFO"
    public_base_url: str = "http://localhost:8000"
    cors_origins: str = "http://localhost:3000"

    # Database
    database_url: str
    database_migration_url: str
    db_pool_size: int = 10
    db_max_overflow: int = 5
    db_statement_timeout_ms: int = 15_000

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    ingest_stream: str = "rag:ingest"
    ingest_group: str = "ingest-workers"
    ingest_dead_stream: str = "rag:ingest:dead"
    ingest_claim_idle_ms: int = 300_000
    ingest_sweeper_interval_seconds: int = 60
    worker_job_timeout_seconds: int = 900
    worker_concurrency: int = 2

    # Object storage
    s3_endpoint_url: str | None = None
    s3_access_key_id: str = "minioadmin"
    s3_secret_access_key: str = "minioadmin"
    s3_bucket: str = "rag-documents"
    s3_region: str = "us-east-1"
    max_upload_bytes: int = 52_428_800

    # Auth
    jwt_secret: str = "dev-only-change-me"
    jwt_issuer: str = "winnow"
    jwt_audience: str = "rag-api"
    access_token_ttl_seconds: int = 1_800
    refresh_token_ttl_seconds: int = 604_800
    allow_tenant_signup: bool = True
    argon2_time_cost: int = 3
    argon2_memory_cost: int = 65_536
    argon2_parallelism: int = 2

    # Embeddings
    gemini_api_key: str | None = None
    embedding_provider: EmbeddingProviderName = "gemini"
    embedding_model: str = "gemini-embedding-001"
    embedding_dimension: int = 1_536
    embedding_document_task: Literal["RETRIEVAL_DOCUMENT"] = "RETRIEVAL_DOCUMENT"
    embedding_query_task: Literal["RETRIEVAL_QUERY"] = "RETRIEVAL_QUERY"
    embedding_batch_size: int = 16
    embedding_concurrency: int = 2
    embedding_requests_per_minute: int = 60
    embedding_timeout_seconds: int = 30
    embedding_max_retries: int = 5
    embedding_query_cache_ttl_seconds: int = 3_600

    # Chunking
    chunk_target_chars: int = 3_000
    chunk_overlap_chars: int = 400
    chunk_max_chars: int = 6_000
    chunk_min_chars: int = 200
    pdf_max_pages: int = 1_000

    # Search
    search_candidates: int = 50
    search_top_k_default: int = 10
    search_top_k_max: int = 50
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    hnsw_ef_search: int = 100
    hnsw_iterative_scan: Literal["strict_order", "relaxed_order", "off"] = "strict_order"
    hnsw_max_scan_tuples: int = 20_000

    # Rate limits
    rl_login_per_minute: int = 10
    rl_register_per_hour: int = 5
    rl_upload_per_hour: int = 30
    rl_search_per_minute: int = 120

    # Maintenance (scripts/purge.py)
    soft_delete_retention_days: int = 30

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @model_validator(mode="after")
    def _validate_invariants(self) -> Settings:
        if self.embedding_dimension != 1_536:
            raise ValueError(
                "EMBEDDING_DIMENSION must be 1536; changing it requires scripts/reembed.py "
                "and a schema migration"
            )
        if self.app_env != "development" and len(self.jwt_secret.encode()) < 32:
            raise ValueError("JWT_SECRET must be at least 32 bytes outside development")
        if self.app_env == "production":
            if self.allow_tenant_signup:
                raise ValueError("ALLOW_TENANT_SIGNUP must be false in production")
            if self.embedding_provider == "fake":
                raise ValueError("EMBEDDING_PROVIDER=fake is not allowed in production")
        if self.embedding_provider == "gemini" and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when EMBEDDING_PROVIDER=gemini")
        if self.chunk_overlap_chars >= self.chunk_target_chars:
            raise ValueError("CHUNK_OVERLAP_CHARS must be smaller than CHUNK_TARGET_CHARS")
        if self.search_top_k_default > self.search_top_k_max:
            raise ValueError("SEARCH_TOP_K_DEFAULT must not exceed SEARCH_TOP_K_MAX")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
