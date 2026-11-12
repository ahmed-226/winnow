from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings

REQUIRED = {
    "database_url": "postgresql+asyncpg://rag_app:rag_app@localhost:5432/rag",
    "database_migration_url": "postgresql+asyncpg://rag_owner:rag_owner@localhost:5432/rag",
    "embedding_provider": "fake",
}


def make_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **{**REQUIRED, **overrides})  # type: ignore[arg-type]


def test_defaults_are_development_and_fixed_dimension() -> None:
    settings = make_settings()
    assert settings.app_env == "development"
    assert settings.embedding_dimension == 1536
    assert settings.embedding_document_task == "RETRIEVAL_DOCUMENT"
    assert settings.embedding_query_task == "RETRIEVAL_QUERY"
    assert settings.cors_origins_list == ["http://localhost:3000"]


def test_rejects_non_1536_dimension() -> None:
    with pytest.raises(ValidationError, match="EMBEDDING_DIMENSION"):
        make_settings(embedding_dimension=768)


def test_rejects_fake_embeddings_in_production() -> None:
    with pytest.raises(ValidationError, match="EMBEDDING_PROVIDER=fake"):
        make_settings(app_env="production", jwt_secret="x" * 32, allow_tenant_signup=False)


def test_production_requires_signup_disabled() -> None:
    with pytest.raises(ValidationError, match="ALLOW_TENANT_SIGNUP"):
        make_settings(
            app_env="production",
            jwt_secret="x" * 32,
            embedding_provider="gemini",
            gemini_api_key="test-key",
        )


def test_gemini_requires_api_key() -> None:
    with pytest.raises(ValidationError, match="GEMINI_API_KEY"):
        make_settings(embedding_provider="gemini", gemini_api_key=None)


def test_non_development_requires_long_jwt_secret() -> None:
    with pytest.raises(ValidationError, match="JWT_SECRET"):
        make_settings(app_env="test", jwt_secret="short")


def test_chunk_overlap_must_be_smaller_than_target() -> None:
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP_CHARS"):
        make_settings(chunk_target_chars=500, chunk_overlap_chars=500)


def test_cors_origins_are_split_and_trimmed() -> None:
    settings = make_settings(cors_origins="http://a.example, http://b.example ,")
    assert settings.cors_origins_list == ["http://a.example", "http://b.example"]
