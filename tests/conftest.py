"""Test defaults applied before any `app` import.

Unit tests must not read a developer's `.env` for required connection strings.
"""

import os

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://rag_app:rag_app@localhost:5432/rag")
os.environ.setdefault(
    "DATABASE_MIGRATION_URL", "postgresql+asyncpg://rag_owner:rag_owner@localhost:5432/rag"
)
os.environ.setdefault("EMBEDDING_PROVIDER", "fake")
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("JWT_SECRET", "dev-only-change-me-at-least-32-bytes")
