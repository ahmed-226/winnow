# AGENTS.md

## Project

Multi-tenant hybrid search & RAG backend: ingest PDF/Markdown/HTML, chunk, embed with
Gemini, and serve ACL-safe hybrid retrieval (pgvector dense + Postgres full-text, fused
with RRF). There is **no LLM answer generation**: `/search` returns ranked chunks.

## Source of truth

- Spec: `dev/docs/implementation-plan.md`. Read the relevant section before changing
  behavior: decisions §2, data model §7, tenancy/RLS §8, ingestion §9, chunking §10,
  embeddings §11, search §12, API §13, config §14, testing §17, phases §19.
- Background: `dev/docs/overview.md`. Where the two conflict, the plan wins.
- §2 "Decisions" is settled. §22 open questions do not block Phases 0–5.

## Status

Greenfield: only the two docs above exist. No code, manifests, migrations, or CI yet.
Build in the phase order of §19 and do not start a phase before the previous phase's
exit criterion is demonstrated locally. Put new code at the repo root per the layout in
§6 (`app/`, `alembic/`, `tests/`, `scripts/`), not under `dev/`.

## Stack (locked — do not substitute)

Python 3.12, FastAPI, SQLAlchemy 2 async + asyncpg, Alembic, PostgreSQL 17 + pgvector
>= 0.8, Redis Streams consumer groups, MinIO, Gemini `gemini-embedding-001` at 1536
dims, `uv`, `ruff`, `mypy`, `pytest`.

## Non-negotiable invariants

- Tenant id and roles come only from the validated JWT. Never trust `X-Tenant-ID` or
  `X-Roles`; the JWT `roles` claim is informational — current roles are read from the DB.
- Every tenant-table query carries an explicit `tenant_id` predicate **and** runs under
  forced RLS. Set context with transaction-local `set_config('app.tenant_id', ..., true)`
  (never session-level `SET`; pooled connections leak it).
- Cross-tenant or ACL-hidden resources return **404, not 403**.
- Empty `access_roles` means tenant-wide. `tenant_admin` bypasses document ACLs but never
  tenant isolation, and must never appear in `documents.access_roles`.
- Keep the composite FKs so a chunk/job tenant always equals its document's tenant.
- Chunks are written only after every embedding succeeds, in one transaction. Reprocess
  does delete+insert under a transaction-scoped advisory lock; never write partial vectors.
- Embedding task types are fixed: `RETRIEVAL_DOCUMENT` for chunks, `RETRIEVAL_QUERY` for
  queries, always `output_dimensionality=1536`. Validate dimension/finiteness; never pad,
  truncate, or silently change model/dimension (that requires `scripts/reembed.py`).
- RRF fuses **ranks only**. ACL and tenant filters belong inside both CTEs, before `LIMIT`.
- All SQL is parameterized. User text reaches FTS only through `websearch_to_tsquery`.
- Never log document text, query text, passwords, tokens, or API keys.

## Layout and ownership

- `app/api/` routes are thin; business rules live in `app/services/`.
- Vector/hybrid SQL exists only in `app/services/search.py` and `app/services/ingestion.py`.
- `app/workers/ingest.py` is the consumer, sweeper, and DLQ entrypoint; API and worker
  share one image and differ only by command.
- Migrations live in `alembic/versions/`, one per phase. `alembic check` must report no drift.

## Testing

- Unit tests need no I/O. Integration tests use a real Postgres+pgvector container
  (`testcontainers`) — never SQLite: generated `tsvector`, RLS, `set_config`, and HNSW
  cannot be faithfully tested otherwise.
- Keep `EMBEDDING_PROVIDER=fake` in CI and integration tests. Live Gemini calls are
  opt-in and marked `@pytest.mark.live`.
- The ACL matrix in §17.2 and the 18-step acceptance scenario in §20 are required
  coverage, not optional examples.

## Target commands (not scaffolded yet — create them in Phase 0)

```bash
cp .env.example .env                      # then set GEMINI_API_KEY or EMBEDDING_PROVIDER=fake
docker compose up -d postgres redis minio
uv sync
uv run alembic upgrade head               # migrations need postgres up first
uv run uvicorn app.main:app --reload --port 8000
uv run python -m app.workers.ingest
uv run pytest
uv run pytest tests/unit/test_chunking.py::test_name   # single test
uv run ruff check && uv run ruff format --check
uv run mypy app
```

Windows PowerShell equivalents belong in `README.md`; do not add a `make` dependency.

## Out of scope for v1 (do not add)

LLM answer generation, OCR, web UI, zip uploads, cross-tenant sharing, per-chunk ACLs,
billing/metering, Kubernetes, real BM25 via ParadeDB, automatic embedding-model migration.
See §3.2 before extending scope.
