# Winnow

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17-4169E1?logo=postgresql&logoColor=white)
![pgvector](https://img.shields.io/badge/pgvector-0.8+-336791?logo=postgresql&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-Streams-DC382D?logo=redis&logoColor=white)
![MinIO](https://img.shields.io/badge/MinIO-S3-C72E49?logo=minio&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-embedding--001-4285F4?logo=google&logoColor=white)
![uv](https://img.shields.io/badge/uv-managed-DE5FE9?logo=astral&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)

**Winnow** — a multi-tenant hybrid search & RAG engine: PDF/Markdown/HTML in, ACL-safe
ranked chunks out (pgvector dense + PostgreSQL full-text, fused with RRF). No LLM answer
generation.

## API overview

All paths are under `/api/v1` except health and metrics. Authenticated endpoints expect
`Authorization: Bearer <access_token>`. The tenant comes only from the signed token;
`X-Tenant-ID` and `X-Roles` headers are ignored.

### Health and metrics

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/healthz` | none | Liveness; no dependency checks |
| `GET` | `/readyz` | none | Database readiness; `503` when Postgres is unreachable |
| `GET` | `/readyz?full=true` | none | Also checks Redis and the object-storage bucket |
| `GET` | `/metrics` | none | Prometheus metrics |
| `GET` | `/docs` | none | OpenAPI UI |

### Auth

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/api/v1/auth/register-tenant` | none | Creates a tenant plus its first `tenant_admin`; `409` on duplicate slug/email |
| `POST` | `/api/v1/auth/login` | none | Returns access token, refresh token, and user |
| `POST` | `/api/v1/auth/refresh` | refresh token | Rotates the refresh token; reusing a revoked token revokes the whole family |
| `POST` | `/api/v1/auth/logout` | refresh token | Revokes the supplied refresh token |
| `GET` | `/api/v1/auth/me` | bearer | Current user and tenant from the database |

Login failures are generic (`401 Invalid credentials`) for unknown tenant, unknown email,
wrong password, and inactive users alike.

### Users (tenant admin only)

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/users` | Create a user with optional roles |
| `GET` | `/api/v1/users` | List users |
| `GET` | `/api/v1/users/{id}` | Get a user; cross-tenant returns `404` |
| `PATCH` | `/api/v1/users/{id}` | Change email, password, roles, or `is_active` |

The last active `tenant_admin` cannot be demoted or deactivated (`409`). Role changes and
deactivation take effect immediately; access tokens are rechecked against the database on
every request.

### Documents

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/api/v1/documents` | bearer | Multipart upload (`file`, optional `title`, repeated `access_roles`); `202` new / `200` duplicate |
| `GET` | `/api/v1/documents` | bearer | Cursor-paginated list of visible documents (`limit`, `cursor`) |
| `GET` | `/api/v1/documents/{id}` | bearer | Get one document |
| `PATCH` | `/api/v1/documents/{id}` | admin or uploader | Change `title` and/or `access_roles` |
| `DELETE` | `/api/v1/documents/{id}` | admin or uploader | Soft delete; the same bytes can be re-uploaded afterwards |
| `POST` | `/api/v1/documents/{id}/reprocess` | admin | Re-enqueue ingestion |

Behavior:

- Supported types: PDF, Markdown, HTML. Extension/content mismatches return `415`,
  oversized uploads `413`, empty files `415`.
- Files are stored in a private object-storage bucket; duplicates of the same bytes within
  a tenant return the original document (`duplicate: true`, same id).
- Members may only assign access roles they hold; `tenant_admin` may assign any role except
  `tenant_admin`. Empty ACLs are tenant-wide.
- ACL-hidden or cross-tenant documents return `404`, not `403`.
- Documents stay `pending` until the async ingestion worker processes them.

### Search

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/api/v1/search` | bearer | Hybrid dense + sparse retrieval, RRF-fused, ACL-filtered |

Request: `{"query": "how many weeks of parental leave?", "top_k": 8}` — `query` is 1–1,000
characters (trimmed, no control characters), `top_k` defaults to 10 and is capped at 50.

Response: ranked chunks with `score` (RRF), plus `dense_rank`, `sparse_rank`,
`sparse_score`, and `cosine_distance` so relevance can be debugged. `mode` is `hybrid`, or
`sparse_only` when the query embedding failed after retries — a degraded response is
always visible, never silent. Only `ready` documents are searchable; empty ACLs are
tenant-wide and `tenant_admin` bypasses document ACLs but never tenant isolation.

## Error format

All errors use `application/problem+json` (`type`, `title`, `status`, `detail`, optional
`errors`) and include a `request_id`. Validation failures return `422`; unexpected failures
return a generic `500`.

## Rate limits

Fixed-window counters in Redis, per user and route (`429` with a `Retry-After` header).
A Redis outage fails open: authentication still applies, and a cache outage never locks
users out.

| Route | Limit | Window | Key |
|---|---|---|---|
| `POST /api/v1/auth/login` | `RL_LOGIN_PER_MINUTE` (10) | 60 s | IP + tenant + email |
| `POST /api/v1/auth/register-tenant` | `RL_REGISTER_PER_HOUR` (5) | 3600 s | client IP |
| `POST /api/v1/documents` | `RL_UPLOAD_PER_HOUR` (30) | 3600 s | user id |
| `POST /api/v1/search` | `RL_SEARCH_PER_MINUTE` (120) | 60 s | user id |

Rejected requests are counted in `rate_limit_rejections_total{scope}`.

## Metrics

`GET /metrics` serves Prometheus, combining the HTTP instrumentator
(`http_requests_total`, `http_request_duration_seconds`) with application metrics:

| Metric | Type | Labels |
|---|---|---|
| `documents_uploaded_total` | counter | `source_type`, `duplicate` |
| `document_ingest_duration_seconds` | histogram | `source_type`, `outcome` |
| `document_ingest_failures_total` | counter | `error_code` (ingestion error taxonomy) |
| `search_requests_total` | counter | `mode`, `outcome` |
| `search_duration_seconds` | histogram | `mode` |
| `search_results_count` | histogram | — |
| `embedding_requests_total` | counter | `operation` (document/query), `status` |
| `embedding_request_duration_seconds` | histogram | `operation` |
| `rate_limit_rejections_total` | counter | `scope` |
| `db_pool_in_use` | gauge | — |
| `queue_depth` | gauge | `stream` (API observes Redis for the worker) |
| `queue_pending_messages` | gauge | `group` |

Alert-worthy: queue depth growing for 10 minutes, ingest failure rate > 5%, search p95
> 1 s, embedding 429 rate > 10%, `readyz` failing.

Label values are bounded enums — never document text, query text, or tenant IDs.

## Operations

### Ingestion worker

`uv run python -m app.workers.ingest` (or `docker compose up -d worker`). The DB job row
is authoritative; every 60 s the sweeper re-enqueues jobs whose Redis message was lost
(`queued` > 2 min stale) or whose worker died (`running` > 15 min stale), and `XAUTOCLAIM`
recovers messages abandoned by dead consumers. Jobs that exhaust their attempts are
dead-lettered to `rag:ingest:dead` and marked `failed` with a reason in
`documents.error_message`; `POST /api/v1/documents/{id}/reprocess` (admin) retries them.

### Soft-delete purge and orphan objects

Soft-deleted documents are kept for `SOFT_DELETE_RETENTION_DAYS` (default 30), then
hard-deleted together with their stored object:

```bash
uv run python scripts/purge.py                  # purge expired + report orphan objects
uv run python scripts/purge.py --delete-orphans # also delete unreferenced bucket objects
```

Orphan objects (leftovers from failed uploads or failed deletes) are reported by default;
deletion is opt-in. The script runs as `rag_owner` (BYPASSRLS) and needs MinIO up.
Schedule it (cron / Task Scheduler) in any real deployment.

### Backup and restore

Postgres (physical layout irrelevant, logical dump is enough for v1):

```bash
docker exec winnow-postgres-1 pg_dump -U postgres -d rag -Fc -f /tmp/rag.dump
docker exec winnow-postgres-1 createdb -U postgres rag_restore_check
docker exec winnow-postgres-1 pg_restore -U postgres -d rag_restore_check /tmp/rag.dump
# verify: compare row counts, confirm pgvector extension, HNSW/GIN indexes, and the
# 6 RLS policies exist in the restored database, then drop it
docker exec winnow-postgres-1 dropdb -U postgres rag_restore_check
```

MinIO: `mc alias set local http://localhost:9000 minioadmin minioadmin && mc mirror local/rag-documents /backup/minio` — restore by mirroring back; the DB `storage_key` values
match object keys, so no re-pointing is needed. Verify a restore by fetching one object.

### Load test

```bash
uv run python scripts/loadtest.py --tenant acme --email admin@acme.example \
    --password '...' --concurrency 10 --duration 30
```

Reports req/s and p50/p95/p99/max. For load testing, temporarily raise
`RL_SEARCH_PER_MINUTE` (or expect and accept the `429` tail — the script reports its
count). Baseline on a dev box (10 concurrent, 20 s, empty tenant index, fake
provider): 126.5 req/s, p50 78 ms, p95 94 ms, p99 125 ms, max 157 ms, zero errors —
this measures request overhead; hybrid SQL latency at scale is benchmarked separately
(5,001 chunks, HNSW ≈ 78 ms).

### Runbook

| Symptom | Check | Action |
|---|---|---|
| Uploads return 202 but never become `ready` | `queue_depth`, worker logs, `ingestion_jobs` | Start the worker; check Gemini key/quota; check sweeper |
| Search returns fewer results than expected | `mode`, `dense_rank`/`sparse_rank` in the response | Check document ACL and status (`ready` only); raise `SEARCH_CANDIDATES`/`HNSW_EF_SEARCH` |
| Search is slow | `search_duration_seconds`, DB pool | Tune `HNSW_EF_SEARCH`; check DB CPU; `db_pool_in_use` near pool size → raise `DB_POOL_SIZE` |
| Documents stuck in `processing` | `ingestion_jobs.updated_at` | Sweeper requeues after 15 min; otherwise `POST .../reprocess` |
| Cross-tenant data suspected | logs, `tenant_id` predicates | Treat as an incident: revoke tokens, audit queries, restore isolation tests |
| Gemini 429 storm | `embedding_requests_total{status="error"}` | Lower `EMBEDDING_CONCURRENCY`/`EMBEDDING_BATCH_SIZE`; reprocess failed docs |
| Uploads return 503 | MinIO down | `docker compose up -d minio`; search still works (does not need MinIO) |
| Disk filling up | Postgres/MinIO volumes | Run `scripts/purge.py`; expand volume |

## Security

Isolation is enforced in depth, not just in application code:

- **Identity and tenancy** come only from the signed JWT; `X-Tenant-ID` and `X-Roles`
  headers are ignored. Roles are re-read from the database on every request, so a
  revoked or deactivated account takes effect immediately.
- **Tenant boundaries** are enforced twice: every query carries an explicit
  `tenant_id` predicate, and PostgreSQL row-level security on a `NOBYPASSRLS` runtime
  role enforces the same boundary if a predicate is ever missed. Tenant context is
  transaction-local, so pooled connections cannot leak it.
- **Authorization failures are indistinguishable**: cross-tenant and ACL-hidden
  resources both return `404`. Empty document ACLs mean tenant-wide; `tenant_admin`
  bypasses document ACLs but never tenant isolation.
- **Credentials**: passwords are Argon2-hashed; refresh tokens are stored hashed,
  rotated on every use, and family-revoked on reuse.
- **Ingestion** validates upload size while streaming, verifies MIME type by magic
  bytes, sanitizes file names, and keeps the object bucket private.
- **Input handling**: all SQL is parameterized and user text only reaches full-text
  search through `websearch_to_tsquery`.
- **Abuse protection**: login, registration, upload, and search are rate-limited
  (`429` with `Retry-After`); CORS is an explicit allowlist.
- **Operational hygiene**: secrets come from the environment, errors never expose
  stack traces or SQL, and logs exclude document text, query text, tokens, and
  passwords.

Not in v1: OCR, virus scanning, per-chunk ACLs, presigned downloads, dependency
scanning in CI, and security headers at a reverse proxy.
