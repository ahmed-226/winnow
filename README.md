# Multi-Tenant Hybrid Search & RAG Engine

Tenant-isolated ingestion and hybrid retrieval backend: PDF/Markdown/HTML in, ACL-safe
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
