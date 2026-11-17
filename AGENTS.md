# AGENTS.md

## Project

Multi-tenant hybrid search & RAG backend: ingest PDF/Markdown/HTML, chunk, embed with
Gemini, and serve ACL-safe hybrid retrieval (pgvector dense + Postgres full-text, fused
with RRF). There is **no LLM answer generation**: `/search` returns ranked chunks.

## Source of truth

- Spec: `dev/docs/implementation-plan.md` (local, gitignored). Read the relevant section
  before changing behavior: decisions §2, data model §7, tenancy/RLS §8, ingestion §9,
  chunking §10, embeddings §11, search §12, API §13, config §14, testing §17, phases §19.
- Background: `dev/docs/overview.md` (local). Where the two conflict, the plan wins.
- §2 "Decisions" is settled. §22 open questions do not block Phases 0–6.

## Status

Phases 0–6 are implemented and committed: auth/tenancy, documents, ingestion worker,
embeddings, hybrid search, hardening (rate limits, metrics, purge tooling). Run
`uv run pytest` to see the passing suite. New work must follow the invariants below
and the phase order in the (local) plan. Put new code at the repo root per the layout
in §6 (`app/`, `alembic/`, `tests/`, `scripts/`), not under `dev/`.

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

## Container image (build rules)

There is exactly **one** image, `winnow-api:latest`, built from the root `Dockerfile`
with `build: .` in `docker-compose.yml`. Both `api` and `worker` reuse it and differ
only by `command`. Never add a second Dockerfile or a per-service image.

### Canonical build

Run from WSL inside the repo root (not PowerShell, to avoid path translation):

```bash
docker compose build api        # builds winnow-api; worker reuses the same image
docker compose up -d            # api, worker, postgres, redis, minio
docker compose ps
curl -fsS http://localhost:8000/healthz
```

- Build **once** with `docker compose build api`. Do not run `--build` on every
  `up` and do not build `api` and `worker` separately — that only churns cache.
- If compose is unavailable, `docker buildx build --load -t winnow-api:latest .`
  is acceptable; the tag must stay `winnow-api:latest`.
- **Never** pass `--no-cache` unless `pyproject.toml`/`uv.lock` changed or the
  build is provably broken. Each no-cache rebuild recreates every layer and is the
  main source of multi-GB BuildKit cache growth.
- Do not add `--pull` to routine builds. Base images are pinned:
  `python:3.12-slim` and `ghcr.io/astral-sh/uv:0.13.1`.

### Layer and dependency rules

- `COPY pyproject.toml uv.lock ./` + `uv sync --frozen --no-dev --no-install-project`
  must stay **before** `COPY . .`. Never move source copying above dependency
  installation.
- Only `uv sync --frozen`. Never run `uv lock`, `uv add`, or `pip install` inside
  the build; update `uv.lock` in the workspace and commit it.
- Any new OS package goes in one combined layer placed before `COPY . .`:
  `RUN apt-get update && apt-get install -y --no-install-recommends <pkgs> && rm -rf /var/lib/apt/lists/*`
- Keep `USER appuser` (uid 10001). The image must never run as root.
- Runtime config comes from `env_file: .env` in compose. Never `COPY .env`,
  keys, or tokens into the image.

### `.dockerignore` is part of the build contract

The build context stays under 1 MB. Keep excluding `.git`,
`.venv` (~488 MB), `dev/`, `tests/`, `docker/`, `deploy/`, `__pycache__`, `*.py[cod]`,
`.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `.env`. Also add if missing:
`AGENTS.md`, `README.md`, `.github/`, `.gitattributes`, `.gitignore`.

Verify the context stayed small with:

```bash
du -sh --exclude=.git --exclude=.venv --exclude=dev --exclude=tests \
      --exclude=docker --exclude=.pytest_cache --exclude=.mypy_cache \
      --exclude=.ruff_cache .
```

Never add a `COPY` of files that `.dockerignore` excludes.

### After a build session: cap the cache

BuildKit cache lives in `/var/lib/containerd` inside WSL and grows ~1 GB per
rebuild of this image. Reclaim it after heavy rebuild days:

```bash
docker system df                              # inspect RECLAIMABLE
docker buildx prune -f --keep-storage=3GB     # keep recent layers, drop older ones
docker image prune -f                         # dangling <none> images only
```

**Never** run `docker system prune -a --volumes`, `docker volume prune`, or
`docker compose down -v`: they delete tagged images and the `pgdata` /
`redisdata` / `miniodata` volumes (all tenant data). Tagged images and named
volumes are off-limits unless the user explicitly asks.

If WSL storage is tight:

```bash
sudo fstrim -av
# then in Windows: wsl --shutdown
# elevated: diskpart -> select vdisk file="...\ext4.vhdx"
#           -> attach vdisk readonly -> compact vdisk -> detach vdisk
```

The `ext4.vhdx` is non-sparse, so it only shrinks after `fstrim` plus an offline
compact. Do not use `wsl --manage Ubuntu --set-sparse true --allow-unsafe`.

### Verify the build

```bash
docker compose config --images    # only winnow-api for api/worker
docker image inspect winnow-api:latest \
  --format '{{.Config.User}} | {{.Config.Cmd}} | {{.Config.ExposedPorts}}'
# expect: appuser | [uvicorn app.main:app --host 0.0.0.0 --port 8000] | map[8000/tcp:{}]
docker compose run --rm --no-deps api python -c "import app.main; print('ok')"
```

`tests/` is excluded from the image, so run `uv run pytest` on the host, not in
the container. `docker compose run api uv run pytest` will fail by design.
