"""Phase 5 integration tests: hybrid search, ACL matrix, fusion, degraded mode.

Covers plan §17.2 (mandatory ACL matrix through search), §17.4 (hybrid quality
fixtures, RRF math against the SQL), §12.4 (sparse-only fallback), index-usage
verification, and a recall benchmark against an exact scan.
"""

from __future__ import annotations

import json
import math
import random
import uuid
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.schemas import SearchRequest
from app.security import CurrentUser
from app.services.embeddings import EmbeddingRequestError
from app.services.search import SearchService, rrf_score
from app.workers.ingest import ensure_group, handle_message
from tests.integration.helpers import bearer, create_user, login, register_tenant
from tests.integration.test_ingest_pipeline import get_document, make_worker_context

_SEED_TOKEN = "zgrix"


def _unique_body(tag: str) -> str:
    """A body with a rare token so a keyword query can single it out."""
    return f"Synergy quantum flux {tag}. The {_SEED_TOKEN} protocol applies here. " * 12


async def drain(app: Any) -> None:
    settings = app.state.settings
    await ensure_group(app.state.redis, stream=settings.ingest_stream, group=settings.ingest_group)
    batches = await app.state.redis.xreadgroup(
        settings.ingest_group, "test-consumer", {settings.ingest_stream: ">"}, count=10
    )
    ctx = make_worker_context(app)
    for _stream, entries in batches:
        for message_id, fields in entries:
            await handle_message(ctx, app.state.redis, message_id, fields)


async def upload_and_wait(
    client: AsyncClient,
    app: Any,
    token: str,
    *,
    body: str,
    name: str = "doc.md",
    title: str = "Test Document",
    roles: list[str] | None = None,
) -> dict[str, Any]:
    data: dict[str, str | list[str]] = {"title": title}
    if roles is not None:
        data["access_roles"] = roles
    response = await client.post(
        "/api/v1/documents",
        files={"file": (name, body.encode(), "text/markdown")},
        data=data,
        headers=bearer(token),
    )
    assert response.status_code == 202, response.text
    document_id = str(response.json()["id"])
    await drain(app)
    final = await get_document(client, token, document_id)
    assert final["status"] == "ready", final
    return final


async def search(client: AsyncClient, token: str, query: str, *, top_k: int | None = None) -> Any:
    payload: dict[str, Any] = {"query": query}
    if top_k is not None:
        payload["top_k"] = top_k
    return await client.post("/api/v1/search", json=payload, headers=bearer(token))


async def fetch_chunk_contents(app: Any, tenant_id: str, document_id: str) -> list[str]:
    async with app.state.sessionmaker() as session:
        async with session.begin():
            await session.execute(
                text("SELECT set_config('app.tenant_id', :tid, true)"), {"tid": tenant_id}
            )
            rows = (
                await session.execute(
                    text(
                        "SELECT content FROM document_chunks "
                        "WHERE document_id = :doc ORDER BY chunk_index"
                    ),
                    {"doc": document_id},
                )
            ).scalars()
    return list(rows)


# ------------------------------------------------------------------ basic flow


async def test_search_returns_hybrid_results_with_ranks(app, client: AsyncClient) -> None:
    await register_tenant(client, slug="acme")
    tokens = await login(client, slug="acme")
    token = str(tokens["access_token"])
    document = await upload_and_wait(client, app, token, body=_unique_body("alpha"), title="Alpha")

    response = await search(client, token, _SEED_TOKEN)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["query"] == _SEED_TOKEN
    assert payload["mode"] == "hybrid"
    assert isinstance(payload["took_ms"], int)
    assert payload["results"], "expected at least one chunk"
    for item in payload["results"]:
        assert item["document_id"] == str(document["id"])
        assert item["title"] == "Alpha"
        assert item["dense_rank"] is not None or item["sparse_rank"] is not None
        assert item["score"] > 0
        assert isinstance(item["metadata"], dict)
    # Rare-token query: the keyword leg must find the chunk.
    assert any(item["sparse_rank"] is not None for item in payload["results"])


async def test_exact_content_query_is_dense_top1_and_hybrid_top1(app, client: AsyncClient) -> None:
    await register_tenant(client, slug="acme")
    tokens = await login(client, slug="acme")
    token = str(tokens["access_token"])
    tenant_id = str(tokens["user"]["tenant_id"])
    target = await upload_and_wait(client, app, token, body=_unique_body("bravo"))
    # A second document with disjoint vocabulary: without it, top-1 agreement
    # would be trivial.
    await upload_and_wait(client, app, token, body=_unique_body("charlie"), name="other.md")

    contents = await fetch_chunk_contents(app, tenant_id, str(target["id"]))
    query = contents[0]

    response = await search(client, token, query)
    assert response.status_code == 200, response.text
    results = response.json()["results"]
    assert results
    top = results[0]
    # Identical text -> identical vector -> cosine distance 0 -> dense rank 1,
    # and the full keyword overlap -> sparse rank 1 -> hybrid top 1.
    assert top["dense_rank"] == 1
    assert top["sparse_rank"] == 1
    assert top["chunk_id"] == next(item["chunk_id"] for item in results if item["dense_rank"] == 1)
    assert top["chunk_id"] == next(item["chunk_id"] for item in results if item["sparse_rank"] == 1)


async def test_rrf_scores_match_the_sql_formula_and_ordering(app, client: AsyncClient) -> None:
    await register_tenant(client, slug="acme")
    tokens = await login(client, slug="acme")
    token = str(tokens["access_token"])
    await upload_and_wait(client, app, token, body=_unique_body("delta"))
    await upload_and_wait(client, app, token, body=_unique_body("echo"), name="other.md")

    response = await search(client, token, _SEED_TOKEN, top_k=20)
    assert response.status_code == 200, response.text
    results = response.json()["results"]
    settings = app.state.settings
    scores = []
    for item in results:
        expected = rrf_score(
            item["dense_rank"],
            item["sparse_rank"],
            k=settings.rrf_k,
            dense_weight=settings.dense_weight,
            sparse_weight=settings.sparse_weight,
        )
        assert item["score"] == pytest.approx(expected, abs=1e-9)
        scores.append(item["score"])
    assert scores == sorted(scores, reverse=True)


# ------------------------------------------------------------------ ACL matrix


@pytest.mark.parametrize(
    ("doc_roles", "user_roles", "visible"),
    [
        ([], [], True),  # case 1
        ([], ["engineering"], True),  # case 2
        (["engineering"], ["engineering"], True),  # case 3
        (["engineering"], ["hr"], False),  # case 4
        (["engineering"], [], False),  # case 5
        (["engineering", "hr"], ["hr"], True),  # case 6
    ],
    ids=[
        "1-empty-any",
        "2-empty-engineering",
        "3-overlap",
        "4-no-overlap",
        "5-empty-user",
        "6-multi-overlap",
    ],
)
async def test_acl_matrix(
    app, client: AsyncClient, doc_roles: list[str], user_roles: list[str], visible: bool
) -> None:
    await register_tenant(client, slug="acme")
    tokens = await login(client, slug="acme")
    admin_token = str(tokens["access_token"])
    await upload_and_wait(client, app, admin_token, body=_unique_body("matrix"), roles=doc_roles)

    email = f"member-{uuid.uuid4().hex[:8]}@acme.example.com"
    await create_user(client, admin_token=admin_token, email=email, roles=user_roles)
    member = await login(client, slug="acme", email=email)

    response = await search(client, str(member["access_token"]), _SEED_TOKEN)
    assert response.status_code == 200, response.text
    results = response.json()["results"]
    if visible:
        assert results, f"doc with {doc_roles} should be visible to {user_roles}"
        assert all(item["title"] for item in results)
    else:
        assert results == [], f"doc with {doc_roles} must be hidden from {user_roles}"


async def test_acl_matrix_case7_admin_bypasses_document_acl(app, client: AsyncClient) -> None:
    await register_tenant(client, slug="acme")
    tokens = await login(client, slug="acme")
    admin_token = str(tokens["access_token"])
    await upload_and_wait(
        client, app, admin_token, body=_unique_body("hotel"), roles=["engineering"]
    )

    response = await search(client, admin_token, _SEED_TOKEN)
    assert response.status_code == 200, response.text
    results = response.json()["results"]
    assert results, "tenant_admin bypasses document ACLs (but never tenant isolation)"


async def test_acl_matrix_cases_8_9_cross_tenant_is_hidden(app, client: AsyncClient) -> None:
    await register_tenant(client, slug="acme")
    acme = await login(client, slug="acme")
    await upload_and_wait(client, app, str(acme["access_token"]), body=_unique_body("india"))

    # Case 9: tenant B user with no roles; case 8 is the admin variant below.
    await register_tenant(client, slug="globex")
    globex_admin = await login(client, slug="globex")
    other_email = f"member-{uuid.uuid4().hex[:8]}@globex.example.com"
    await create_user(
        client, admin_token=str(globex_admin["access_token"]), email=other_email, roles=[]
    )
    other_member = await login(client, slug="globex", email=other_email)

    for token in (globex_admin["access_token"], other_member["access_token"]):
        response = await search(client, str(token), _SEED_TOKEN)
        assert response.status_code == 200, response.text
        assert response.json()["results"] == [], "cross-tenant content must never leak"


# --------------------------------------------------------------- degraded mode


class _FailingQueryProvider:
    """Simulates a query-embedding outage; the service must fall back."""

    model = "fake-embedding-001"
    dimension = 1536

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        raise EmbeddingRequestError("outage")

    async def embed_query(self, text: str) -> list[float]:
        raise EmbeddingRequestError("outage")


async def test_embedding_failure_degrades_to_sparse_only(app, client: AsyncClient) -> None:
    await register_tenant(client, slug="acme")
    tokens = await login(client, slug="acme")
    token = str(tokens["access_token"])
    await upload_and_wait(client, app, token, body=_unique_body("juliet"))

    app.state.embeddings = _FailingQueryProvider()
    response = await search(client, token, _SEED_TOKEN)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["mode"] == "sparse_only"
    assert payload["results"], "sparse-only fallback must still return keyword matches"
    for item in payload["results"]:
        assert item["dense_rank"] is None
        assert item["cosine_distance"] is None
        assert item["sparse_rank"] is not None


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "x" * 1001},
        {"query": "bad\x00query"},
        {"query": _SEED_TOKEN, "top_k": 0},
        {"query": _SEED_TOKEN, "top_k": 51},
        {"query": _SEED_TOKEN, "extra": True},
    ],
    ids=["empty", "blank", "too-long", "control-char", "topk-0", "topk-51", "extra-key"],
)
async def test_search_validation_rejects(app, client: AsyncClient, payload: dict[str, Any]) -> None:
    await register_tenant(client, slug="acme")
    tokens = await login(client, slug="acme")
    response = await client.post(
        "/api/v1/search", json=payload, headers=bearer(str(tokens["access_token"]))
    )
    assert response.status_code == 422, response.text


async def test_search_validation_boundaries_accepted(app, client: AsyncClient) -> None:
    await register_tenant(client, slug="acme")
    tokens = await login(client, slug="acme")
    token = str(tokens["access_token"])
    # Exactly 1000 characters, empty tenant: valid request, empty results.
    response = await search(client, token, "word " * 200)
    assert response.status_code == 200, response.text
    assert response.json()["results"] == []
    assert SearchRequest(query="x", top_k=50).top_k == 50


async def test_search_requires_auth(app, client: AsyncClient) -> None:
    response = await client.post("/api/v1/search", json={"query": _SEED_TOKEN})
    assert response.status_code == 401


# ------------------------------------------------------ index usage + recall


async def test_explain_uses_hnsw_and_gin(app, test_database, client: AsyncClient) -> None:
    await register_tenant(client, slug="acme")
    tokens = await login(client, slug="acme")
    token = str(tokens["access_token"])
    tenant_id = str(tokens["user"]["tenant_id"])
    document = await upload_and_wait(client, app, token, body=_unique_body("kilo"))

    # A small table lets the planner order via the tenant index + sort; at
    # realistic scale (the plan's exit criterion is a 100k-chunk corpus) the
    # HNSW scan is the cheapest plan, so use enough filler rows to reproduce
    # that shape locally.
    owner_engine = create_async_engine(test_database.owner_url)
    try:
        async with owner_engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO document_chunks (document_id, tenant_id, chunk_index, "
                    "content, embedding, embedding_model) "
                    "SELECT :doc, :tid, 1000 + i, 'filler chunk unrelated words ' || i, "
                    "('[' || (SELECT string_agg((((i * j) % 1000)::float8 / 1000.0)::text, ',') "
                    "FROM generate_series(1, 1536) AS g(j)) || ']')::vector, 'filler' "
                    "FROM generate_series(0, 4999) AS i"
                ),
                {"doc": document["id"], "tid": tenant_id},
            )
            # Refresh stats and flush the GIN pending list so the planner sees
            # the steady-state cost: a bulk insert leaves thousands of entries
            # in the fastupdate pending list, which makes the GIN bitmap scan
            # look more expensive than the tenant btree. In production
            # autovacuum does this; here we do it explicitly.
            await conn.execute(text("SELECT gin_clean_pending_list('chunks_fts_gin')"))
            await conn.execute(text("ANALYZE document_chunks"))
            await conn.execute(text("ANALYZE documents"))
    finally:
        await owner_engine.dispose()

    vec = "[" + ",".join(["0.1"] * 1536) + "]"

    async with app.state.sessionmaker() as session:
        async with session.begin():
            await session.execute(
                text("SELECT set_config('app.tenant_id', :tid, true)"), {"tid": tenant_id}
            )
            # Mirror the service's per-transaction HNSW settings (§12.3).
            await session.execute(
                text("SELECT set_config('hnsw.iterative_scan', 'strict_order', true)")
            )
            await session.execute(text("SELECT set_config('hnsw.ef_search', '100', true)"))
            await session.execute(text("SELECT set_config('hnsw.max_scan_tuples', '20000', true)"))
            await session.execute(text("SET LOCAL enable_seqscan = off"))
            # Same shape as the production CTEs: the tenant predicate goes
            # through the `p` CTE. A direct parameter would fold into the RLS
            # policy qual and stop the planner from using the GIN index.
            dense_plan = " ".join(
                json.dumps(row) if not isinstance(row, str) else row
                for row in (
                    await session.execute(
                        text(
                            "EXPLAIN (FORMAT JSON) "
                            "WITH p AS (SELECT CAST(:tid AS uuid) AS tenant_id) "
                            "SELECT c.id FROM document_chunks c "
                            "JOIN documents d ON d.id = c.document_id "
                            "CROSS JOIN p "
                            "WHERE c.tenant_id = p.tenant_id AND d.tenant_id = p.tenant_id "
                            "AND d.deleted_at IS NULL AND d.status = 'ready' "
                            "ORDER BY c.embedding <=> CAST(:v AS vector), c.id LIMIT 50"
                        ),
                        {"tid": tenant_id, "v": vec},
                    )
                ).scalars()
            )
            sparse_plan = " ".join(
                json.dumps(row) if not isinstance(row, str) else row
                for row in (
                    await session.execute(
                        text(
                            "EXPLAIN (FORMAT JSON) "
                            "WITH p AS (SELECT CAST(:tid AS uuid) AS tenant_id) "
                            "SELECT c.id FROM document_chunks c "
                            "CROSS JOIN p "
                            "WHERE c.tenant_id = p.tenant_id "
                            "AND c.fts_doc @@ websearch_to_tsquery('english', :q)"
                        ),
                        {"tid": tenant_id, "q": _SEED_TOKEN},
                    )
                ).scalars()
            )

    assert str(document["id"])  # data exists
    assert "chunks_embedding_hnsw" in dense_plan, dense_plan
    assert "chunks_fts_gin" in sparse_plan, sparse_plan


class _FixedVectorProvider:
    """Returns one fixed vector for every query; for the recall benchmark."""

    model = "bench-embedding"
    dimension = 1536

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        return [self._vector for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector


async def test_recall_hnsw_vs_exact_scan(app, test_database, client: AsyncClient) -> None:
    """Recall benchmark vs exact scan on a synthetic corpus (plan §12.3).

    1,500 random vectors in one tenant; recall@10 of the HNSW path
    (iterative_scan=strict_order, ef_search=100) must be at least 0.9 vs the
    exact sequential-scan answer. Latencies are printed for the phase record.
    """
    import time

    owner_engine = create_async_engine(test_database.owner_url)
    chunk_count = 1_500
    random.seed(42)

    def random_vector() -> tuple[list[float], str]:
        raw = [random.uniform(-1.0, 1.0) for _ in range(1536)]
        norm = math.sqrt(sum(value * value for value in raw))
        normalized = [value / norm for value in raw]
        literal = "[" + ",".join(f"{value:.6f}" for value in normalized) + "]"
        return normalized, literal

    vectors: list[str] = []
    for _ in range(chunk_count):
        vectors.append(random_vector()[1])
    query_vec, query_literal = random_vector()

    try:
        async with owner_engine.begin() as conn:
            tenant_id = (
                await conn.execute(
                    text("INSERT INTO tenants (name, slug) VALUES ('Bench', :slug) RETURNING id"),
                    {"slug": f"bench-{uuid.uuid4().hex[:10]}"},
                )
            ).scalar_one()
            document_id = (
                await conn.execute(
                    text(
                        "INSERT INTO documents (tenant_id, title, file_name, mime_type, "
                        "source_type, storage_key, file_hash, size_bytes, access_roles, "
                        "status) VALUES (:tid, 'Bench corpus', 'bench.md', 'text/markdown', "
                        "'markdown', 'bench/key', :file_hash, 10, '{}'::text[], 'ready') "
                        "RETURNING id"
                    ),
                    {"tid": tenant_id, "file_hash": uuid.uuid4().hex + uuid.uuid4().hex},
                )
            ).scalar_one()
            await conn.execute(
                text(
                    "INSERT INTO document_chunks (document_id, tenant_id, chunk_index, "
                    "content, metadata, embedding, embedding_model) "
                    "VALUES (:doc, :tid, :i, :content, '{}'::jsonb, CAST(:vec AS vector), "
                    "'bench-embedding')"
                ),
                [
                    {
                        "doc": document_id,
                        "tid": tenant_id,
                        "i": index,
                        "content": f"Bench chunk {index} lorem ipsum dolor sit amet",
                        "vec": vectors[index],
                    }
                    for index in range(chunk_count)
                ],
            )
    finally:
        await owner_engine.dispose()

    service = SearchService(app.state.settings, _FixedVectorProvider(query_vec))
    user = CurrentUser(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        email="bench@bench.example.com",
        roles=(),
        is_active=True,
    )

    async with app.state.sessionmaker() as session:
        async with session.begin():
            await session.execute(
                text("SELECT set_config('app.tenant_id', :tid, true)"), {"tid": str(tenant_id)}
            )
            started = time.monotonic()
            # A token that matches no chunk text keeps the sparse CTE empty, so
            # the fused ordering is exactly the HNSW dense ordering and the
            # comparison below is a pure ANN recall measurement.
            result = await service.search(session, user=user, query="zzqq", top_k=10)
            hnsw_seconds = time.monotonic() - started

    hnsw_ids = [row.chunk_id for row in result.rows]
    assert result.mode == "hybrid"
    assert len(hnsw_ids) == 10
    # With an empty sparse leg every row must carry a dense rank.
    assert all(row.dense_rank is not None for row in result.rows)

    async with owner_engine.begin() as conn:
        await conn.execute(text("SET LOCAL enable_indexscan = off"))
        await conn.execute(text("SET LOCAL enable_bitmapscan = off"))
        started = time.monotonic()
        exact_rows = (
            await conn.execute(
                text(
                    "SELECT c.id FROM document_chunks c WHERE c.tenant_id = :tid "
                    "ORDER BY c.embedding <=> CAST(:v AS vector), c.id LIMIT 10"
                ),
                {"tid": tenant_id, "v": query_literal},
            )
        ).scalars()
        exact_seconds = time.monotonic() - started
        exact_ids = list(exact_rows)

    overlap = len(set(hnsw_ids) & set(exact_ids))
    recall = overlap / 10
    print(
        f"\n[recall-benchmark] chunks={chunk_count} recall@10={recall} "
        f"hnsw_ms={hnsw_seconds * 1000:.1f} exact_ms={exact_seconds * 1000:.1f}"
    )
    assert recall >= 0.9, f"HNSW recall@10={recall} vs exact scan"
