"""Integration tests for the Phase 3 ingestion pipeline.

Drives the real worker functions (claim, parse, chunk, embed, commit, sweep)
against the test Postgres + fakeredis + moto stack.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pymupdf
import pytest
from httpx import AsyncClient
from sqlalchemy import text

from app.services.embeddings import EmbeddingRequestError, FakeEmbeddingProvider
from app.services.ingestion import WorkerContext, claim_job
from app.workers.ingest import ensure_group, handle_message, run_sweep
from tests.integration.helpers import bearer, create_user, login, register_tenant

_MD_BODY = "Parental leave is 16 weeks. " * 40


class _FailingProvider:
    model = "fake-embedding-001"
    dimension = 1536

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingRequestError("quota exhausted")

    async def embed_query(self, text: str) -> list[float]:
        raise EmbeddingRequestError("quota exhausted")


def make_worker_context(app: Any, provider: object | None = None) -> WorkerContext:
    return WorkerContext(
        settings=app.state.settings,
        sessionmaker=app.state.sessionmaker,
        storage=app.state.storage,
        embeddings=provider or FakeEmbeddingProvider(app.state.settings.embedding_dimension),
    )


async def drain(app: Any, ctx: WorkerContext, *, limit: int = 10) -> int:
    """Read and process pending stream messages exactly once (no blocking)."""
    settings = app.state.settings
    await ensure_group(app.state.redis, stream=settings.ingest_stream, group=settings.ingest_group)
    batches = await app.state.redis.xreadgroup(
        settings.ingest_group, "test-consumer", {settings.ingest_stream: ">"}, count=limit
    )
    handled = 0
    for _stream, entries in batches:
        for message_id, fields in entries:
            await handle_message(ctx, app.state.redis, message_id, fields)
            handled += 1
    return handled


async def stream_length(app: Any, stream: str) -> int:
    return int(await app.state.redis.xlen(stream))


async def fetch_job(app: Any, tenant_id: str, document_id: str) -> dict[str, Any]:
    async with app.state.sessionmaker() as session:
        async with session.begin():
            await session.execute(
                text("SELECT set_config('app.tenant_id', :tid, true)"), {"tid": tenant_id}
            )
            row = (
                await session.execute(
                    text(
                        "SELECT id::text, status::text, attempts, max_attempts, last_error "
                        "FROM ingestion_jobs WHERE document_id = :doc"
                    ),
                    {"doc": document_id},
                )
            ).one()
    return dict(row._mapping)


async def fetch_chunks(app: Any, tenant_id: str, document_id: str) -> list[Any]:
    async with app.state.sessionmaker() as session:
        async with session.begin():
            await session.execute(
                text("SELECT set_config('app.tenant_id', :tid, true)"), {"tid": tenant_id}
            )
            return (
                await session.execute(
                    text(
                        "SELECT chunk_index, content, vector_dims(embedding) AS dims, "
                        "embedding_model, metadata FROM document_chunks "
                        "WHERE document_id = :doc ORDER BY chunk_index"
                    ),
                    {"doc": document_id},
                )
            ).all()


async def upload_markdown(
    client: AsyncClient, token: str, *, name: str = "handbook.md", body: str = _MD_BODY
) -> dict[str, Any]:
    response = await client.post(
        "/api/v1/documents",
        files={"file": (name, body.encode(), "text/markdown")},
        data={"title": "Employee Handbook"},
        headers=bearer(token),
    )
    assert response.status_code == 202, response.text
    return response.json()


async def get_document(client: AsyncClient, token: str, document_id: str) -> dict[str, Any]:
    response = await client.get(f"/api/v1/documents/{document_id}", headers=bearer(token))
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture
async def acme_admin(client: AsyncClient) -> AsyncIterator[dict[str, str]]:
    await register_tenant(client, slug="acme")
    tokens = await login(client, slug="acme")
    user = tokens["user"]
    assert isinstance(user, dict)
    yield {
        "token": str(tokens["access_token"]),
        "tenant_id": str(user["tenant_id"]),
    }


async def test_upload_ingests_to_ready(
    client: AsyncClient, app: Any, acme_admin: dict[str, str]
) -> None:
    document = await upload_markdown(client, acme_admin["token"])
    assert document["status"] == "pending"

    handled = await drain(app, make_worker_context(app))
    assert handled == 1

    final = await get_document(client, acme_admin["token"], str(document["id"]))
    assert final["status"] == "ready"
    assert final["chunk_count"] > 0

    chunks = await fetch_chunks(app, acme_admin["tenant_id"], str(document["id"]))
    assert len(chunks) == final["chunk_count"]
    assert all(chunk.dims == 1536 for chunk in chunks)
    assert all(chunk.embedding_model == "fake-embedding-001" for chunk in chunks)
    raw_metadata = chunks[0].metadata
    metadata = raw_metadata if isinstance(raw_metadata, dict) else json.loads(raw_metadata)
    assert metadata["page_start"] is None
    assert metadata["char_start"] == 0
    assert "Employee Handbook" in final["title"]


async def test_indexes_exist(app: Any) -> None:
    async with app.state.sessionmaker() as session:
        names = {
            row[0]
            for row in await session.execute(
                text("SELECT indexname FROM pg_indexes WHERE tablename = 'document_chunks'")
            )
        }
    assert {"chunks_embedding_hnsw", "chunks_fts_gin", "chunks_tenant_idx"} <= names


async def test_duplicate_message_is_idempotent(
    client: AsyncClient, app: Any, acme_admin: dict[str, str]
) -> None:
    document = await upload_markdown(client, acme_admin["token"])
    await drain(app, make_worker_context(app))
    ready = await get_document(client, acme_admin["token"], str(document["id"]))
    assert ready["status"] == "ready"

    # Redeliver the same job (e.g. another worker got the same message).
    job = await fetch_job(app, acme_admin["tenant_id"], str(document["id"]))
    await app.state.redis.xadd(
        app.state.settings.ingest_stream,
        {
            "version": "1",
            "job_id": job["id"],
            "document_id": str(document["id"]),
            "tenant_id": acme_admin["tenant_id"],
            "enqueued_at": "2026-11-12T00:00:00+00:00",
        },
    )
    await drain(app, make_worker_context(app))

    after = await get_document(client, acme_admin["token"], str(document["id"]))
    assert after["status"] == "ready"
    assert after["chunk_count"] == ready["chunk_count"]
    chunks = await fetch_chunks(app, acme_admin["tenant_id"], str(document["id"]))
    assert len(chunks) == ready["chunk_count"]


async def test_reprocess_replaces_chunks_without_duplicates(
    client: AsyncClient, app: Any, acme_admin: dict[str, str]
) -> None:
    document = await upload_markdown(client, acme_admin["token"])
    await drain(app, make_worker_context(app))
    ready = await get_document(client, acme_admin["token"], str(document["id"]))
    assert ready["status"] == "ready"
    ready_chunks = await fetch_chunks(app, acme_admin["tenant_id"], str(document["id"]))

    response = await client.post(
        f"/api/v1/documents/{document['id']}/reprocess", headers=bearer(acme_admin["token"])
    )
    assert response.status_code == 202, response.text
    reset = response.json()
    assert reset["status"] == "pending"

    await drain(app, make_worker_context(app))
    final = await get_document(client, acme_admin["token"], str(document["id"]))
    assert final["status"] == "ready"
    assert final["chunk_count"] == ready["chunk_count"]
    final_chunks = await fetch_chunks(app, acme_admin["tenant_id"], str(document["id"]))
    indices = [chunk.chunk_index for chunk in final_chunks]
    assert len(indices) == len(set(indices)) == len(ready_chunks)


async def test_reprocess_forbidden_for_member(
    client: AsyncClient, app: Any, acme_admin: dict[str, str]
) -> None:
    await create_user(client, admin_token=acme_admin["token"], email="alice@acme.example.com")
    document = await upload_markdown(client, acme_admin["token"])
    tokens = await login(client, slug="acme", email="alice@acme.example.com")
    response = await client.post(
        f"/api/v1/documents/{document['id']}/reprocess",
        headers=bearer(str(tokens["access_token"])),
    )
    assert response.status_code == 403


async def test_no_text_pdf_fails_cleanly(
    client: AsyncClient, app: Any, acme_admin: dict[str, str]
) -> None:
    document = pymupdf.open()
    document.new_page()  # blank page: no extractable text
    data = document.tobytes()
    document.close()

    uploaded = await client.post(
        "/api/v1/documents",
        files={"file": ("scan.pdf", data, "application/pdf")},
        headers=bearer(acme_admin["token"]),
    )
    assert uploaded.status_code == 202, uploaded.text
    document_id = str(uploaded.json()["id"])

    await drain(app, make_worker_context(app))
    failed = await get_document(client, acme_admin["token"], document_id)
    assert failed["status"] == "failed"
    assert failed["error_message"] is not None
    assert "no_extractable_text" in failed["error_message"]
    assert await fetch_chunks(app, acme_admin["tenant_id"], document_id) == []


async def test_bad_pdf_fails_cleanly(
    client: AsyncClient, app: Any, acme_admin: dict[str, str]
) -> None:
    uploaded = await client.post(
        "/api/v1/documents",
        files={"file": ("broken.pdf", b"%PDF-1.4 garbage bytes", "application/pdf")},
        headers=bearer(acme_admin["token"]),
    )
    assert uploaded.status_code == 202, uploaded.text
    document_id = str(uploaded.json()["id"])

    await drain(app, make_worker_context(app))
    failed = await get_document(client, acme_admin["token"], document_id)
    assert failed["status"] == "failed"
    assert failed["error_message"] is not None


async def test_sweeper_recovers_lost_job(
    client: AsyncClient, app: Any, acme_admin: dict[str, str]
) -> None:
    document = await upload_markdown(client, acme_admin["token"])
    tenant_id = acme_admin["tenant_id"]
    document_id = str(document["id"])
    job = await fetch_job(app, tenant_id, document_id)

    # Simulate a worker crash after claiming: job running, message never acked.
    ctx = make_worker_context(app)
    claimed = await claim_job(ctx, job_id=uuid.UUID(job["id"]), tenant_id=uuid.UUID(tenant_id))
    assert hasattr(claimed, "storage_key")

    before = await stream_length(app, app.state.settings.ingest_stream)
    await run_sweep(ctx, app.state.redis, queued_stale_seconds=0, running_stale_seconds=0)
    after = await stream_length(app, app.state.settings.ingest_stream)
    assert after > before

    await drain(app, ctx)
    final = await get_document(client, acme_admin["token"], document_id)
    assert final["status"] == "ready"
    assert final["chunk_count"] > 0


async def test_dead_letter_after_exhausted_attempts_then_reprocess_recovers(
    client: AsyncClient, app: Any, acme_admin: dict[str, str]
) -> None:
    tenant_id = acme_admin["tenant_id"]
    document = await upload_markdown(client, acme_admin["token"])
    document_id = str(document["id"])
    failing_ctx = make_worker_context(app, _FailingProvider())

    # One attempt only, so the failure is terminal immediately.
    async with app.state.sessionmaker() as session:
        async with session.begin():
            await session.execute(
                text("SELECT set_config('app.tenant_id', :tid, true)"), {"tid": tenant_id}
            )
            await session.execute(
                text("UPDATE ingestion_jobs SET max_attempts = 1 WHERE document_id = :doc"),
                {"doc": document_id},
            )

    await drain(app, failing_ctx)
    failed = await get_document(client, acme_admin["token"], document_id)
    assert failed["status"] == "failed"
    assert failed["error_message"] is not None
    assert "embedding_failed" in failed["error_message"]
    job = await fetch_job(app, tenant_id, document_id)
    assert job["status"] == "failed"
    assert await stream_length(app, app.state.settings.ingest_dead_stream) == 1

    # Reprocess resets attempts and requeues; the good provider then succeeds.
    response = await client.post(
        f"/api/v1/documents/{document_id}/reprocess", headers=bearer(acme_admin["token"])
    )
    assert response.status_code == 202, response.text
    await drain(app, make_worker_context(app))
    final = await get_document(client, acme_admin["token"], document_id)
    assert final["status"] == "ready"
    assert final["chunk_count"] > 0


async def test_deleted_document_job_completes_without_chunks(
    client: AsyncClient, app: Any, acme_admin: dict[str, str]
) -> None:
    document = await upload_markdown(client, acme_admin["token"])
    document_id = str(document["id"])
    response = await client.delete(
        f"/api/v1/documents/{document_id}", headers=bearer(acme_admin["token"])
    )
    assert response.status_code == 204

    await drain(app, make_worker_context(app))
    job = await fetch_job(app, acme_admin["tenant_id"], document_id)
    assert job["status"] == "done"
    gone = await client.get(f"/api/v1/documents/{document_id}", headers=bearer(acme_admin["token"]))
    assert gone.status_code == 404


async def test_chunks_are_tenant_isolated(
    client: AsyncClient, app: Any, acme_admin: dict[str, str]
) -> None:
    document = await upload_markdown(client, acme_admin["token"])
    await drain(app, make_worker_context(app))

    # Without a tenant context, rag_app must see zero chunks (RLS).
    async with app.state.sessionmaker() as session:
        count = (
            await session.execute(
                text("SELECT count(*) FROM document_chunks WHERE document_id = :doc"),
                {"doc": str(document["id"])},
            )
        ).scalar_one()
    assert count == 0
