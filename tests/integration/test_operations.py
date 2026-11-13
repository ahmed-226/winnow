"""Phase 6 operations: rate limits, custom metrics, purge, orphan-object scan."""

from __future__ import annotations

import uuid
from pathlib import Path

import boto3
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.services.maintenance import (
    delete_orphan_objects,
    find_orphan_objects,
    purge_expired_documents,
)
from app.services.storage import StorageService
from tests.integration.helpers import bearer, login, register_tenant

MARKDOWN_CONTENT = b"# Employee Handbook\n\nParental leave is 16 weeks.\n"
OTHER_CONTENT = b"# Architecture\n\nThe ingestion pipeline is asynchronous.\n"


async def _setup_admin(client: AsyncClient, slug: str) -> str:
    await register_tenant(client, slug=slug)
    payload = await login(client, slug=slug)
    return str(payload["access_token"])


async def _upload(client: AsyncClient, token: str, *, content: bytes, name: str = "handbook.md"):
    return await client.post(
        "/api/v1/documents",
        files={"file": (name, content, "text/markdown")},
        headers=bearer(token),
    )


@pytest.fixture
def bucket_s3(app):
    settings = app.state.settings
    return boto3.client(
        "s3",
        region_name=settings.s3_region,
        aws_access_key_id=settings.s3_access_key_id,
        aws_secret_access_key=settings.s3_secret_access_key,
    )


async def test_search_rate_limit_returns_429_with_retry_after(app_factory) -> None:
    async with app_factory(rl_search_per_minute=2) as application:
        async with AsyncClient(
            transport=ASGITransport(app=application), base_url="http://test"
        ) as client:
            token = await _setup_admin(client, "rlsearch")
            headers = bearer(token)
            for _ in range(2):
                ok = await client.post(
                    "/api/v1/search", json={"query": "parental leave"}, headers=headers
                )
                assert ok.status_code == 200, ok.text
            blocked = await client.post(
                "/api/v1/search", json={"query": "parental leave"}, headers=headers
            )
            assert blocked.status_code == 429, blocked.text
            assert int(blocked.headers["retry-after"]) >= 1

            metrics = await client.get("/metrics")
            assert metrics.status_code == 200
            assert 'rate_limit_rejections_total{scope="search"}' in metrics.text
            assert 'search_requests_total{mode="hybrid",outcome="ok"}' in metrics.text


async def test_upload_rate_limit_returns_429(app_factory) -> None:
    async with app_factory(rl_upload_per_hour=1) as application:
        async with AsyncClient(
            transport=ASGITransport(app=application), base_url="http://test"
        ) as client:
            token = await _setup_admin(client, "rlupload")
            first = await _upload(client, token, content=MARKDOWN_CONTENT)
            assert first.status_code == 202, first.text
            second = await _upload(client, token, content=OTHER_CONTENT)
            assert second.status_code == 429, second.text
            assert "retry-after" in second.headers
            # The rejected upload left no object behind.
            settings = application.state.settings
            s3 = boto3.client(
                "s3",
                region_name=settings.s3_region,
                aws_access_key_id=settings.s3_access_key_id,
                aws_secret_access_key=settings.s3_secret_access_key,
            )
            keys = [
                entry["Key"]
                for entry in s3.list_objects_v2(Bucket=settings.s3_bucket).get("Contents", [])
            ]
            assert len(keys) == 1, keys


async def test_metrics_report_search_and_upload_counters(app, client: AsyncClient) -> None:
    token = await _setup_admin(client, "metrics")
    upload = await _upload(client, token, content=MARKDOWN_CONTENT)
    assert upload.status_code == 202, upload.text
    search = await client.post(
        "/api/v1/search", json={"query": "parental leave"}, headers=bearer(token)
    )
    assert search.status_code == 200, search.text

    metrics = await client.get("/metrics")
    assert 'documents_uploaded_total{duplicate="false",source_type="markdown"}' in metrics.text
    assert 'search_requests_total{mode="hybrid",outcome="ok"}' in metrics.text
    assert "search_duration_seconds_count" in metrics.text
    assert "search_results_count_count" in metrics.text
    assert "db_pool_in_use" in metrics.text
    assert 'queue_depth{stream="rag:ingest"}' in metrics.text
    assert 'queue_pending_messages{group="ingest-workers"}' in metrics.text


async def test_purge_expired_documents(app, client: AsyncClient, test_database, bucket_s3) -> None:
    token = await _setup_admin(client, "purge")
    upload = await _upload(client, token, content=MARKDOWN_CONTENT)
    assert upload.status_code == 202, upload.text
    document_id = upload.json()["id"]

    deleted = await client.delete(f"/api/v1/documents/{document_id}", headers=bearer(token))
    assert deleted.status_code == 204, deleted.text

    settings = app.state.settings
    storage = StorageService(settings)

    # Within the retention window nothing is purged.
    report = await purge_expired_documents(settings, storage=storage)
    assert (report.purged_documents, report.object_deletes_failed) == (0, 0)

    # Backdate the soft delete past the retention window, then purge.
    engine = create_async_engine(test_database.admin_db_url, isolation_level="AUTOCOMMIT")
    try:
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    "UPDATE documents SET deleted_at = now() - interval '40 days' "
                    "WHERE deleted_at IS NOT NULL"
                )
            )

        report = await purge_expired_documents(settings, storage=storage)
        assert (report.purged_documents, report.object_deletes_failed) == (1, 0)

        async with engine.connect() as connection:
            remaining = (
                await connection.execute(
                    text("SELECT count(*) FROM documents WHERE id = :id"),
                    {"id": uuid.UUID(document_id)},
                )
            ).scalar_one()
    finally:
        await engine.dispose()
    assert remaining == 0
    keys = [
        entry["Key"]
        for entry in bucket_s3.list_objects_v2(Bucket=settings.s3_bucket).get("Contents", [])
    ]
    assert keys == [], keys


async def test_orphan_scan_finds_and_deletes_unreferenced_objects(
    app, client: AsyncClient, test_database, bucket_s3
) -> None:
    token = await _setup_admin(client, "orphans")
    upload = await _upload(client, token, content=MARKDOWN_CONTENT)
    assert upload.status_code == 202, upload.text

    settings = app.state.settings
    storage = StorageService(settings)

    document_keys: list[str] = []
    engine = create_async_engine(test_database.admin_db_url, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as connection:
            document_keys = list(
                await connection.scalars(text("SELECT storage_key FROM documents"))
            )
    finally:
        await engine.dispose()
    assert len(document_keys) == 1

    # A leftover object with no documents row (e.g. a failed upload).
    bogus_key = "tenant/someone-else/doc/leftover.md"
    bucket_s3.put_object(Bucket=settings.s3_bucket, Key=bogus_key, Body=b"orphan")

    orphans = await find_orphan_objects(settings, storage=storage)
    assert orphans == [bogus_key]

    deleted = await delete_orphan_objects(storage, orphans)
    assert deleted == 1

    remaining_keys = [
        entry["Key"]
        for entry in bucket_s3.list_objects_v2(Bucket=settings.s3_bucket).get("Contents", [])
    ]
    assert remaining_keys == document_keys


@pytest.mark.parametrize("target", [Path("scripts/purge.py"), Path("scripts/loadtest.py")])
def test_operations_scripts_exist(target: Path) -> None:
    assert (Path(__file__).resolve().parents[2] / target).is_file()
