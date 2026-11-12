from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.main import create_app

REQUIRED = {
    "database_url": "postgresql+asyncpg://rag_app:rag_app@localhost:5432/rag",
    "database_migration_url": "postgresql+asyncpg://rag_owner:rag_owner@localhost:5432/rag",
    "embedding_provider": "fake",
}


def make_app() -> object:
    return create_app(Settings(_env_file=None, **REQUIRED))  # type: ignore[arg-type]


async def test_healthz_is_dependency_free() -> None:
    app = make_app()
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/healthz", headers={"X-Request-ID": "test-request"})
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["x-request-id"] == "test-request"


async def test_readyz_reports_database_unavailable_without_lifespan() -> None:
    app = make_app()
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["checks"]["database"] == "unavailable"


@pytest.mark.parametrize("path", ["/does-not-exist"])
async def test_errors_use_problem_json_and_request_id(path: str) -> None:
    app = make_app()
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(path)
    body = response.json()
    assert response.status_code == 404
    assert response.headers["content-type"] == "application/problem+json"
    assert body["status"] == 404
    assert body["title"] == "Not Found"
    assert "request_id" in body
    assert body["request_id"] == response.headers["x-request-id"]


async def test_invalid_request_id_header_is_replaced() -> None:
    app = make_app()
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/healthz", headers={"X-Request-ID": "not valid!"})
    assert response.status_code == 200
    assert response.headers["x-request-id"] != "not valid!"
