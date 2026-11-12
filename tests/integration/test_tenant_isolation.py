from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import create_async_engine

from tests.integration.conftest import DatabaseHandle
from tests.integration.helpers import bearer, login, register_tenant


async def test_rls_blocks_reads_without_context(test_database: DatabaseHandle) -> None:
    engine = create_async_engine(test_database.app_url)
    try:
        async with engine.connect() as connection:
            users = (await connection.execute(text("SELECT count(*) FROM users"))).scalar_one()
            tenants = (await connection.execute(text("SELECT count(*) FROM tenants"))).scalar_one()
    finally:
        await engine.dispose()

    assert users == 0
    assert tenants == 0


async def test_rls_blocks_mismatched_context_and_cross_tenant_insert(
    client: AsyncClient, test_database: DatabaseHandle
) -> None:
    acme = await register_tenant(client, slug="acme")
    globex = await register_tenant(client, slug="globex")
    acme_tenant = acme["tenant"]["id"]
    acme_user = acme["user"]["id"]
    globex_user = globex["user"]["id"]

    engine = create_async_engine(test_database.app_url)
    try:
        async with engine.begin() as connection:
            await connection.execute(
                text("SELECT set_config('app.tenant_id', :tid, true)"),
                {"tid": acme_tenant},
            )
            visible_users = list(
                (await connection.execute(text("SELECT id::text FROM users"))).scalars()
            )
            hidden = (
                await connection.execute(
                    text("SELECT count(*) FROM users WHERE id = :id"),
                    {"id": globex_user},
                )
            ).scalar_one()
            own = (
                await connection.execute(
                    text("SELECT count(*) FROM users WHERE id = :id"),
                    {"id": acme_user},
                )
            ).scalar_one()

        assert visible_users == [acme_user]
        assert hidden == 0
        assert own == 1

        with pytest.raises(DBAPIError):
            async with engine.begin() as connection:
                await connection.execute(
                    text("SELECT set_config('app.tenant_id', :tid, true)"),
                    {"tid": acme_tenant},
                )
                await connection.execute(
                    text(
                        "INSERT INTO users (tenant_id, email, password_hash) "
                        "VALUES (:tid, 'intruder@globex.example.com', 'x')"
                    ),
                    {"tid": globex["tenant"]["id"]},
                )
    finally:
        await engine.dispose()


async def test_cross_tenant_admin_gets_404_and_headers_are_ignored(
    client: AsyncClient,
) -> None:
    acme = await register_tenant(client, slug="acme")
    globex = await register_tenant(client, slug="globex")
    acme_tokens = await login(client, slug="acme")
    headers = bearer(str(acme_tokens["access_token"]))
    headers["X-Tenant-ID"] = str(globex["tenant"]["id"])
    headers["X-Roles"] = "tenant_admin,engineering"

    me = await client.get("/api/v1/auth/me", headers=headers)
    assert me.status_code == 200
    assert me.json()["tenant"]["id"] == acme["tenant"]["id"]

    users = await client.get("/api/v1/users", headers=headers)
    assert users.status_code == 200
    assert [user["id"] for user in users.json()] == [acme["user"]["id"]]

    cross_tenant = await client.get(f"/api/v1/users/{globex['user']['id']}", headers=headers)
    assert cross_tenant.status_code == 404
