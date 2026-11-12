from __future__ import annotations

from httpx import AsyncClient

from tests.integration.helpers import (
    DEFAULT_PASSWORD,
    bearer,
    login,
    register_tenant,
)


async def _admin_headers(client: AsyncClient, slug: str = "acme") -> dict[str, str]:
    tokens = await login(client, slug=slug)
    return bearer(str(tokens["access_token"]))


async def test_admin_can_create_list_get_and_update_users(client: AsyncClient) -> None:
    await register_tenant(client, slug="acme")
    headers = await _admin_headers(client)

    created = await client.post(
        "/api/v1/users",
        headers=headers,
        json={
            "email": "alice@acme.example.com",
            "password": DEFAULT_PASSWORD,
            "roles": ["Engineering", "engineering"],
        },
    )
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["roles"] == ["engineering"]
    assert body["is_active"] is True

    listed = await client.get("/api/v1/users", headers=headers)
    assert listed.status_code == 200
    assert {user["email"] for user in listed.json()} == {
        "admin@acme.example.com",
        "alice@acme.example.com",
    }

    fetched = await client.get(f"/api/v1/users/{body['id']}", headers=headers)
    assert fetched.status_code == 200
    assert fetched.json()["email"] == "alice@acme.example.com"

    updated = await client.patch(
        f"/api/v1/users/{body['id']}",
        headers=headers,
        json={"roles": ["hr", "engineering"], "is_active": False},
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["roles"] == ["hr", "engineering"]
    assert updated.json()["is_active"] is False


async def test_duplicate_email_is_conflict(client: AsyncClient) -> None:
    await register_tenant(client, slug="acme")
    headers = await _admin_headers(client)
    payload = {
        "email": "alice@acme.example.com",
        "password": DEFAULT_PASSWORD,
        "roles": [],
    }
    first = await client.post("/api/v1/users", headers=headers, json=payload)
    assert first.status_code == 201

    second = await client.post("/api/v1/users", headers=headers, json=payload)
    assert second.status_code == 409


async def test_member_cannot_manage_users(client: AsyncClient) -> None:
    await register_tenant(client, slug="acme")
    admin_headers = await _admin_headers(client)
    await client.post(
        "/api/v1/users",
        headers=admin_headers,
        json={
            "email": "alice@acme.example.com",
            "password": DEFAULT_PASSWORD,
            "roles": ["engineering"],
        },
    )

    member = await login(client, slug="acme", email="alice@acme.example.com")
    member_headers = bearer(str(member["access_token"]))

    listed = await client.get("/api/v1/users", headers=member_headers)
    assert listed.status_code == 403

    created = await client.post(
        "/api/v1/users",
        headers=member_headers,
        json={
            "email": "bob@acme.example.com",
            "password": DEFAULT_PASSWORD,
            "roles": [],
        },
    )
    assert created.status_code == 403


async def test_last_admin_cannot_be_demoted_or_deactivated(client: AsyncClient) -> None:
    registered = await register_tenant(client, slug="acme")
    admin_id = registered["user"]["id"]
    headers = await _admin_headers(client)

    demote = await client.patch(f"/api/v1/users/{admin_id}", headers=headers, json={"roles": []})
    assert demote.status_code == 409
    assert demote.json()["title"] == "Conflict"

    deactivate = await client.patch(
        f"/api/v1/users/{admin_id}", headers=headers, json={"is_active": False}
    )
    assert deactivate.status_code == 409


async def test_admin_can_be_demoted_once_another_admin_exists(
    client: AsyncClient,
) -> None:
    registered = await register_tenant(client, slug="acme")
    first_admin_id = registered["user"]["id"]
    headers = await _admin_headers(client)

    second = await client.post(
        "/api/v1/users",
        headers=headers,
        json={
            "email": "second-admin@acme.example.com",
            "password": DEFAULT_PASSWORD,
            "roles": ["tenant_admin"],
        },
    )
    assert second.status_code == 201, second.text

    demoted = await client.patch(
        f"/api/v1/users/{first_admin_id}", headers=headers, json={"roles": ["engineering"]}
    )
    assert demoted.status_code == 200, demoted.text
    assert demoted.json()["roles"] == ["engineering"]

    second_tokens = await login(client, slug="acme", email="second-admin@acme.example.com")
    second_headers = bearer(str(second_tokens["access_token"]))
    last = await client.patch(
        f"/api/v1/users/{second.json()['id']}",
        headers=second_headers,
        json={"roles": []},
    )
    assert last.status_code == 409
