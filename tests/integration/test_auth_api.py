from __future__ import annotations

import time
import uuid

import jwt
from httpx import AsyncClient

from tests.integration.helpers import (
    DEFAULT_PASSWORD,
    bearer,
    email_for,
    login,
    register_tenant,
)

ACCESS = "/api/v1/auth/me"


def _token(
    settings: object,
    *,
    secret: str | None = None,
    **overrides: object,
) -> str:
    now = int(time.time())
    payload: dict[str, object] = {
        "iss": settings.jwt_issuer,  # type: ignore[attr-defined]
        "aud": settings.jwt_audience,  # type: ignore[attr-defined]
        "sub": str(uuid.uuid4()),
        "tid": str(uuid.uuid4()),
        "roles": ["tenant_admin"],
        "typ": "access",
        "jti": uuid.uuid4().hex,
        "iat": now,
        "exp": now + 60,
    }
    payload.update(overrides)
    return jwt.encode(
        payload,
        secret or settings.jwt_secret,  # type: ignore[attr-defined]
        algorithm="HS256",
    )


async def test_register_login_and_me(client: AsyncClient) -> None:
    registered = await register_tenant(client, slug="acme")

    tokens = await login(client, slug="acme")
    response = await client.get(ACCESS, headers=bearer(str(tokens["access_token"])))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["user"]["id"] == registered["user"]["id"]
    assert body["user"]["roles"] == ["tenant_admin"]
    assert body["tenant"]["slug"] == "acme"
    assert "password" not in response.text
    assert "refresh_token" not in response.text


async def test_login_failures_are_generic(client: AsyncClient) -> None:
    await register_tenant(client, slug="acme")

    unknown_tenant = await client.post(
        "/api/v1/auth/login",
        json={"tenant_slug": "globex", "email": email_for("acme"), "password": DEFAULT_PASSWORD},
    )
    unknown_email = await client.post(
        "/api/v1/auth/login",
        json={
            "tenant_slug": "acme",
            "email": "nobody@acme.example.com",
            "password": DEFAULT_PASSWORD,
        },
    )
    wrong_password = await client.post(
        "/api/v1/auth/login",
        json={"tenant_slug": "acme", "email": email_for("acme"), "password": "wrong password!!"},
    )

    for response in (unknown_tenant, unknown_email, wrong_password):
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid credentials."


async def test_expired_and_wrong_tokens_are_rejected(client: AsyncClient, app: object) -> None:
    settings = app.state.settings  # type: ignore[attr-defined]
    now = int(time.time())

    cases = {
        "expired": _token(settings, exp=now - 1),
        "wrong audience": _token(settings, aud="another-api"),
        "wrong issuer": _token(settings, iss="another-issuer"),
        "wrong type": _token(settings, typ="refresh"),
        "bad subject": _token(settings, sub="not-a-uuid"),
        "wrong signature": _token(settings, secret="x" * 64),
    }

    for name, token in cases.items():
        response = await client.get(ACCESS, headers=bearer(token))
        assert response.status_code == 401, name


async def test_refresh_rotation_and_reuse_detection(client: AsyncClient) -> None:
    await register_tenant(client, slug="acme")
    tokens = await login(client, slug="acme")
    first_refresh = str(tokens["refresh_token"])

    rotated = await client.post("/api/v1/auth/refresh", json={"refresh_token": first_refresh})
    assert rotated.status_code == 200, rotated.text
    second_refresh = rotated.json()["refresh_token"]
    assert second_refresh != first_refresh

    reuse = await client.post("/api/v1/auth/refresh", json={"refresh_token": first_refresh})
    assert reuse.status_code == 401

    family_revoked = await client.post(
        "/api/v1/auth/refresh", json={"refresh_token": second_refresh}
    )
    assert family_revoked.status_code == 401


async def test_logout_revokes_refresh_token(client: AsyncClient) -> None:
    await register_tenant(client, slug="acme")
    tokens = await login(client, slug="acme")
    refresh_token = str(tokens["refresh_token"])

    logout = await client.post("/api/v1/auth/logout", json={"refresh_token": refresh_token})
    assert logout.status_code == 204

    after = await client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
    assert after.status_code == 401


async def test_deactivated_user_is_rejected_immediately(client: AsyncClient) -> None:
    registered = await register_tenant(client, slug="acme")
    admin = await login(client, slug="acme")
    admin_headers = bearer(str(admin["access_token"]))

    created = await client.post(
        "/api/v1/users",
        headers=admin_headers,
        json={
            "email": "alice@acme.example.com",
            "password": DEFAULT_PASSWORD,
            "roles": ["engineering"],
        },
    )
    assert created.status_code == 201, created.text
    alice_id = created.json()["id"]

    alice = await login(client, slug="acme", email="alice@acme.example.com")
    alice_headers = bearer(str(alice["access_token"]))

    deactivated = await client.patch(
        f"/api/v1/users/{alice_id}", headers=admin_headers, json={"is_active": False}
    )
    assert deactivated.status_code == 200, deactivated.text

    assert (await client.get(ACCESS, headers=alice_headers)).status_code == 401
    refreshed = await client.post(
        "/api/v1/auth/refresh", json={"refresh_token": alice["refresh_token"]}
    )
    assert refreshed.status_code == 401

    assert registered["user"]["id"] != alice_id


async def test_signup_can_be_disabled(client: AsyncClient, app: object) -> None:
    app.state.settings.allow_tenant_signup = False  # type: ignore[attr-defined]
    response = await client.post(
        "/api/v1/auth/register-tenant",
        json={
            "tenant_name": "Acme",
            "tenant_slug": "acme",
            "email": email_for("acme"),
            "password": DEFAULT_PASSWORD,
        },
    )
    assert response.status_code == 403
    assert response.json()["title"] == "Forbidden"


async def test_login_rate_limit_returns_retry_after(client: AsyncClient, app: object) -> None:
    await register_tenant(client, slug="acme")
    await app.state.redis.flushall()  # type: ignore[attr-defined]
    app.state.settings.rl_login_per_minute = 1  # type: ignore[attr-defined]

    first = await client.post(
        "/api/v1/auth/login",
        json={"tenant_slug": "acme", "email": email_for("acme"), "password": DEFAULT_PASSWORD},
    )
    assert first.status_code == 200

    second = await client.post(
        "/api/v1/auth/login",
        json={"tenant_slug": "acme", "email": email_for("acme"), "password": DEFAULT_PASSWORD},
    )
    assert second.status_code == 429
    assert int(second.headers["retry-after"]) >= 1
