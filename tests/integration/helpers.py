from __future__ import annotations

from httpx import AsyncClient

DEFAULT_PASSWORD = "correct horse battery staple"


def email_for(slug: str) -> str:
    return f"admin@{slug}.example.com"


async def register_tenant(
    client: AsyncClient,
    *,
    slug: str,
    name: str | None = None,
    email: str | None = None,
    password: str = DEFAULT_PASSWORD,
) -> dict[str, object]:
    response = await client.post(
        "/api/v1/auth/register-tenant",
        json={
            "tenant_name": name or slug.title(),
            "tenant_slug": slug,
            "email": email or email_for(slug),
            "password": password,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


async def login(
    client: AsyncClient,
    *,
    slug: str,
    email: str | None = None,
    password: str = DEFAULT_PASSWORD,
) -> dict[str, object]:
    response = await client.post(
        "/api/v1/auth/login",
        json={
            "tenant_slug": slug,
            "email": email or email_for(slug),
            "password": password,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def bearer(access_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {access_token}"}


async def create_user(
    client: AsyncClient,
    *,
    admin_token: str,
    email: str,
    password: str = DEFAULT_PASSWORD,
    roles: list[str] | None = None,
) -> dict[str, object]:
    response = await client.post(
        "/api/v1/users",
        json={"email": email, "password": password, "roles": roles or []},
        headers=bearer(admin_token),
    )
    assert response.status_code == 201, response.text
    return response.json()
