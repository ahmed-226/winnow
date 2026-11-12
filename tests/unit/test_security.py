from __future__ import annotations

import time
import uuid

import jwt
import pytest

from app.config import Settings
from app.security import (
    PasswordPolicyError,
    RoleValidationError,
    TokenError,
    create_access_token,
    decode_access_token,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    normalize_roles,
    validate_password,
    validate_password_bytes,
    verify_password,
)

REQUIRED = {
    "database_url": "postgresql+asyncpg://rag_app:rag_app@localhost:5432/rag",
    "database_migration_url": "postgresql+asyncpg://rag_owner:rag_owner@localhost:5432/rag",
    "embedding_provider": "fake",
}


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        **REQUIRED,
        "app_env": "test",
        "jwt_secret": "unit-test-secret-that-is-long-enough",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


def test_password_hash_roundtrip() -> None:
    settings = make_settings()
    password_hash = hash_password(settings, "correct horse battery staple")

    assert password_hash.startswith("$argon2id$")
    assert verify_password(settings, password_hash, "correct horse battery staple")
    assert not verify_password(settings, password_hash, "wrong password")


def test_password_policy_rejects_short_and_oversized() -> None:
    with pytest.raises(PasswordPolicyError):
        validate_password("too-short")
    with pytest.raises(PasswordPolicyError):
        validate_password("é" * 65)
    validate_password_bytes("a" * 128)


def test_role_normalization() -> None:
    assert normalize_roles(["Engineering", "engineering", "hr-2"]) == [
        "engineering",
        "hr-2",
    ]
    assert normalize_roles([]) == []
    with pytest.raises(RoleValidationError):
        normalize_roles(["bad role"])
    with pytest.raises(RoleValidationError):
        normalize_roles([f"role{i}" for i in range(21)])


def test_access_token_roundtrip() -> None:
    settings = make_settings()
    user_id = uuid.uuid4()
    tenant_id = uuid.uuid4()

    token, expires_in = create_access_token(
        settings, user_id=user_id, tenant_id=tenant_id, roles=["tenant_admin"]
    )
    principal = decode_access_token(settings, token)

    assert expires_in == settings.access_token_ttl_seconds
    assert principal.user_id == user_id
    assert principal.tenant_id == tenant_id
    assert principal.jti


def _craft_token(settings: Settings, **overrides: object) -> str:
    now = int(time.time())
    payload: dict[str, object] = {
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "sub": str(uuid.uuid4()),
        "tid": str(uuid.uuid4()),
        "roles": [],
        "typ": "access",
        "jti": uuid.uuid4().hex,
        "iat": now,
        "exp": now + 60,
    }
    secret = overrides.pop("_secret", settings.jwt_secret)
    payload.update(overrides)
    return jwt.encode(payload, str(secret), algorithm="HS256")


@pytest.mark.parametrize(
    "overrides",
    [
        {"exp": int(time.time()) - 1},
        {"aud": "wrong-audience"},
        {"iss": "wrong-issuer"},
        {"typ": "refresh"},
        {"sub": "not-a-uuid"},
        {"tid": "not-a-uuid"},
        {"_secret": "x" * 64},
    ],
)
def test_invalid_access_tokens_are_rejected(overrides: dict[str, object]) -> None:
    settings = make_settings()
    token = _craft_token(settings, **overrides)
    with pytest.raises(TokenError):
        decode_access_token(settings, token)


def test_refresh_tokens_are_random_and_hashed() -> None:
    first = generate_refresh_token()
    second = generate_refresh_token()

    assert first != second
    assert len(hash_refresh_token(first)) == 64
    assert hash_refresh_token(first) == hash_refresh_token(first)
