from __future__ import annotations

import hashlib
import re
import secrets
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from app.config import Settings

ACCESS_TOKEN_TYPE = "access"
JWT_ALGORITHM = "HS256"
REFRESH_TOKEN_BYTES = 32
PASSWORD_MIN_CHARS = 12
PASSWORD_MAX_BYTES = 128
TENANT_ADMIN_ROLE = "tenant_admin"
MAX_ROLES = 20
ROLE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class TokenError(Exception):
    """An access token is malformed, expired, or otherwise invalid."""


class PasswordPolicyError(ValueError):
    """A password does not satisfy the minimum policy."""


class RoleValidationError(ValueError):
    """A role name is invalid or there are too many roles."""


@dataclass(frozen=True, slots=True)
class Principal:
    """Authenticated identity extracted from a signed access token.

    Roles are deliberately not part of the principal: authorization reads the
    current roles from the database on every request.
    """

    user_id: uuid.UUID
    tenant_id: uuid.UUID
    jti: str


@lru_cache(maxsize=8)
def _hasher(time_cost: int, memory_cost: int, parallelism: int) -> PasswordHasher:
    return PasswordHasher(
        time_cost=time_cost,
        memory_cost=memory_cost,
        parallelism=parallelism,
    )


def password_hasher(settings: Settings) -> PasswordHasher:
    return _hasher(
        settings.argon2_time_cost,
        settings.argon2_memory_cost,
        settings.argon2_parallelism,
    )


def validate_password(password: str) -> None:
    if len(password) < PASSWORD_MIN_CHARS:
        raise PasswordPolicyError(f"Password must be at least {PASSWORD_MIN_CHARS} characters.")
    if len(password.encode("utf-8")) > PASSWORD_MAX_BYTES:
        raise PasswordPolicyError(f"Password must be at most {PASSWORD_MAX_BYTES} bytes.")


def validate_password_bytes(password: str) -> None:
    if len(password.encode("utf-8")) > PASSWORD_MAX_BYTES:
        raise PasswordPolicyError(f"Password must be at most {PASSWORD_MAX_BYTES} bytes.")


def hash_password(settings: Settings, password: str) -> str:
    validate_password(password)
    return password_hasher(settings).hash(password)


def verify_password(settings: Settings, password_hash: str, password: str) -> bool:
    try:
        return password_hasher(settings).verify(password_hash, password)
    except (VerificationError, InvalidHashError):
        return False


def password_needs_rehash(settings: Settings, password_hash: str) -> bool:
    try:
        return password_hasher(settings).check_needs_rehash(password_hash)
    except InvalidHashError:
        return True


@lru_cache(maxsize=8)
def dummy_password_hash(time_cost: int, memory_cost: int, parallelism: int) -> str:
    """A hash used to equalize login timing when no user row exists."""
    return _hasher(time_cost, memory_cost, parallelism).hash(
        "dummy-password-that-is-never-accepted"
    )


def create_access_token(
    settings: Settings,
    *,
    user_id: uuid.UUID,
    tenant_id: uuid.UUID,
    roles: Iterable[str],
    now: datetime | None = None,
    ttl_seconds: int | None = None,
) -> tuple[str, int]:
    issued_at = now or datetime.now(UTC)
    ttl = settings.access_token_ttl_seconds if ttl_seconds is None else ttl_seconds
    payload = {
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "sub": str(user_id),
        "tid": str(tenant_id),
        "roles": list(roles),
        "typ": ACCESS_TOKEN_TYPE,
        "jti": uuid.uuid4().hex,
        "iat": int(issued_at.timestamp()),
        "exp": int((issued_at + timedelta(seconds=ttl)).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=JWT_ALGORITHM), ttl


def decode_access_token(settings: Settings, token: str) -> Principal:
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[JWT_ALGORITHM],
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
            options={
                "require": ["exp", "iat", "iss", "aud", "sub", "tid", "typ", "jti"],
            },
        )
    except jwt.InvalidTokenError as exc:
        raise TokenError("Invalid or expired access token.") from exc

    if payload.get("typ") != ACCESS_TOKEN_TYPE:
        raise TokenError("Wrong token type.")

    try:
        user_id = uuid.UUID(str(payload["sub"]))
        tenant_id = uuid.UUID(str(payload["tid"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise TokenError("Token subject or tenant is not a UUID.") from exc

    jti = payload.get("jti")
    if not isinstance(jti, str) or not jti:
        raise TokenError("Token has no jti.")

    return Principal(user_id=user_id, tenant_id=tenant_id, jti=jti)


def generate_refresh_token() -> str:
    return secrets.token_urlsafe(REFRESH_TOKEN_BYTES)


def hash_refresh_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def normalize_roles(roles: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in roles:
        role = raw.strip().lower()
        if not ROLE_PATTERN.fullmatch(role):
            raise RoleValidationError("Roles must match ^[a-z0-9][a-z0-9_-]{0,63}$ (lowercase).")
        if role not in seen:
            seen.add(role)
            normalized.append(role)
    if len(normalized) > MAX_ROLES:
        raise RoleValidationError(f"At most {MAX_ROLES} roles are allowed.")
    return normalized


def is_tenant_admin(roles: Iterable[str]) -> bool:
    return TENANT_ADMIN_ROLE in roles


@dataclass(frozen=True, slots=True)
class CurrentUser:
    """Database-authoritative identity for an authenticated request."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    email: str
    roles: tuple[str, ...]
    is_active: bool
