from __future__ import annotations

import asyncio
import hashlib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import structlog
from redis.asyncio import Redis
from sqlalchemy import select, text, update
from sqlalchemy.engine import RowMapping
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.errors import InvalidCredentialsError, TenantAlreadyExistsError
from app.models import RefreshToken, User
from app.security import (
    TENANT_ADMIN_ROLE,
    create_access_token,
    dummy_password_hash,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    password_needs_rehash,
    verify_password,
)
from app.services.rate_limit import enforce_rate_limit

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TenantSnapshot:
    id: uuid.UUID
    name: str
    slug: str


@dataclass(frozen=True, slots=True)
class UserSnapshot:
    id: uuid.UUID
    tenant_id: uuid.UUID
    email: str
    roles: list[str]
    is_active: bool


@dataclass(frozen=True, slots=True)
class TokenBundle:
    access_token: str
    expires_in: int
    refresh_token: str
    user: UserSnapshot


async def _set_tenant_context(session: AsyncSession, tenant_id: uuid.UUID) -> None:
    await session.execute(
        text("SELECT set_config('app.tenant_id', :tid, true)"),
        {"tid": str(tenant_id)},
    )


def _as_user_snapshot(user: User) -> UserSnapshot:
    return UserSnapshot(
        id=user.id,
        tenant_id=user.tenant_id,
        email=user.email,
        roles=list(user.roles),
        is_active=user.is_active,
    )


class AuthService:
    def __init__(self, settings: Settings, redis: Redis) -> None:
        self._settings = settings
        self._redis = redis

    async def register_tenant(
        self,
        session: AsyncSession,
        *,
        tenant_name: str,
        tenant_slug: str,
        email: str,
        password: str,
        client_ip: str,
    ) -> tuple[TenantSnapshot, UserSnapshot]:
        await enforce_rate_limit(
            self._redis,
            scope="register",
            key=f"rag:rl:register:{client_ip}",
            limit=self._settings.rl_register_per_hour,
            window_seconds=3600,
        )

        password_hash = await asyncio.to_thread(hash_password, self._settings, password)
        try:
            async with session.begin():
                row = (
                    await session.execute(
                        text(
                            "SELECT tenant_id, user_id "
                            "FROM auth_create_tenant_with_admin("
                            ":name, :slug, :email, :password_hash)"
                        ),
                        {
                            "name": tenant_name,
                            "slug": tenant_slug,
                            "email": email,
                            "password_hash": password_hash,
                        },
                    )
                ).one()
        except IntegrityError as exc:
            raise TenantAlreadyExistsError() from exc

        tenant = TenantSnapshot(id=row.tenant_id, name=tenant_name, slug=tenant_slug)
        user = UserSnapshot(
            id=row.user_id,
            tenant_id=row.tenant_id,
            email=email,
            roles=[TENANT_ADMIN_ROLE],
            is_active=True,
        )
        log.info(
            "auth.tenant_registered",
            tenant_id=str(tenant.id),
            user_id=str(user.id),
        )
        return tenant, user

    async def login(
        self,
        session: AsyncSession,
        *,
        tenant_slug: str,
        email: str,
        password: str,
        client_ip: str,
    ) -> TokenBundle:
        email_key = hashlib.sha256(f"{tenant_slug}:{email}".encode()).hexdigest()[:24]
        await enforce_rate_limit(
            self._redis,
            scope="login",
            key=f"rag:rl:login:{client_ip}:{email_key}",
            limit=self._settings.rl_login_per_minute,
            window_seconds=60,
        )

        async with session.begin():
            row = (
                (
                    await session.execute(
                        text(
                            "SELECT user_id, tenant_id, password_hash, roles, is_active "
                            "FROM auth_get_user_for_login(:slug, :email)"
                        ),
                        {"slug": tenant_slug, "email": email},
                    )
                )
                .mappings()
                .one_or_none()
            )

            if row is None:
                await asyncio.to_thread(
                    verify_password,
                    self._settings,
                    dummy_password_hash(
                        self._settings.argon2_time_cost,
                        self._settings.argon2_memory_cost,
                        self._settings.argon2_parallelism,
                    ),
                    password,
                )
                log.info("auth.login_failed", tenant_slug=tenant_slug, reason="unknown")
                raise InvalidCredentialsError()

            valid = await asyncio.to_thread(
                verify_password, self._settings, row["password_hash"], password
            )
            if not valid or not row["is_active"]:
                log.info(
                    "auth.login_failed",
                    tenant_id=str(row["tenant_id"]),
                    reason="inactive" if valid else "password",
                )
                raise InvalidCredentialsError()

            await _set_tenant_context(session, row["tenant_id"])

            if await asyncio.to_thread(password_needs_rehash, self._settings, row["password_hash"]):
                replacement = await asyncio.to_thread(hash_password, self._settings, password)
                await session.execute(
                    update(User)
                    .where(
                        User.id == row["user_id"],
                        User.tenant_id == row["tenant_id"],
                    )
                    .values(password_hash=replacement)
                )

            user = UserSnapshot(
                id=row["user_id"],
                tenant_id=row["tenant_id"],
                email=email,
                roles=list(row["roles"]),
                is_active=True,
            )
            family_id = uuid.uuid4()
            refresh_token = await self._issue_refresh_token(
                session,
                user_id=user.id,
                tenant_id=user.tenant_id,
                family_id=family_id,
            )

        access_token, expires_in = create_access_token(
            self._settings,
            user_id=user.id,
            tenant_id=user.tenant_id,
            roles=user.roles,
        )
        log.info("auth.login_succeeded", tenant_id=str(user.tenant_id), user_id=str(user.id))
        return TokenBundle(
            access_token=access_token,
            expires_in=expires_in,
            refresh_token=refresh_token,
            user=user,
        )

    async def refresh(self, session: AsyncSession, *, refresh_token: str) -> TokenBundle:
        token_hash = hash_refresh_token(refresh_token)
        reuse_detected = False

        async with session.begin():
            row = await self._lookup_refresh_token(session, token_hash)
            if row is None:
                raise InvalidCredentialsError()

            tenant_id = row["tenant_id"]
            await _set_tenant_context(session, tenant_id)

            locked = (
                await session.execute(
                    select(RefreshToken)
                    .where(
                        RefreshToken.id == row["token_id"],
                        RefreshToken.tenant_id == tenant_id,
                    )
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if locked is None:
                raise InvalidCredentialsError()

            if locked.revoked_at is not None:
                await self._revoke_family(session, locked.family_id, locked.tenant_id)
                reuse_detected = True
            else:
                now = datetime.now(UTC)
                if locked.expires_at <= now:
                    raise InvalidCredentialsError()

                user = (
                    await session.execute(
                        select(User).where(
                            User.id == locked.user_id,
                            User.tenant_id == locked.tenant_id,
                        )
                    )
                ).scalar_one_or_none()
                if user is None or not user.is_active:
                    raise InvalidCredentialsError()

                locked.revoked_at = now
                new_refresh_token = await self._issue_refresh_token(
                    session,
                    user_id=user.id,
                    tenant_id=user.tenant_id,
                    family_id=locked.family_id,
                )
                snapshot = _as_user_snapshot(user)

        if reuse_detected:
            log.warning("auth.refresh_reuse_detected", tenant_id=str(tenant_id))
            raise InvalidCredentialsError()

        access_token, expires_in = create_access_token(
            self._settings,
            user_id=snapshot.id,
            tenant_id=snapshot.tenant_id,
            roles=snapshot.roles,
        )
        log.info(
            "auth.token_refreshed",
            tenant_id=str(snapshot.tenant_id),
            user_id=str(snapshot.id),
        )
        return TokenBundle(
            access_token=access_token,
            expires_in=expires_in,
            refresh_token=new_refresh_token,
            user=snapshot,
        )

    async def logout(self, session: AsyncSession, *, refresh_token: str) -> None:
        token_hash = hash_refresh_token(refresh_token)
        async with session.begin():
            row = await self._lookup_refresh_token(session, token_hash)
            if row is None:
                return
            await _set_tenant_context(session, row["tenant_id"])
            await session.execute(
                update(RefreshToken)
                .where(
                    RefreshToken.id == row["token_id"],
                    RefreshToken.tenant_id == row["tenant_id"],
                    RefreshToken.revoked_at.is_(None),
                )
                .values(revoked_at=datetime.now(UTC))
            )

    async def _lookup_refresh_token(
        self, session: AsyncSession, token_hash: str
    ) -> RowMapping | None:
        result = await session.execute(
            text(
                "SELECT token_id, user_id, tenant_id, family_id, expires_at, revoked_at "
                "FROM auth_get_refresh_token(CAST(:token_hash AS char(64)))"
            ),
            {"token_hash": token_hash},
        )
        return result.mappings().one_or_none()

    async def _revoke_family(
        self, session: AsyncSession, family_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> None:
        await session.execute(
            update(RefreshToken)
            .where(
                RefreshToken.family_id == family_id,
                RefreshToken.tenant_id == tenant_id,
                RefreshToken.revoked_at.is_(None),
            )
            .values(revoked_at=datetime.now(UTC))
        )

    async def _issue_refresh_token(
        self,
        session: AsyncSession,
        *,
        user_id: uuid.UUID,
        tenant_id: uuid.UUID,
        family_id: uuid.UUID,
    ) -> str:
        raw_token = generate_refresh_token()
        session.add(
            RefreshToken(
                user_id=user_id,
                tenant_id=tenant_id,
                family_id=family_id,
                token_hash=hash_refresh_token(raw_token),
                expires_at=datetime.now(UTC)
                + timedelta(seconds=self._settings.refresh_token_ttl_seconds),
            )
        )
        await session.flush()
        return raw_token
