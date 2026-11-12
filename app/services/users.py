from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime

from sqlalchemy import select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.errors import (
    EmailAlreadyExistsError,
    InvalidRoleError,
    LastAdminError,
    UserNotFoundError,
)
from app.models import RefreshToken, User
from app.security import (
    TENANT_ADMIN_ROLE,
    RoleValidationError,
    hash_password,
    normalize_roles,
)


class UserService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def list_users(
        self, session: AsyncSession, *, tenant_id: uuid.UUID, limit: int
    ) -> list[User]:
        result = await session.execute(
            select(User)
            .where(User.tenant_id == tenant_id)
            .order_by(User.created_at, User.id)
            .limit(limit)
        )
        return list(result.scalars())

    async def get_user(
        self, session: AsyncSession, *, tenant_id: uuid.UUID, user_id: uuid.UUID
    ) -> User:
        user = (
            await session.execute(
                select(User).where(
                    User.id == user_id,
                    User.tenant_id == tenant_id,
                )
            )
        ).scalar_one_or_none()
        if user is None:
            raise UserNotFoundError()
        return user

    async def create_user(
        self,
        session: AsyncSession,
        *,
        tenant_id: uuid.UUID,
        email: str,
        password: str,
        roles: list[str],
    ) -> User:
        normalized = self._normalize(roles)
        password_hash = await asyncio.to_thread(hash_password, self._settings, password)
        user = User(
            tenant_id=tenant_id,
            email=email,
            password_hash=password_hash,
            roles=normalized,
            is_active=True,
        )
        session.add(user)
        try:
            await session.flush()
        except IntegrityError as exc:
            raise EmailAlreadyExistsError() from exc
        return user

    async def update_user(
        self,
        session: AsyncSession,
        *,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        email: str | None = None,
        password: str | None = None,
        roles: list[str] | None = None,
        is_active: bool | None = None,
    ) -> User:
        user = await self.get_user(session, tenant_id=tenant_id, user_id=user_id)

        new_roles = user.roles if roles is None else self._normalize(roles)
        new_active = user.is_active if is_active is None else is_active
        was_admin = TENANT_ADMIN_ROLE in user.roles
        will_be_admin = TENANT_ADMIN_ROLE in new_roles

        if was_admin and (not will_be_admin or not new_active):
            await self._assert_not_last_admin(session, tenant_id, user.id)

        user.roles = list(new_roles)
        user.is_active = new_active
        if email is not None:
            user.email = email
        if password is not None:
            user.password_hash = await asyncio.to_thread(hash_password, self._settings, password)

        try:
            await session.flush()
        except IntegrityError as exc:
            raise EmailAlreadyExistsError() from exc

        if was_admin and not new_active:
            await session.execute(
                update(RefreshToken)
                .where(
                    RefreshToken.user_id == user.id,
                    RefreshToken.tenant_id == tenant_id,
                    RefreshToken.revoked_at.is_(None),
                )
                .values(revoked_at=datetime.now(UTC))
            )
        return user

    def _normalize(self, roles: list[str]) -> list[str]:
        try:
            return normalize_roles(roles)
        except RoleValidationError as exc:
            raise InvalidRoleError(str(exc)) from exc

    async def _assert_not_last_admin(
        self, session: AsyncSession, tenant_id: uuid.UUID, target_id: uuid.UUID
    ) -> None:
        admin_ids = list(
            (
                await session.execute(
                    text(
                        "SELECT id FROM users "
                        "WHERE tenant_id = :tid AND is_active "
                        "AND :admin_role = ANY(roles) FOR UPDATE"
                    ),
                    {"tid": str(tenant_id), "admin_role": TENANT_ADMIN_ROLE},
                )
            ).scalars()
        )
        if target_id in admin_ids and len(admin_ids) == 1:
            raise LastAdminError()
