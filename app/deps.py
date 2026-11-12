from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from redis.asyncio import Redis
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.errors import ForbiddenError, InvalidTokenError, UnauthorizedError
from app.models import User
from app.security import (
    TENANT_ADMIN_ROLE,
    Principal,
    TokenError,
    decode_access_token,
)
from app.security import CurrentUser as CurrentUser  # explicit re-export
from app.services.auth import AuthService
from app.services.documents import DocumentService
from app.services.storage import StorageService
from app.services.users import UserService

_bearer = HTTPBearer(auto_error=False)


def get_app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_redis(request: Request) -> Redis:
    redis: Redis = request.app.state.redis
    return redis


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Plain session for flows that run before a tenant context exists."""
    async with request.app.state.sessionmaker() as session:
        yield session


async def principal_from_jwt(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    settings: Settings = Depends(get_app_settings),
) -> Principal:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise UnauthorizedError()
    try:
        return decode_access_token(settings, credentials.credentials)
    except TokenError as exc:
        raise InvalidTokenError() from exc


async def tenant_session(
    request: Request,
    principal: Principal = Depends(principal_from_jwt),
) -> AsyncIterator[AsyncSession]:
    """Open a transaction with transaction-local RLS context for the JWT tenant."""
    async with request.app.state.sessionmaker() as session:
        async with session.begin():
            await session.execute(
                text("SELECT set_config('app.tenant_id', :tid, true)"),
                {"tid": str(principal.tenant_id)},
            )
            yield session


async def current_user(
    principal: Principal = Depends(principal_from_jwt),
    session: AsyncSession = Depends(tenant_session),
) -> CurrentUser:
    user = (
        await session.execute(
            select(User).where(
                User.id == principal.user_id,
                User.tenant_id == principal.tenant_id,
            )
        )
    ).scalar_one_or_none()
    if user is None or not user.is_active:
        raise UnauthorizedError("User is missing or inactive.")
    return CurrentUser(
        id=user.id,
        tenant_id=user.tenant_id,
        email=user.email,
        roles=tuple(user.roles),
        is_active=user.is_active,
    )


async def require_admin(user: CurrentUser = Depends(current_user)) -> CurrentUser:
    if TENANT_ADMIN_ROLE not in user.roles:
        raise ForbiddenError("Tenant admin role required.")
    return user


def get_auth_service(request: Request) -> AuthService:
    return AuthService(
        settings=request.app.state.settings,
        redis=request.app.state.redis,
    )


def get_user_service(request: Request) -> UserService:
    return UserService(settings=request.app.state.settings)


def get_storage_service(request: Request) -> StorageService:
    storage: StorageService = request.app.state.storage
    return storage


def get_document_service(
    request: Request, storage: StorageService = Depends(get_storage_service)
) -> DocumentService:
    return DocumentService(request.app.state.settings, storage)
