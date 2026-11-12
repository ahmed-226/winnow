from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps import CurrentUser, get_user_service, require_admin, tenant_session
from app.schemas import UserCreateRequest, UserOut, UserUpdateRequest
from app.services.users import UserService

router = APIRouter(prefix="/api/v1/users", tags=["users"])


@router.post("", status_code=201, response_model=UserOut)
async def create_user(
    payload: UserCreateRequest,
    admin: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(tenant_session),
    service: UserService = Depends(get_user_service),
) -> UserOut:
    user = await service.create_user(
        session,
        tenant_id=admin.tenant_id,
        email=payload.email,
        password=payload.password,
        roles=payload.roles,
    )
    return UserOut.model_validate(user)


@router.get("", response_model=list[UserOut])
async def list_users(
    limit: int = Query(default=50, ge=1, le=100),
    admin: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(tenant_session),
    service: UserService = Depends(get_user_service),
) -> list[UserOut]:
    users = await service.list_users(session, tenant_id=admin.tenant_id, limit=limit)
    return [UserOut.model_validate(user) for user in users]


@router.get("/{user_id}", response_model=UserOut)
async def get_user(
    user_id: uuid.UUID,
    admin: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(tenant_session),
    service: UserService = Depends(get_user_service),
) -> UserOut:
    user = await service.get_user(session, tenant_id=admin.tenant_id, user_id=user_id)
    return UserOut.model_validate(user)


@router.patch("/{user_id}", response_model=UserOut)
async def update_user(
    user_id: uuid.UUID,
    payload: UserUpdateRequest,
    admin: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(tenant_session),
    service: UserService = Depends(get_user_service),
) -> UserOut:
    user = await service.update_user(
        session,
        tenant_id=admin.tenant_id,
        user_id=user_id,
        email=payload.email,
        password=payload.password,
        roles=payload.roles,
        is_active=payload.is_active,
    )
    return UserOut.model_validate(user)
