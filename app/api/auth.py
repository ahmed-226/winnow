from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.deps import (
    CurrentUser,
    current_user,
    get_app_settings,
    get_auth_service,
    get_session,
    tenant_session,
)
from app.errors import SignupDisabledError, UnauthorizedError
from app.models import Tenant
from app.schemas import (
    LoginRequest,
    MeResponse,
    RefreshTokenRequest,
    RegisterTenantRequest,
    RegisterTenantResponse,
    TenantOut,
    TokenResponse,
    UserOut,
)
from app.services.auth import AuthService, TenantSnapshot, UserSnapshot

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _tenant_out(tenant: TenantSnapshot) -> TenantOut:
    return TenantOut(id=tenant.id, name=tenant.name, slug=tenant.slug)


def _user_out(user: UserSnapshot) -> UserOut:
    return UserOut(
        id=user.id,
        tenant_id=user.tenant_id,
        email=user.email,
        roles=user.roles,
        is_active=user.is_active,
    )


@router.post("/register-tenant", status_code=201, response_model=RegisterTenantResponse)
async def register_tenant(
    payload: RegisterTenantRequest,
    request: Request,
    settings: Settings = Depends(get_app_settings),
    session: AsyncSession = Depends(get_session),
    service: AuthService = Depends(get_auth_service),
) -> RegisterTenantResponse:
    if not settings.allow_tenant_signup:
        raise SignupDisabledError()

    tenant, user = await service.register_tenant(
        session,
        tenant_name=payload.tenant_name,
        tenant_slug=payload.tenant_slug,
        email=payload.email,
        password=payload.password,
        client_ip=_client_ip(request),
    )
    return RegisterTenantResponse(tenant=_tenant_out(tenant), user=_user_out(user))


@router.post("/login", response_model=TokenResponse)
async def login(
    payload: LoginRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
    service: AuthService = Depends(get_auth_service),
) -> TokenResponse:
    bundle = await service.login(
        session,
        tenant_slug=payload.tenant_slug,
        email=payload.email,
        password=payload.password,
        client_ip=_client_ip(request),
    )
    return TokenResponse(
        access_token=bundle.access_token,
        expires_in=bundle.expires_in,
        refresh_token=bundle.refresh_token,
        user=_user_out(bundle.user),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    payload: RefreshTokenRequest,
    session: AsyncSession = Depends(get_session),
    service: AuthService = Depends(get_auth_service),
) -> TokenResponse:
    bundle = await service.refresh(session, refresh_token=payload.refresh_token)
    return TokenResponse(
        access_token=bundle.access_token,
        expires_in=bundle.expires_in,
        refresh_token=bundle.refresh_token,
        user=_user_out(bundle.user),
    )


@router.post("/logout", status_code=204)
async def logout(
    payload: RefreshTokenRequest,
    session: AsyncSession = Depends(get_session),
    service: AuthService = Depends(get_auth_service),
) -> Response:
    await service.logout(session, refresh_token=payload.refresh_token)
    return Response(status_code=204)


@router.get("/me", response_model=MeResponse)
async def me(
    user: CurrentUser = Depends(current_user),
    session: AsyncSession = Depends(tenant_session),
) -> MeResponse:
    tenant = (
        await session.execute(select(Tenant).where(Tenant.id == user.tenant_id))
    ).scalar_one_or_none()
    if tenant is None:
        raise UnauthorizedError("Tenant is missing.")

    return MeResponse(
        user=UserOut(
            id=user.id,
            tenant_id=user.tenant_id,
            email=user.email,
            roles=list(user.roles),
            is_active=user.is_active,
        ),
        tenant=TenantOut(id=tenant.id, name=tenant.name, slug=tenant.slug),
    )
