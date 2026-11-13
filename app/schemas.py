from __future__ import annotations

import unicodedata
import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.security import (
    PasswordPolicyError,
    RoleValidationError,
    normalize_roles,
    validate_password,
    validate_password_bytes,
)

TENANT_SLUG_PATTERN = r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$"


class _RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RegisterTenantRequest(_RequestModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "tenant_name": "Acme Corp",
                    "tenant_slug": "acme",
                    "email": "admin@acme.example",
                    "password": "correct horse battery staple",
                }
            ]
        },
    )

    tenant_name: str = Field(min_length=1, max_length=255)
    tenant_slug: str = Field(min_length=1, max_length=63, pattern=TENANT_SLUG_PATTERN)
    email: EmailStr
    password: str

    @field_validator("tenant_name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Tenant name must not be blank.")
        return value

    @field_validator("tenant_slug", mode="before")
    @classmethod
    def _normalize_slug(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("email", mode="before")
    @classmethod
    def _normalize_email(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("password")
    @classmethod
    def _validate_password(cls, value: str) -> str:
        try:
            validate_password(value)
        except PasswordPolicyError as exc:
            raise ValueError(str(exc)) from exc
        return value


class LoginRequest(_RequestModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "tenant_slug": "acme",
                    "email": "admin@acme.example",
                    "password": "correct horse battery staple",
                }
            ]
        },
    )

    tenant_slug: str = Field(min_length=1, max_length=63)
    email: EmailStr
    password: str

    @field_validator("tenant_slug", mode="before")
    @classmethod
    def _normalize_slug(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("email", mode="before")
    @classmethod
    def _normalize_email(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("password")
    @classmethod
    def _validate_password_bytes(cls, value: str) -> str:
        try:
            validate_password_bytes(value)
        except PasswordPolicyError as exc:
            raise ValueError(str(exc)) from exc
        return value


class RefreshTokenRequest(_RequestModel):
    refresh_token: str = Field(min_length=1, max_length=512)


class TenantOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    slug: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    tenant_id: uuid.UUID
    email: str
    roles: list[str]
    is_active: bool


class RegisterTenantResponse(BaseModel):
    tenant: TenantOut
    user: UserOut


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int
    refresh_token: str
    user: UserOut


class MeResponse(BaseModel):
    user: UserOut
    tenant: TenantOut


class UserCreateRequest(_RequestModel):
    email: EmailStr
    password: str
    roles: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("email", mode="before")
    @classmethod
    def _normalize_email(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("password")
    @classmethod
    def _validate_password(cls, value: str) -> str:
        try:
            validate_password(value)
        except PasswordPolicyError as exc:
            raise ValueError(str(exc)) from exc
        return value

    @field_validator("roles")
    @classmethod
    def _validate_roles(cls, value: list[str]) -> list[str]:
        try:
            return normalize_roles(value)
        except RoleValidationError as exc:
            raise ValueError(str(exc)) from exc


class UserUpdateRequest(_RequestModel):
    email: EmailStr | None = None
    password: str | None = None
    roles: list[str] | None = Field(default=None, max_length=20)
    is_active: bool | None = None

    @field_validator("email", mode="before")
    @classmethod
    def _normalize_email(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("password")
    @classmethod
    def _validate_password(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            validate_password(value)
        except PasswordPolicyError as exc:
            raise ValueError(str(exc)) from exc
        return value

    @field_validator("roles")
    @classmethod
    def _validate_roles(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        try:
            return normalize_roles(value)
        except RoleValidationError as exc:
            raise ValueError(str(exc)) from exc


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    file_name: str
    source_type: str
    status: str
    access_roles: list[str]
    size_bytes: int
    chunk_count: int
    error_message: str | None
    created_by: uuid.UUID | None
    created_at: datetime
    updated_at: datetime


class UploadDocumentResponse(DocumentOut):
    duplicate: bool


class DocumentListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    status: str
    access_roles: list[str]
    chunk_count: int
    created_at: datetime


class DocumentListResponse(BaseModel):
    items: list[DocumentListItem]
    next_cursor: str | None


class DocumentPatchRequest(_RequestModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    access_roles: list[str] | None = Field(default=None, max_length=20)

    @field_validator("title")
    @classmethod
    def _strip_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("Title must not be blank.")
        return stripped

    @field_validator("access_roles")
    @classmethod
    def _validate_roles(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        try:
            return normalize_roles(value)
        except RoleValidationError as exc:
            raise ValueError(str(exc)) from exc


class SearchRequest(_RequestModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{"query": "how many weeks of parental leave do I get?", "top_k": 8}]
        },
    )

    query: str = Field(min_length=1, max_length=1_000)
    top_k: int = Field(default=10, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def _clean_query(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Query must not be blank.")
        if any(unicodedata.category(char) == "Cc" and char not in "\t\n\r" for char in stripped):
            raise ValueError("Query must not contain control characters.")
        return stripped


class SearchResultItem(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    title: str
    source_type: str
    chunk_index: int
    content: str
    metadata: dict[str, Any]
    score: float
    dense_rank: int | None
    sparse_rank: int | None
    sparse_score: float | None
    cosine_distance: float | None


class SearchResponse(BaseModel):
    query: str
    mode: Literal["hybrid", "sparse_only"]
    took_ms: int
    results: list[SearchResultItem]
