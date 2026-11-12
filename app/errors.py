from __future__ import annotations

from collections.abc import Mapping
from http import HTTPStatus
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.logging import get_request_id

log = structlog.get_logger(__name__)


class AppError(Exception):
    """Base for errors that map to a client-facing problem response."""

    status_code = 400
    title = "Bad Request"
    type_slug = "error"
    default_detail = "Request failed."

    def __init__(
        self, detail: str | None = None, *, headers: Mapping[str, str] | None = None
    ) -> None:
        self.detail = detail or self.default_detail
        self.headers = dict(headers) if headers is not None else None
        super().__init__(self.detail)


class UnauthorizedError(AppError):
    status_code = 401
    title = "Unauthorized"
    type_slug = "unauthorized"
    default_detail = "Authentication required."


class ForbiddenError(AppError):
    status_code = 403
    title = "Forbidden"
    type_slug = "forbidden"
    default_detail = "You do not have permission to perform this action."


class NotFoundError(AppError):
    status_code = 404
    title = "Not Found"
    type_slug = "not-found"
    default_detail = "Resource not found."


class ConflictError(AppError):
    status_code = 409
    title = "Conflict"
    type_slug = "conflict"
    default_detail = "The request conflicts with the current state of the resource."


class UnprocessableError(AppError):
    status_code = 422
    title = "Unprocessable Entity"
    type_slug = "unprocessable"
    default_detail = "The request could not be processed."


class BadRequestError(AppError):
    status_code = 400
    title = "Bad Request"
    type_slug = "bad-request"
    default_detail = "The request is malformed."


class UploadTooLargeError(AppError):
    status_code = 413
    title = "Payload Too Large"
    type_slug = "upload-too-large"
    default_detail = "The uploaded file exceeds the size limit."


class UnsupportedFileTypeError(AppError):
    status_code = 415
    title = "Unsupported Media Type"
    type_slug = "unsupported-file-type"
    default_detail = "The file type is not supported or does not match its extension."


class ServiceUnavailableError(AppError):
    status_code = 503
    title = "Service Unavailable"
    type_slug = "service-unavailable"
    default_detail = "A required dependency is unavailable."


class RateLimitedError(AppError):
    status_code = 429
    title = "Too Many Requests"
    type_slug = "rate-limited"
    default_detail = "Too many requests. Try again later."

    def __init__(self, *, retry_after: int, detail: str | None = None) -> None:
        super().__init__(detail, headers={"Retry-After": str(retry_after)})


class InvalidCredentialsError(UnauthorizedError):
    type_slug = "invalid-credentials"
    default_detail = "Invalid credentials."


class InvalidTokenError(UnauthorizedError):
    type_slug = "invalid-token"
    default_detail = "Invalid or expired token."


class SignupDisabledError(ForbiddenError):
    type_slug = "signup-disabled"
    default_detail = "Tenant signup is disabled."


class TenantAlreadyExistsError(ConflictError):
    type_slug = "tenant-exists"
    default_detail = "Tenant slug or admin email is already registered."


class EmailAlreadyExistsError(ConflictError):
    type_slug = "email-exists"
    default_detail = "A user with this email already exists."


class UserNotFoundError(NotFoundError):
    type_slug = "user-not-found"
    default_detail = "User not found."


class LastAdminError(ConflictError):
    type_slug = "last-admin"
    default_detail = "The last active tenant admin cannot be removed."


class InvalidRoleError(UnprocessableError):
    type_slug = "invalid-role"
    default_detail = "One or more roles are invalid."


class DocumentNotFoundError(NotFoundError):
    type_slug = "document-not-found"
    default_detail = "Document not found."


class RoleNotHeldError(ForbiddenError):
    type_slug = "role-not-held"
    default_detail = "Access roles can only be assigned if you hold them yourself."


class InvalidCursorError(BadRequestError):
    type_slug = "invalid-cursor"
    default_detail = "The pagination cursor is invalid."


class StorageUnavailableError(ServiceUnavailableError):
    type_slug = "storage-unavailable"
    default_detail = "Object storage is unavailable."


def problem_response(
    *,
    status_code: int,
    title: str,
    detail: str,
    type_slug: str | None = None,
    errors: list[dict[str, Any]] | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {
        "type": f"https://errors.example.com/{type_slug or title.lower().replace(' ', '-')}",
        "title": title,
        "status": status_code,
        "detail": detail,
    }
    request_id = get_request_id()
    if request_id is not None:
        body["request_id"] = request_id
    if errors is not None:
        body["errors"] = errors
    return JSONResponse(
        body,
        status_code=status_code,
        headers=headers,
        media_type="application/problem+json",
    )


def _title_for(status_code: int) -> str:
    try:
        return HTTPStatus(status_code).phrase
    except ValueError:
        return "Error"


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else _title_for(exc.status_code)
    return problem_response(
        status_code=exc.status_code,
        title=_title_for(exc.status_code),
        detail=detail,
        headers=exc.headers,
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    errors = [
        {
            "loc": [str(part) for part in error.get("loc", ())],
            "msg": str(error.get("msg", "invalid value")),
            "type": str(error.get("type", "value_error")),
        }
        for error in exc.errors()
    ]
    return problem_response(
        status_code=422,
        title="Unprocessable Entity",
        detail="Request validation failed.",
        type_slug="validation-error",
        errors=errors,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("request.unhandled_error", path=request.url.path)
    return problem_response(
        status_code=500,
        title="Internal Server Error",
        detail="An unexpected error occurred.",
        type_slug="internal-error",
    )


async def app_exception_handler(request: Request, exc: AppError) -> JSONResponse:
    return problem_response(
        status_code=exc.status_code,
        title=exc.title,
        detail=exc.detail,
        type_slug=exc.type_slug,
        headers=exc.headers,
    )


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, app_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)
