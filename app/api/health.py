from __future__ import annotations

import asyncio
from typing import Any

import boto3
import structlog
from botocore.config import Config as BotoConfig
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.config import Settings

log = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness: the process is running. No dependency checks."""
    return {"status": "ok"}


async def _database_ok(engine: AsyncEngine) -> bool:
    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))
    return True


async def _redis_ok(settings: Settings) -> bool:
    client = Redis.from_url(settings.redis_url)
    try:
        return bool(await client.ping())
    finally:
        await client.aclose()


def _storage_ok_blocking(settings: Settings) -> bool:
    client = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key_id,
        aws_secret_access_key=settings.s3_secret_access_key,
        region_name=settings.s3_region,
        config=BotoConfig(connect_timeout=2, read_timeout=2, retries={"max_attempts": 1}),
    )
    client.head_bucket(Bucket=settings.s3_bucket)
    return True


async def _storage_ok(settings: Settings) -> bool:
    return await asyncio.to_thread(_storage_ok_blocking, settings)


@router.get("/readyz")
async def readyz(request: Request, full: bool = False) -> JSONResponse:
    """Readiness: database is required; Redis and object storage only with `full=true`."""
    settings: Settings = request.app.state.settings
    engine: AsyncEngine | None = getattr(request.app.state, "engine", None)
    checks: dict[str, str] = {}

    if engine is None:
        checks["database"] = "unavailable"
    else:
        try:
            await _database_ok(engine)
            checks["database"] = "ok"
        except Exception:
            log.exception("readiness.database_failed")
            checks["database"] = "error"

    if full:
        try:
            checks["redis"] = "ok" if await _redis_ok(settings) else "error"
        except Exception:
            log.exception("readiness.redis_failed")
            checks["redis"] = "error"

        try:
            checks["storage"] = "ok" if await _storage_ok(settings) else "error"
        except Exception:
            log.exception("readiness.storage_failed")
            checks["storage"] = "error"

    ready = all(status == "ok" for status in checks.values())
    payload: dict[str, Any] = {"status": "ready" if ready else "unavailable", "checks": checks}
    return JSONResponse(payload, status_code=200 if ready else 503)
