from __future__ import annotations

import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from redis import Redis as SyncRedis
from redis.asyncio import Redis

from app.api.auth import router as auth_router
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.search import router as search_router
from app.api.users import router as users_router
from app.config import Settings, get_settings
from app.db import create_engine, create_sessionmaker
from app.errors import install_error_handlers
from app.logging import configure_logging, request_id_var
from app.metrics import install_process_gauges
from app.services.embeddings import build_embedding_provider
from app.services.storage import StorageService

log = structlog.get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    engine = create_engine(settings)
    app.state.engine = engine
    app.state.sessionmaker = create_sessionmaker(engine)
    app.state.redis = Redis.from_url(settings.redis_url, decode_responses=True)
    app.state.storage = StorageService(settings)
    # The API embeds queries only; the worker embeds documents with its own
    # provider instance. The Redis client doubles as the query-embedding cache.
    app.state.embeddings = build_embedding_provider(settings, cache=app.state.redis)
    # A separate sync client feeds the scrape-time gauges (queue depth/pending,
    # DB pool usage). The worker serves its own /metrics, but the API's queue
    # gauges stay here so one scrape shows queue state next to HTTP traffic.
    metrics_redis = SyncRedis.from_url(settings.redis_url, decode_responses=True)
    install_process_gauges(
        engine,
        metrics_redis,
        stream=settings.ingest_stream,
        group=settings.ingest_group,
    )
    try:
        await app.state.storage.ensure_bucket()
    except Exception as exc:  # pragma: no cover - depends on MinIO availability
        log.warning(
            "app.storage_bootstrap_failed",
            bucket=settings.s3_bucket,
            error=type(exc).__name__,
        )
    log.info("app.started", app_env=settings.app_env)
    try:
        yield
    finally:
        await app.state.redis.aclose()
        metrics_redis.close()
        await engine.dispose()
        log.info("app.stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings)

    app = FastAPI(
        title="Winnow",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[REQUEST_ID_HEADER],
    )

    install_error_handlers(app)

    @app.middleware("http")
    async def request_id_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id_header = request.headers.get(REQUEST_ID_HEADER)
        request_id = (
            request_id_header
            if request_id_header and _REQUEST_ID_PATTERN.fullmatch(request_id_header)
            else uuid.uuid4().hex
        )
        token = request_id_var.set(request_id)
        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            response = await call_next(request)
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            structlog.contextvars.clear_contextvars()
            request_id_var.reset(token)

    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(users_router)
    app.include_router(documents_router)
    app.include_router(search_router)

    instrumentator = Instrumentator(
        excluded_handlers=["/metrics", "/healthz", "/readyz"],
    )
    instrumentator.instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

    return app


app = create_app()
