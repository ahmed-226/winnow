from __future__ import annotations

import logging as stdlib_logging
from contextvars import ContextVar
from typing import Any

import structlog

from app.config import Settings

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def configure_logging(settings: Settings) -> None:
    level = getattr(stdlib_logging, settings.log_level)
    stdlib_logging.basicConfig(level=level, format="%(message)s")

    renderer: structlog.types.Processor
    if settings.app_env == "production":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )


def get_request_id() -> str | None:
    return request_id_var.get()


def get_logger(name: str) -> Any:
    return structlog.get_logger(name)
