from __future__ import annotations

import asyncio

from app.config import Settings
from app.logging import get_logger

log = get_logger(__name__)


async def run() -> None:
    """Phase 0 placeholder.

    The Phase 3 worker replaces this with the Redis Streams consumer, sweeper,
    and dead-letter handling. Keeping the process alive here lets the compose
    worker service start alongside the API.
    """
    log.info("worker.placeholder_started", detail="ingestion consumer lands in Phase 3")
    await asyncio.Event().wait()


def main() -> None:
    from app.logging import configure_logging

    configure_logging(Settings())  # type: ignore[call-arg]
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
