"""Prometheus metrics beyond the HTTP instrumentator (plan §15.2).

All metrics live on the default registry so `/metrics` (exposed by
`prometheus-fastapi-instrumentator`) serves them together with the HTTP
counters. Never put document text, query text, or tenant identifiers into
label values; labels are bounded enums only.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

RATE_LIMIT_REJECTIONS = Counter(
    "rate_limit_rejections_total",
    "Requests rejected with 429 by rate-limit scope.",
    ["scope"],
)

DOCUMENTS_UPLOADED = Counter(
    "documents_uploaded_total",
    "Document uploads accepted (duplicates included).",
    ["source_type", "duplicate"],
)

INGEST_DURATION = Histogram(
    "document_ingest_duration_seconds",
    "End-to-end ingestion duration per job outcome.",
    ["source_type", "outcome"],
)

INGEST_FAILURES = Counter(
    "document_ingest_failures_total",
    "Ingestion failures by error code (§9.4 taxonomy).",
    ["error_code"],
)

SEARCH_REQUESTS = Counter(
    "search_requests_total",
    "Search requests by response mode and outcome.",
    ["mode", "outcome"],
)

SEARCH_DURATION = Histogram(
    "search_duration_seconds",
    "Search endpoint duration.",
    ["mode"],
)

SEARCH_RESULTS = Histogram(
    "search_results_count",
    "Number of results returned per search request.",
    buckets=(0, 1, 3, 5, 10, 20, 50),
)

EMBEDDING_REQUESTS = Counter(
    "embedding_requests_total",
    "Embedding API requests by operation and status.",
    ["operation", "status"],
)

EMBEDDING_DURATION = Histogram(
    "embedding_request_duration_seconds",
    "Embedding API request duration.",
    ["operation"],
)

DB_POOL_IN_USE = Gauge(
    "db_pool_in_use",
    "SQLAlchemy connections currently checked out by the API process.",
)

QUEUE_DEPTH = Gauge(
    "queue_depth",
    "Ingest backlog: stream entries not yet delivered to the consumer group.",
    ["stream"],
)

QUEUE_PENDING = Gauge(
    "queue_pending_messages",
    "Pending (delivered, unacknowledged) messages in the consumer group.",
    ["group"],
)


def install_process_gauges(engine: object, sync_redis: object, *, stream: str, group: str) -> None:
    """Wire the API-process gauges to live values, evaluated at scrape time.

    The callbacks must never raise: a Redis outage reports 0 rather than
    failing the `/metrics` scrape.
    """
    pool = engine.pool  # type: ignore[attr-defined]
    DB_POOL_IN_USE.set_function(pool.checkedout)

    def _queue_depth() -> float:
        try:
            # XLEN counts consumed history (kept until MAXLEN trim); the group's
            # `lag` is the actual backlog of undelivered entries.
            for group_info in sync_redis.xinfo_groups(stream):  # type: ignore[attr-defined]
                if group_info.get("name") == group:
                    return float(group_info.get("lag") or 0)
            return 0.0
        except Exception:  # a Redis outage must not fail the /metrics scrape
            return 0.0

    def _queue_pending() -> float:
        try:
            summary = sync_redis.xpending(group)  # type: ignore[attr-defined]
            return float(summary.get("pending", 0) if isinstance(summary, dict) else 0)
        except Exception:  # a Redis outage must not fail the /metrics scrape
            return 0.0

    QUEUE_DEPTH.labels(stream=stream).set_function(_queue_depth)
    QUEUE_PENDING.labels(group=group).set_function(_queue_pending)
