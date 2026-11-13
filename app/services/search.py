"""Hybrid search: query embedding, hybrid SQL, RRF fusion (plan §12).

The dense (pgvector HNSW cosine) and sparse (PostgreSQL FTS `ts_rank_cd`)
rankings are fused with Reciprocal Rank Fusion over *ranks only*, inside both
CTEs the tenant and ACL predicates are applied before the candidate `LIMIT`.
If the query embedding fails after the provider's own retries, the service
degrades to sparse-only and reports `mode="sparse_only"`; it never returns
dense rows pretending to be hybrid.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import structlog
from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, Float, Integer, Text, bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.errors import UnprocessableError
from app.security import TENANT_ADMIN_ROLE, CurrentUser
from app.services.embeddings import EmbeddingError, EmbeddingProvider

log = structlog.get_logger(__name__)

_TENANT_ACL_PREDICATE = """
      c.tenant_id = p.tenant_id
      AND d.tenant_id = p.tenant_id
      AND d.deleted_at IS NULL
      AND d.status = 'ready'
      AND (
            p.is_admin
            OR d.access_roles = '{}'::text[]
            OR d.access_roles && p.user_roles
          )
"""

# The two CTEs share the predicates; `p` keeps the constants in one place so
# they cannot diverge between retrievers (plan §12.2).
_HYBRID_SQL_TEXT = f"""
WITH p AS (
    SELECT
        CAST(:tenant_id AS uuid)     AS tenant_id,
        CAST(:user_roles AS text[])  AS user_roles,
        CAST(:is_admin AS boolean)   AS is_admin,
        CAST(:rrf_k AS integer)      AS rrf_k
),
dense AS (
    SELECT
        c.id,
        row_number() OVER (
            ORDER BY c.embedding <=> CAST(:query_vec AS vector), c.id
        )::int AS dense_rank
    FROM document_chunks c
    JOIN documents d ON d.id = c.document_id
    CROSS JOIN p
    WHERE{_TENANT_ACL_PREDICATE}
    ORDER BY c.embedding <=> CAST(:query_vec AS vector), c.id
    LIMIT :candidates
),
sparse AS (
    SELECT
        c.id,
        row_number() OVER (
            ORDER BY ts_rank_cd(c.fts_doc, q.tsquery) DESC, c.id
        )::int AS sparse_rank,
        ts_rank_cd(c.fts_doc, q.tsquery) AS sparse_score
    FROM document_chunks c
    JOIN documents d ON d.id = c.document_id
    CROSS JOIN p
    CROSS JOIN (SELECT websearch_to_tsquery('english', :query_text) AS tsquery) q
    WHERE{_TENANT_ACL_PREDICATE}
      AND c.fts_doc @@ q.tsquery
    ORDER BY ts_rank_cd(c.fts_doc, q.tsquery) DESC, c.id
    LIMIT :candidates
)
SELECT
    c.id AS chunk_id,
    c.document_id,
    d.title,
    d.source_type,
    c.chunk_index,
    c.content,
    c.metadata,
    dense.dense_rank,
    sparse.sparse_rank,
    sparse.sparse_score,
    (c.embedding <=> CAST(:query_vec AS vector)) AS cosine_distance,
    (
        COALESCE(CAST(:dense_weight AS float8) / (p.rrf_k + dense.dense_rank), 0.0) +
        COALESCE(CAST(:sparse_weight AS float8) / (p.rrf_k + sparse.sparse_rank), 0.0)
    ) AS rrf_score
FROM dense
FULL OUTER JOIN sparse ON dense.id = sparse.id
CROSS JOIN p
JOIN document_chunks c ON c.id = COALESCE(dense.id, sparse.id)
JOIN documents d ON d.id = c.document_id
ORDER BY rrf_score DESC, c.id
LIMIT :top_k
"""

_SPARSE_ONLY_SQL_TEXT = f"""
WITH p AS (
    SELECT
        CAST(:tenant_id AS uuid)     AS tenant_id,
        CAST(:user_roles AS text[])  AS user_roles,
        CAST(:is_admin AS boolean)   AS is_admin,
        CAST(:rrf_k AS integer)      AS rrf_k
),
sparse AS (
    SELECT
        c.id,
        row_number() OVER (
            ORDER BY ts_rank_cd(c.fts_doc, q.tsquery) DESC, c.id
        )::int AS sparse_rank,
        ts_rank_cd(c.fts_doc, q.tsquery) AS sparse_score
    FROM document_chunks c
    JOIN documents d ON d.id = c.document_id
    CROSS JOIN p
    CROSS JOIN (SELECT websearch_to_tsquery('english', :query_text) AS tsquery) q
    WHERE{_TENANT_ACL_PREDICATE}
      AND c.fts_doc @@ q.tsquery
    ORDER BY ts_rank_cd(c.fts_doc, q.tsquery) DESC, c.id
    LIMIT :candidates
)
SELECT
    c.id AS chunk_id,
    c.document_id,
    d.title,
    d.source_type,
    c.chunk_index,
    c.content,
    c.metadata,
    NULL::int AS dense_rank,
    sparse.sparse_rank,
    sparse.sparse_score,
    NULL::float8 AS cosine_distance,
    COALESCE(CAST(:sparse_weight AS float8) / (p.rrf_k + sparse.sparse_rank), 0.0)
        AS rrf_score
FROM sparse
CROSS JOIN p
JOIN document_chunks c ON c.id = sparse.id
JOIN documents d ON d.id = c.document_id
ORDER BY rrf_score DESC, c.id
LIMIT :top_k
"""

_HYBRID_SQL = text(_HYBRID_SQL_TEXT).bindparams(
    bindparam("tenant_id", type_=UUID),
    bindparam("user_roles", type_=ARRAY(Text)),
    bindparam("is_admin", type_=Boolean),
    bindparam("rrf_k", type_=Integer),
    bindparam("query_vec", type_=Vector(1536)),
    bindparam("query_text", type_=Text),
    bindparam("candidates", type_=Integer),
    bindparam("top_k", type_=Integer),
    bindparam("dense_weight", type_=Float),
    bindparam("sparse_weight", type_=Float),
)

_SPARSE_ONLY_SQL = text(_SPARSE_ONLY_SQL_TEXT).bindparams(
    bindparam("tenant_id", type_=UUID),
    bindparam("user_roles", type_=ARRAY(Text)),
    bindparam("is_admin", type_=Boolean),
    bindparam("rrf_k", type_=Integer),
    bindparam("query_text", type_=Text),
    bindparam("candidates", type_=Integer),
    bindparam("top_k", type_=Integer),
    bindparam("sparse_weight", type_=Float),
)


def rrf_score(
    dense_rank: int | None,
    sparse_rank: int | None,
    *,
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> float:
    """Reciprocal Rank Fusion over ranks only (plan §2, §12.5).

    This is the Python mirror of the expression the hybrid SQL computes; the
    integration tests assert both produce identical values. A missing rank
    (chunk found by only one retriever) contributes nothing, never a penalty.
    """
    total = 0.0
    if dense_rank is not None:
        total += dense_weight / (k + dense_rank)
    if sparse_rank is not None:
        total += sparse_weight / (k + sparse_rank)
    return total


@dataclass(frozen=True, slots=True)
class SearchRow:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    title: str
    source_type: str
    chunk_index: int
    content: str
    metadata: dict[str, Any]
    dense_rank: int | None
    sparse_rank: int | None
    sparse_score: float | None
    cosine_distance: float | None
    rrf_score: float


@dataclass(frozen=True, slots=True)
class SearchResult:
    mode: str  # "hybrid" or "sparse_only"
    rows: list[SearchRow]


class SearchService:
    """Query embedding + hybrid SQL, executed in the caller's tenant session."""

    def __init__(self, settings: Settings, embeddings: EmbeddingProvider) -> None:
        self._settings = settings
        self._embeddings = embeddings

    async def search(
        self, session: AsyncSession, *, user: CurrentUser, query: str, top_k: int
    ) -> SearchResult:
        if not 1 <= top_k <= self._settings.search_top_k_max:
            raise UnprocessableError(
                f"top_k must be between 1 and {self._settings.search_top_k_max}."
            )
        tenant_id = user.tenant_id
        is_admin = TENANT_ADMIN_ROLE in user.roles
        candidates = self._settings.search_candidates

        await self._set_hnsw_settings(session)
        try:
            query_vec = await self._embeddings.embed_query(query)
        except EmbeddingError as exc:
            log.warning(
                "search.query_embedding_failed",
                mode="sparse_only",
                error=type(exc).__name__,
                tenant_id=str(tenant_id),
            )
            rows = await self._run(
                session,
                _SPARSE_ONLY_SQL,
                {
                    "tenant_id": tenant_id,
                    "user_roles": list(user.roles),
                    "is_admin": is_admin,
                    "rrf_k": self._settings.rrf_k,
                    "query_text": query,
                    "candidates": candidates,
                    "top_k": top_k,
                    "sparse_weight": self._settings.sparse_weight,
                },
            )
            return SearchResult(mode="sparse_only", rows=rows)

        rows = await self._run(
            session,
            _HYBRID_SQL,
            {
                "tenant_id": tenant_id,
                "user_roles": list(user.roles),
                "is_admin": is_admin,
                "rrf_k": self._settings.rrf_k,
                "query_vec": query_vec,
                "query_text": query,
                "candidates": candidates,
                "top_k": top_k,
                "dense_weight": self._settings.dense_weight,
                "sparse_weight": self._settings.sparse_weight,
            },
        )
        return SearchResult(mode="hybrid", rows=rows)

    async def _set_hnsw_settings(self, session: AsyncSession) -> None:
        """Transaction-local HNSW settings for filtered ANN recall (§12.3)."""
        for name, value in (
            ("hnsw.iterative_scan", self._settings.hnsw_iterative_scan),
            ("hnsw.ef_search", str(self._settings.hnsw_ef_search)),
            ("hnsw.max_scan_tuples", str(self._settings.hnsw_max_scan_tuples)),
        ):
            await session.execute(
                text("SELECT set_config(:name, :value, true)"),
                {"name": name, "value": value},
            )

    async def _run(
        self, session: AsyncSession, sql: Any, params: dict[str, Any]
    ) -> list[SearchRow]:
        result = await session.execute(sql, params)
        return [
            SearchRow(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                title=row["title"],
                source_type=row["source_type"],
                chunk_index=row["chunk_index"],
                content=row["content"],
                metadata=row["metadata"] or {},
                dense_rank=row["dense_rank"],
                sparse_rank=row["sparse_rank"],
                sparse_score=row["sparse_score"],
                cosine_distance=row["cosine_distance"],
                rrf_score=float(row["rrf_score"]),
            )
            for row in result.mappings()
        ]
