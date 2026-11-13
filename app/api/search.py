from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps import CurrentUser, current_user, tenant_session
from app.metrics import SEARCH_DURATION, SEARCH_REQUESTS, SEARCH_RESULTS
from app.schemas import SearchRequest, SearchResponse, SearchResultItem
from app.services.rate_limit import enforce_rate_limit
from app.services.search import SearchService

router = APIRouter(prefix="/api/v1", tags=["search"])


def get_search_service(request: Request) -> SearchService:
    embeddings = request.app.state.embeddings
    return SearchService(request.app.state.settings, embeddings)


@router.post("/search", response_model=SearchResponse)
async def search(
    payload: SearchRequest,
    request: Request,
    user: CurrentUser = Depends(current_user),
    session: AsyncSession = Depends(tenant_session),
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    await enforce_rate_limit(
        request.app.state.redis,
        scope="search",
        key=f"rag:rl:search:{user.id}",
        limit=request.app.state.settings.rl_search_per_minute,
        window_seconds=60,
    )
    started = time.monotonic()
    try:
        result = await service.search(session, user=user, query=payload.query, top_k=payload.top_k)
    except Exception:
        SEARCH_REQUESTS.labels(mode="error", outcome="error").inc()
        raise
    duration = time.monotonic() - started
    mode = result.mode
    SEARCH_REQUESTS.labels(mode=mode, outcome="ok").inc()
    SEARCH_DURATION.labels(mode=mode).observe(duration)
    SEARCH_RESULTS.observe(len(result.rows))
    return SearchResponse(
        query=payload.query,
        mode=mode,  # type: ignore[arg-type]
        took_ms=round(duration * 1000),
        results=[
            SearchResultItem(
                chunk_id=row.chunk_id,
                document_id=row.document_id,
                title=row.title,
                source_type=row.source_type,
                chunk_index=row.chunk_index,
                content=row.content,
                metadata=row.metadata,
                score=row.rrf_score,
                dense_rank=row.dense_rank,
                sparse_rank=row.sparse_rank,
                sparse_score=row.sparse_score,
                cosine_distance=row.cosine_distance,
            )
            for row in result.rows
        ],
    )
