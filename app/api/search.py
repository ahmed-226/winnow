from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps import CurrentUser, current_user, tenant_session
from app.schemas import SearchRequest, SearchResponse, SearchResultItem
from app.services.search import SearchService

router = APIRouter(prefix="/api/v1", tags=["search"])


def get_search_service(request: Request) -> SearchService:
    embeddings = request.app.state.embeddings
    return SearchService(request.app.state.settings, embeddings)


@router.post("/search", response_model=SearchResponse)
async def search(
    payload: SearchRequest,
    user: CurrentUser = Depends(current_user),
    session: AsyncSession = Depends(tenant_session),
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    started = time.monotonic()
    result = await service.search(session, user=user, query=payload.query, top_k=payload.top_k)
    return SearchResponse(
        query=payload.query,
        mode=result.mode,  # type: ignore[arg-type]
        took_ms=round((time.monotonic() - started) * 1000),
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
