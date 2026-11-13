from __future__ import annotations

import uuid

from fastapi import (
    APIRouter,
    Depends,
    Form,
    Query,
    Request,
    Response,
    UploadFile,
)
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps import (
    CurrentUser,
    current_user,
    get_document_service,
    get_redis,
    get_storage_service,
    tenant_session,
)
from app.schemas import (
    DocumentListItem,
    DocumentListResponse,
    DocumentOut,
    DocumentPatchRequest,
    UploadDocumentResponse,
)
from app.services.documents import DocumentService
from app.services.rate_limit import enforce_rate_limit
from app.services.storage import StorageService

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post("", response_model=UploadDocumentResponse)
async def upload_document(
    file: UploadFile,
    response: Response,
    request: Request,
    title: str | None = Form(default=None),
    access_roles: list[str] = Form(default=[]),
    user: CurrentUser = Depends(current_user),
    session: AsyncSession = Depends(tenant_session),
    redis: Redis = Depends(get_redis),
    service: DocumentService = Depends(get_document_service),
) -> UploadDocumentResponse:
    await enforce_rate_limit(
        request.app.state.redis,
        scope="upload",
        key=f"rag:rl:upload:{user.id}",
        limit=request.app.state.settings.rl_upload_per_hour,
        window_seconds=3600,
    )
    result = await service.upload(
        session,
        user=user,
        redis=redis,
        file_name=file.filename or "",
        read=file.read,
        title=title,
        access_roles=access_roles,
    )
    response.status_code = 200 if result.duplicate else 202
    data = DocumentOut.model_validate(result.document).model_dump(mode="python")
    return UploadDocumentResponse.model_validate({**data, "duplicate": result.duplicate})


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(default=None),
    user: CurrentUser = Depends(current_user),
    session: AsyncSession = Depends(tenant_session),
    service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    documents, next_cursor = await service.list_documents(
        session, user=user, limit=limit, cursor=cursor
    )
    return DocumentListResponse(
        items=[DocumentListItem.model_validate(document) for document in documents],
        next_cursor=next_cursor,
    )


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(
    document_id: uuid.UUID,
    user: CurrentUser = Depends(current_user),
    session: AsyncSession = Depends(tenant_session),
    service: DocumentService = Depends(get_document_service),
) -> DocumentOut:
    document = await service.get_visible(session, user=user, document_id=document_id)
    return DocumentOut.model_validate(document)


@router.patch("/{document_id}", response_model=DocumentOut)
async def patch_document(
    document_id: uuid.UUID,
    payload: DocumentPatchRequest,
    user: CurrentUser = Depends(current_user),
    session: AsyncSession = Depends(tenant_session),
    service: DocumentService = Depends(get_document_service),
) -> DocumentOut:
    document = await service.get_visible(session, user=user, document_id=document_id)
    updated = await service.patch_document(
        session,
        user=user,
        document=document,
        title=payload.title,
        access_roles=payload.access_roles,
    )
    return DocumentOut.model_validate(updated)


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: uuid.UUID,
    user: CurrentUser = Depends(current_user),
    session: AsyncSession = Depends(tenant_session),
    storage: StorageService = Depends(get_storage_service),
    service: DocumentService = Depends(get_document_service),
) -> None:
    document = await service.get_visible(session, user=user, document_id=document_id)
    storage_key = await service.delete_document(session, user=user, document=document)
    # The object delete is best effort and runs after the row update.
    await storage.delete_object_quietly(key=storage_key)


@router.post("/{document_id}/reprocess", status_code=202, response_model=DocumentOut)
async def reprocess_document(
    document_id: uuid.UUID,
    user: CurrentUser = Depends(current_user),
    session: AsyncSession = Depends(tenant_session),
    redis: Redis = Depends(get_redis),
    service: DocumentService = Depends(get_document_service),
) -> DocumentOut:
    document = await service.get_visible(session, user=user, document_id=document_id)
    updated = await service.reprocess(session, user=user, redis=redis, document=document)
    return DocumentOut.model_validate(updated)
