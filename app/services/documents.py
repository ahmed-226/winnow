"""Document upload, listing, ACL rules, and soft delete.

Business rules for the Phase 2 document API live here; routes stay thin.
Upload streams the file to a temp file (enforcing the size limit), hashes
it, sniffs magic bytes, uploads the object, then inserts the row. A
duplicate `(tenant_id, file_hash)` returns the existing document.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import filetype
import structlog
from redis.asyncio import Redis
from sqlalchemy import or_, select, tuple_
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import TEXT
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.deps import CurrentUser
from app.errors import (
    DocumentNotFoundError,
    ForbiddenError,
    InvalidCursorError,
    InvalidRoleError,
    RoleNotHeldError,
    UnprocessableError,
    UnsupportedFileTypeError,
    UploadTooLargeError,
)
from app.metrics import DOCUMENTS_UPLOADED
from app.models import Document, DocumentStatus, IngestionJob
from app.security import TENANT_ADMIN_ROLE, normalize_roles
from app.services.ingestion import reset_job_for_reprocess
from app.services.queue import enqueue_ingest
from app.services.storage import StorageService

log = structlog.get_logger(__name__)

_SOURCE_TYPES: dict[str, tuple[str, str]] = {
    ".pdf": ("pdf", "application/pdf"),
    ".md": ("markdown", "text/markdown"),
    ".markdown": ("markdown", "text/markdown"),
    ".html": ("html", "text/html"),
    ".htm": ("html", "text/html"),
}

# Magic-byte results that are still acceptable for text-based source types.
_TEXTUAL_MIMES = frozenset(
    {
        "text/plain",
        "text/html",
        "text/markdown",
        "text/xml",
        "application/xml",
        "application/xhtml+xml",
    }
)

_READ_CHUNK_BYTES = 1_048_576
_SNIFF_BYTES = 65_536
_KEY_NAME_MAX_CHARS = 120
_TITLE_MAX_CHARS = 255


@dataclass(frozen=True, slots=True)
class UploadResult:
    document: Document
    duplicate: bool


class _SpooledUpload:
    """Temp file that stages an upload before hashing and object upload."""

    def __init__(self) -> None:
        descriptor, self._path = tempfile.mkstemp(prefix="rag-upload-")
        self._handle = os.fdopen(descriptor, "wb")

    @property
    def path(self) -> Path:
        return Path(self._path)

    def write(self, data: bytes) -> None:
        self._handle.write(data)

    def flush(self) -> None:
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()
        self._handle = None  # type: ignore[assignment]

    def __enter__(self) -> _SpooledUpload:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
        try:
            Path(self._path).unlink(missing_ok=True)
        except OSError:
            log.warning("documents.temp_file_cleanup_failed", path=self._path)


def sanitize_file_name(raw: str) -> str:
    """Reduce a client-supplied name to a safe single path component."""
    name = Path(raw).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return cleaned[:_KEY_NAME_MAX_CHARS] or "document"


def source_type_for(file_name: str) -> str:
    suffix = Path(file_name).suffix.lower()
    try:
        return _SOURCE_TYPES[suffix][0]
    except KeyError as exc:
        raise UnsupportedFileTypeError(
            "File extension must be one of: pdf, md, markdown, html, htm."
        ) from exc


def source_type_mime(source_type: str) -> str:
    for mapped_source, mapped_mime in _SOURCE_TYPES.values():
        if mapped_source == source_type:
            return mapped_mime
    raise UnsupportedFileTypeError()


def encode_cursor(*, created_at: datetime, document_id: uuid.UUID) -> str:
    payload = {"created_at": created_at.isoformat(), "id": str(document_id)}
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()


def decode_cursor(cursor: str) -> tuple[datetime, uuid.UUID]:
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor.encode()))
        created_at = datetime.fromisoformat(str(payload["created_at"]))
        document_id = uuid.UUID(str(payload["id"]))
    except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
        raise InvalidCursorError() from exc
    if created_at.tzinfo is None:
        raise InvalidCursorError()
    return created_at, document_id


def _title_from(raw_title: str | None, file_name: str) -> str:
    title = (raw_title or "").strip()
    if not title:
        title = Path(file_name).stem.strip() or "Untitled"
    return title[:_TITLE_MAX_CHARS]


def _validate_access_roles(*, roles: list[str], user: CurrentUser, is_admin: bool) -> list[str]:
    try:
        normalized = normalize_roles(roles)
    except ValueError as exc:
        raise InvalidRoleError(str(exc)) from exc
    if TENANT_ADMIN_ROLE in normalized:
        raise InvalidRoleError("tenant_admin cannot be assigned to documents.")
    if not is_admin:
        held = set(user.roles)
        if any(role not in held for role in normalized):
            raise RoleNotHeldError()
    return normalized


def _check_magic_bytes(source_type: str, head: bytes) -> None:
    detected = filetype.guess(head)
    if source_type == "pdf":
        if detected is None or detected.mime != "application/pdf":
            raise UnsupportedFileTypeError("File content does not match the pdf extension.")
        return
    if detected is not None and detected.mime not in _TEXTUAL_MIMES:
        raise UnsupportedFileTypeError("File content does not match a text-based document type.")


def _visibility_clauses(user: CurrentUser) -> list[Any]:
    """ACL filter: empty ACL is tenant-wide; admins bypass document ACLs."""
    if TENANT_ADMIN_ROLE in user.roles:
        return []
    clauses: list[Any] = [Document.access_roles == []]
    if user.roles:
        role_array = postgresql.array(list(user.roles)).cast(postgresql.ARRAY(TEXT()))
        clauses.append(Document.access_roles.op("&&")(role_array))
    return [or_(*clauses)]


def _ensure_can_modify(user: CurrentUser, document: Document) -> None:
    if TENANT_ADMIN_ROLE not in user.roles and document.created_by != user.id:
        raise ForbiddenError("Only the tenant admin or the uploader can modify a document.")


async def _enqueue_ingest(
    settings: Settings,
    redis: Redis,
    *,
    document_id: uuid.UUID,
    tenant_id: uuid.UUID,
    job_id: uuid.UUID,
) -> None:
    """Best-effort enqueue; the sweeper recovers lost messages."""
    try:
        await enqueue_ingest(
            redis,
            stream=settings.ingest_stream,
            document_id=document_id,
            tenant_id=tenant_id,
            job_id=job_id,
        )
    except Exception as exc:
        log.warning(
            "documents.enqueue_failed",
            document_id=str(document_id),
            error=type(exc).__name__,
        )


class DocumentService:
    def __init__(self, settings: Settings, storage: StorageService) -> None:
        self._settings = settings
        self._storage = storage

    # ------------------------------------------------------------------ upload

    async def upload(
        self,
        session: AsyncSession,
        *,
        user: CurrentUser,
        redis: Redis,
        file_name: str,
        read: Callable[[int], Awaitable[bytes]],
        title: str | None,
        access_roles: list[str],
    ) -> UploadResult:
        is_admin = TENANT_ADMIN_ROLE in user.roles
        roles = _validate_access_roles(roles=access_roles, user=user, is_admin=is_admin)
        safe_name = sanitize_file_name(file_name)
        source_type = source_type_for(safe_name)
        doc_title = _title_from(title, safe_name)

        with _SpooledUpload() as staged:
            digest, size, head = await self._stage_file(read, staged)
            _check_magic_bytes(source_type, head)
            file_hash = digest.hexdigest()

            existing = await self._find_live_by_hash(
                session, tenant_id=user.tenant_id, file_hash=file_hash
            )
            if existing is not None:
                DOCUMENTS_UPLOADED.labels(source_type=source_type, duplicate="true").inc()
                return UploadResult(document=existing, duplicate=True)

            document_id = uuid.uuid4()
            storage_key = f"tenant/{user.tenant_id}/doc/{document_id}/{file_hash[:16]}-{safe_name}"
            await self._storage.ensure_bucket()
            await self._storage.put_object(
                key=storage_key,
                file_path=staged.path,
                content_type=source_type_mime(source_type),
            )

            savepoint = await session.begin_nested()
            job = IngestionJob(id=uuid.uuid4(), document_id=document_id, tenant_id=user.tenant_id)
            document = Document(
                id=document_id,
                tenant_id=user.tenant_id,
                title=doc_title,
                file_name=safe_name,
                mime_type=source_type_mime(source_type),
                source_type=source_type,
                storage_key=storage_key,
                file_hash=file_hash,
                size_bytes=size,
                access_roles=roles,
                status=DocumentStatus.PENDING,
                created_by=user.id,
            )
            session.add(document)
            session.add(job)
            try:
                await session.flush()
            except IntegrityError:
                # A concurrent upload of the same bytes won the unique index.
                await savepoint.rollback()
                winner = await self._find_live_by_hash(
                    session, tenant_id=user.tenant_id, file_hash=file_hash
                )
                if winner is None:
                    raise
                await self._storage.delete_object_quietly(key=storage_key)
                DOCUMENTS_UPLOADED.labels(source_type=source_type, duplicate="true").inc()
                return UploadResult(document=winner, duplicate=True)
            else:
                await savepoint.commit()

            await _enqueue_ingest(
                self._settings,
                redis,
                document_id=document.id,
                tenant_id=user.tenant_id,
                job_id=job.id,
            )

        log.info(
            "documents.uploaded",
            document_id=str(document.id),
            tenant_id=str(user.tenant_id),
            size_bytes=size,
            duplicate=False,
        )
        DOCUMENTS_UPLOADED.labels(source_type=source_type, duplicate="false").inc()
        return UploadResult(document=document, duplicate=False)

    async def _stage_file(
        self, read: Callable[[int], Awaitable[bytes]], staged: _SpooledUpload
    ) -> tuple[Any, int, bytes]:
        digest = hashlib.sha256()
        size = 0
        head = b""
        first = True
        while True:
            chunk = await read(_READ_CHUNK_BYTES)
            if not chunk:
                break
            if first:
                head = chunk[:_SNIFF_BYTES]
                first = False
            size += len(chunk)
            if size > self._settings.max_upload_bytes:
                raise UploadTooLargeError()
            digest.update(chunk)
            staged.write(chunk)
        staged.flush()
        if size == 0:
            raise UnsupportedFileTypeError("The uploaded file is empty.")
        return digest, size, head

    async def _find_live_by_hash(
        self, session: AsyncSession, *, tenant_id: uuid.UUID, file_hash: str
    ) -> Document | None:
        result = await session.execute(
            select(Document).where(
                Document.tenant_id == tenant_id,
                Document.file_hash == file_hash,
                Document.deleted_at.is_(None),
            )
        )
        return result.scalar_one_or_none()

    # ------------------------------------------------------------------ queries

    async def get_visible(
        self, session: AsyncSession, *, user: CurrentUser, document_id: uuid.UUID
    ) -> Document:
        document = (
            await session.execute(
                select(Document).where(
                    Document.id == document_id,
                    Document.tenant_id == user.tenant_id,
                    Document.deleted_at.is_(None),
                    *_visibility_clauses(user),
                )
            )
        ).scalar_one_or_none()
        if document is None:
            raise DocumentNotFoundError()
        return document

    async def list_documents(
        self,
        session: AsyncSession,
        *,
        user: CurrentUser,
        limit: int,
        cursor: str | None,
    ) -> tuple[list[Document], str | None]:
        clauses: list[Any] = [
            Document.tenant_id == user.tenant_id,
            Document.deleted_at.is_(None),
            *_visibility_clauses(user),
        ]
        if cursor is not None:
            cursor_created_at, cursor_document_id = decode_cursor(cursor)
            clauses.append(
                tuple_(Document.created_at, Document.id)
                < tuple_(cursor_created_at, cursor_document_id)
            )

        result = await session.execute(
            select(Document)
            .where(*clauses)
            .order_by(Document.created_at.desc(), Document.id.desc())
            .limit(limit + 1)
        )
        documents = list(result.scalars())

        next_cursor: str | None = None
        if len(documents) > limit:
            documents = documents[:limit]
            last = documents[-1]
            next_cursor = encode_cursor(created_at=last.created_at, document_id=last.id)
        return documents, next_cursor

    # ------------------------------------------------------------------ mutations

    async def patch_document(
        self,
        session: AsyncSession,
        *,
        user: CurrentUser,
        document: Document,
        title: str | None,
        access_roles: list[str] | None,
    ) -> Document:
        _ensure_can_modify(user, document)
        if access_roles is not None:
            is_admin = TENANT_ADMIN_ROLE in user.roles
            document.access_roles = _validate_access_roles(
                roles=access_roles, user=user, is_admin=is_admin
            )
        if title is not None:
            stripped = title.strip()
            if not stripped:
                raise UnprocessableError("Title must not be blank.")
            document.title = stripped[:_TITLE_MAX_CHARS]
        await session.flush()
        return document

    async def delete_document(
        self, session: AsyncSession, *, user: CurrentUser, document: Document
    ) -> str:
        _ensure_can_modify(user, document)
        document.deleted_at = datetime.now(UTC)
        await session.flush()
        return document.storage_key

    async def reprocess(
        self,
        session: AsyncSession,
        *,
        user: CurrentUser,
        redis: Redis,
        document: Document,
    ) -> Document:
        """Reset the job to queued, mark the document pending, and enqueue.

        Safe to call repeatedly: the worker replaces the old chunks in its
        final transaction, so no duplicates can appear.
        """
        if TENANT_ADMIN_ROLE not in user.roles:
            raise ForbiddenError("Only tenant_admin can reprocess documents.")
        await reset_job_for_reprocess(session, document_id=document.id, tenant_id=user.tenant_id)
        document.status = DocumentStatus.PENDING
        await session.flush()
        job = (
            await session.execute(
                select(IngestionJob).where(
                    IngestionJob.document_id == document.id,
                    IngestionJob.tenant_id == user.tenant_id,
                )
            )
        ).scalar_one()
        await _enqueue_ingest(
            self._settings,
            redis,
            document_id=document.id,
            tenant_id=user.tenant_id,
            job_id=job.id,
        )
        return document
