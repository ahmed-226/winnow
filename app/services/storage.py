"""S3/MinIO object storage abstraction.

Thin wrapper around boto3 with synchronous calls pushed to a thread so the
event loop is not blocked. Works against moto in tests and MinIO locally.
The bucket is always private; nothing here grants public access.
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import boto3
import structlog
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

from app.config import Settings
from app.errors import StorageUnavailableError

log = structlog.get_logger(__name__)

_S3_ERROR_CODES = frozenset({"404", "NoSuchBucket", "NoSuchKey", "NotFound", "BucketNotFoundError"})


def _client(settings: Settings) -> Any:
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key_id,
        aws_secret_access_key=settings.s3_secret_access_key,
        region_name=settings.s3_region,
        config=BotoConfig(
            connect_timeout=5,
            read_timeout=30,
            retries={"max_attempts": 2, "mode": "standard"},
        ),
    )


def _is_not_found(error: ClientError) -> bool:
    code = error.response.get("Error", {}).get("Code", "")
    status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in _S3_ERROR_CODES or status == 404


class StorageService:
    """Owns all object-storage operations for the application."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = _client(settings)

    @property
    def bucket(self) -> str:
        return self._settings.s3_bucket

    async def _run(self, function: Callable[[], Any]) -> Any:
        try:
            return await asyncio.to_thread(function)
        except (BotoCoreError, ClientError, OSError) as exc:
            log.error("storage.operation_failed", error=type(exc).__name__)
            raise StorageUnavailableError() from exc

    async def ensure_bucket(self) -> None:
        """Create the bucket if missing. Idempotent; keeps it private."""

        def _ensure() -> None:
            try:
                self._client.head_bucket(Bucket=self.bucket)
                return
            except ClientError as error:
                if not _is_not_found(error):
                    raise
            if self._settings.s3_region in ("us-east-1", ""):
                self._client.create_bucket(Bucket=self.bucket)
            else:
                self._client.create_bucket(
                    Bucket=self.bucket,
                    CreateBucketConfiguration={"LocationConstraint": self._settings.s3_region},
                )
            log.info("storage.bucket_created", bucket=self.bucket)

        await self._run(functools.partial(_ensure))

    async def put_object(self, *, key: str, file_path: Path, content_type: str) -> None:
        def _put() -> None:
            with file_path.open("rb") as handle:
                self._client.upload_fileobj(
                    handle,
                    self.bucket,
                    key,
                    ExtraArgs={"ContentType": content_type},
                )

        await self._run(functools.partial(_put))

    async def delete_object(self, *, key: str) -> None:
        def _delete() -> None:
            self._client.delete_object(Bucket=self.bucket, Key=key)

        await self._run(functools.partial(_delete))

    async def delete_object_quietly(self, *, key: str) -> None:
        """Best-effort delete used on rollback and soft delete."""
        try:
            await self.delete_object(key=key)
        except StorageUnavailableError:
            log.warning("storage.delete_best_effort_failed", key=key)
