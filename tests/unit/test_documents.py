from __future__ import annotations

import base64
import json
import uuid
from datetime import UTC, datetime

import pytest

from app.deps import CurrentUser
from app.errors import (
    InvalidCursorError,
    InvalidRoleError,
    RoleNotHeldError,
    UnsupportedFileTypeError,
)
from app.services.documents import (
    _check_magic_bytes,
    _validate_access_roles,
    decode_cursor,
    encode_cursor,
    sanitize_file_name,
    source_type_for,
    source_type_mime,
)

PDF_HEAD = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n"
PNG_HEAD = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def _user(roles: list[str]) -> CurrentUser:
    return CurrentUser(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        email="user@example.com",
        roles=tuple(roles),
        is_active=True,
    )


def test_sanitize_file_name_strips_paths_and_characters() -> None:
    assert sanitize_file_name("../../etc/passwd") == "passwd"
    assert sanitize_file_name("my report (<2026>).md") == "my_report_2026_.md"
    assert sanitize_file_name("") == "document"
    assert len(sanitize_file_name("x" * 500 + ".pdf")) <= 120


def test_source_type_mapping() -> None:
    assert source_type_for("a/b.PDF") == "pdf"
    assert source_type_for("notes.md") == "markdown"
    assert source_type_for("page.htm") == "html"
    assert source_type_mime("pdf") == "application/pdf"
    with pytest.raises(UnsupportedFileTypeError):
        source_type_for("archive.zip")


def test_magic_bytes_accept_and_reject() -> None:
    _check_magic_bytes("pdf", PDF_HEAD)  # valid PDF stays
    with pytest.raises(UnsupportedFileTypeError):
        _check_magic_bytes("pdf", b"<html><body>hi</body></html>")
    with pytest.raises(UnsupportedFileTypeError):
        _check_magic_bytes("markdown", PNG_HEAD)
    _check_magic_bytes("markdown", b"# Heading\n\nSome text.\n")
    _check_magic_bytes("html", b"<html><body>hi</body></html>")


def test_cursor_roundtrip_and_rejection() -> None:
    moment = datetime(2026, 11, 12, 10, 30, 0, tzinfo=UTC)
    document_id = __import__("uuid").uuid4()
    cursor = encode_cursor(created_at=moment, document_id=document_id)
    restored_created_at, restored_id = decode_cursor(cursor)
    assert restored_created_at == moment
    assert restored_id == document_id

    with pytest.raises(InvalidCursorError):
        decode_cursor(base64.urlsafe_b64encode(b"not json").decode())
    with pytest.raises(InvalidCursorError):
        decode_cursor(
            base64.urlsafe_b64encode(
                json.dumps({"created_at": "2026-11-12T10:30:00", "id": "x"}).encode()
            ).decode()
        )


def test_access_role_validation() -> None:
    admin = _user(["tenant_admin"])
    assert _validate_access_roles(roles=["HR", "hr"], user=admin, is_admin=True) == ["hr"]
    assert _validate_access_roles(roles=[], user=admin, is_admin=True) == []

    with pytest.raises(InvalidRoleError):
        _validate_access_roles(roles=["tenant_admin"], user=admin, is_admin=True)
    with pytest.raises(InvalidRoleError):
        _validate_access_roles(roles=["bad role!"], user=admin, is_admin=True)

    member = _user(["engineering"])
    assert _validate_access_roles(roles=["engineering"], user=member, is_admin=False) == [
        "engineering"
    ]
    with pytest.raises(RoleNotHeldError):
        _validate_access_roles(roles=["hr"], user=member, is_admin=False)
