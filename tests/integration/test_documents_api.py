"""Document API integration tests: upload, idempotency, ACLs, soft delete."""

from __future__ import annotations

import uuid

import boto3
from httpx import AsyncClient

from tests.integration.helpers import bearer, create_user, login, register_tenant

MARKDOWN_CONTENT = b"# Employee Handbook\n\nParental leave is 16 weeks.\n"
OTHER_CONTENT = b"# Architecture\n\nThe ingestion pipeline is asynchronous.\n"


async def _upload(
    client: AsyncClient,
    token: str,
    *,
    file_name: str = "handbook.md",
    content: bytes = MARKDOWN_CONTENT,
    title: str | None = None,
    roles: list[str] | None = None,
):
    data: dict[str, str | list[str]] = {}
    if title is not None:
        data["title"] = title
    if roles is not None:
        data["access_roles"] = roles
    return await client.post(
        "/api/v1/documents",
        files={"file": (file_name, content, "text/markdown")},
        data=data,
        headers=bearer(token),
    )


async def _admin_setup(client: AsyncClient, slug: str) -> str:
    await register_tenant(client, slug=slug)
    login_payload = await login(client, slug=slug)
    return str(login_payload["access_token"])


async def test_upload_then_duplicate_is_idempotent(app, client: AsyncClient) -> None:
    token = await _admin_setup(client, "dup")
    first = await _upload(client, token, title="Handbook")
    assert first.status_code == 202, first.text
    body = first.json()
    assert body["status"] == "pending"
    assert body["duplicate"] is False
    assert body["file_name"] == "handbook.md"
    assert body["source_type"] == "markdown"

    second = await _upload(client, token)
    assert second.status_code == 200, second.text
    duplicate = second.json()
    assert duplicate["duplicate"] is True
    assert duplicate["id"] == body["id"]

    # The object exists in the (moto) bucket with the full content.
    settings = app.state.settings
    s3 = boto3.client(
        "s3",
        region_name=settings.s3_region,
        aws_access_key_id=settings.s3_access_key_id,
        aws_secret_access_key=settings.s3_secret_access_key,
    )
    objects = s3.list_objects_v2(Bucket=settings.s3_bucket)
    keys = [item["Key"] for item in objects.get("Contents", [])]
    assert len(keys) == 1, keys
    stored = s3.get_object(Bucket=settings.s3_bucket, Key=keys[0])["Body"].read()
    assert stored == MARKDOWN_CONTENT

    listing = await client.get("/api/v1/documents", headers=bearer(token))
    assert listing.status_code == 200
    assert len(listing.json()["items"]) == 1


async def test_oversized_upload_rejected(app, client: AsyncClient) -> None:
    token = await _admin_setup(client, "size")
    app.state.settings.max_upload_bytes = 10
    response = await _upload(client, token, content=b"x" * 100)
    assert response.status_code == 413, response.text
    assert response.json()["type"].endswith("upload-too-large")

    # Empty upload is rejected as unsupported.
    empty = await _upload(client, token, content=b"")
    assert empty.status_code == 415


async def test_mime_mismatch_rejected(client: AsyncClient) -> None:
    token = await _admin_setup(client, "mime")
    response = await _upload(client, token, file_name="fake.pdf", content=b"<html>not a pdf</html>")
    assert response.status_code == 415, response.text
    listing = await client.get("/api/v1/documents", headers=bearer(token))
    assert listing.json()["items"] == []


async def test_acl_visibility_matrix(app, client: AsyncClient) -> None:
    token = await _admin_setup(client, "acl")
    await create_user(client, admin_token=token, email="hr@acl.example.com", roles=["hr"])
    await create_user(client, admin_token=token, email="eng@acl.example.com", roles=["engineering"])
    hr_token = str((await login(client, slug="acl", email="hr@acl.example.com"))["access_token"])
    eng_token = str((await login(client, slug="acl", email="eng@acl.example.com"))["access_token"])

    restricted = await _upload(client, token, roles=["hr"], title="Restricted")
    assert restricted.status_code == 202, restricted.text
    document_id = restricted.json()["id"]

    tenant_wide = await _upload(client, token, content=OTHER_CONTENT, title="Public")
    assert tenant_wide.status_code == 202

    # Admin sees both.
    admin_list = (await client.get("/api/v1/documents", headers=bearer(token))).json()
    assert {item["title"] for item in admin_list["items"]} == {"Restricted", "Public"}

    # HR member sees both (empty ACL is tenant-wide).
    hr_list = (await client.get("/api/v1/documents", headers=bearer(hr_token))).json()
    assert {item["title"] for item in hr_list["items"]} == {"Restricted", "Public"}
    assert (
        await client.get(f"/api/v1/documents/{document_id}", headers=bearer(hr_token))
    ).status_code == 200

    # Engineering member sees only the tenant-wide document.
    eng_list = (await client.get("/api/v1/documents", headers=bearer(eng_token))).json()
    assert {item["title"] for item in eng_list["items"]} == {"Public"}
    assert (
        await client.get(f"/api/v1/documents/{document_id}", headers=bearer(eng_token))
    ).status_code == 404


async def test_member_role_assignment_rules(client: AsyncClient) -> None:
    token = await _admin_setup(client, "roles")
    await create_user(
        client,
        admin_token=token,
        email="eng@roles.example.com",
        roles=["engineering"],
    )
    eng_token = str(
        (await login(client, slug="roles", email="eng@roles.example.com"))["access_token"]
    )

    held = await _upload(client, eng_token, roles=["engineering"])
    assert held.status_code == 202, held.text

    not_held = await _upload(client, eng_token, roles=["hr"])
    assert not_held.status_code == 403, not_held.text

    system_role = await _upload(client, token, roles=["tenant_admin"])
    assert system_role.status_code == 422

    admin_any = await _upload(
        client, token, content=b"# Finance\n\nBudget cycle.\n", roles=["hr", "finance"]
    )
    assert admin_any.status_code == 202


async def test_patch_title_and_roles(client: AsyncClient) -> None:
    token = await _admin_setup(client, "patch")
    uploaded = await _upload(client, token, roles=["hr"])
    document_id = uploaded.json()["id"]

    patched = await client.patch(
        f"/api/v1/documents/{document_id}",
        json={"title": "New title", "access_roles": ["management"]},
        headers=bearer(token),
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["title"] == "New title"
    assert patched.json()["access_roles"] == ["management"]

    blank = await client.patch(
        f"/api/v1/documents/{document_id}",
        json={"title": "   "},
        headers=bearer(token),
    )
    assert blank.status_code == 422


async def test_delete_is_soft_and_reupload_allowed(client: AsyncClient) -> None:
    token = await _admin_setup(client, "delete")
    uploaded = await _upload(client, token)
    document_id = uploaded.json()["id"]

    deleted = await client.delete(f"/api/v1/documents/{document_id}", headers=bearer(token))
    assert deleted.status_code == 204

    assert (
        await client.get(f"/api/v1/documents/{document_id}", headers=bearer(token))
    ).status_code == 404
    listing = (await client.get("/api/v1/documents", headers=bearer(token))).json()
    assert listing["items"] == []

    reuploaded = await _upload(client, token)
    assert reuploaded.status_code == 202
    assert reuploaded.json()["duplicate"] is False
    assert reuploaded.json()["id"] != document_id


async def test_cross_tenant_document_returns_404(client: AsyncClient) -> None:
    acme_token = await _admin_setup(client, "acmedocs")
    globex_token = await _admin_setup(client, "globexdocs")

    uploaded = await _upload(client, acme_token)
    document_id = uploaded.json()["id"]

    response = await client.get(f"/api/v1/documents/{document_id}", headers=bearer(globex_token))
    assert response.status_code == 404

    globex_list = (await client.get("/api/v1/documents", headers=bearer(globex_token))).json()
    assert globex_list["items"] == []


async def test_reprocess_permissions(client: AsyncClient) -> None:
    token = await _admin_setup(client, "reproc")
    await create_user(
        client,
        admin_token=token,
        email="eng@reproc.example.com",
        roles=["engineering"],
    )
    eng_token = str(
        (await login(client, slug="reproc", email="eng@reproc.example.com"))["access_token"]
    )

    uploaded = await _upload(client, token, content=OTHER_CONTENT, roles=["engineering"])
    document_id = uploaded.json()["id"]

    member_attempt = await client.post(
        f"/api/v1/documents/{document_id}/reprocess", headers=bearer(eng_token)
    )
    assert member_attempt.status_code == 403, member_attempt.text

    admin_attempt = await client.post(
        f"/api/v1/documents/{document_id}/reprocess", headers=bearer(token)
    )
    assert admin_attempt.status_code == 202, admin_attempt.text


async def test_pagination_cursor(client: AsyncClient) -> None:
    token = await _admin_setup(client, "paging")
    for index in range(3):
        response = await _upload(client, token, content=MARKDOWN_CONTENT + f"page {index}".encode())
        assert response.status_code == 202

    page_one = (await client.get("/api/v1/documents?limit=2", headers=bearer(token))).json()
    assert len(page_one["items"]) == 2
    assert page_one["next_cursor"] is not None

    cursor = page_one["next_cursor"]
    page_two = (
        await client.get(f"/api/v1/documents?limit=2&cursor={cursor}", headers=bearer(token))
    ).json()
    assert len(page_two["items"]) == 1
    assert page_two["next_cursor"] is None

    bad = await client.get("/api/v1/documents?cursor=bogus", headers=bearer(token))
    assert bad.status_code == 400


async def test_forge_is_ignored_and_deleted_hidden(app, client: AsyncClient) -> None:
    token = await _admin_setup(client, "forge")
    other_token = await _admin_setup(client, "otherforge")
    uploaded = await _upload(client, token)
    document_id = uploaded.json()["id"]

    # A forged tenant id in the request has no effect: the tenant comes from the JWT.
    response = await client.get(
        f"/api/v1/documents/{document_id}",
        headers={**bearer(other_token), "X-Tenant-ID": str(uuid.uuid4())},
    )
    assert response.status_code == 404

    # Soft-deleted rows are never returned even by id.
    await client.delete(f"/api/v1/documents/{document_id}", headers=bearer(token))
    assert (
        await client.get(f"/api/v1/documents/{document_id}", headers=bearer(token))
    ).status_code == 404
