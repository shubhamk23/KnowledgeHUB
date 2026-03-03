"""
Integration tests for admin note endpoints:
  GET    /api/admin/notes
  GET    /api/admin/notes/{id}
  POST   /api/admin/notes
  PUT    /api/admin/notes/{id}
  DELETE /api/admin/notes/{id}
"""

import pytest


# ── Auth guard tests ──────────────────────────────────────────────────────────

@pytest.mark.integration
async def test_admin_notes_requires_auth(client):
    resp = await client.get("/api/admin/notes")
    assert resp.status_code == 401


@pytest.mark.integration
async def test_admin_create_note_requires_auth(client):
    resp = await client.post("/api/admin/notes", json={"title": "T", "section_slug": "nlp", "content": "Body"})
    assert resp.status_code == 401


# ── List notes ────────────────────────────────────────────────────────────────

@pytest.mark.integration
async def test_admin_list_notes_empty(client, auth_headers):
    resp = await client.get("/api/admin/notes", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.integration
async def test_admin_list_notes_includes_seeded_note(client, auth_headers, seeded_note, seeded_section):
    resp = await client.get("/api/admin/notes", headers=auth_headers)
    assert resp.status_code == 200
    slugs = [n["slug"] for n in resp.json()]
    assert "transformers" in slugs


@pytest.mark.integration
async def test_admin_list_includes_drafts(client, auth_headers, db_session, seeded_section, tmp_content_dir):
    """Admin list should include draft notes unlike the public endpoint."""
    from sqlalchemy import text
    from app.models import Note

    md_path = tmp_content_dir / "nlp" / "my-draft.md"
    md_path.write_text("---\ntitle: Draft\nslug: my-draft\nvisibility: draft\n---\n\nSecret.", encoding="utf-8")
    note = Note(
        slug="my-draft",
        section_id=seeded_section.id,
        title="Draft",
        summary="Secret.",
        tags="[]",
        visibility="draft",
        file_path=str(md_path),
        word_count=1,
        read_time=1,
    )
    db_session.add(note)
    await db_session.flush()
    await db_session.execute(
        text("INSERT INTO notes_fts(rowid, title, content, tags, summary) VALUES (:rowid, :title, :content, :tags, :summary)"),
        {"rowid": note.id, "title": "Draft", "content": "Secret.", "tags": "[]", "summary": "Secret."},
    )
    await db_session.commit()

    resp = await client.get("/api/admin/notes", headers=auth_headers)
    assert resp.status_code == 200
    slugs = [n["slug"] for n in resp.json()]
    assert "my-draft" in slugs


# ── Get single note ───────────────────────────────────────────────────────────

@pytest.mark.integration
async def test_admin_get_note_by_id(client, auth_headers, seeded_note, seeded_section):
    resp = await client.get(f"/api/admin/notes/{seeded_note.id}", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["slug"] == "transformers"
    assert data["section_slug"] == "nlp"


@pytest.mark.integration
async def test_admin_get_note_not_found(client, auth_headers):
    resp = await client.get("/api/admin/notes/99999", headers=auth_headers)
    assert resp.status_code == 404


# ── Create note ───────────────────────────────────────────────────────────────

@pytest.mark.integration
async def test_admin_create_note_success(client, auth_headers, seeded_section, tmp_content_dir):
    payload = {
        "title": "Attention Is All You Need",
        "section_slug": "nlp",
        "content": "Transformers use self-attention mechanisms.",
        "tags": ["attention", "transformers"],
        "visibility": "public",
    }
    resp = await client.post("/api/admin/notes", json=payload, headers=auth_headers)
    assert resp.status_code == 201
    data = resp.json()
    assert data["slug"] == "attention-is-all-you-need"
    assert data["title"] == "Attention Is All You Need"
    assert data["section_slug"] == "nlp"


@pytest.mark.integration
async def test_admin_create_note_writes_file_to_disk(client, auth_headers, seeded_section, tmp_content_dir):
    payload = {"title": "File Test", "section_slug": "nlp", "content": "File body.", "tags": [], "visibility": "public"}
    resp = await client.post("/api/admin/notes", json=payload, headers=auth_headers)
    assert resp.status_code == 201
    file_path = tmp_content_dir / "nlp" / "file-test.md"
    assert file_path.exists()
    assert "File body." in file_path.read_text()


@pytest.mark.integration
async def test_admin_create_note_custom_slug(client, auth_headers, seeded_section, tmp_content_dir):
    payload = {"title": "Custom Slug Note", "section_slug": "nlp", "content": "Body.", "slug": "my-custom-slug", "tags": [], "visibility": "public"}
    resp = await client.post("/api/admin/notes", json=payload, headers=auth_headers)
    assert resp.status_code == 201
    assert resp.json()["slug"] == "my-custom-slug"


@pytest.mark.integration
async def test_admin_create_note_duplicate_slug_409(client, auth_headers, seeded_note, seeded_section, tmp_content_dir):
    payload = {"title": "Transformers", "section_slug": "nlp", "content": "Duplicate.", "slug": "transformers", "tags": [], "visibility": "public"}
    resp = await client.post("/api/admin/notes", json=payload, headers=auth_headers)
    assert resp.status_code == 409


@pytest.mark.integration
async def test_admin_create_note_unknown_section(client, auth_headers):
    payload = {"title": "Test", "section_slug": "nonexistent", "content": "Body.", "tags": [], "visibility": "public"}
    resp = await client.post("/api/admin/notes", json=payload, headers=auth_headers)
    assert resp.status_code == 404


@pytest.mark.integration
async def test_admin_create_note_missing_required_fields(client, auth_headers):
    resp = await client.post("/api/admin/notes", json={"title": "Only title"}, headers=auth_headers)
    assert resp.status_code == 422


# ── Update note ───────────────────────────────────────────────────────────────

@pytest.mark.integration
async def test_admin_update_note_title(client, auth_headers, seeded_note, seeded_section, tmp_content_dir):
    resp = await client.put(
        f"/api/admin/notes/{seeded_note.id}",
        json={"title": "Updated Transformers"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["title"] == "Updated Transformers"


@pytest.mark.integration
async def test_admin_update_note_visibility(client, auth_headers, seeded_note, seeded_section, tmp_content_dir):
    resp = await client.put(
        f"/api/admin/notes/{seeded_note.id}",
        json={"visibility": "draft"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["visibility"] == "draft"


@pytest.mark.integration
async def test_admin_update_note_not_found(client, auth_headers):
    resp = await client.put("/api/admin/notes/99999", json={"title": "X"}, headers=auth_headers)
    assert resp.status_code == 404


# ── Delete note ───────────────────────────────────────────────────────────────

@pytest.mark.integration
async def test_admin_delete_note_success(client, auth_headers, seeded_note, seeded_section, tmp_content_dir):
    note_id = seeded_note.id
    resp = await client.delete(f"/api/admin/notes/{note_id}", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True

    # Verify it's gone
    get_resp = await client.get(f"/api/admin/notes/{note_id}", headers=auth_headers)
    assert get_resp.status_code == 404


@pytest.mark.integration
async def test_admin_delete_removes_file_from_disk(client, auth_headers, seeded_note, seeded_section, tmp_content_dir):
    from pathlib import Path
    file_path = Path(seeded_note.file_path)
    assert file_path.exists()

    await client.delete(f"/api/admin/notes/{seeded_note.id}", headers=auth_headers)
    assert not file_path.exists()


@pytest.mark.integration
async def test_admin_delete_note_not_found(client, auth_headers):
    resp = await client.delete("/api/admin/notes/99999", headers=auth_headers)
    assert resp.status_code == 404
