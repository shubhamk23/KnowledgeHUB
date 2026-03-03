"""
Integration tests for GET /api/notes/{section_slug}/{note_slug}
"""

import pytest


@pytest.mark.integration
async def test_get_note_success(client, seeded_note, seeded_section):
    resp = await client.get("/api/notes/nlp/transformers")
    assert resp.status_code == 200
    data = resp.json()
    assert data["slug"] == "transformers"
    assert data["section_slug"] == "nlp"
    assert data["title"] == "Transformers"


@pytest.mark.integration
async def test_get_note_content_is_stripped_of_frontmatter(client, seeded_note, seeded_section):
    resp = await client.get("/api/notes/nlp/transformers")
    assert resp.status_code == 200
    content = resp.json()["content"]
    assert "---" not in content
    assert "Transformers are great." in content


@pytest.mark.integration
async def test_get_note_unknown_section(client, seeded_note):
    resp = await client.get("/api/notes/unknown-section/transformers")
    assert resp.status_code == 404


@pytest.mark.integration
async def test_get_note_unknown_slug(client, seeded_section):
    resp = await client.get("/api/notes/nlp/does-not-exist")
    assert resp.status_code == 404


@pytest.mark.integration
async def test_get_draft_note_returns_404_on_public_endpoint(client, db_session, seeded_section, tmp_content_dir):
    """Draft notes must NOT be accessible via the public endpoint."""
    from sqlalchemy import text
    from app.models import Note

    md_path = tmp_content_dir / "nlp" / "secret.md"
    md_path.write_text("---\ntitle: Secret\nslug: secret\nvisibility: draft\n---\n\nTop secret.", encoding="utf-8")
    note = Note(
        slug="secret",
        section_id=seeded_section.id,
        title="Secret",
        summary="Top secret.",
        tags="[]",
        visibility="draft",
        file_path=str(md_path),
        word_count=2,
        read_time=1,
    )
    db_session.add(note)
    await db_session.flush()
    await db_session.execute(
        text("INSERT INTO notes_fts(rowid, title, content, tags, summary) VALUES (:rowid, :title, :content, :tags, :summary)"),
        {"rowid": note.id, "title": "Secret", "content": "Top secret.", "tags": "[]", "summary": "Top secret."},
    )
    await db_session.commit()

    resp = await client.get("/api/notes/nlp/secret")
    assert resp.status_code == 404


@pytest.mark.integration
async def test_note_response_shape(client, seeded_note, seeded_section):
    resp = await client.get("/api/notes/nlp/transformers")
    data = resp.json()
    for field in ("id", "slug", "section_slug", "title", "content", "tags", "read_time", "word_count", "visibility"):
        assert field in data, f"Missing field: {field}"
