"""
Integration tests for GET /api/search?q=...
"""

import pytest


@pytest.mark.integration
async def test_search_empty_db_returns_empty(client):
    resp = await client.get("/api/search?q=transformer")
    assert resp.status_code == 200
    data = resp.json()
    assert data["results"] == []
    assert data["total"] == 0
    assert data["query"] == "transformer"


@pytest.mark.integration
async def test_search_finds_existing_note(client, seeded_note, seeded_section):
    resp = await client.get("/api/search?q=Transformers")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 1
    assert any(r["slug"] == "transformers" for r in data["results"])


@pytest.mark.integration
async def test_search_result_has_excerpt(client, seeded_note, seeded_section):
    resp = await client.get("/api/search?q=Transformers")
    assert resp.status_code == 200
    result = resp.json()["results"][0]
    assert "excerpt" in result
    assert isinstance(result["excerpt"], str)


@pytest.mark.integration
async def test_search_missing_q_param(client):
    resp = await client.get("/api/search")
    assert resp.status_code == 422  # q is required


@pytest.mark.integration
async def test_search_empty_q_string(client):
    # min_length=1 on q means empty string fails validation
    resp = await client.get("/api/search?q=")
    assert resp.status_code == 422


@pytest.mark.integration
async def test_search_no_match_returns_empty(client, seeded_note, seeded_section):
    resp = await client.get("/api/search?q=absolutelynothing12345xyz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["results"] == []


@pytest.mark.integration
async def test_search_response_shape(client, seeded_note, seeded_section):
    resp = await client.get("/api/search?q=Transformers")
    data = resp.json()
    assert "results" in data
    assert "total" in data
    assert "query" in data


@pytest.mark.integration
async def test_search_limit_param(client, seeded_note, seeded_section):
    resp = await client.get("/api/search?q=Transformers&limit=1")
    assert resp.status_code == 200
    assert len(resp.json()["results"]) <= 1


@pytest.mark.integration
async def test_search_does_not_return_draft_notes(client, db_session, seeded_section, tmp_content_dir):
    """Draft notes must be excluded from search results."""
    from sqlalchemy import text
    from app.models import Note

    md_path = tmp_content_dir / "nlp" / "draft-note.md"
    md_path.write_text("---\ntitle: Draft\nslug: draft-note\nvisibility: draft\n---\n\nDraft content here.", encoding="utf-8")
    note = Note(
        slug="draft-note",
        section_id=seeded_section.id,
        title="Draft",
        summary="Draft content here.",
        tags="[]",
        visibility="draft",
        file_path=str(md_path),
        word_count=3,
        read_time=1,
    )
    db_session.add(note)
    await db_session.flush()
    await db_session.execute(
        text("INSERT INTO notes_fts(rowid, title, content, tags, summary) VALUES (:rowid, :title, :content, :tags, :summary)"),
        {"rowid": note.id, "title": "Draft", "content": "Draft content here.", "tags": "[]", "summary": "Draft content here."},
    )
    await db_session.commit()

    resp = await client.get("/api/search?q=Draft")
    assert resp.status_code == 200
    # Draft note should NOT appear
    slugs = [r["slug"] for r in resp.json()["results"]]
    assert "draft-note" not in slugs
