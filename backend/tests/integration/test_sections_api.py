"""
Integration tests for:
  GET /api/sections
  GET /api/sections/{section_slug}
"""

import pytest


@pytest.mark.integration
async def test_list_sections_empty(client):
    resp = await client.get("/api/sections")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.integration
async def test_list_sections_returns_seeded_section(client, seeded_section):
    resp = await client.get("/api/sections")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["slug"] == "nlp"
    assert data[0]["title"] == "NLP"


@pytest.mark.integration
async def test_list_sections_note_count_zero_no_notes(client, seeded_section):
    resp = await client.get("/api/sections")
    assert resp.status_code == 200
    assert resp.json()[0]["note_count"] == 0


@pytest.mark.integration
async def test_list_sections_note_count_with_note(client, seeded_note, seeded_section):
    resp = await client.get("/api/sections")
    assert resp.status_code == 200
    section_data = next(s for s in resp.json() if s["slug"] == "nlp")
    assert section_data["note_count"] == 1


@pytest.mark.integration
async def test_get_section_by_slug(client, seeded_section):
    resp = await client.get("/api/sections/nlp")
    assert resp.status_code == 200
    data = resp.json()
    assert data["section"]["slug"] == "nlp"
    assert data["section"]["title"] == "NLP"
    assert isinstance(data["notes"], list)


@pytest.mark.integration
async def test_get_section_includes_public_note(client, seeded_note, seeded_section):
    resp = await client.get("/api/sections/nlp")
    assert resp.status_code == 200
    notes = resp.json()["notes"]
    assert len(notes) == 1
    assert notes[0]["slug"] == "transformers"


@pytest.mark.integration
async def test_get_section_not_found(client):
    resp = await client.get("/api/sections/nonexistent")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


@pytest.mark.integration
async def test_section_response_has_required_fields(client, seeded_section):
    resp = await client.get("/api/sections")
    s = resp.json()[0]
    for field in ("id", "slug", "title", "note_count", "sort_order"):
        assert field in s, f"Missing field: {field}"
