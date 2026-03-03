"""
Integration tests for admin section endpoints:
  GET  /api/admin/sections
  POST /api/admin/sections
  PUT  /api/admin/sections/{id}
"""

import pytest


@pytest.mark.integration
async def test_admin_list_sections_requires_auth(client):
    resp = await client.get("/api/admin/sections")
    assert resp.status_code == 401


@pytest.mark.integration
async def test_admin_list_sections_empty(client, auth_headers):
    resp = await client.get("/api/admin/sections", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.integration
async def test_admin_list_sections_includes_seeded(client, auth_headers, seeded_section):
    resp = await client.get("/api/admin/sections", headers=auth_headers)
    assert resp.status_code == 200
    slugs = [s["slug"] for s in resp.json()]
    assert "nlp" in slugs


@pytest.mark.integration
async def test_admin_create_section_success(client, auth_headers, tmp_content_dir):
    payload = {"slug": "vision", "title": "Computer Vision", "description": "CV notes", "icon": "Vision", "sort_order": 2}
    resp = await client.post("/api/admin/sections", json=payload, headers=auth_headers)
    assert resp.status_code == 201
    data = resp.json()
    assert data["slug"] == "vision"
    assert data["title"] == "Computer Vision"


@pytest.mark.integration
async def test_admin_create_section_creates_directory(client, auth_headers, tmp_content_dir):
    await client.post(
        "/api/admin/sections",
        json={"slug": "cv", "title": "CV", "sort_order": 0},
        headers=auth_headers,
    )
    assert (tmp_content_dir / "cv").is_dir()
    assert (tmp_content_dir / "cv" / "_section.json").exists()


@pytest.mark.integration
async def test_admin_create_section_duplicate_409(client, auth_headers, seeded_section, tmp_content_dir):
    payload = {"slug": "nlp", "title": "NLP Again", "sort_order": 0}
    resp = await client.post("/api/admin/sections", json=payload, headers=auth_headers)
    assert resp.status_code == 409


@pytest.mark.integration
async def test_admin_create_section_missing_required_fields(client, auth_headers):
    resp = await client.post("/api/admin/sections", json={"slug": "only-slug"}, headers=auth_headers)
    assert resp.status_code == 422


@pytest.mark.integration
async def test_admin_update_section_title(client, auth_headers, seeded_section, tmp_content_dir):
    resp = await client.put(
        f"/api/admin/sections/{seeded_section.id}",
        json={"title": "Natural Language Processing"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["title"] == "Natural Language Processing"


@pytest.mark.integration
async def test_admin_update_section_not_found(client, auth_headers):
    resp = await client.put("/api/admin/sections/99999", json={"title": "X"}, headers=auth_headers)
    assert resp.status_code == 404
