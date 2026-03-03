"""
Integration tests for GET /api/health
"""

import pytest


@pytest.mark.integration
async def test_health_returns_ok(client):
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["version"] == "1.0.0"
    assert "note_count" in data


@pytest.mark.integration
async def test_health_note_count_zero_on_empty_db(client):
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["note_count"] == 0


@pytest.mark.integration
async def test_health_note_count_reflects_public_notes(client, seeded_note):
    """note_count must count only public notes."""
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["note_count"] == 1
