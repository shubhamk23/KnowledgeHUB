"""
Integration tests for POST /api/auth/token
"""

import pytest
from app.config import settings


@pytest.mark.integration
async def test_login_valid_credentials(client, seeded_admin):
    resp = await client.post(
        "/api/auth/token",
        json={"username": settings.admin_username, "password": settings.admin_password},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["expires_in"] > 0


@pytest.mark.integration
async def test_login_wrong_password(client, seeded_admin):
    resp = await client.post(
        "/api/auth/token",
        json={"username": settings.admin_username, "password": "wrongpassword"},
    )
    assert resp.status_code == 401


@pytest.mark.integration
async def test_login_unknown_user(client, seeded_admin):
    resp = await client.post(
        "/api/auth/token",
        json={"username": "ghost", "password": "whatever"},
    )
    assert resp.status_code == 401


@pytest.mark.integration
async def test_login_missing_body_fields(client):
    resp = await client.post("/api/auth/token", json={})
    assert resp.status_code == 422  # Pydantic validation error


@pytest.mark.integration
async def test_token_is_valid_jwt(client, seeded_admin):
    from jose import jwt as _jwt
    resp = await client.post(
        "/api/auth/token",
        json={"username": settings.admin_username, "password": settings.admin_password},
    )
    token = resp.json()["access_token"]
    payload = _jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    assert payload["sub"] == settings.admin_username
