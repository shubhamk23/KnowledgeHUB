"""
End-to-end journey tests — simulate a real user session through the full API.

Each test is a multi-step scenario that mirrors how a user would actually
interact with the system: authenticate → create content → verify it's public
→ edit it → delete it.
"""

import pytest


@pytest.mark.e2e
async def test_full_note_lifecycle(client, seeded_admin, tmp_content_dir):
    """
    Complete lifecycle:
    1. Admin logs in
    2. Creates a section
    3. Creates a note in that section
    4. Verifies note appears on the public section endpoint
    5. Reads the note via the public note endpoint
    6. Updates the note title
    7. Verifies update is reflected
    8. Deletes the note
    9. Verifies it is gone from both admin and public endpoints
    """
    # ── Step 1: Login ──────────────────────────────────────────────────────
    from app.config import settings
    login = await client.post(
        "/api/auth/token",
        json={"username": settings.admin_username, "password": settings.admin_password},
    )
    assert login.status_code == 200
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # ── Step 2: Create a section ───────────────────────────────────────────
    section_resp = await client.post(
        "/api/admin/sections",
        json={"slug": "ml", "title": "Machine Learning", "icon": "ML", "sort_order": 1},
        headers=headers,
    )
    assert section_resp.status_code == 201
    section_id = section_resp.json()["id"]

    # ── Step 3: Create a note ──────────────────────────────────────────────
    create_resp = await client.post(
        "/api/admin/notes",
        json={
            "title": "Gradient Descent",
            "section_slug": "ml",
            "content": "Gradient descent is an optimization algorithm.",
            "tags": ["optimization", "ml"],
            "visibility": "public",
        },
        headers=headers,
    )
    assert create_resp.status_code == 201
    note = create_resp.json()
    note_id = note["id"]
    assert note["slug"] == "gradient-descent"
    assert note["section_slug"] == "ml"

    # ── Step 4: Verify note appears in public section listing ──────────────
    section_notes_resp = await client.get("/api/sections/ml")
    assert section_notes_resp.status_code == 200
    public_notes = section_notes_resp.json()["notes"]
    assert any(n["slug"] == "gradient-descent" for n in public_notes)

    # ── Step 5: Read note via public endpoint ──────────────────────────────
    note_resp = await client.get("/api/notes/ml/gradient-descent")
    assert note_resp.status_code == 200
    assert "optimization algorithm" in note_resp.json()["content"]

    # ── Step 6: Update the note ────────────────────────────────────────────
    update_resp = await client.put(
        f"/api/admin/notes/{note_id}",
        json={"title": "Gradient Descent (Updated)"},
        headers=headers,
    )
    assert update_resp.status_code == 200
    assert update_resp.json()["title"] == "Gradient Descent (Updated)"

    # ── Step 7: Verify title is updated in admin listing ───────────────────
    admin_notes = await client.get("/api/admin/notes", headers=headers)
    titles = [n["title"] for n in admin_notes.json()]
    assert "Gradient Descent (Updated)" in titles

    # ── Step 8: Delete the note ────────────────────────────────────────────
    delete_resp = await client.delete(f"/api/admin/notes/{note_id}", headers=headers)
    assert delete_resp.status_code == 200
    assert delete_resp.json()["deleted"] is True

    # ── Step 9a: Verify gone from admin ────────────────────────────────────
    get_resp = await client.get(f"/api/admin/notes/{note_id}", headers=headers)
    assert get_resp.status_code == 404

    # ── Step 9b: Verify gone from public endpoint ──────────────────────────
    public_get = await client.get("/api/notes/ml/gradient-descent")
    assert public_get.status_code == 404


@pytest.mark.e2e
async def test_search_workflow(client, seeded_admin, tmp_content_dir):
    """
    Search journey:
    1. Admin logs in and creates a section + note
    2. Public user searches and finds the note
    3. Searches for unrelated term — no results
    """
    from app.config import settings
    login = await client.post(
        "/api/auth/token",
        json={"username": settings.admin_username, "password": settings.admin_password},
    )
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Create section
    await client.post(
        "/api/admin/sections",
        json={"slug": "recsys", "title": "RecSys", "icon": "RecSys", "sort_order": 3},
        headers=headers,
    )

    # Create note
    await client.post(
        "/api/admin/notes",
        json={
            "title": "Collaborative Filtering",
            "section_slug": "recsys",
            "content": "Collaborative filtering is a recommendation technique based on user behaviour.",
            "tags": ["cf", "recommendations"],
            "visibility": "public",
        },
        headers=headers,
    )

    # Search — should find it
    search_resp = await client.get("/api/search?q=collaborative")
    assert search_resp.status_code == 200
    results = search_resp.json()["results"]
    assert any(r["slug"] == "collaborative-filtering" for r in results)

    # Search — no match
    no_match = await client.get("/api/search?q=absolutelyrandomxyzterm")
    assert no_match.status_code == 200
    assert no_match.json()["total"] == 0


@pytest.mark.e2e
async def test_draft_note_not_visible_publicly(client, seeded_admin, tmp_content_dir):
    """
    Privacy journey:
    1. Admin creates a draft note
    2. Public endpoints do NOT return it
    3. Admin CAN see it via admin endpoint
    """
    from app.config import settings
    login = await client.post(
        "/api/auth/token",
        json={"username": settings.admin_username, "password": settings.admin_password},
    )
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Create section
    await client.post(
        "/api/admin/sections",
        json={"slug": "drafts", "title": "Drafts", "icon": "Blog", "sort_order": 99},
        headers=headers,
    )

    # Create draft note
    create = await client.post(
        "/api/admin/notes",
        json={
            "title": "Secret Research",
            "section_slug": "drafts",
            "content": "Top secret findings.",
            "tags": [],
            "visibility": "draft",
        },
        headers=headers,
    )
    assert create.status_code == 201
    note_id = create.json()["id"]

    # Public section listing should show 0 notes (only public counted)
    pub_section = await client.get("/api/sections/drafts")
    assert pub_section.status_code == 200
    assert pub_section.json()["section"]["note_count"] == 0

    # Public note endpoint returns 404
    pub_note = await client.get("/api/notes/drafts/secret-research")
    assert pub_note.status_code == 404

    # Search does not surface it
    search = await client.get("/api/search?q=secret")
    slugs = [r["slug"] for r in search.json()["results"]]
    assert "secret-research" not in slugs

    # Admin CAN see it
    admin_get = await client.get(f"/api/admin/notes/{note_id}", headers=headers)
    assert admin_get.status_code == 200
    assert admin_get.json()["visibility"] == "draft"


@pytest.mark.e2e
async def test_invalid_token_rejected_on_all_admin_endpoints(client, tmp_content_dir):
    """Tampered tokens must be rejected with 401 on every admin route."""
    bad_headers = {"Authorization": "Bearer totally.invalid.token"}

    endpoints = [
        ("GET", "/api/admin/notes"),
        ("GET", "/api/admin/sections"),
        ("POST", "/api/admin/notes"),
        ("POST", "/api/admin/sections"),
        ("POST", "/api/admin/reindex"),
    ]

    for method, path in endpoints:
        if method == "GET":
            resp = await client.get(path, headers=bad_headers)
        else:
            resp = await client.post(path, json={}, headers=bad_headers)
        assert resp.status_code == 401, f"Expected 401 for {method} {path}, got {resp.status_code}"


@pytest.mark.e2e
async def test_section_and_health_reflect_note_count_changes(client, seeded_admin, tmp_content_dir):
    """
    Health and section note_count must update after create/delete.
    """
    from app.config import settings
    login = await client.post(
        "/api/auth/token",
        json={"username": settings.admin_username, "password": settings.admin_password},
    )
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Setup section
    await client.post(
        "/api/admin/sections",
        json={"slug": "ml2", "title": "ML Part 2", "icon": "ML", "sort_order": 5},
        headers=headers,
    )

    # Health before
    health_before = await client.get("/api/health")
    count_before = health_before.json()["note_count"]

    # Create note
    create = await client.post(
        "/api/admin/notes",
        json={"title": "New Note", "section_slug": "ml2", "content": "Content.", "tags": [], "visibility": "public"},
        headers=headers,
    )
    note_id = create.json()["id"]

    # Health after create
    health_after_create = await client.get("/api/health")
    assert health_after_create.json()["note_count"] == count_before + 1

    # Delete note
    await client.delete(f"/api/admin/notes/{note_id}", headers=headers)

    # Health after delete
    health_after_delete = await client.get("/api/health")
    assert health_after_delete.json()["note_count"] == count_before
