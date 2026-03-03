"""
Shared pytest fixtures for all test layers.

Strategy
--------
* Each test gets a FRESH in-memory SQLite database — fast and fully isolated.
* The FastAPI app is created with all lifespan startup bypassed so tests
  control exactly what state exists (no file-system scans, no watchdog).
* A helper `auth_headers` fixture logs in and returns an Authorization header
  that every admin endpoint test can reuse.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# ── Override settings BEFORE any app imports ──────────────────────────────────
os.environ.setdefault("ADMIN_USERNAME", "testadmin")
os.environ.setdefault("ADMIN_PASSWORD", "testpassword")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-tests-only")

# ── Now it is safe to import app modules ──────────────────────────────────────
from app.auth import hash_password
from app.config import settings
from app.database import Base
from app.models import AdminUser, Note, Section


# ──────────────────────────────────────────────────────────────────────────────
# In-memory async engine — one per test function
# ──────────────────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture()
async def db_engine():
    """Create a brand-new in-memory SQLite engine for each test."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Create FTS5 virtual table
        await conn.execute(
            text(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
                    title, content, tags, summary,
                    tokenize='porter unicode61'
                )
                """
            )
        )
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture()
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Provide an open async DB session backed by the in-memory engine."""
    session_factory = async_sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        yield session


# ──────────────────────────────────────────────────────────────────────────────
# Temporary content directory — each test writes to a fresh temp folder
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_content_dir(tmp_path: Path, monkeypatch) -> Path:
    """
    Point settings.content_dir to a fresh temporary directory.
    Also patch the module-level `settings` object used inside routers.
    """
    content = tmp_path / "content"
    content.mkdir()
    monkeypatch.setattr(settings, "content_dir", str(content))
    return content


# ──────────────────────────────────────────────────────────────────────────────
# Pre-seeded test data helpers
# ──────────────────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture()
async def seeded_section(db_session: AsyncSession, tmp_content_dir: Path) -> Section:
    """Insert a Section row and create its directory on disk."""
    section = Section(slug="nlp", title="NLP", description="Natural Language Processing", icon="NLP", sort_order=1)
    db_session.add(section)
    await db_session.commit()
    await db_session.refresh(section)
    (tmp_content_dir / "nlp").mkdir(exist_ok=True)
    return section


@pytest_asyncio.fixture()
async def seeded_note(db_session: AsyncSession, seeded_section: Section, tmp_content_dir: Path) -> Note:
    """Insert a Note row and write a minimal .md file to disk."""
    md_path = tmp_content_dir / "nlp" / "transformers.md"
    md_path.write_text(
        '---\ntitle: "Transformers"\nslug: transformers\nvisibility: public\ntags: []\n---\n\nTransformers are great.',
        encoding="utf-8",
    )
    note = Note(
        slug="transformers",
        section_id=seeded_section.id,
        title="Transformers",
        summary="Transformers are great.",
        tags="[]",
        visibility="public",
        file_path=str(md_path),
        word_count=3,
        read_time=1,
    )
    db_session.add(note)
    await db_session.flush()

    await db_session.execute(
        text(
            "INSERT INTO notes_fts(rowid, title, content, tags, summary) "
            "VALUES (:rowid, :title, :content, :tags, :summary)"
        ),
        {"rowid": note.id, "title": note.title, "content": "Transformers are great.", "tags": "[]", "summary": note.summary},
    )
    await db_session.commit()
    await db_session.refresh(note)
    return note


@pytest_asyncio.fixture()
async def seeded_admin(db_session: AsyncSession) -> AdminUser:
    """Seed an admin user with a known username/password."""
    user = AdminUser(
        username=settings.admin_username,
        password_hash=hash_password(settings.admin_password),
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI test client — overrides DB dependency
# ──────────────────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture()
async def client(db_engine, tmp_content_dir, monkeypatch) -> AsyncGenerator[AsyncClient, None]:
    """
    Return an httpx.AsyncClient wired to the FastAPI app.

    The lifespan is bypassed (lifespan=None) so tests control DB state directly.
    The `get_db` dependency is overridden to use the in-memory engine.
    """
    from app.main import app
    from app.database import get_db

    session_factory = async_sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False
    )

    async def _override_get_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest_asyncio.fixture()
async def auth_headers(client: AsyncClient, seeded_admin) -> dict:
    """
    Log in as the test admin and return headers suitable for admin endpoints.
    Depends on `seeded_admin` to ensure the user exists before logging in.
    """
    resp = await client.post(
        "/api/auth/token",
        json={"username": settings.admin_username, "password": settings.admin_password},
    )
    assert resp.status_code == 200, f"Login failed: {resp.text}"
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
