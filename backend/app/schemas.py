from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List


# ── Section Schemas ──────────────────────────────────────────

class SectionOut(BaseModel):
    id: int
    slug: str
    title: str
    description: Optional[str] = None
    icon: Optional[str] = None
    sort_order: int
    note_count: int = 0

    model_config = {"from_attributes": True}


class SectionAdminOut(SectionOut):
    created_at: datetime
    updated_at: Optional[datetime] = None


class SectionCreateRequest(BaseModel):
    slug: str
    title: str
    description: Optional[str] = None
    icon: Optional[str] = None
    sort_order: int = 0


class SectionUpdateRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None
    sort_order: Optional[int] = None


# ── Note Schemas ─────────────────────────────────────────────

class NoteCardOut(BaseModel):
    id: int
    slug: str
    section_slug: str
    title: str
    summary: Optional[str] = None
    tags: List[str] = []
    read_time: int = 0
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class NoteDetailOut(NoteCardOut):
    content: str          # raw markdown — frontend renders it
    visibility: str
    word_count: int = 0


class NoteAdminOut(NoteDetailOut):
    file_path: str


class NoteCreateRequest(BaseModel):
    title: str
    section_slug: str
    content: str
    tags: List[str] = Field(default_factory=list)
    visibility: str = "public"
    slug: Optional[str] = None   # auto-generated from title if absent


class NoteUpdateRequest(BaseModel):
    title: Optional[str] = None
    section_slug: Optional[str] = None
    content: Optional[str] = None
    tags: Optional[List[str]] = None
    visibility: Optional[str] = None
    slug: Optional[str] = None


# ── Search Schemas ───────────────────────────────────────────

class SearchResultOut(BaseModel):
    id: int
    slug: str
    section_slug: str
    title: str
    excerpt: str          # FTS5 snippet() output, may contain <mark> tags
    tags: List[str] = []


class SearchResponse(BaseModel):
    results: List[SearchResultOut]
    total: int
    query: str


# ── Auth Schemas ─────────────────────────────────────────────

class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


# ── Health Schema ─────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    note_count: int
