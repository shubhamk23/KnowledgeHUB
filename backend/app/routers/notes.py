import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Note, Section
from app.schemas import NoteDetailOut

router = APIRouter()


@router.get("/notes/{section_slug}/{note_slug}", response_model=NoteDetailOut)
async def get_note(section_slug: str, note_slug: str, db: AsyncSession = Depends(get_db)):
    section_result = await db.execute(select(Section).where(Section.slug == section_slug))
    section = section_result.scalar_one_or_none()
    if not section:
        raise HTTPException(status_code=404, detail="Section not found")

    note_result = await db.execute(
        select(Note).where(
            Note.slug == note_slug,
            Note.section_id == section.id,
            Note.visibility == "public",
        )
    )
    note = note_result.scalar_one_or_none()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    # Read raw markdown from disk
    try:
        file_path = Path(note.file_path)
        raw = file_path.read_text(encoding="utf-8")
        # Strip frontmatter block before returning to frontend
        import frontmatter
        post = frontmatter.loads(raw)
        content = post.content
    except Exception:
        content = ""

    return NoteDetailOut(
        id=note.id,
        slug=note.slug,
        section_slug=section.slug,
        title=note.title,
        summary=note.summary,
        tags=json.loads(note.tags or "[]"),
        read_time=note.read_time,
        word_count=note.word_count,
        visibility=note.visibility,
        content=content,
        created_at=note.created_at,
        updated_at=note.updated_at,
    )
