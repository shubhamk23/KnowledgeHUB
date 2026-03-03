import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Section, Note
from app.schemas import SectionOut, NoteCardOut

router = APIRouter()


@router.get("/sections", response_model=list[SectionOut])
async def list_sections(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Section).order_by(Section.sort_order, Section.title))
    sections = result.scalars().all()

    output = []
    for s in sections:
        count_result = await db.execute(
            select(func.count(Note.id)).where(
                Note.section_id == s.id, Note.visibility == "public"
            )
        )
        note_count = count_result.scalar_one()
        output.append(
            SectionOut(
                id=s.id,
                slug=s.slug,
                title=s.title,
                description=s.description,
                icon=s.icon,
                sort_order=s.sort_order,
                note_count=note_count,
            )
        )
    return output


@router.get("/sections/{section_slug}")
async def get_section(section_slug: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Section).where(Section.slug == section_slug))
    section = result.scalar_one_or_none()
    if not section:
        raise HTTPException(status_code=404, detail="Section not found")

    notes_result = await db.execute(
        select(Note)
        .where(Note.section_id == section.id, Note.visibility == "public")
        .order_by(Note.created_at.desc())
    )
    notes = notes_result.scalars().all()

    note_count = len(notes)
    section_out = SectionOut(
        id=section.id,
        slug=section.slug,
        title=section.title,
        description=section.description,
        icon=section.icon,
        sort_order=section.sort_order,
        note_count=note_count,
    )

    notes_out = [
        NoteCardOut(
            id=n.id,
            slug=n.slug,
            section_slug=section.slug,
            title=n.title,
            summary=n.summary,
            tags=json.loads(n.tags or "[]"),
            read_time=n.read_time,
            created_at=n.created_at,
            updated_at=n.updated_at,
        )
        for n in notes
    ]

    return {"section": section_out, "notes": notes_out}
