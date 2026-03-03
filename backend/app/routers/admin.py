import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_admin
from app.config import settings
from app.database import get_db
from app.indexer import full_reindex, _upsert_section
from app.markdown_utils import slugify, build_frontmatter_string, parse_note_file
from app.models import Note, Section
from app.schemas import (
    NoteAdminOut,
    NoteCreateRequest,
    NoteUpdateRequest,
    SectionAdminOut,
    SectionCreateRequest,
    SectionUpdateRequest,
)

router = APIRouter()


# ── Notes ─────────────────────────────────────────────────────

@router.get("/notes", response_model=list[NoteAdminOut])
async def admin_list_notes(
    db: AsyncSession = Depends(get_db),
    _: str = Depends(get_current_admin),
):
    result = await db.execute(
        select(Note, Section.slug.label("section_slug"))
        .join(Section, Note.section_id == Section.id)
        .order_by(Note.updated_at.desc())
    )
    rows = result.all()

    notes = []
    for note, section_slug in rows:
        try:
            content = Path(note.file_path).read_text(encoding="utf-8")
            import frontmatter
            post = frontmatter.loads(content)
            body = post.content
        except Exception:
            body = ""

        notes.append(
            NoteAdminOut(
                id=note.id,
                slug=note.slug,
                section_slug=section_slug,
                title=note.title,
                summary=note.summary,
                tags=json.loads(note.tags or "[]"),
                read_time=note.read_time,
                word_count=note.word_count,
                visibility=note.visibility,
                content=body,
                file_path=note.file_path,
                created_at=note.created_at,
                updated_at=note.updated_at,
            )
        )
    return notes


@router.get("/notes/{note_id}", response_model=NoteAdminOut)
async def admin_get_note(
    note_id: int,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(get_current_admin),
):
    result = await db.execute(
        select(Note, Section.slug.label("section_slug"))
        .join(Section, Note.section_id == Section.id)
        .where(Note.id == note_id)
    )
    row = result.first()
    if not row:
        raise HTTPException(status_code=404, detail="Note not found")

    note, section_slug = row
    try:
        content = Path(note.file_path).read_text(encoding="utf-8")
        import frontmatter
        post = frontmatter.loads(content)
        body = post.content
    except Exception:
        body = ""

    return NoteAdminOut(
        id=note.id,
        slug=note.slug,
        section_slug=section_slug,
        title=note.title,
        summary=note.summary,
        tags=json.loads(note.tags or "[]"),
        read_time=note.read_time,
        word_count=note.word_count,
        visibility=note.visibility,
        content=body,
        file_path=note.file_path,
        created_at=note.created_at,
        updated_at=note.updated_at,
    )


@router.post("/notes", response_model=NoteAdminOut, status_code=status.HTTP_201_CREATED)
async def admin_create_note(
    req: NoteCreateRequest,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(get_current_admin),
):
    section_result = await db.execute(select(Section).where(Section.slug == req.section_slug))
    section = section_result.scalar_one_or_none()
    if not section:
        raise HTTPException(status_code=404, detail=f"Section '{req.section_slug}' not found")

    slug = req.slug or slugify(req.title)

    # Check slug uniqueness within section
    existing = await db.execute(
        select(Note).where(Note.slug == slug, Note.section_id == section.id)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail=f"Note with slug '{slug}' already exists in this section")

    content_dir = Path(settings.content_dir).resolve()
    folder = content_dir / req.section_slug
    folder.mkdir(parents=True, exist_ok=True)

    from app.markdown_utils import extract_first_paragraph
    summary = extract_first_paragraph(req.content) if req.content else ""

    file_content = build_frontmatter_string(
        title=req.title,
        slug=slug,
        tags=req.tags,
        visibility=req.visibility,
        summary=summary,
        content=req.content,
    )

    file_path = folder / f"{slug}.md"
    file_path.write_text(file_content, encoding="utf-8")

    word_count = len(req.content.split())
    tags_json = json.dumps(req.tags)

    note = Note(
        slug=slug,
        section_id=section.id,
        title=req.title,
        summary=summary,
        tags=tags_json,
        visibility=req.visibility,
        file_path=str(file_path),
        word_count=word_count,
        read_time=max(1, word_count // 200),
    )
    db.add(note)
    await db.flush()

    await db.execute(
        text(
            "INSERT INTO notes_fts(rowid, title, content, tags, summary) "
            "VALUES (:rowid, :title, :content, :tags, :summary)"
        ),
        {"rowid": note.id, "title": req.title, "content": req.content, "tags": tags_json, "summary": summary},
    )
    await db.commit()
    await db.refresh(note)

    return NoteAdminOut(
        id=note.id,
        slug=note.slug,
        section_slug=req.section_slug,
        title=note.title,
        summary=note.summary,
        tags=json.loads(note.tags or "[]"),
        read_time=note.read_time,
        word_count=note.word_count,
        visibility=note.visibility,
        content=req.content,
        file_path=note.file_path,
        created_at=note.created_at,
        updated_at=note.updated_at,
    )


@router.put("/notes/{note_id}", response_model=NoteAdminOut)
async def admin_update_note(
    note_id: int,
    req: NoteUpdateRequest,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(get_current_admin),
):
    result = await db.execute(
        select(Note, Section.slug.label("section_slug"))
        .join(Section, Note.section_id == Section.id)
        .where(Note.id == note_id)
    )
    row = result.first()
    if not row:
        raise HTTPException(status_code=404, detail="Note not found")

    note, current_section_slug = row

    # Load current content from disk
    try:
        raw = Path(note.file_path).read_text(encoding="utf-8")
        import frontmatter
        post = frontmatter.loads(raw)
        current_content = post.content
    except Exception:
        current_content = ""

    # Apply updates
    new_title = req.title if req.title is not None else note.title
    new_content = req.content if req.content is not None else current_content
    new_tags = req.tags if req.tags is not None else json.loads(note.tags or "[]")
    new_visibility = req.visibility if req.visibility is not None else note.visibility
    new_slug = req.slug if req.slug is not None else note.slug
    new_section_slug = req.section_slug if req.section_slug is not None else current_section_slug

    # Resolve section
    if new_section_slug != current_section_slug:
        sec_result = await db.execute(select(Section).where(Section.slug == new_section_slug))
        new_section = sec_result.scalar_one_or_none()
        if not new_section:
            raise HTTPException(status_code=404, detail=f"Section '{new_section_slug}' not found")
        note.section_id = new_section.id
    else:
        new_section = None

    from app.markdown_utils import extract_first_paragraph
    new_summary = extract_first_paragraph(new_content)

    content_dir = Path(settings.content_dir).resolve()
    old_file = Path(note.file_path)

    # Write to new location if slug or section changed
    new_folder = content_dir / new_section_slug
    new_folder.mkdir(parents=True, exist_ok=True)
    new_file_path = new_folder / f"{new_slug}.md"

    file_content = build_frontmatter_string(
        title=new_title,
        slug=new_slug,
        tags=new_tags,
        visibility=new_visibility,
        summary=new_summary,
        content=new_content,
    )
    new_file_path.write_text(file_content, encoding="utf-8")

    # Remove old file if path changed
    if old_file != new_file_path and old_file.exists():
        old_file.unlink()

    word_count = len(new_content.split())
    tags_json = json.dumps(new_tags)

    note.slug = new_slug
    note.title = new_title
    note.summary = new_summary
    note.tags = tags_json
    note.visibility = new_visibility
    note.file_path = str(new_file_path)
    note.word_count = word_count
    note.read_time = max(1, word_count // 200)

    # FTS5 doesn't support ON CONFLICT — delete + re-insert
    await db.execute(
        text("DELETE FROM notes_fts WHERE rowid = :rowid"),
        {"rowid": note.id},
    )
    await db.execute(
        text(
            "INSERT INTO notes_fts(rowid, title, content, tags, summary) "
            "VALUES (:rowid, :title, :content, :tags, :summary)"
        ),
        {"rowid": note.id, "title": new_title, "content": new_content, "tags": tags_json, "summary": new_summary},
    )
    await db.commit()
    await db.refresh(note)

    return NoteAdminOut(
        id=note.id,
        slug=note.slug,
        section_slug=new_section_slug,
        title=note.title,
        summary=note.summary,
        tags=json.loads(note.tags or "[]"),
        read_time=note.read_time,
        word_count=note.word_count,
        visibility=note.visibility,
        content=new_content,
        file_path=note.file_path,
        created_at=note.created_at,
        updated_at=note.updated_at,
    )


@router.delete("/notes/{note_id}")
async def admin_delete_note(
    note_id: int,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(get_current_admin),
):
    result = await db.execute(select(Note).where(Note.id == note_id))
    note = result.scalar_one_or_none()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    # Remove from FTS
    await db.execute(text("DELETE FROM notes_fts WHERE rowid = :rowid"), {"rowid": note.id})

    # Remove file from disk
    try:
        file_path = Path(note.file_path)
        if file_path.exists():
            file_path.unlink()
    except Exception:
        pass

    await db.delete(note)
    await db.commit()
    return {"deleted": True}


@router.post("/reindex")
async def admin_reindex(
    db: AsyncSession = Depends(get_db),
    _: str = Depends(get_current_admin),
):
    result = await full_reindex(db)
    return {"indexed": result.indexed, "errors": result.errors}


# ── Sections ──────────────────────────────────────────────────

@router.get("/sections", response_model=list[SectionAdminOut])
async def admin_list_sections(
    db: AsyncSession = Depends(get_db),
    _: str = Depends(get_current_admin),
):
    from sqlalchemy import func
    result = await db.execute(select(Section).order_by(Section.sort_order, Section.title))
    sections = result.scalars().all()

    output = []
    for s in sections:
        count_result = await db.execute(
            select(func.count(Note.id)).where(Note.section_id == s.id)
        )
        note_count = count_result.scalar_one()
        output.append(
            SectionAdminOut(
                id=s.id,
                slug=s.slug,
                title=s.title,
                description=s.description,
                icon=s.icon,
                sort_order=s.sort_order,
                note_count=note_count,
                created_at=s.created_at,
                updated_at=s.updated_at,
            )
        )
    return output


@router.post("/sections", response_model=SectionAdminOut, status_code=status.HTTP_201_CREATED)
async def admin_create_section(
    req: SectionCreateRequest,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(get_current_admin),
):
    existing = await db.execute(select(Section).where(Section.slug == req.slug))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail=f"Section '{req.slug}' already exists")

    folder = Path(settings.content_dir).resolve() / req.slug
    folder.mkdir(parents=True, exist_ok=True)

    section_json = {
        "title": req.title,
        "description": req.description,
        "icon": req.icon,
        "sort_order": req.sort_order,
    }
    (folder / "_section.json").write_text(json.dumps(section_json, indent=2), encoding="utf-8")

    section = Section(
        slug=req.slug,
        title=req.title,
        description=req.description,
        icon=req.icon,
        sort_order=req.sort_order,
    )
    db.add(section)
    await db.commit()
    await db.refresh(section)

    return SectionAdminOut(
        id=section.id,
        slug=section.slug,
        title=section.title,
        description=section.description,
        icon=section.icon,
        sort_order=section.sort_order,
        note_count=0,
        created_at=section.created_at,
        updated_at=section.updated_at,
    )


@router.put("/sections/{section_id}", response_model=SectionAdminOut)
async def admin_update_section(
    section_id: int,
    req: SectionUpdateRequest,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(get_current_admin),
):
    from sqlalchemy import func
    result = await db.execute(select(Section).where(Section.id == section_id))
    section = result.scalar_one_or_none()
    if not section:
        raise HTTPException(status_code=404, detail="Section not found")

    if req.title is not None:
        section.title = req.title
    if req.description is not None:
        section.description = req.description
    if req.icon is not None:
        section.icon = req.icon
    if req.sort_order is not None:
        section.sort_order = req.sort_order

    # Update _section.json on disk
    folder = Path(settings.content_dir).resolve() / section.slug
    folder.mkdir(parents=True, exist_ok=True)
    section_json = {
        "title": section.title,
        "description": section.description,
        "icon": section.icon,
        "sort_order": section.sort_order,
    }
    (folder / "_section.json").write_text(json.dumps(section_json, indent=2), encoding="utf-8")

    await db.commit()
    await db.refresh(section)

    count_result = await db.execute(
        select(func.count(Note.id)).where(Note.section_id == section.id)
    )
    note_count = count_result.scalar_one()

    return SectionAdminOut(
        id=section.id,
        slug=section.slug,
        title=section.title,
        description=section.description,
        icon=section.icon,
        sort_order=section.sort_order,
        note_count=note_count,
        created_at=section.created_at,
        updated_at=section.updated_at,
    )
