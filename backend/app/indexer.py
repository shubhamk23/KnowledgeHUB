import json
import logging
import threading
from pathlib import Path
from typing import Optional

from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.markdown_utils import parse_note_file
from app.models import Note, Section

logger = logging.getLogger(__name__)


class IndexResult:
    def __init__(self):
        self.indexed = 0
        self.errors: list[str] = []


async def _upsert_section(db: AsyncSession, folder: Path) -> Optional[int]:
    """Read _section.json and upsert section row. Returns section id."""
    section_json_path = folder / "_section.json"
    if not section_json_path.exists():
        # Auto-create section from folder name
        meta = {
            "title": folder.name.replace("-", " ").title(),
            "description": None,
            "icon": None,
            "sort_order": 99,
        }
    else:
        try:
            meta = json.loads(section_json_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Failed to parse {section_json_path}: {e}")
            return None

    slug = folder.name
    result = await db.execute(select(Section).where(Section.slug == slug))
    section = result.scalar_one_or_none()

    if section is None:
        section = Section(
            slug=slug,
            title=meta.get("title", slug),
            description=meta.get("description"),
            icon=meta.get("icon"),
            sort_order=meta.get("sort_order", 99),
        )
        db.add(section)
        await db.flush()
    else:
        section.title = meta.get("title", section.title)
        section.description = meta.get("description", section.description)
        section.icon = meta.get("icon", section.icon)
        section.sort_order = meta.get("sort_order", section.sort_order)
        await db.flush()

    return section.id


async def _upsert_note(db: AsyncSession, md_path: Path, section_id: int, section_slug: str) -> bool:
    """Parse .md file and upsert note + FTS row. Returns True on success."""
    try:
        parsed = parse_note_file(md_path)
    except Exception as e:
        logger.error(f"Failed to parse {md_path}: {e}")
        return False

    file_path_str = str(md_path.resolve())
    tags_json = json.dumps(parsed["tags"])

    result = await db.execute(select(Note).where(Note.file_path == file_path_str))
    note = result.scalar_one_or_none()

    if note is None:
        note = Note(
            slug=parsed["slug"],
            section_id=section_id,
            title=parsed["title"],
            summary=parsed["summary"],
            tags=tags_json,
            visibility=parsed["visibility"],
            file_path=file_path_str,
            word_count=parsed["word_count"],
            read_time=parsed["read_time"],
        )
        db.add(note)
        await db.flush()

        # Insert into FTS
        await db.execute(
            text(
                "INSERT INTO notes_fts(rowid, title, content, tags, summary) "
                "VALUES (:rowid, :title, :content, :tags, :summary)"
            ),
            {
                "rowid": note.id,
                "title": parsed["title"],
                "content": parsed["content"],
                "tags": tags_json,
                "summary": parsed["summary"] or "",
            },
        )
    else:
        note.slug = parsed["slug"]
        note.section_id = section_id
        note.title = parsed["title"]
        note.summary = parsed["summary"]
        note.tags = tags_json
        note.visibility = parsed["visibility"]
        note.word_count = parsed["word_count"]
        note.read_time = parsed["read_time"]
        await db.flush()

        # Update FTS: FTS5 doesn't support ON CONFLICT — delete + re-insert
        await db.execute(
            text("DELETE FROM notes_fts WHERE rowid = :rowid"),
            {"rowid": note.id},
        )
        await db.execute(
            text(
                "INSERT INTO notes_fts(rowid, title, content, tags, summary) "
                "VALUES (:rowid, :title, :content, :tags, :summary)"
            ),
            {
                "rowid": note.id,
                "title": parsed["title"],
                "content": parsed["content"],
                "tags": tags_json,
                "summary": parsed["summary"] or "",
            },
        )

    return True


async def full_reindex(db: AsyncSession) -> IndexResult:
    """Scan content/ directory and rebuild all sections + notes in SQLite."""
    result = IndexResult()
    content_dir = Path(settings.content_dir).resolve()

    if not content_dir.exists():
        content_dir.mkdir(parents=True)
        logger.info(f"Created content directory: {content_dir}")

    for folder in sorted(content_dir.iterdir()):
        if not folder.is_dir():
            continue

        section_id = await _upsert_section(db, folder)
        if section_id is None:
            result.errors.append(f"Could not index section: {folder.name}")
            continue

        for md_path in sorted(folder.glob("*.md")):
            success = await _upsert_note(db, md_path, section_id, folder.name)
            if success:
                result.indexed += 1
            else:
                result.errors.append(str(md_path))

    await db.commit()
    logger.info(f"Reindex complete: {result.indexed} notes indexed, {len(result.errors)} errors")
    return result


class _WatcherHandler:
    """Handles file system events from watchdog."""

    def __init__(self, session_factory):
        self._session_factory = session_factory

    def _run_async(self, coro):
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(coro)
        finally:
            loop.close()

    async def _handle_md_change(self, path: str):
        md_path = Path(path)
        if not md_path.exists() or md_path.name == "_section.json":
            return
        section_folder = md_path.parent
        async with self._session_factory() as db:
            section_id = await _upsert_section(db, section_folder)
            if section_id:
                await _upsert_note(db, md_path, section_id, section_folder.name)
                await db.commit()
                logger.info(f"Re-indexed: {md_path.name}")

    async def _handle_delete(self, path: str):
        md_path = Path(path)
        file_path_str = str(md_path.resolve())
        async with self._session_factory() as db:
            result = await db.execute(select(Note).where(Note.file_path == file_path_str))
            note = result.scalar_one_or_none()
            if note:
                await db.execute(
                    text("DELETE FROM notes_fts WHERE rowid = :rowid"), {"rowid": note.id}
                )
                await db.delete(note)
                await db.commit()
                logger.info(f"Removed from index: {md_path.name}")

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(".md"):
            self._run_async(self._handle_md_change(event.src_path))

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".md"):
            self._run_async(self._handle_md_change(event.src_path))

    def on_deleted(self, event):
        if not event.is_directory and event.src_path.endswith(".md"):
            self._run_async(self._handle_delete(event.src_path))


_observer = None


def start_watcher(session_factory):
    """Start the watchdog observer in a background thread."""
    global _observer
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        handler_impl = _WatcherHandler(session_factory)

        class _FSHandler(FileSystemEventHandler):
            def on_modified(self, event):
                handler_impl.on_modified(event)

            def on_created(self, event):
                handler_impl.on_created(event)

            def on_deleted(self, event):
                handler_impl.on_deleted(event)

        content_dir = str(Path(settings.content_dir).resolve())
        _observer = Observer()
        _observer.schedule(_FSHandler(), content_dir, recursive=True)
        _observer.start()
        logger.info(f"File watcher started on: {content_dir}")
    except Exception as e:
        logger.warning(f"Could not start file watcher: {e}")


def stop_watcher():
    global _observer
    if _observer:
        _observer.stop()
        _observer.join()
        logger.info("File watcher stopped")
