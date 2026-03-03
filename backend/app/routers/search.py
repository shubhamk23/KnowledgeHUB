import json
import logging

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas import SearchResponse, SearchResultOut

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/search", response_model=SearchResponse)
async def search_notes(
    q: str = Query(..., min_length=1, max_length=200),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    if not q.strip():
        return SearchResponse(results=[], total=0, query=q)

    # Sanitize query for FTS5 — escape special chars
    safe_q = q.replace('"', '""').strip()
    fts_query = f'"{safe_q}"'

    # Search using FTS5 with snippet highlighting
    sql = text(
        """
        SELECT
            n.id,
            n.slug,
            s.slug AS section_slug,
            n.title,
            n.tags,
            snippet(notes_fts, 1, '<mark>', '</mark>', '...', 32) AS excerpt
        FROM notes_fts
        JOIN notes n ON notes_fts.rowid = n.id
        JOIN sections s ON n.section_id = s.id
        WHERE notes_fts MATCH :query
          AND n.visibility = 'public'
        ORDER BY rank
        LIMIT :limit OFFSET :offset
        """
    )

    count_sql = text(
        """
        SELECT COUNT(*)
        FROM notes_fts
        JOIN notes n ON notes_fts.rowid = n.id
        WHERE notes_fts MATCH :query
          AND n.visibility = 'public'
        """
    )

    try:
        rows = await db.execute(sql, {"query": fts_query, "limit": limit, "offset": offset})
        count_row = await db.execute(count_sql, {"query": fts_query})
        total = count_row.scalar_one()

        results = [
            SearchResultOut(
                id=row.id,
                slug=row.slug,
                section_slug=row.section_slug,
                title=row.title,
                excerpt=row.excerpt or "",
                tags=json.loads(row.tags or "[]"),
            )
            for row in rows
        ]
    except Exception:
        # FTS5 query parse error — return empty results but log for debugging
        logger.exception("FTS5 search error for query %r", q)
        results = []
        total = 0

    return SearchResponse(results=results, total=total, query=q)
