import json
import logging

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.schemas import SearchResponse, SearchResultOut

logger = logging.getLogger(__name__)
router = APIRouter()


def _build_pg_search_sql(limit: int, offset: int):
    """Build tsvector-based search queries for PostgreSQL."""
    sql = text(
        """
        SELECT
            n.id,
            n.slug,
            s.slug AS section_slug,
            n.title,
            n.tags,
            ts_headline(
                'english',
                COALESCE(n.summary, '') || ' ' || COALESCE(n.tags, ''),
                plainto_tsquery('english', :query),
                'MaxFragments=1,MaxWords=32,MinWords=5,StartSel=<mark>,StopSel=</mark>'
            ) AS excerpt
        FROM notes n
        JOIN sections s ON n.section_id = s.id
        WHERE n.search_vector @@ plainto_tsquery('english', :query)
          AND n.visibility = 'public'
        ORDER BY ts_rank(n.search_vector, plainto_tsquery('english', :query)) DESC
        LIMIT :limit OFFSET :offset
        """
    )
    count_sql = text(
        """
        SELECT COUNT(*)
        FROM notes n
        WHERE n.search_vector @@ plainto_tsquery('english', :query)
          AND n.visibility = 'public'
        """
    )
    return sql, count_sql


def _build_sqlite_search_sql(fts_query: str, limit: int, offset: int):
    """Build FTS5-based search queries for SQLite."""
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
    return sql, count_sql


@router.get("/search", response_model=SearchResponse)
async def search_notes(
    q: str = Query(..., min_length=1, max_length=200),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    if not q.strip():
        return SearchResponse(results=[], total=0, query=q)

    try:
        if settings.is_postgres:
            sql, count_sql = _build_pg_search_sql(limit, offset)
            params = {"query": q.strip(), "limit": limit, "offset": offset}
            count_params = {"query": q.strip()}
        else:
            safe_q = q.replace('"', '""').strip()
            fts_query = f'"{safe_q}"'
            sql, count_sql = _build_sqlite_search_sql(fts_query, limit, offset)
            params = {"query": fts_query, "limit": limit, "offset": offset}
            count_params = {"query": fts_query}

        rows = await db.execute(sql, params)
        count_row = await db.execute(count_sql, count_params)
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
        logger.exception("Search error for query %r", q)
        results = []
        total = 0

    return SearchResponse(results=results, total=total, query=q)
