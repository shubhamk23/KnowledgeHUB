import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import hash_password, get_current_admin
from app.config import settings
from app.database import create_tables, get_db, AsyncSessionLocal
from app.indexer import full_reindex, start_watcher, stop_watcher
from app.models import AdminUser
from app.routers import auth_router, sections, notes, search, admin
from app.schemas import HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def _create_fts_table():
    """Create FTS5 virtual table if it doesn't exist."""
    async with AsyncSessionLocal() as db:
        await db.execute(
            text(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
                    title,
                    content,
                    tags,
                    summary,
                    tokenize='porter unicode61'
                )
                """
            )
        )
        await db.commit()


async def _seed_admin_user():
    """Create the default admin user from settings if it doesn't exist."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(AdminUser).where(AdminUser.username == settings.admin_username)
        )
        user = result.scalar_one_or_none()
        if not user:
            hashed = hash_password(settings.admin_password)
            user = AdminUser(username=settings.admin_username, password_hash=hashed)
            db.add(user)
            await db.commit()
            logger.info(f"Admin user '{settings.admin_username}' created")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    logger.info("Starting AI Notes Knowledge Hub...")

    await create_tables()
    await _create_fts_table()
    await _seed_admin_user()

    async with AsyncSessionLocal() as db:
        result = await full_reindex(db)
        logger.info(f"Initial index: {result.indexed} notes, {len(result.errors)} errors")

    start_watcher(AsyncSessionLocal)

    yield

    # ── Shutdown ──
    stop_watcher()
    logger.info("Shutdown complete")


app = FastAPI(
    title="AI Notes Knowledge Hub",
    description="A personal knowledge hub for AI/ML notes and research",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────
app.include_router(auth_router.router, prefix="/api/auth", tags=["auth"])
app.include_router(sections.router, prefix="/api", tags=["sections"])
app.include_router(notes.router, prefix="/api", tags=["notes"])
app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(
    admin.router,
    prefix="/api/admin",
    tags=["admin"],
)


@app.get("/api/health", response_model=HealthResponse, tags=["health"])
async def health(db: AsyncSession = Depends(get_db)):
    from sqlalchemy import func
    from app.models import Note
    count_result = await db.execute(
        select(func.count(Note.id)).where(Note.visibility == "public")
    )
    note_count = count_result.scalar_one()
    return HealthResponse(status="ok", version="1.0.0", note_count=note_count)
