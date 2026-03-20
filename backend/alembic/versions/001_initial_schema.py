"""Initial schema

Revision ID: 001
Revises:
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "sections",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("slug", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("icon", sa.String(), nullable=True),
        sa.Column("sort_order", sa.Integer(), default=0),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=True),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("slug"),
    )

    op.create_table(
        "notes",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("slug", sa.String(), nullable=False),
        sa.Column("section_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("tags", sa.Text(), default="[]"),
        sa.Column("visibility", sa.String(), default="public"),
        sa.Column("level", sa.String(), default="beginner"),
        sa.Column("file_path", sa.String(), nullable=False),
        sa.Column("word_count", sa.Integer(), default=0),
        sa.Column("read_time", sa.Integer(), default=0),
        sa.Column(
            "search_vector",
            sa.Text().with_variant(
                sa.dialects.postgresql.TSVECTOR() if hasattr(sa.dialects, "postgresql") else sa.Text(),
                "postgresql",
            ),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=True),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=True),
        sa.ForeignKeyConstraint(["section_id"], ["sections.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("file_path"),
    )

    # GIN index on search_vector for fast full-text search (PostgreSQL only)
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_notes_search_vector
        ON notes USING GIN(search_vector)
        """
    )

    # Trigger function: auto-update search_vector on INSERT/UPDATE
    op.execute(
        """
        CREATE OR REPLACE FUNCTION update_notes_search_vector()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.search_vector :=
                setweight(to_tsvector('english', coalesce(NEW.title, '')), 'A') ||
                setweight(to_tsvector('english', coalesce(NEW.summary, '')), 'B') ||
                setweight(to_tsvector('english', coalesce(NEW.tags, '')), 'C');
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )

    op.execute(
        """
        CREATE TRIGGER notes_search_vector_update
        BEFORE INSERT OR UPDATE ON notes
        FOR EACH ROW EXECUTE FUNCTION update_notes_search_vector();
        """
    )

    op.create_table(
        "admin_users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("username", sa.String(), nullable=False),
        sa.Column("password_hash", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("username"),
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS notes_search_vector_update ON notes")
    op.execute("DROP FUNCTION IF EXISTS update_notes_search_vector")
    op.drop_table("admin_users")
    op.drop_table("notes")
    op.drop_table("sections")
