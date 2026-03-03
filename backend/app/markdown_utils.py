import re
from pathlib import Path
from typing import Optional
import frontmatter


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def extract_first_paragraph(content: str) -> str:
    """Extract the first non-heading, non-empty paragraph (up to 300 chars)."""
    lines = content.split("\n")
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("```"):
            # Remove inline markdown
            clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", stripped)
            clean = re.sub(r"[*_`~]", "", clean)
            return clean[:300]
    return ""


def parse_note_file(file_path: Path) -> dict:
    """
    Parse a .md file and return a dict with all note metadata + content.
    Frontmatter fields take precedence; sane defaults are derived from the file.
    """
    post = frontmatter.load(str(file_path))

    slug = str(post.get("slug", "") or slugify(file_path.stem))
    title = str(post.get("title", "") or file_path.stem.replace("-", " ").title())
    content: str = post.content or ""
    tags = post.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    elif not isinstance(tags, list):
        tags = []

    summary = str(post.get("summary", "") or extract_first_paragraph(content))
    visibility = str(post.get("visibility", "public"))
    word_count = len(content.split())
    read_time = max(1, word_count // 200)

    created_at = post.get("created_at", None)

    return {
        "title": title,
        "slug": slug,
        "tags": tags,
        "summary": summary,
        "visibility": visibility,
        "content": content,
        "word_count": word_count,
        "read_time": read_time,
        "created_at": created_at,
    }


def build_frontmatter_string(
    title: str,
    slug: str,
    tags: list,
    visibility: str,
    summary: Optional[str],
    content: str,
) -> str:
    """Build a full .md file string including YAML frontmatter."""
    tags_yaml = ", ".join(f'"{t}"' for t in tags) if tags else ""
    summary_line = f'summary: "{summary}"' if summary else 'summary: ""'
    return f"""---
title: "{title}"
slug: {slug}
{summary_line}
tags: [{tags_yaml}]
visibility: {visibility}
---

{content}
"""
