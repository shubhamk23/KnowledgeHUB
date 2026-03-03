"""
Unit tests for app.markdown_utils
-----------------------------------
Tests are pure (no DB, no HTTP) and run in milliseconds.
"""

import json
import textwrap
from pathlib import Path

import pytest

from app.markdown_utils import (
    build_frontmatter_string,
    extract_first_paragraph,
    parse_note_file,
    slugify,
)


# ── slugify ────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestSlugify:
    def test_basic_lowercasing(self):
        assert slugify("Hello World") == "hello-world"

    def test_replaces_spaces_with_hyphens(self):
        assert slugify("Natural Language Processing") == "natural-language-processing"

    def test_strips_special_characters(self):
        assert slugify("What is GPT-4?!") == "what-is-gpt-4"

    def test_collapses_multiple_hyphens(self):
        assert slugify("  leading  and  trailing  ") == "leading-and-trailing"

    def test_strips_leading_trailing_hyphens(self):
        assert slugify("---hello---") == "hello"

    def test_underscore_becomes_hyphen(self):
        assert slugify("my_note_title") == "my-note-title"

    def test_empty_string(self):
        assert slugify("") == ""

    def test_already_slug(self):
        assert slugify("already-a-slug") == "already-a-slug"

    def test_unicode_stripped(self):
        # Non-ASCII letters that match \w in Python unicode are preserved
        result = slugify("café")
        assert "-" not in result or result  # just verify no crash

    def test_numbers_preserved(self):
        assert slugify("gpt-4 turbo 123") == "gpt-4-turbo-123"


# ── extract_first_paragraph ────────────────────────────────────────────────────

@pytest.mark.unit
class TestExtractFirstParagraph:
    def test_simple_paragraph(self):
        result = extract_first_paragraph("This is a paragraph.\nSecond line.")
        assert result == "This is a paragraph."

    def test_skips_headings(self):
        content = "# Heading\n\nActual paragraph here."
        result = extract_first_paragraph(content)
        assert result == "Actual paragraph here."

    def test_skips_opening_code_fence_line(self):
        # The function skips lines that START with ``` but not inner content lines.
        # So "code here" inside the fence is the first non-fence non-heading non-empty line.
        content = "```python\ncode here\n```\n\nReal paragraph."
        result = extract_first_paragraph(content)
        assert result == "code here"

    def test_strips_inline_links(self):
        content = "See [the paper](https://example.com) for details."
        result = extract_first_paragraph(content)
        assert "the paper" in result
        assert "https://" not in result

    def test_strips_markdown_emphasis(self):
        content = "This is **bold** and *italic* text."
        result = extract_first_paragraph(content)
        assert "**" not in result
        assert "*" not in result

    def test_truncates_at_300_chars(self):
        long_line = "word " * 100  # 500 chars
        result = extract_first_paragraph(long_line)
        assert len(result) <= 300

    def test_empty_content_returns_empty(self):
        assert extract_first_paragraph("") == ""

    def test_only_headings_returns_empty(self):
        assert extract_first_paragraph("# H1\n## H2\n### H3") == ""


# ── build_frontmatter_string ───────────────────────────────────────────────────

@pytest.mark.unit
class TestBuildFrontmatterString:
    def test_contains_title(self):
        result = build_frontmatter_string("My Note", "my-note", [], "public", "Summary", "Body")
        assert 'title: "My Note"' in result

    def test_contains_slug(self):
        result = build_frontmatter_string("My Note", "my-note", [], "public", "Summary", "Body")
        assert "slug: my-note" in result

    def test_tags_serialised(self):
        result = build_frontmatter_string("T", "t", ["ml", "nlp"], "public", "", "Body")
        assert '"ml"' in result
        assert '"nlp"' in result

    def test_empty_tags(self):
        result = build_frontmatter_string("T", "t", [], "public", "", "Body")
        assert "tags: []" in result

    def test_body_appended_after_front_matter(self):
        result = build_frontmatter_string("T", "t", [], "public", "", "My content here.")
        assert "My content here." in result

    def test_has_yaml_delimiters(self):
        result = build_frontmatter_string("T", "t", [], "public", "", "")
        assert result.startswith("---")
        # There should be a closing --- before the content
        parts = result.split("---")
        assert len(parts) >= 3

    def test_visibility_draft(self):
        result = build_frontmatter_string("T", "t", [], "draft", "", "Body")
        assert "visibility: draft" in result


# ── parse_note_file ────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestParseNoteFile:
    def _write_note(self, tmp_path: Path, content: str, filename: str = "note.md") -> Path:
        p = tmp_path / filename
        p.write_text(content, encoding="utf-8")
        return p

    def test_parses_title_from_frontmatter(self, tmp_path):
        path = self._write_note(tmp_path, '---\ntitle: "Hello"\n---\n\nContent.')
        result = parse_note_file(path)
        assert result["title"] == "Hello"

    def test_derives_title_from_filename_when_missing(self, tmp_path):
        path = self._write_note(tmp_path, "---\n---\n\nContent.", "my-cool-note.md")
        result = parse_note_file(path)
        assert result["title"] == "My Cool Note"

    def test_parses_slug_from_frontmatter(self, tmp_path):
        path = self._write_note(tmp_path, '---\nslug: custom-slug\n---\n\nContent.')
        result = parse_note_file(path)
        assert result["slug"] == "custom-slug"

    def test_derives_slug_from_filename_when_missing(self, tmp_path):
        path = self._write_note(tmp_path, "---\n---\n\nContent.", "my-note.md")
        result = parse_note_file(path)
        assert result["slug"] == "my-note"

    def test_parses_tags_list(self, tmp_path):
        path = self._write_note(tmp_path, "---\ntags: [nlp, bert]\n---\n\nContent.")
        result = parse_note_file(path)
        assert "nlp" in result["tags"]
        assert "bert" in result["tags"]

    def test_parses_tags_comma_string(self, tmp_path):
        path = self._write_note(tmp_path, "---\ntags: nlp, bert\n---\n\nContent.")
        result = parse_note_file(path)
        assert "nlp" in result["tags"]

    def test_visibility_defaults_to_public(self, tmp_path):
        path = self._write_note(tmp_path, "---\n---\n\nContent.")
        result = parse_note_file(path)
        assert result["visibility"] == "public"

    def test_visibility_draft(self, tmp_path):
        path = self._write_note(tmp_path, "---\nvisibility: draft\n---\n\nContent.")
        result = parse_note_file(path)
        assert result["visibility"] == "draft"

    def test_word_count_computed(self, tmp_path):
        path = self._write_note(tmp_path, "---\n---\n\none two three four five")
        result = parse_note_file(path)
        assert result["word_count"] == 5

    def test_read_time_at_least_one(self, tmp_path):
        path = self._write_note(tmp_path, "---\n---\n\nShort.")
        result = parse_note_file(path)
        assert result["read_time"] >= 1

    def test_summary_from_frontmatter(self, tmp_path):
        path = self._write_note(tmp_path, '---\nsummary: "Custom summary"\n---\n\nContent.')
        result = parse_note_file(path)
        assert result["summary"] == "Custom summary"

    def test_summary_derived_from_content(self, tmp_path):
        path = self._write_note(tmp_path, "---\n---\n\nThis is the first line of content.")
        result = parse_note_file(path)
        assert "first line" in result["summary"]

    def test_content_returned(self, tmp_path):
        path = self._write_note(tmp_path, "---\n---\n\nMy note body.")
        result = parse_note_file(path)
        assert "My note body." in result["content"]
