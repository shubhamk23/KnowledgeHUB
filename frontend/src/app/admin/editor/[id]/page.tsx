"use client";

import { useState, useEffect } from "react";
import { useRouter, useParams } from "next/navigation";
import { adminGetNote, adminUpdateNote, adminGetSections } from "@/lib/api";
import { SplitEditor } from "@/components/admin/SplitEditor";
import { toast } from "sonner";
import { Save, ArrowLeft, ExternalLink } from "lucide-react";
import Link from "next/link";
import type { Section, NoteAdminDetail } from "@/types";

export default function EditNotePage() {
  const params = useParams();
  const noteId = parseInt(params.id as string, 10);

  const [note, setNote] = useState<NoteAdminDetail | null>(null);
  const [title, setTitle] = useState("");
  const [sectionSlug, setSectionSlug] = useState("");
  const [content, setContent] = useState("");
  const [tags, setTags] = useState("");
  const [visibility, setVisibility] = useState("public");
  const [sections, setSections] = useState<Section[]>([]);
  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    Promise.all([adminGetNote(noteId), adminGetSections()])
      .then(([noteData, sectionsData]) => {
        setNote(noteData);
        setTitle(noteData.title);
        setSectionSlug(noteData.section_slug);
        setContent(noteData.content);
        setTags(noteData.tags.join(", "));
        setVisibility(noteData.visibility);
        setSections(sectionsData);
      })
      .catch(() => toast.error("Failed to load note"))
      .finally(() => setLoading(false));
  }, [noteId]);

  const handleSave = async () => {
    if (!title.trim()) return toast.error("Title is required");
    setSaving(true);
    try {
      await adminUpdateNote(noteId, {
        title: title.trim(),
        section_slug: sectionSlug,
        content,
        tags: tags
          .split(",")
          .map((t) => t.trim())
          .filter(Boolean),
        visibility,
      });
      toast.success("Note updated");
    } catch (err: any) {
      toast.error(err.message || "Failed to update note");
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div style={{ textAlign: "center", color: "var(--text-muted)", padding: "4rem" }}>
        Loading...
      </div>
    );
  }

  return (
    <div>
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "12px",
          marginBottom: "1.5rem",
        }}
      >
        <Link href="/admin">
          <button
            style={{
              background: "none",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius)",
              padding: "8px 12px",
              cursor: "pointer",
              color: "var(--text-secondary)",
              display: "flex",
              alignItems: "center",
              gap: "6px",
              fontSize: "0.85rem",
            }}
          >
            <ArrowLeft size={14} />
            Back
          </button>
        </Link>
        <h1
          style={{
            fontSize: "1.5rem",
            fontWeight: 700,
            color: "var(--text-primary)",
            margin: 0,
            flex: 1,
          }}
        >
          Edit Note
        </h1>
        {note && (
          <Link href={`/${note.section_slug}/${note.slug}`} target="_blank">
            <button
              style={{
                display: "flex",
                alignItems: "center",
                gap: "6px",
                padding: "9px 16px",
                background: "none",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius)",
                fontSize: "0.85rem",
                color: "var(--text-secondary)",
                cursor: "pointer",
              }}
            >
              <ExternalLink size={13} />
              View
            </button>
          </Link>
        )}
        <button
          onClick={handleSave}
          disabled={saving}
          style={{
            display: "flex",
            alignItems: "center",
            gap: "6px",
            padding: "9px 20px",
            background: saving ? "var(--text-muted)" : "var(--accent)",
            color: "#fff",
            border: "none",
            borderRadius: "var(--radius)",
            fontSize: "0.9rem",
            fontWeight: 600,
            cursor: saving ? "not-allowed" : "pointer",
          }}
        >
          <Save size={15} />
          {saving ? "Saving..." : "Save"}
        </button>
      </div>

      {/* Metadata */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 200px 200px 150px",
          gap: "12px",
          marginBottom: "1.25rem",
          alignItems: "end",
        }}
      >
        <div>
          <label style={labelStyle}>Title *</label>
          <input
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            style={inputStyle}
          />
        </div>
        <div>
          <label style={labelStyle}>Section *</label>
          <select
            value={sectionSlug}
            onChange={(e) => setSectionSlug(e.target.value)}
            style={inputStyle}
          >
            {sections.map((s) => (
              <option key={s.slug} value={s.slug}>
                {s.title}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label style={labelStyle}>Tags</label>
          <input
            value={tags}
            onChange={(e) => setTags(e.target.value)}
            style={inputStyle}
          />
        </div>
        <div>
          <label style={labelStyle}>Visibility</label>
          <select
            value={visibility}
            onChange={(e) => setVisibility(e.target.value)}
            style={inputStyle}
          >
            <option value="public">Public</option>
            <option value="draft">Draft</option>
          </select>
        </div>
      </div>

      <SplitEditor value={content} onChange={setContent} />
    </div>
  );
}

const labelStyle: React.CSSProperties = {
  display: "block",
  fontSize: "0.75rem",
  fontWeight: 600,
  color: "var(--text-secondary)",
  marginBottom: "5px",
  textTransform: "uppercase",
  letterSpacing: "0.05em",
};

const inputStyle: React.CSSProperties = {
  width: "100%",
  padding: "9px 12px",
  background: "var(--bg-card)",
  border: "1px solid var(--border)",
  borderRadius: "var(--radius)",
  fontSize: "0.9rem",
  color: "var(--text-primary)",
  outline: "none",
};
