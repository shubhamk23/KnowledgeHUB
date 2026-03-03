"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { adminCreateNote, adminGetSections } from "@/lib/api";
import { SplitEditor } from "@/components/admin/SplitEditor";
import { toast } from "sonner";
import { Save, ArrowLeft } from "lucide-react";
import Link from "next/link";
import type { Section } from "@/types";

export default function NewNotePage() {
  const [title, setTitle] = useState("");
  const [sectionSlug, setSectionSlug] = useState("");
  const [content, setContent] = useState("# Title\n\nStart writing your note here...\n");
  const [tags, setTags] = useState("");
  const [visibility, setVisibility] = useState("public");
  const [sections, setSections] = useState<Section[]>([]);
  const [saving, setSaving] = useState(false);
  const router = useRouter();

  useEffect(() => {
    adminGetSections()
      .then((data) => {
        setSections(data);
        if (data.length > 0) setSectionSlug(data[0].slug);
      })
      .catch(() => toast.error("Failed to load sections"));
  }, []);

  const handleSave = async () => {
    if (!title.trim()) return toast.error("Title is required");
    if (!sectionSlug) return toast.error("Section is required");

    setSaving(true);
    try {
      const note = await adminCreateNote({
        title: title.trim(),
        section_slug: sectionSlug,
        content,
        tags: tags
          .split(",")
          .map((t) => t.trim())
          .filter(Boolean),
        visibility,
      });
      toast.success("Note created successfully");
      router.push(`/admin/editor/${note.id}`);
    } catch (err: any) {
      toast.error(err.message || "Failed to create note");
    } finally {
      setSaving(false);
    }
  };

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
          New Note
        </h1>
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
          {saving ? "Saving..." : "Save Note"}
        </button>
      </div>

      {/* Metadata form */}
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
            placeholder="Note title..."
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
          <label style={labelStyle}>Tags (comma-separated)</label>
          <input
            value={tags}
            onChange={(e) => setTags(e.target.value)}
            placeholder="nlp, attention, bert"
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

      {/* Split editor */}
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
