"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { adminDeleteNote, adminGetNotes, adminReindex } from "@/lib/api";
import type { NoteAdminDetail } from "@/types";
import { toast } from "sonner";
import { Plus, Pencil, Trash2, RefreshCw, LogOut } from "lucide-react";
import { useAuthStore } from "@/store/authStore";
import { useRouter } from "next/navigation";

export default function AdminDashboard() {
  const [notes, setNotes] = useState<NoteAdminDetail[]>([]);
  const [loading, setLoading] = useState(true);
  const [reindexing, setReindexing] = useState(false);
  const [deleteId, setDeleteId] = useState<number | null>(null);
  const { clearToken } = useAuthStore();
  const router = useRouter();

  const fetchNotes = async () => {
    try {
      const data = await adminGetNotes();
      setNotes(data);
    } catch {
      toast.error("Failed to load notes");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchNotes();
  }, []);

  const handleDelete = async (id: number) => {
    if (!confirm("Are you sure you want to delete this note? This will also remove the file from disk.")) return;
    try {
      await adminDeleteNote(id);
      setNotes((prev) => prev.filter((n) => n.id !== id));
      toast.success("Note deleted");
    } catch {
      toast.error("Failed to delete note");
    }
  };

  const handleReindex = async () => {
    setReindexing(true);
    try {
      const res = await adminReindex();
      toast.success(`Reindexed ${res.indexed} notes`);
      fetchNotes();
    } catch {
      toast.error("Reindex failed");
    } finally {
      setReindexing(false);
    }
  };

  const handleLogout = () => {
    clearToken();
    router.replace("/admin/login");
  };

  return (
    <div>
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: "2rem",
          flexWrap: "wrap",
          gap: "12px",
        }}
      >
        <h1
          style={{
            fontSize: "1.75rem",
            fontWeight: 800,
            color: "var(--text-primary)",
            margin: 0,
          }}
        >
          Admin Dashboard
        </h1>

        <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
          <button
            onClick={handleReindex}
            disabled={reindexing}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "6px",
              padding: "8px 16px",
              background: "var(--bg-secondary)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius)",
              fontSize: "0.85rem",
              color: "var(--text-secondary)",
              cursor: reindexing ? "not-allowed" : "pointer",
              fontWeight: 500,
            }}
          >
            <RefreshCw size={14} style={{ animation: reindexing ? "spin 1s linear infinite" : "none" }} />
            {reindexing ? "Reindexing..." : "Reindex"}
          </button>

          <Link href="/admin/editor">
            <button
              style={{
                display: "flex",
                alignItems: "center",
                gap: "6px",
                padding: "8px 16px",
                background: "var(--accent)",
                border: "none",
                borderRadius: "var(--radius)",
                fontSize: "0.85rem",
                color: "#fff",
                cursor: "pointer",
                fontWeight: 600,
              }}
            >
              <Plus size={14} />
              New Note
            </button>
          </Link>

          <button
            onClick={handleLogout}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "6px",
              padding: "8px 14px",
              background: "none",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius)",
              fontSize: "0.85rem",
              color: "var(--text-secondary)",
              cursor: "pointer",
            }}
          >
            <LogOut size={14} />
            Logout
          </button>
        </div>
      </div>

      {/* Stats */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
          gap: "1rem",
          marginBottom: "2rem",
        }}
      >
        {[
          { label: "Total Notes", value: notes.length },
          { label: "Published", value: notes.filter((n) => n.visibility === "public").length },
          { label: "Drafts", value: notes.filter((n) => n.visibility !== "public").length },
        ].map((stat) => (
          <div
            key={stat.label}
            style={{
              background: "var(--bg-card)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius)",
              padding: "1.25rem",
            }}
          >
            <div
              style={{
                fontSize: "2rem",
                fontWeight: 800,
                color: "var(--accent)",
                lineHeight: 1,
              }}
            >
              {stat.value}
            </div>
            <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "4px" }}>
              {stat.label}
            </div>
          </div>
        ))}
      </div>

      {/* Notes table */}
      {loading ? (
        <div style={{ textAlign: "center", color: "var(--text-muted)", padding: "3rem" }}>
          Loading notes...
        </div>
      ) : (
        <div
          style={{
            background: "var(--bg-card)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius)",
            overflow: "hidden",
          }}
        >
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.875rem" }}>
            <thead>
              <tr style={{ background: "var(--bg-secondary)" }}>
                {["Title", "Section", "Tags", "Status", "Updated", "Actions"].map((h) => (
                  <th
                    key={h}
                    style={{
                      padding: "12px 16px",
                      textAlign: "left",
                      fontWeight: 600,
                      color: "var(--text-secondary)",
                      borderBottom: "1px solid var(--border)",
                      fontSize: "0.8rem",
                      textTransform: "uppercase",
                      letterSpacing: "0.05em",
                    }}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {notes.map((note, i) => (
                <tr
                  key={note.id}
                  style={{
                    background: i % 2 === 0 ? "transparent" : "var(--bg-secondary)",
                    borderBottom: "1px solid var(--border)",
                  }}
                >
                  <td style={{ padding: "12px 16px", color: "var(--text-primary)", fontWeight: 500 }}>
                    <Link
                      href={`/${note.section_slug}/${note.slug}`}
                      target="_blank"
                      style={{ color: "inherit", textDecoration: "none" }}
                    >
                      {note.title}
                    </Link>
                  </td>
                  <td style={{ padding: "12px 16px", color: "var(--text-secondary)" }}>
                    {note.section_slug}
                  </td>
                  <td style={{ padding: "12px 16px" }}>
                    <div style={{ display: "flex", gap: "4px", flexWrap: "wrap" }}>
                      {note.tags.slice(0, 3).map((tag) => (
                        <span
                          key={tag}
                          style={{
                            background: "var(--tag-bg)",
                            color: "var(--tag-text)",
                            borderRadius: "4px",
                            padding: "2px 7px",
                            fontSize: "0.7rem",
                          }}
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td style={{ padding: "12px 16px" }}>
                    <span
                      style={{
                        background: note.visibility === "public" ? "#dcfce7" : "#fef9c3",
                        color: note.visibility === "public" ? "#166534" : "#854d0e",
                        borderRadius: "4px",
                        padding: "2px 8px",
                        fontSize: "0.75rem",
                        fontWeight: 600,
                      }}
                    >
                      {note.visibility}
                    </span>
                  </td>
                  <td style={{ padding: "12px 16px", color: "var(--text-muted)", fontSize: "0.8rem" }}>
                    {note.updated_at
                      ? new Date(note.updated_at).toLocaleDateString()
                      : "—"}
                  </td>
                  <td style={{ padding: "12px 16px" }}>
                    <div style={{ display: "flex", gap: "8px" }}>
                      <Link href={`/admin/editor/${note.id}`}>
                        <button
                          style={{
                            background: "none",
                            border: "1px solid var(--border)",
                            borderRadius: "4px",
                            padding: "4px 8px",
                            cursor: "pointer",
                            color: "var(--accent)",
                            display: "flex",
                            alignItems: "center",
                          }}
                        >
                          <Pencil size={12} />
                        </button>
                      </Link>
                      <button
                        onClick={() => handleDelete(note.id)}
                        style={{
                          background: "none",
                          border: "1px solid #fecaca",
                          borderRadius: "4px",
                          padding: "4px 8px",
                          cursor: "pointer",
                          color: "#ef4444",
                          display: "flex",
                          alignItems: "center",
                        }}
                      >
                        <Trash2 size={12} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
              {notes.length === 0 && (
                <tr>
                  <td
                    colSpan={6}
                    style={{
                      padding: "3rem",
                      textAlign: "center",
                      color: "var(--text-muted)",
                    }}
                  >
                    No notes yet.{" "}
                    <Link href="/admin/editor" style={{ color: "var(--accent)" }}>
                      Create your first note
                    </Link>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}
