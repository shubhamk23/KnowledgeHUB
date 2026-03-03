"use client";

import Link from "next/link";
import type { NoteCard as NoteCardType } from "@/types";

interface Props {
  note: NoteCardType;
}

export function NoteCard({ note }: Props) {
  const date = new Date(note.created_at).toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });

  return (
    <Link
      href={`/${note.section_slug}/${note.slug}`}
      style={{ textDecoration: "none" }}
    >
      <article
        style={{
          background: "var(--bg-card)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius)",
          padding: "1.25rem",
          height: "100%",
          transition: "border-color 0.15s ease, box-shadow 0.15s ease",
          cursor: "pointer",
        }}
        onMouseOver={(e) => {
          (e.currentTarget as HTMLElement).style.borderColor = "var(--accent)";
          (e.currentTarget as HTMLElement).style.boxShadow = "var(--shadow-md)";
        }}
        onMouseOut={(e) => {
          (e.currentTarget as HTMLElement).style.borderColor = "var(--border)";
          (e.currentTarget as HTMLElement).style.boxShadow = "none";
        }}
      >
        <h3
          style={{
            margin: "0 0 0.5rem",
            fontSize: "1rem",
            fontWeight: 600,
            color: "var(--text-primary)",
            lineHeight: 1.4,
          }}
        >
          {note.title}
        </h3>
        {note.summary && (
          <p
            style={{
              margin: "0 0 0.75rem",
              fontSize: "0.85rem",
              color: "var(--text-secondary)",
              lineHeight: 1.5,
              display: "-webkit-box",
              WebkitLineClamp: 2,
              WebkitBoxOrient: "vertical",
              overflow: "hidden",
            }}
          >
            {note.summary}
          </p>
        )}

        {note.tags.length > 0 && (
          <div style={{ display: "flex", flexWrap: "wrap", gap: "4px", marginBottom: "0.75rem" }}>
            {note.tags.slice(0, 4).map((tag) => (
              <span
                key={tag}
                style={{
                  background: "var(--tag-bg)",
                  color: "var(--tag-text)",
                  border: "1px solid var(--border)",
                  borderRadius: "4px",
                  padding: "2px 8px",
                  fontSize: "0.7rem",
                  fontWeight: 500,
                }}
              >
                {tag}
              </span>
            ))}
          </div>
        )}

        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "12px",
            fontSize: "0.75rem",
            color: "var(--text-muted)",
          }}
        >
          <span>{date}</span>
          <span>&bull;</span>
          <span>{note.read_time} min read</span>
        </div>
      </article>
    </Link>
  );
}
