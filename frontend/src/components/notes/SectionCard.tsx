"use client";

import Link from "next/link";
import type { Section } from "@/types";

const SECTION_COLORS: Record<string, string> = {
  NLP: "#6366f1",
  ML: "#8b5cf6",
  RecSys: "#ec4899",
  Models: "#f59e0b",
  Infra: "#10b981",
  Vision: "#3b82f6",
  Multi: "#ef4444",
  Blog: "#14b8a6",
};

export function SectionCard({ section }: { section: Section }) {
  const color = SECTION_COLORS[section.icon ?? ""] ?? "#6366f1";

  return (
    <Link href={`/${section.slug}`} style={{ textDecoration: "none" }}>
      <div
        style={{
          background: "var(--bg-card)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius)",
          padding: "1.5rem",
          height: "100%",
          cursor: "pointer",
          transition: "border-color 0.15s ease, box-shadow 0.15s ease, transform 0.15s ease",
          position: "relative",
          overflow: "hidden",
        }}
        onMouseOver={(e) => {
          const el = e.currentTarget as HTMLElement;
          el.style.borderColor = color;
          el.style.boxShadow = `0 4px 20px ${color}20`;
          el.style.transform = "translateY(-2px)";
        }}
        onMouseOut={(e) => {
          const el = e.currentTarget as HTMLElement;
          el.style.borderColor = "var(--border)";
          el.style.boxShadow = "none";
          el.style.transform = "translateY(0)";
        }}
      >
        {/* Accent bar at top */}
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            height: "3px",
            background: color,
          }}
        />

        {/* Icon label */}
        <div
          style={{
            display: "inline-block",
            background: `${color}15`,
            color: color,
            borderRadius: "6px",
            padding: "4px 10px",
            fontSize: "0.75rem",
            fontWeight: 700,
            letterSpacing: "0.05em",
            marginBottom: "0.75rem",
          }}
        >
          {section.icon}
        </div>

        <h2
          style={{
            margin: "0 0 0.5rem",
            fontSize: "1.1rem",
            fontWeight: 700,
            color: "var(--text-primary)",
          }}
        >
          {section.title}
        </h2>

        {section.description && (
          <p
            style={{
              margin: "0 0 1rem",
              fontSize: "0.85rem",
              color: "var(--text-secondary)",
              lineHeight: 1.5,
              display: "-webkit-box",
              WebkitLineClamp: 2,
              WebkitBoxOrient: "vertical",
              overflow: "hidden",
            }}
          >
            {section.description}
          </p>
        )}

        <div
          style={{
            fontSize: "0.8rem",
            color: color,
            fontWeight: 600,
          }}
        >
          {section.note_count} {section.note_count === 1 ? "note" : "notes"}
        </div>
      </div>
    </Link>
  );
}
