"use client";

import { useEffect, useState } from "react";
import type { TOCItem } from "@/types";

interface Props {
  toc: TOCItem[];
}

export function Sidebar({ toc }: Props) {
  const [activeId, setActiveId] = useState<string>("");

  useEffect(() => {
    const headings = document.querySelectorAll(".prose h1, .prose h2, .prose h3, .prose h4");
    if (!headings.length) return;

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setActiveId(entry.target.id);
            break;
          }
        }
      },
      { rootMargin: "-80px 0px -60% 0px", threshold: 0 }
    );

    headings.forEach((h) => observer.observe(h));
    return () => observer.disconnect();
  }, []);

  if (!toc.length) return null;

  return (
    <aside
      style={{
        position: "sticky",
        top: "5rem",
        maxHeight: "calc(100vh - 6rem)",
        overflowY: "auto",
        width: "240px",
        flexShrink: 0,
      }}
    >
      <div
        style={{
          fontSize: "0.7rem",
          fontWeight: 700,
          textTransform: "uppercase",
          letterSpacing: "0.1em",
          color: "var(--text-muted)",
          marginBottom: "0.75rem",
          paddingLeft: "0.5rem",
        }}
      >
        On this page
      </div>
      <TOCList items={toc} activeId={activeId} depth={0} />
    </aside>
  );
}

function TOCList({
  items,
  activeId,
  depth,
}: {
  items: TOCItem[];
  activeId: string;
  depth: number;
}) {
  return (
    <ul style={{ listStyle: "none", margin: 0, padding: 0 }}>
      {items.map((item) => (
        <li key={item.id}>
          <a
            href={`#${item.id}`}
            style={{
              display: "block",
              padding: "4px 8px",
              paddingLeft: `${8 + depth * 12}px`,
              borderRadius: "4px",
              fontSize: "0.8rem",
              textDecoration: "none",
              lineHeight: 1.5,
              transition: "all 0.1s ease",
              color:
                activeId === item.id ? "var(--accent)" : "var(--text-secondary)",
              background:
                activeId === item.id ? "var(--accent-light)" : "transparent",
              fontWeight: activeId === item.id ? 600 : 400,
              borderLeft: activeId === item.id ? `2px solid var(--accent)` : "2px solid transparent",
              marginBottom: "2px",
            }}
            onClick={(e) => {
              e.preventDefault();
              document.getElementById(item.id)?.scrollIntoView({ behavior: "smooth" });
            }}
          >
            {item.text}
          </a>
          {item.children.length > 0 && (
            <TOCList items={item.children} activeId={activeId} depth={depth + 1} />
          )}
        </li>
      ))}
    </ul>
  );
}
