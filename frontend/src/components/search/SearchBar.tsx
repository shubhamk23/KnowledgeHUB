"use client";

import { Search } from "lucide-react";
import { useEffect } from "react";

interface Props {
  onOpen: () => void;
}

export function SearchBar({ onOpen }: Props) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        onOpen();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onOpen]);

  return (
    <button
      onClick={onOpen}
      aria-label="Open search (Cmd+K)"
      style={{
        display: "flex",
        alignItems: "center",
        gap: "8px",
        background: "var(--nav-hover)",
        border: "none",
        borderRadius: "6px",
        padding: "6px 12px",
        cursor: "pointer",
        color: "var(--text-muted)",
        fontSize: "0.875rem",
        transition: "background 0.15s ease",
        minWidth: "180px",
      }}
    >
      <Search size={14} />
      <span style={{ flex: 1, textAlign: "left" }}>Search notes...</span>
      <kbd
        style={{
          background: "rgba(255,255,255,0.1)",
          border: "1px solid rgba(255,255,255,0.15)",
          borderRadius: "4px",
          padding: "1px 5px",
          fontSize: "0.7rem",
          color: "var(--text-muted)",
        }}
      >
        ⌘K
      </kbd>
    </button>
  );
}
