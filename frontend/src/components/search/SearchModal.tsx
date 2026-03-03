"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Search, X, ArrowRight } from "lucide-react";
import { searchNotes } from "@/lib/api";
import type { SearchResult } from "@/types";

interface Props {
  open: boolean;
  onClose: () => void;
}

export function SearchModal({ open, onClose }: Props) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();

  useEffect(() => {
    if (open) {
      setTimeout(() => inputRef.current?.focus(), 50);
      setQuery("");
      setResults([]);
    }
  }, [open]);

  useEffect(() => {
    if (!query.trim()) {
      setResults([]);
      return;
    }
    const timer = setTimeout(async () => {
      setLoading(true);
      try {
        const res = await searchNotes(query);
        setResults(res.results.slice(0, 10));
        setSelected(0);
      } catch {
        setResults([]);
      } finally {
        setLoading(false);
      }
    }, 300);
    return () => clearTimeout(timer);
  }, [query]);

  const navigate = (result: SearchResult) => {
    router.push(`/${result.section_slug}/${result.slug}`);
    onClose();
  };

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") onClose();
    if (e.key === "ArrowDown") setSelected((s) => Math.min(s + 1, results.length - 1));
    if (e.key === "ArrowUp") setSelected((s) => Math.max(s - 1, 0));
    if (e.key === "Enter" && results[selected]) navigate(results[selected]);
  };

  if (!open) return null;

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 1000,
        display: "flex",
        alignItems: "flex-start",
        justifyContent: "center",
        paddingTop: "10vh",
        background: "rgba(0,0,0,0.5)",
        backdropFilter: "blur(4px)",
      }}
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        style={{
          width: "100%",
          maxWidth: "580px",
          background: "var(--bg-card)",
          borderRadius: "12px",
          border: "1px solid var(--border)",
          boxShadow: "0 20px 40px rgba(0,0,0,0.3)",
          overflow: "hidden",
        }}
        onKeyDown={handleKey}
      >
        {/* Input */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "12px",
            padding: "16px 20px",
            borderBottom: "1px solid var(--border)",
          }}
        >
          <Search size={18} style={{ color: "var(--text-muted)", flexShrink: 0 }} />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search notes, topics, tags..."
            style={{
              flex: 1,
              background: "transparent",
              border: "none",
              outline: "none",
              fontSize: "1rem",
              color: "var(--text-primary)",
            }}
          />
          <button
            onClick={onClose}
            style={{
              background: "none",
              border: "none",
              cursor: "pointer",
              color: "var(--text-muted)",
              padding: "2px",
            }}
          >
            <X size={16} />
          </button>
        </div>

        {/* Results */}
        <div style={{ maxHeight: "420px", overflowY: "auto" }}>
          {loading && (
            <div
              style={{
                padding: "24px",
                textAlign: "center",
                color: "var(--text-muted)",
                fontSize: "0.875rem",
              }}
            >
              Searching...
            </div>
          )}
          {!loading && query && results.length === 0 && (
            <div
              style={{
                padding: "24px",
                textAlign: "center",
                color: "var(--text-muted)",
                fontSize: "0.875rem",
              }}
            >
              No results for &quot;{query}&quot;
            </div>
          )}
          {!loading && !query && (
            <div
              style={{
                padding: "24px",
                textAlign: "center",
                color: "var(--text-muted)",
                fontSize: "0.875rem",
              }}
            >
              Type to search across all notes...
            </div>
          )}
          {results.map((r, i) => (
            <button
              key={r.id}
              onClick={() => navigate(r)}
              style={{
                width: "100%",
                display: "flex",
                alignItems: "flex-start",
                gap: "12px",
                padding: "14px 20px",
                background: i === selected ? "var(--accent-light)" : "transparent",
                border: "none",
                borderBottom: "1px solid var(--border)",
                cursor: "pointer",
                textAlign: "left",
                transition: "background 0.1s ease",
              }}
              onMouseEnter={() => setSelected(i)}
            >
              <div style={{ flex: 1, minWidth: 0 }}>
                <div
                  style={{
                    fontWeight: 600,
                    fontSize: "0.9rem",
                    color: "var(--text-primary)",
                    marginBottom: "4px",
                  }}
                >
                  {r.title}
                </div>
                <div
                  style={{
                    fontSize: "0.8rem",
                    color: "var(--text-secondary)",
                    overflow: "hidden",
                    display: "-webkit-box",
                    WebkitLineClamp: 2,
                    WebkitBoxOrient: "vertical",
                  }}
                  dangerouslySetInnerHTML={{ __html: r.excerpt }}
                />
                <div
                  style={{
                    fontSize: "0.75rem",
                    color: "var(--accent)",
                    marginTop: "4px",
                    textTransform: "uppercase",
                    letterSpacing: "0.05em",
                  }}
                >
                  {r.section_slug}
                </div>
              </div>
              <ArrowRight
                size={14}
                style={{ color: "var(--text-muted)", flexShrink: 0, marginTop: "4px" }}
              />
            </button>
          ))}
        </div>

        {/* Footer */}
        <div
          style={{
            display: "flex",
            gap: "16px",
            padding: "10px 20px",
            borderTop: "1px solid var(--border)",
            fontSize: "0.75rem",
            color: "var(--text-muted)",
          }}
        >
          <span>↑↓ navigate</span>
          <span>↵ select</span>
          <span>esc close</span>
        </div>
      </div>
    </div>
  );
}
