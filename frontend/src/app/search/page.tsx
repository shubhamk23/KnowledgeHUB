"use client";

import { useSearchParams, useRouter } from "next/navigation";
import { useEffect, useState, Suspense } from "react";
import { searchNotes } from "@/lib/api";
import type { SearchResult } from "@/types";
import Link from "next/link";
import { Search, ArrowRight } from "lucide-react";

function SearchContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const [query, setQuery] = useState(searchParams.get("q") ?? "");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const q = searchParams.get("q") ?? "";
    setQuery(q);
    if (!q) return;

    setLoading(true);
    searchNotes(q)
      .then((res) => {
        setResults(res.results);
        setTotal(res.total);
      })
      .catch(() => setResults([]))
      .finally(() => setLoading(false));
  }, [searchParams]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      router.push(`/search?q=${encodeURIComponent(query.trim())}`);
    }
  };

  return (
    <div style={{ maxWidth: "800px", margin: "0 auto", padding: "2.5rem 1.5rem" }}>
      <h1
        style={{
          fontSize: "1.75rem",
          fontWeight: 800,
          color: "var(--text-primary)",
          marginBottom: "1.5rem",
        }}
      >
        Search Notes
      </h1>

      {/* Search form */}
      <form onSubmit={handleSubmit} style={{ marginBottom: "2rem" }}>
        <div
          style={{
            display: "flex",
            gap: "12px",
          }}
        >
          <div
            style={{
              flex: 1,
              display: "flex",
              alignItems: "center",
              gap: "10px",
              background: "var(--bg-card)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius)",
              padding: "10px 16px",
            }}
          >
            <Search size={18} style={{ color: "var(--text-muted)", flexShrink: 0 }} />
            <input
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
              autoFocus
            />
          </div>
          <button
            type="submit"
            style={{
              background: "var(--accent)",
              color: "#fff",
              border: "none",
              borderRadius: "var(--radius)",
              padding: "10px 24px",
              fontWeight: 600,
              cursor: "pointer",
              fontSize: "0.9rem",
            }}
          >
            Search
          </button>
        </div>
      </form>

      {/* Results */}
      {loading && (
        <div style={{ textAlign: "center", color: "var(--text-muted)", padding: "3rem" }}>
          Searching...
        </div>
      )}

      {!loading && searchParams.get("q") && results.length === 0 && (
        <div style={{ textAlign: "center", color: "var(--text-muted)", padding: "3rem" }}>
          No results for &quot;{searchParams.get("q")}&quot;
        </div>
      )}

      {!loading && results.length > 0 && (
        <>
          <p style={{ fontSize: "0.875rem", color: "var(--text-muted)", marginBottom: "1rem" }}>
            {total} result{total !== 1 ? "s" : ""} for &quot;{searchParams.get("q")}&quot;
          </p>
          <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
            {results.map((r) => (
              <Link
                key={r.id}
                href={`/${r.section_slug}/${r.slug}`}
                style={{ textDecoration: "none" }}
              >
                <div
                  style={{
                    background: "var(--bg-card)",
                    border: "1px solid var(--border)",
                    borderRadius: "var(--radius)",
                    padding: "1.25rem",
                    display: "flex",
                    alignItems: "flex-start",
                    gap: "12px",
                    transition: "border-color 0.15s ease",
                    cursor: "pointer",
                  }}
                  onMouseOver={(e) =>
                    ((e.currentTarget as HTMLElement).style.borderColor = "var(--accent)")
                  }
                  onMouseOut={(e) =>
                    ((e.currentTarget as HTMLElement).style.borderColor = "var(--border)")
                  }
                >
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div
                      style={{
                        fontSize: "0.75rem",
                        color: "var(--accent)",
                        fontWeight: 600,
                        textTransform: "uppercase",
                        letterSpacing: "0.05em",
                        marginBottom: "4px",
                      }}
                    >
                      {r.section_slug}
                    </div>
                    <h3
                      style={{
                        margin: "0 0 0.5rem",
                        fontSize: "1rem",
                        fontWeight: 600,
                        color: "var(--text-primary)",
                      }}
                    >
                      {r.title}
                    </h3>
                    <p
                      style={{
                        margin: 0,
                        fontSize: "0.85rem",
                        color: "var(--text-secondary)",
                        lineHeight: 1.5,
                      }}
                      dangerouslySetInnerHTML={{ __html: r.excerpt }}
                    />
                    {r.tags.length > 0 && (
                      <div style={{ display: "flex", gap: "6px", flexWrap: "wrap", marginTop: "8px" }}>
                        {r.tags.map((tag) => (
                          <span
                            key={tag}
                            style={{
                              background: "var(--tag-bg)",
                              color: "var(--tag-text)",
                              border: "1px solid var(--border)",
                              borderRadius: "4px",
                              padding: "2px 7px",
                              fontSize: "0.7rem",
                            }}
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                  <ArrowRight size={16} style={{ color: "var(--text-muted)", flexShrink: 0, marginTop: "4px" }} />
                </div>
              </Link>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

export default function SearchPage() {
  return (
    <Suspense fallback={<div style={{ padding: "3rem", textAlign: "center", color: "var(--text-muted)" }}>Loading...</div>}>
      <SearchContent />
    </Suspense>
  );
}
