"use client";

import Link from "next/link";
import { useState } from "react";
import { ThemeToggle } from "@/components/ui/ThemeToggle";
import { SearchBar } from "@/components/search/SearchBar";
import { SearchModal } from "@/components/search/SearchModal";
import type { Section } from "@/types";

interface Props {
  sections: Section[];
}

export function Navbar({ sections }: Props) {
  const [searchOpen, setSearchOpen] = useState(false);

  return (
    <>
      <nav
        style={{
          background: "var(--nav-bg)",
          position: "sticky",
          top: 0,
          zIndex: 100,
          boxShadow: "0 1px 3px rgba(0,0,0,0.3)",
        }}
      >
        <div
          style={{
            maxWidth: "1400px",
            margin: "0 auto",
            padding: "0 1.5rem",
            display: "flex",
            alignItems: "center",
            gap: "1rem",
            height: "56px",
          }}
        >
          {/* Logo */}
          <Link
            href="/"
            style={{
              color: "var(--nav-text)",
              fontWeight: 700,
              fontSize: "1.1rem",
              textDecoration: "none",
              flexShrink: 0,
              letterSpacing: "-0.02em",
            }}
          >
            Knowledge Hub
          </Link>

          {/* Section links — scrollable */}
          <div
            style={{
              flex: 1,
              display: "flex",
              gap: "4px",
              overflowX: "auto",
              scrollbarWidth: "none",
              msOverflowStyle: "none",
            }}
          >
            {sections.map((s) => (
              <Link
                key={s.slug}
                href={`/${s.slug}`}
                style={{
                  color: "var(--nav-text)",
                  textDecoration: "none",
                  padding: "6px 12px",
                  borderRadius: "6px",
                  fontSize: "0.85rem",
                  whiteSpace: "nowrap",
                  flexShrink: 0,
                  transition: "background 0.15s ease",
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                }}
                onMouseOver={(e) =>
                  ((e.currentTarget as HTMLAnchorElement).style.background =
                    "var(--nav-hover)")
                }
                onMouseOut={(e) =>
                  ((e.currentTarget as HTMLAnchorElement).style.background =
                    "transparent")
                }
              >
                {s.title}
                {s.note_count > 0 && (
                  <span
                    style={{
                      background: "rgba(255,255,255,0.15)",
                      borderRadius: "10px",
                      padding: "1px 6px",
                      fontSize: "0.7rem",
                    }}
                  >
                    {s.note_count}
                  </span>
                )}
              </Link>
            ))}
          </div>

          {/* Right controls */}
          <div style={{ display: "flex", alignItems: "center", gap: "8px", flexShrink: 0 }}>
            <SearchBar onOpen={() => setSearchOpen(true)} />
            <ThemeToggle />
          </div>
        </div>
      </nav>

      <SearchModal open={searchOpen} onClose={() => setSearchOpen(false)} />
    </>
  );
}
