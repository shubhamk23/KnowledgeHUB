import { notFound } from "next/navigation";
import Link from "next/link";
import { getNote } from "@/lib/api";
import { renderMarkdown } from "@/lib/markdown";
import { extractTOC } from "@/lib/toc";
import { NoteRenderer } from "@/components/notes/NoteRenderer";
import { Sidebar } from "@/components/layout/Sidebar";
import type { Metadata } from "next";

interface Props {
  params: Promise<{ section: string; slug: string }>;
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { section, slug } = await params;
  try {
    const note = await getNote(section, slug);
    return {
      title: note.title,
      description: note.summary ?? undefined,
    };
  } catch {
    return { title: "Note" };
  }
}

export default async function NotePage({ params }: Props) {
  const { section: sectionSlug, slug } = await params;

  let note: Awaited<ReturnType<typeof getNote>>;
  try {
    note = await getNote(sectionSlug, slug);
  } catch {
    notFound();
  }

  const html = await renderMarkdown(note!.content);
  const toc = extractTOC(html);

  const date = new Date(note!.created_at).toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });

  return (
    <div style={{ maxWidth: "1400px", margin: "0 auto", padding: "2.5rem 1.5rem" }}>
      {/* Breadcrumb */}
      <div
        style={{
          fontSize: "0.8rem",
          color: "var(--text-muted)",
          marginBottom: "2rem",
          display: "flex",
          alignItems: "center",
          gap: "6px",
        }}
      >
        <Link href="/" style={{ color: "var(--accent)", textDecoration: "none" }}>
          Home
        </Link>
        <span>/</span>
        <Link
          href={`/${sectionSlug}`}
          style={{ color: "var(--accent)", textDecoration: "none" }}
        >
          {sectionSlug}
        </Link>
        <span>/</span>
        <span>{note!.title}</span>
      </div>

      <div style={{ display: "flex", gap: "3rem", alignItems: "flex-start" }}>
        {/* Main content */}
        <article style={{ flex: 1, minWidth: 0 }}>
          {/* Header */}
          <header style={{ marginBottom: "2rem" }}>
            <h1
              style={{
                fontSize: "clamp(1.75rem, 4vw, 2.5rem)",
                fontWeight: 800,
                color: "var(--text-primary)",
                margin: "0 0 1rem",
                letterSpacing: "-0.02em",
                lineHeight: 1.2,
              }}
            >
              {note!.title}
            </h1>

            {note!.summary && (
              <p
                style={{
                  fontSize: "1.05rem",
                  color: "var(--text-secondary)",
                  margin: "0 0 1rem",
                  lineHeight: 1.6,
                }}
              >
                {note!.summary}
              </p>
            )}

            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "16px",
                flexWrap: "wrap",
                fontSize: "0.825rem",
                color: "var(--text-muted)",
                paddingBottom: "1.5rem",
                borderBottom: "1px solid var(--border)",
              }}
            >
              <span>{date}</span>
              <span>&bull;</span>
              <span>{note!.read_time} min read</span>
              <span>&bull;</span>
              <span>{note!.word_count.toLocaleString()} words</span>

              {note!.tags.length > 0 && (
                <>
                  <span>&bull;</span>
                  <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
                    {note!.tags.map((tag) => (
                      <span
                        key={tag}
                        style={{
                          background: "var(--tag-bg)",
                          color: "var(--accent)",
                          border: "1px solid var(--border)",
                          borderRadius: "4px",
                          padding: "2px 8px",
                          fontSize: "0.75rem",
                          fontWeight: 500,
                        }}
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </>
              )}
            </div>
          </header>

          {/* Note content */}
          <NoteRenderer html={html} />
        </article>

        {/* TOC Sidebar */}
        {toc.length > 0 && (
          <div
            style={{ display: "none" }}
            className="toc-sidebar"
          >
            <Sidebar toc={toc} />
          </div>
        )}

        {/* Sidebar — visible on large screens */}
        {toc.length > 0 && (
          <div
            style={{
              width: "240px",
              flexShrink: 0,
            }}
            className="toc-visible"
          >
            <Sidebar toc={toc} />
          </div>
        )}
      </div>

      <style>{`
        @media (max-width: 1024px) {
          .toc-visible { display: none !important; }
        }
        @media (min-width: 1025px) {
          .toc-sidebar { display: none !important; }
        }
      `}</style>
    </div>
  );
}
