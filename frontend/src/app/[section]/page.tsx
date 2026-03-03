import { notFound } from "next/navigation";
import Link from "next/link";
import { getSection } from "@/lib/api";
import { NoteCard } from "@/components/notes/NoteCard";
import type { Metadata } from "next";

interface Props {
  params: Promise<{ section: string }>;
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { section } = await params;
  try {
    const data = await getSection(section);
    return {
      title: data.section.title,
      description: data.section.description ?? undefined,
    };
  } catch {
    return { title: section };
  }
}

export default async function SectionPage({ params }: Props) {
  const { section: sectionSlug } = await params;

  let data: Awaited<ReturnType<typeof getSection>>;
  try {
    data = await getSection(sectionSlug);
  } catch {
    notFound();
  }

  const { section, notes } = data!;

  return (
    <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "2.5rem 1.5rem" }}>
      {/* Breadcrumb */}
      <div
        style={{
          fontSize: "0.8rem",
          color: "var(--text-muted)",
          marginBottom: "1.5rem",
          display: "flex",
          alignItems: "center",
          gap: "6px",
        }}
      >
        <Link href="/" style={{ color: "var(--accent)", textDecoration: "none" }}>
          Home
        </Link>
        <span>/</span>
        <span>{section.title}</span>
      </div>

      {/* Header */}
      <div style={{ marginBottom: "2.5rem" }}>
        <h1
          style={{
            fontSize: "2rem",
            fontWeight: 800,
            color: "var(--text-primary)",
            margin: "0 0 0.75rem",
            letterSpacing: "-0.02em",
          }}
        >
          {section.title}
        </h1>
        {section.description && (
          <p
            style={{
              fontSize: "1rem",
              color: "var(--text-secondary)",
              margin: 0,
              maxWidth: "65ch",
              lineHeight: 1.6,
            }}
          >
            {section.description}
          </p>
        )}
        <div
          style={{
            marginTop: "1rem",
            fontSize: "0.85rem",
            color: "var(--text-muted)",
          }}
        >
          {notes.length} {notes.length === 1 ? "note" : "notes"}
        </div>
      </div>

      {/* Notes grid */}
      {notes.length > 0 ? (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))",
            gap: "1rem",
          }}
        >
          {notes.map((note) => (
            <NoteCard key={note.id} note={note} />
          ))}
        </div>
      ) : (
        <div
          style={{
            textAlign: "center",
            color: "var(--text-muted)",
            padding: "4rem",
          }}
        >
          <p>No notes in this section yet.</p>
        </div>
      )}
    </div>
  );
}
