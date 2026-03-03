import { getSections } from "@/lib/api";
import { SectionCard } from "@/components/notes/SectionCard";
import type { Section } from "@/types";

export const revalidate = 60;

export default async function HomePage() {
  let sections: Section[] = [];
  try {
    sections = await getSections();
  } catch {
    // API not available
  }

  const totalNotes = sections.reduce((acc, s) => acc + s.note_count, 0);

  return (
    <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "3rem 1.5rem" }}>
      <div style={{ textAlign: "center", marginBottom: "3.5rem" }}>
        <h1
          style={{
            fontSize: "clamp(2rem, 5vw, 3rem)",
            fontWeight: 800,
            color: "var(--text-primary)",
            marginBottom: "1rem",
            letterSpacing: "-0.03em",
            lineHeight: 1.2,
          }}
        >
          Personal Knowledge Hub
        </h1>
        <p
          style={{
            fontSize: "1.1rem",
            color: "var(--text-secondary)",
            maxWidth: "580px",
            margin: "0 auto 2rem",
            lineHeight: 1.7,
          }}
        >
          Curated notes on AI/ML, research paper reviews, and technical articles.
          Bite-sized information that is easy to understand.
        </p>
        <div style={{ display: "flex", justifyContent: "center", gap: "12px", flexWrap: "wrap" }}>
          <div
            style={{
              background: "var(--accent-light)",
              color: "var(--accent)",
              borderRadius: "20px",
              padding: "6px 16px",
              fontSize: "0.85rem",
              fontWeight: 600,
            }}
          >
            {totalNotes} Notes
          </div>
          <div
            style={{
              background: "var(--accent-light)",
              color: "var(--accent)",
              borderRadius: "20px",
              padding: "6px 16px",
              fontSize: "0.85rem",
              fontWeight: 600,
            }}
          >
            {sections.length} Sections
          </div>
        </div>
      </div>

      {sections.length > 0 ? (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
            gap: "1.25rem",
          }}
        >
          {sections.map((s) => (
            <SectionCard key={s.slug} section={s} />
          ))}
        </div>
      ) : (
        <div style={{ textAlign: "center", color: "var(--text-muted)", padding: "4rem" }}>
          <p>No sections yet. Start the backend and add some notes!</p>
        </div>
      )}
    </div>
  );
}
