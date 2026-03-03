"use client";

import { useState, useEffect, useCallback } from "react";
import dynamic from "next/dynamic";
import { NoteRenderer } from "@/components/notes/NoteRenderer";
import { renderMarkdown } from "@/lib/markdown";

const CodeMirror = dynamic(() => import("@uiw/react-codemirror"), { ssr: false });

interface Props {
  value: string;
  onChange: (val: string) => void;
}

export function SplitEditor({ value, onChange }: Props) {
  const [html, setHtml] = useState("");
  const [view, setView] = useState<"split" | "editor" | "preview">("split");

  useEffect(() => {
    const timer = setTimeout(async () => {
      try {
        const rendered = await renderMarkdown(value);
        setHtml(rendered);
      } catch {
        setHtml("<p style='color:red'>Render error</p>");
      }
    }, 300);
    return () => clearTimeout(timer);
  }, [value]);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "calc(100vh - 280px)", minHeight: "500px" }}>
      {/* Toolbar */}
      <div
        style={{
          display: "flex",
          gap: "4px",
          marginBottom: "8px",
          alignItems: "center",
        }}
      >
        {(["split", "editor", "preview"] as const).map((v) => (
          <button
            key={v}
            onClick={() => setView(v)}
            style={{
              padding: "5px 14px",
              fontSize: "0.8rem",
              fontWeight: 500,
              border: "1px solid var(--border)",
              borderRadius: "4px",
              cursor: "pointer",
              background: view === v ? "var(--accent)" : "var(--bg-secondary)",
              color: view === v ? "#fff" : "var(--text-secondary)",
              textTransform: "capitalize",
            }}
          >
            {v}
          </button>
        ))}
        <span style={{ marginLeft: "auto", fontSize: "0.75rem", color: "var(--text-muted)" }}>
          {value.split(/\s+/).filter(Boolean).length} words
        </span>
      </div>

      {/* Editor panes */}
      <div
        style={{
          display: "flex",
          flex: 1,
          gap: "1rem",
          overflow: "hidden",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius)",
        }}
      >
        {/* Editor pane */}
        {view !== "preview" && (
          <div style={{ flex: 1, overflow: "auto", minWidth: 0 }}>
            <EditorPane value={value} onChange={onChange} />
          </div>
        )}

        {/* Divider */}
        {view === "split" && (
          <div
            style={{
              width: "1px",
              background: "var(--border)",
              flexShrink: 0,
            }}
          />
        )}

        {/* Preview pane */}
        {view !== "editor" && (
          <div
            style={{
              flex: 1,
              overflow: "auto",
              padding: "1.5rem",
              minWidth: 0,
              background: "var(--bg-primary)",
            }}
          >
            <NoteRenderer html={html} />
          </div>
        )}
      </div>
    </div>
  );
}

function EditorPane({ value, onChange }: Props) {
  const [extensions, setExtensions] = useState<any[]>([]);

  useEffect(() => {
    Promise.all([
      import("@codemirror/lang-markdown"),
      import("@codemirror/theme-one-dark"),
    ]).then(([{ markdown }, { oneDark }]) => {
      setExtensions([markdown(), oneDark]);
    });
  }, []);

  if (!extensions.length) {
    return (
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        style={{
          width: "100%",
          height: "100%",
          border: "none",
          outline: "none",
          padding: "1.25rem",
          background: "#1e293b",
          color: "#f1f5f9",
          fontFamily: "JetBrains Mono, Fira Code, Consolas, monospace",
          fontSize: "0.875rem",
          resize: "none",
          lineHeight: 1.6,
        }}
      />
    );
  }

  return (
    <CodeMirror
      value={value}
      onChange={onChange}
      extensions={extensions}
      height="100%"
      style={{ height: "100%", fontSize: "0.875rem" }}
      basicSetup={{
        lineNumbers: true,
        foldGutter: false,
        dropCursor: false,
        allowMultipleSelections: false,
        indentOnInput: true,
      }}
    />
  );
}
