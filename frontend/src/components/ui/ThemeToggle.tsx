"use client";

import { useTheme } from "next-themes";
import { Sun, Moon } from "lucide-react";
import { useEffect, useState } from "react";

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => setMounted(true), []);

  if (!mounted) {
    return <div className="w-8 h-8" />;
  }

  return (
    <button
      onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
      aria-label="Toggle theme"
      style={{
        background: "var(--nav-hover)",
        border: "none",
        borderRadius: "6px",
        padding: "6px",
        cursor: "pointer",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: "var(--nav-text)",
        transition: "background 0.15s ease",
      }}
      onMouseOver={(e) =>
        ((e.currentTarget as HTMLButtonElement).style.background = "#475569")
      }
      onMouseOut={(e) =>
        ((e.currentTarget as HTMLButtonElement).style.background =
          "var(--nav-hover)")
      }
    >
      {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
    </button>
  );
}
