import type {
  AuthToken,
  NoteAdminDetail,
  NoteCard,
  NoteCreatePayload,
  NoteDetail,
  NoteUpdatePayload,
  SearchResponse,
  Section,
} from "@/types";

const BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api";

async function apiFetch<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...opts,
    headers: {
      "Content-Type": "application/json",
      ...opts?.headers,
    },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

function adminHeaders(): Record<string, string> {
  if (typeof window === "undefined") return {};
  const token = localStorage.getItem("admin_token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

// ── Public ───────────────────────────────────────────────────

export const getSections = () => apiFetch<Section[]>("/sections");

export const getSection = (slug: string) =>
  apiFetch<{ section: Section; notes: NoteCard[] }>(`/sections/${slug}`);

export const getNote = (section: string, slug: string) =>
  apiFetch<NoteDetail>(`/notes/${section}/${slug}`);

export const searchNotes = (q: string, limit = 20) =>
  apiFetch<SearchResponse>(
    `/search?q=${encodeURIComponent(q)}&limit=${limit}`
  );

export const getHealth = () =>
  apiFetch<{ status: string; version: string; note_count: number }>("/health");

// ── Auth ─────────────────────────────────────────────────────

export const login = (username: string, password: string) =>
  apiFetch<AuthToken>("/auth/token", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });

// ── Admin ─────────────────────────────────────────────────────

export const adminGetNotes = () =>
  apiFetch<NoteAdminDetail[]>("/admin/notes", {
    headers: adminHeaders(),
  });

export const adminGetNote = (id: number) =>
  apiFetch<NoteAdminDetail>(`/admin/notes/${id}`, {
    headers: adminHeaders(),
  });

export const adminCreateNote = (data: NoteCreatePayload) =>
  apiFetch<NoteAdminDetail>("/admin/notes", {
    method: "POST",
    body: JSON.stringify(data),
    headers: adminHeaders(),
  });

export const adminUpdateNote = (id: number, data: NoteUpdatePayload) =>
  apiFetch<NoteAdminDetail>(`/admin/notes/${id}`, {
    method: "PUT",
    body: JSON.stringify(data),
    headers: adminHeaders(),
  });

export const adminDeleteNote = (id: number) =>
  apiFetch<{ deleted: boolean }>(`/admin/notes/${id}`, {
    method: "DELETE",
    headers: adminHeaders(),
  });

export const adminReindex = () =>
  apiFetch<{ indexed: number; errors: string[] }>("/admin/reindex", {
    method: "POST",
    headers: adminHeaders(),
  });

export const adminGetSections = () =>
  apiFetch<Section[]>("/admin/sections", {
    headers: adminHeaders(),
  });
