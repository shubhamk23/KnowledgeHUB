export interface Section {
  id: number;
  slug: string;
  title: string;
  description: string | null;
  icon: string | null;
  sort_order: number;
  note_count: number;
}

export interface NoteCard {
  id: number;
  slug: string;
  section_slug: string;
  title: string;
  summary: string | null;
  tags: string[];
  read_time: number;
  created_at: string;
  updated_at: string | null;
}

export interface NoteDetail extends NoteCard {
  content: string;
  visibility: string;
  word_count: number;
}

export interface NoteAdminDetail extends NoteDetail {
  file_path: string;
}

export interface SearchResult {
  id: number;
  slug: string;
  section_slug: string;
  title: string;
  excerpt: string;
  tags: string[];
}

export interface SearchResponse {
  results: SearchResult[];
  total: number;
  query: string;
}

export interface AuthToken {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export interface TOCItem {
  id: string;
  text: string;
  level: number;
  children: TOCItem[];
}

export interface NoteCreatePayload {
  title: string;
  section_slug: string;
  content: string;
  tags: string[];
  visibility: string;
  slug?: string;
}

export interface NoteUpdatePayload extends Partial<NoteCreatePayload> {}
