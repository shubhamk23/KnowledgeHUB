# AI Notes Knowledge Hub

A full-stack AI/ML knowledge base application with 31 curated notes across 8 technical sections — featuring full-text search, admin panel, dark mode, and KaTeX math rendering.

---

## Features

- **31 curated AI/ML notes** across 8 sections (NLP, ML Concepts, RecSys, AI Models, ML Infrastructure, Vision, Multimodal, Talks)
- **Full-text search** powered by SQLite FTS5
- **KaTeX math rendering** for formulas and equations
- **Syntax-highlighted code blocks** with copy-to-clipboard
- **Dark / light mode** toggle
- **Admin panel** for content management and re-indexing
- **Auto-indexing** — drop a `.md` file into `backend/content/` and the backend indexes it on startup
- **REST API** with interactive Swagger docs at `/docs`
- **113 passing pytest tests** with 50%+ coverage

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 15 (App Router) + TypeScript + Tailwind CSS v4 |
| Backend | FastAPI + Python 3.11+ |
| Database | SQLite (aiosqlite) + FTS5 full-text search |
| Auth | JWT (python-jose) + bcrypt (passlib) |
| Testing | pytest + pytest-asyncio + httpx |
| Content | Markdown files with YAML frontmatter |

---

## Project Structure

```
AI Notes Project/
├── Makefile                    # Developer commands (test, start, install)
├── start.sh                    # Start both services
├── README.md
│
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI app entrypoint
│   │   ├── config.py           # Settings (pydantic-settings)
│   │   ├── database.py         # Async SQLite + FTS5 setup
│   │   ├── indexer.py          # Markdown → SQLite indexer
│   │   ├── models.py           # SQLAlchemy ORM models
│   │   ├── schemas.py          # Pydantic request/response schemas
│   │   ├── auth.py             # JWT auth helpers
│   │   ├── markdown_utils.py   # YAML frontmatter parsing + MD rendering
│   │   └── routers/
│   │       ├── notes.py        # GET /api/notes, /api/sections
│   │       ├── search.py       # GET /api/search?q=...
│   │       └── admin.py        # POST /api/admin/reindex (protected)
│   │
│   ├── content/                # Markdown knowledge base
│   │   ├── nlp/                # 6 notes
│   │   ├── ml-concepts/        # 5 notes
│   │   ├── recsys/             # 5 notes
│   │   ├── ai-models/          # 10 notes
│   │   ├── ml-infra/           # 2 notes
│   │   ├── vision/             # 1 note
│   │   ├── multimodal/         # 1 note
│   │   └── talks/              # 1 note
│   │
│   ├── tests/                  # 113 pytest tests
│   ├── requirements.txt
│   └── pyproject.toml
│
└── frontend/
    ├── src/
    │   ├── app/                # Next.js App Router pages
    │   │   ├── page.tsx        # Homepage — section cards
    │   │   ├── [section]/      # Section listing page
    │   │   ├── [section]/[slug]/ # Individual note page
    │   │   ├── search/         # Search results page
    │   │   └── admin/          # Admin panel
    │   ├── components/
    │   │   ├── notes/          # NoteCard, SectionCard, NoteContent
    │   │   ├── layout/         # Navbar, Sidebar, Footer
    │   │   ├── search/         # SearchBar component
    │   │   └── ui/             # Shared UI primitives
    │   ├── lib/                # API client, fetchers
    │   ├── store/              # Zustand state (theme, etc.)
    │   └── types/              # TypeScript types
    └── package.json
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ / npm

### 1. Install dependencies

```bash
make install
```

Or manually:

```bash
# Backend
cd backend && pip install -r requirements.txt

# Frontend
cd frontend && npm install
```

### 2. Set up environment

```bash
# Copy example env (edit values as needed)
cp backend/.env.example backend/.env
```

Default `.env` values:

```env
SECRET_KEY=your-secret-key-change-in-production
ADMIN_USERNAME=admin
ADMIN_PASSWORD=changeme123
CONTENT_DIR=content
DATABASE_URL=sqlite+aiosqlite:///./knowledge_hub.db
```

### 3. Start both services

```bash
make start
# or
bash start.sh
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Admin Panel | http://localhost:3000/admin |

---

## Content Format

Each note is a Markdown file with YAML frontmatter placed in the appropriate `backend/content/<section>/` folder:

```markdown
---
title: "Attention Mechanism"
slug: attention
summary: "Deep dive into Bahdanau, Luong, self-attention, and multi-head attention."
tags: ["attention", "transformer", "self-attention", "multi-head"]
visibility: public
---

# Attention Mechanism

## Overview
...

## Formula

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
```

Each section folder contains a `_section.json` descriptor:

```json
{
  "title": "NLP",
  "description": "Natural Language Processing concepts and architectures.",
  "icon": "💬",
  "sort_order": 1
}
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check + note count |
| `GET` | `/api/sections` | List all sections |
| `GET` | `/api/sections/{section}` | Notes in a section |
| `GET` | `/api/notes/{section}/{slug}` | Single note with rendered HTML |
| `GET` | `/api/search?q={query}` | Full-text search |
| `POST` | `/api/admin/reindex` | Re-index content (auth required) |
| `POST` | `/api/auth/token` | Get JWT token |

### Example requests

```bash
# Health check
curl http://localhost:8000/api/health

# Search
curl "http://localhost:8000/api/search?q=attention"

# Get a note
curl http://localhost:8000/api/notes/nlp/attention

# Re-index (requires admin token)
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/token \
  -d "username=admin&password=changeme123" | jq -r .access_token)
curl -X POST http://localhost:8000/api/admin/reindex \
  -H "Authorization: Bearer $TOKEN"
```

---

## Knowledge Base Contents

### NLP (6 notes)
- Attention Mechanism — Bahdanau, Luong, self-attention, multi-head, Linformer
- Transformer Architecture — encoder/decoder, positional encoding, RoPE, KV cache
- Transformers Overview — architecture fundamentals
- Parameter-Efficient Fine-Tuning — LoRA, QLoRA, Adapters, Prefix Tuning
- Retrieval-Augmented Generation (RAG) — indexing, retrieval, generation
- LLM Hallucination — types, causes, detection, mitigation

### ML Concepts (5 notes)
- Loss Functions — cross-entropy, focal, KL divergence, triplet, contrastive
- Activation Functions — ReLU, GELU, SiLU, sigmoid, tanh
- LLM Alignment — RLHF, DPO, Constitutional AI
- Token Sampling — greedy, beam search, top-k, top-p, temperature

### Recommendation Systems (5 notes)
- RecSys Introduction — collaborative filtering, content-based, matrix factorization
- Candidate Generation — two-tower models, ANN, embedding retrieval
- Ranking & Scoring — pointwise, pairwise, listwise LTR
- Evaluation Metrics — NDCG, MAP, Precision@k, AUC
- Cold Start Problem — strategies for new users and items

### AI Models (10 notes)
- LLM Overview — scaling laws, emergent abilities, few-shot learning
- BERT — bidirectional pre-training, MLM, NSP, fine-tuning
- LLaMA — open foundation models, efficient architecture
- CLIP — contrastive language-image pre-training, zero-shot
- Diffusion Models — DDPM, score matching, stable diffusion
- Claude — Constitutional AI, long context, tool use, safety
- GPT Series — GPT-1 through o3, RLHF, reasoning, multimodal
- Gemini — multimodal architecture, Gemini 1.5/2.0 Ultra/Flash
- DeepSeek — MLA attention, aux-loss-free MoE, GRPO, R1 reasoning
- Mixture of Experts — sparse routing, Switch Transformer, Mixtral, DeepSeek-V3

### ML Infrastructure (2 notes)
- Docker for ML — GPU containers, multi-stage builds, vLLM serving
- MLflow — experiment tracking, model registry, deployment

### Vision (1 note)
- Computer Vision Overview — CNNs, ViTs, object detection, segmentation

### Multimodal (1 note)
- Multimodal ML Introduction — VLMs, LLaVA, BLIP-2, Flamingo, GPT-4V

### Talks (1 note)
- Rise of Modern AI Agents — agentic frameworks, tool use, planning

---

## Testing

```bash
# Full test suite (unit + integration + e2e + coverage)
make test

# Unit tests only (fast)
make test-unit

# Integration tests only
make test-int

# End-to-end tests only
make test-e2e

# Open HTML coverage report
make coverage
```

Test results: **113 tests, ~50% coverage** (threshold: 48%)

---

## Adding New Notes

1. Create a `.md` file in the appropriate `backend/content/<section>/` folder
2. Add YAML frontmatter (title, slug, summary, tags, visibility)
3. Restart the backend — it auto-indexes on startup, **or** call the admin reindex endpoint

---

## License

MIT
