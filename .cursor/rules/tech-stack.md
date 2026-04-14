# Tech stack — Employee Handbook Q&A

## Client (`/client`)

- **React 19** — functional components and hooks; avoid legacy class components.
- **Vite** — ESM, `import.meta.env` for client env.
- **TypeScript** — `strict: true`, no `any`; narrow `unknown` at boundaries.
- **Tailwind CSS v4** — `@tailwindcss/vite`, `@import "tailwindcss"` in CSS; utility-first layout.
- **TanStack Query** — server state for health, ingest, and ask; avoid duplicating fetch logic in effects.
- **Lucide React** — icons; keep size/stroke consistent.

## RAG API (`/rag-api`)

- **Python 3.12+** — virtualenv at `rag-api/.venv` (gitignored).
- **FastAPI** + **Uvicorn** — port **3001** in dev; CORS for Vite origin.
- **LangChain / LangGraph** — embeddings, retriever, graph orchestration in `app/rag_graph.py`.
- **Chroma** — local persistence; path configurable via env.

## Shared (`/shared`)

- **Zod** — single source for TS contracts (`@employee-handbook/shared`); mirror shapes in FastAPI Pydantic models.
- Export public API from `src/index.ts`.

## MCP (`/mcp-hr-server`)

- **TypeScript ESM** + **MCP SDK** — stdio server; **`get_leave_balance`** tool.

## General

- Prefer **arrow functions** for React components and small pure utilities.
- Run **`npm run build`** before releases; fix type errors rather than suppressing them.
- Do not commit secrets; use `.env` locally and document variables in **README** / **CLAUDE.md**.
