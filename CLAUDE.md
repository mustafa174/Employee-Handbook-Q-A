# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Employee Handbook Q&A is an AI-powered assistant for internal HR/policy questions.
It ingests handbook files into Chroma, runs a LangGraph-style RAG flow with guardrails and retrieval, and returns grounded answers with citations and escalation when needed.

Core goals:
- Ingest handbook/fixture documents into vector storage.
- Answer policy + employee-context questions through a structured graph.
- Surface pipeline status, cache behavior, and system health in the UI.

## Development Commands

### Quick Start

```bash
npm install
npm run dev
# Frontend: http://localhost:5173 | RAG API: http://localhost:3001
```

### Backend (`rag-api`)

```bash
cd rag-api
python -m venv .venv
# Windows
.venv\Scripts\pip install -r requirements.txt
# macOS/Linux
# .venv/bin/pip install -r requirements.txt
```

Run API manually:

```bash
cd rag-api
# Windows
.venv\Scripts\uvicorn.exe app.main:app --reload --port 3001
# macOS/Linux
# .venv/bin/uvicorn app.main:app --reload --port 3001
```

### Frontend (`client`)

```bash
cd client
npm install
npm run dev
```

### Tests

```bash
# JS workspaces (shared + client)
npm run test:js

# Full tests (includes pytest when rag-api/.venv exists)
npm test
```

### Linting & Type Checking

```bash
# Client type check + build
npm run build -w client

# Shared package checks
npm run build -w shared
```

### Ingest / Bootstrap

```bash
# Default knowledge-base bootstrap
curl -X POST http://127.0.0.1:3001/api/bootstrap
```

## Architecture

### Request Flow

- `POST /api/ingest` or `POST /api/ingest/upload` → Parse + chunk + embed + persist to Chroma
- `POST /api/ask` → Guardrail + route + retrieve/tool + synthesis/judge + structured response
- `POST /api/ask/stream` → SSE stream for live node activity + text chunks + final payload
- `GET /api/cache/stats` / `DELETE /api/cache/purge` / `GET /api/cache/viz` → Semantic cache operations + 2D visualizer data

### Graph Pipeline (`rag-api/app/rag_graph.py`)

The AI orchestration is implemented in `rag_graph.py` and compiled via `StateGraph`.

Primary execution stages:
1. `query` — normalize question + history context
2. `guardrail` — detect sensitive topics and decide escalation
3. `router` — classify intent (`policy`, `personal`, `general`)
4. `chroma` — retrieval path for handbook policy grounding
5. `mcp` — employee profile/tool context when applicable
6. `synthesis` — combine evidence/context into candidate answer
7. `judge` — quality/grounding checks and fallback handling
8. `output` — final structured response for API/UI

### Semantic Cache (`rag-api/app/semantic_cache.py`)

- Stores two windows:
  - Query telemetry entries (`entries`)
  - Full answer cache records (`answers`)
- Cache key includes normalized question + employee + RAG mode.
- Invalidation uses fixtures signature hash (`kb_signature`) so handbook updates naturally break stale hits.
- Visualizer points are deterministic 2D projections grouped into categories (`PTO`, `VPN`, `Personal`, `General`).

### Frontend Routing & UI (`client/src`)

- `App.tsx` controls top-level routes and persistent providers.
- `components/HandbookQA.tsx` hosts dual-pane Naive vs RAG chat experience.
- `components/RAGPipelineVisualizer.tsx` renders live graph state.
- `components/CachePanel.tsx` shows cache stats + vector-space scatter plot.
- `pages/Settings.tsx` provides health/system checks, ingest actions, and storage/cache reset actions.

## Streaming (SSE)

- Backend stream endpoint: `POST /api/ask/stream` (`FastAPI StreamingResponse`).
- Stream events include:
  - `run_start`
  - `node_start`
  - `node_end`
  - `text`
  - `done`
  - `error`
- Frontend stream consumer parses SSE frames in `HandbookQA` and updates:
  - incremental answer text
  - active/done pipeline nodes in the visualizer

## API Endpoints (`rag-api/app/main.py`)

- `GET /api/health`
- `POST /api/ingest`
- `POST /api/ingest/upload`
- `POST /api/bootstrap`
- `POST /api/ask`
- `POST /api/ask/stream`
- `GET /api/cache/stats`
- `DELETE /api/cache/purge`
- `GET /api/cache/viz`

## Environment Variables

Use a single root `.env` file (`.env.example` template). Loaded by backend, frontend dev proxy, scripts, and MCP server.

Required/important:

- `OPENAI_API_KEY` — OpenAI auth (required)
- `OPENAI_EMBEDDING_MODEL` — embedding model id
- `OPENAI_CHAT_MODEL` — generation model id
- `CHROMA_PERSIST_DIR` — Chroma persistence folder
- `CHROMA_COLLECTION` — Chroma collection name
- `DEFAULT_HANDBOOK_PATH` — default ingest path
- `ALLOWED_INGEST_ROOT` — allowed ingest root
- `EMPLOYEES_JSON_PATH` — employee fixture for MCP server
- `RAG_CORS_ORIGINS` — allowed CORS origins
- `RAG_API_PORT` — API port (default 3001)
- `RAG_API_URL` — Vite dev proxy target
- `VITE_API_BASE_URL` — optional direct browser API origin

### `RAG_API_URL` vs `VITE_API_BASE_URL`

- `RAG_API_URL`: dev proxy target used by Vite server.
- `VITE_API_BASE_URL`: optional absolute API base for browser direct calls (bypasses proxy).

## Code Standards

- TypeScript strict mode; do not use `any`.
- React functional components + hooks.
- Keep FastAPI route handlers thin; keep orchestration in graph/service modules.
- Use Zod in `shared/` contracts and keep backend response shapes aligned.
- Prefer additive, non-destructive changes; do not remove unrelated user work.

## Repository Layout

- `client/` — Vite + React + Tailwind + TanStack Query
- `shared/` — shared Zod schemas/types (`@employee-handbook/shared`)
- `rag-api/` — FastAPI app, graph orchestration, ingest, vectorstore, semantic cache
- `fixtures/` — handbook + employee sample data
- `mcp-hr-server/` — stdio MCP server for employee leave balance context
- `docs/` — architecture and system docs
