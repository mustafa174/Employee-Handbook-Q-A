# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Project Overview

Employee Handbook Q&A is an AI-powered assistant for internal HR/policy questions.
It ingests handbook files into Chroma, runs a LangGraph `StateGraph` in `rag-api/app/rag_graph.py` with guardrails and route-specific retrieval, and returns grounded answers with citations, optional employee profile context, escalation when needed, and semantic cache metadata.

Core goals:

- Ingest handbook/fixture documents into vector storage.
- Answer policy and employee-context questions through a structured graph (policy vs personal vs mixed vs general).
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
- `GET /api/employees` → Employee picker options (from `EMPLOYEES_JSON_PATH` / fixtures)
- `POST /api/ask` → Graph run + `_enforce_response_contract` + optional semantic cache read/write
- `POST /api/ask/stream` → Same pipeline via `graph.astream`, SSE for nodes + text + final payload
- `GET /api/cache/stats` / `DELETE /api/cache/purge` / `GET /api/cache/viz` → Semantic cache operations + 2D visualizer data

### Graph Pipeline (`rag-api/app/rag_graph.py`)

Compiled `StateGraph` (not a literal “synthesis/judge” split—the heavy lifting is in `node_generate` and related helpers).

Execution flow:

1. **`query_refiner`** — Normalize the question (`normalize_query`: e.g. vacation → PTO), optional multi-part split, set retrieval queries / `sub_questions`.
2. **`guardrail`** — Sensitive-topic detection; may short-circuit to escalation.
3. **`router`** — `router_node`: combines `intent_policy.classify_query`, lexical/heuristic overrides (loan limit, strong PTO policy phrases, explicit mixed queries), **leave/PTO safety net** when `employee_id` is present (do not let `INTENT_POLICY` alone overwrite `personal`/`mixed`), optional **`semantic_router.semantic_rescue_route`** for weak “general” asks when RAG is on, then sets `route`, `intent`, `use_rag`.
4. **Branch by `route`:**
   - **`policy` / `mixed`** → **`retrieve`** (Chroma) → **`grade_documents`** (may re-search) → **`generate`**
   - **`personal`** → **`balance`** (fixture/MCP-style employee profile + balances) → **`generate`**
   - **`general` / `clarify`** → **`generate`** or **`clarify`** then END (see graph edges in `build_graph`)

Personal route uses deterministic profile resolution where applicable; policy route uses retrieval + LLM with grounding gates.

### Intent & routing helpers

- **`rag-api/app/intent_policy.py`** — Domain classification (`POLICY`, `PROFILE`, `IT`, `OOS`), confidence, clarification hints.
- **`rag-api/app/semantic_router.py`** — Embedding similarity “rescue” to nudge borderline queries toward `personal` or `policy` (sentence-transformers; see env vars below).

### Semantic Cache (`rag-api/app/semantic_cache.py`)

- Telemetry window (`entries`) and full answer records (`answers`).
- Cache key includes normalized question text, **route bucket** from `classify_query`, **`employee_id`**, and **`use_rag`** (`_CACHE_KEY_VERSION` bumps when key shape changes).
- **`kb_signature`** from fixtures fingerprint invalidates stale cached answers when handbook/employees change.
- Visualizer: deterministic 2D projection; categories inferred from query text.

### Response contract (`rag-api/app/main.py`)

After the graph, **`_enforce_response_contract`** sanitizes responses (e.g. policy answers must not leak personal balance lines). For **`PROFILE`/`PERSONAL`** routes without `employee_profile`, the **“couldn’t retrieve your personal data”** replacement runs **only when the request included a non-empty `employee_id`**—so guidance like “select an employee profile” for balance questions without an ID is preserved.

### Frontend Routing & UI (`client/src`)

- **`App.tsx`** — Routes (`/`, `/assistant`, `/settings`), theme, employee dropdown (passes `selectedEmployeeId` into chat). Employee list from **`GET /api/employees`** with local fallbacks.
- **`components/HandbookQA.tsx`** — Naive vs RAG panes; RAG uses **`POST /api/ask/stream`**; sends `employee_id` when a profile is selected.
- **`components/RAGPipelineVisualizer.tsx`** — Live pipeline nodes from stream events.
- **`components/CachePanel.tsx`** — Cache stats + scatter visualization.
- **`pages/Settings.tsx`** — Health, ingest, cache purge, storage helpers.
- **`pages/LandingPage.tsx`** — Marketing/entry route at `/`.

## Streaming (SSE)

- Endpoint: **`POST /api/ask/stream`** (`StreamingResponse`).
- Event types: `run_start`, `node_start`, `node_end`, `text`, `done`, `error`; `done` carries the full final payload (same shape family as `/api/ask`).
- **`HandbookQA`** consumes SSE and updates incremental answer text and pipeline visualizer state.

## API Endpoints (`rag-api/app/main.py`)

- `GET /api/health`
- `GET /api/employees`
- `POST /api/ingest`
- `POST /api/ingest/upload`
- `POST /api/bootstrap`
- `POST /api/ask`
- `POST /api/ask/stream`
- `GET /api/cache/stats`
- `DELETE /api/cache/purge`
- `GET /api/cache/viz`

## Environment Variables

Use a **single root `.env`** (`.env.example` template). Loaded by the backend, Vite dev proxy, scripts, and MCP server.

Required/important:

- `OPENAI_API_KEY` — OpenAI auth (required for ingest/ask)
- `OPENAI_EMBEDDING_MODEL` — embedding model id
- `OPENAI_CHAT_MODEL` — generation model id
- `CHROMA_PERSIST_DIR` — Chroma persistence folder (under `rag-api/` if relative)
- `CHROMA_COLLECTION` — Chroma collection name
- `DEFAULT_HANDBOOK_PATH` — default ingest path
- `ALLOWED_INGEST_ROOT` — allowed ingest root
- `EMPLOYEES_JSON_PATH` — employee fixture for in-process profile / MCP alignment
- `RAG_CORS_ORIGINS` — allowed CORS origins
- `RAG_API_PORT` — API port (default 3001)
- `RAG_API_URL` — Vite dev proxy target
- `VITE_API_BASE_URL` — optional direct browser API origin

### Semantic router (optional tuning)

- `SEMANTIC_ROUTER_MODEL` — sentence-transformers model id (default `intfloat/e5-base-v2`)
- `SEMANTIC_ROUTER_THRESHOLD` — similarity threshold (default `0.58`)
- `SEMANTIC_ROUTER_LOCAL_ONLY` — `1`/`true` to avoid HF hub downloads where possible

### `RAG_API_URL` vs `VITE_API_BASE_URL`

- **`RAG_API_URL`**: dev proxy target used by the Vite server for `/api/*`.
- **`VITE_API_BASE_URL`**: optional absolute API origin for browser calls (requires CORS).

## Code Standards

- TypeScript strict mode; do not use `any`.
- React functional components + hooks.
- Keep FastAPI route handlers thin; keep orchestration in graph/service modules.
- Use Zod in `shared/` contracts and keep backend response shapes aligned where models exist.
- Prefer additive, non-destructive changes; do not remove unrelated user work.

## Repository Layout

- `client/` — Vite + React + Tailwind + TanStack Query
- `shared/` — shared Zod schemas/types (`@employee-handbook/shared`)
- `rag-api/` — FastAPI app, LangGraph orchestration, ingest, vectorstore, semantic cache, intent policy, semantic router
- `fixtures/` — handbook + employee sample data
- `mcp-hr-server/` — stdio MCP server for employee leave balance context
- `docs/` — architecture and system docs
