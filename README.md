# Employee Handbook Q&A

AI-powered handbook assistant for employees and HR teams. It ingests policy documents into **Chroma**, runs a **LangGraph** `StateGraph` in **FastAPI** (`rag-api/`) with guardrails, **intent-based routing** (policy vs personal vs mixed), optional **employee profile** context from fixtures, and returns grounded answers with citations, escalation signals, **semantic cache** metadata, and live pipeline visualization in the **Vite/React** client.

## Architecture

- **Backend:** FastAPI + LangGraph-style graph (`rag_graph.py`)
- **Frontend:** React + Vite + Tailwind + TanStack Query
- **Shared contracts:** Zod schemas in `shared/`
- **LLM + embeddings:** OpenAI chat + embedding models
- **Vector DB:** ChromaDB
- **Intent:** `intent_policy.py` (domain classification); **semantic rescue** optional via `semantic_router.py` (sentence-transformers)
- **Cache:** Semantic answer cache keyed by question + route bucket + `employee_id` + `use_rag`, invalidated by fixture **signature** when the knowledge base changes
- **Streaming:** SSE `POST /api/ask/stream` via `StreamingResponse`
- **MCP demo:** `mcp-hr-server` (`get_leave_balance`) over `fixtures/employees.json` (aligns with `EMPLOYEES_JSON_PATH` in the API)

## Core capabilities

- Ingest `.md`/`.txt` handbook content (and related fixtures) into one vector collection
- Route **policy** questions through Chroma retrieval + grounded generation; route **personal** balance/profile questions through the **balance** node (no handbook RAG on pure personal)
- **Mixed** prompts can combine profile numbers with policy retrieval
- Sensitive-topic guardrails with escalation
- Semantic cache hit/miss; purge and viz endpoints for debugging and demos
- System status, ingest, and cache controls in **Settings**
- Live RAG pipeline node flow in the UI (stream-driven)

## Prerequisites

- Node.js **20+**, npm **9+**
- Python **3.12+** (3.14 works locally; pin CI to your team standard)
- **`OPENAI_API_KEY`** in the **repository root** `.env` (see [`.env.example`](.env.example))

## Quick start

```bash
# Repository root — copy env template (Windows: copy; macOS/Linux: cp)
copy .env.example .env
# Edit .env and set OPENAI_API_KEY

cd rag-api
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cd ..

npm install
npm run dev
```

- **UI:** http://localhost:5173 (landing at `/`, assistant at `/assistant`, settings at `/settings`)
- **API:** http://127.0.0.1:3001 (Vite proxies `/api` using root `.env` → **`RAG_API_URL`**, default `http://127.0.0.1:3001`)

Use **Settings** or **bootstrap** to ingest `fixtures/handbook.md` (or your allowed `.md`/`.txt`), select an **employee** in the header when testing leave balances, then ask questions in **Assistant**.

## Scripts (repository root)

| Script | Description |
|--------|-------------|
| `npm run dev` | Vite client + uvicorn on `:3001` (`scripts/uvicorn-rag.cjs`) |
| `npm run build` | `shared` → `client` |
| `npm test` | Vitest in **shared** + **client**, then **pytest** in `rag-api` if `rag-api/.venv` exists |
| `npm run test:js` | JS workspaces only |

## Local development (split processes)

### Backend only

```bash
cd rag-api
# Windows
.venv\Scripts\uvicorn.exe app.main:app --reload --port 3001
# macOS/Linux
# .venv/bin/uvicorn app.main:app --reload --port 3001
```

### Frontend only

```bash
cd client
npm run dev
```

## Tests and build

```bash
npm run test:js
npm test
npm run build
```

## API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/health` | Liveness and model metadata |
| `GET` | `/api/employees` | Employee picker (`employee_id`, `name`) |
| `POST` | `/api/ingest` | Re-index from allowed path |
| `POST` | `/api/ingest/upload` | Upload and ingest `.md`/`.txt` |
| `POST` | `/api/bootstrap` | Ingest default fixture set |
| `POST` | `/api/ask` | Non-streamed answer + pipeline steps |
| `POST` | `/api/ask/stream` | SSE: nodes + text chunks + `done` with final payload |
| `GET` | `/api/cache/stats` | Semantic cache stats |
| `DELETE` | `/api/cache/purge` | Purge cached answers |
| `GET` | `/api/cache/viz` | Points for cache scatter visualizer |

Request body for ask endpoints includes **`question`**, optional **`employee_id`** (required for personal leave/PTO *balances*), optional **`chat_history`**, **`use_rag`**, and **`skip_cache`**.

## RAG pipeline (actual graph)

1. **`query_refiner`** — Normalize query (e.g. vacation → PTO), optional multi-part retrieval queries
2. **`guardrail`** — Sensitive / restricted topics
3. **`router`** — Intent + execution route; safety nets so “how many vacation/PTO days” with an employee stays **personal** when appropriate; optional semantic rescue from `semantic_router.py`
4. **`retrieve`** + **`grade_documents`** — Chroma search and grading (policy / mixed)
5. **`balance`** — Load employee profile snippet when route is **personal** (or as needed for mixed)
6. **`generate`** — Policy synthesis with citations, or deterministic personal field answers
7. **`clarify`** — Clarification short-circuit when routed

After the graph, **`main._enforce_response_contract`** applies API-level invariants (e.g. no personal balance leakage on pure policy answers; profile-missing handling respects whether **`employee_id`** was sent).

## Streaming (SSE)

`POST /api/ask/stream` emits: `run_start`, `node_start`, `node_end`, `text`, `done` (includes final structured payload), `error`. The client updates the streaming answer and pipeline visualizer from these events.

## Project structure

```text
client/
  src/
    components/      # HandbookQA, cache panel, pipeline visualizer
    pages/           # Landing, Settings
    state/           # Chat history + selected employee id

shared/
  src/               # Zod schemas and TS types

rag-api/
  app/
    main.py              # FastAPI routes, response contract, SSE
    rag_graph.py         # LangGraph state machine + nodes
    intent_policy.py     # Query → domain / confidence
    semantic_router.py   # Optional embedding rescue routing
    semantic_cache.py    # Answer cache + viz points
    vectorstore.py
    handbook_ingest.py

fixtures/            # handbook + employees demo data
mcp-hr-server/       # MCP demo for leave balance
docs/                # architecture docs
```

## Environment

Use **one** file at the **repository root**: **`.env`**. Copy from [`.env.example`](.env.example). Loaded by **rag-api** (`app/config.py`), **Vite** (`client/vite.config.ts` `envDir`), **`scripts/uvicorn-rag.cjs`** / **`scripts/pytest-rag.cjs`**, and **`mcp-hr-server`**.

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Embeddings + chat (required for ingest/ask) |
| `OPENAI_BASE_URL`, `OPENAI_ORGANIZATION` | Optional OpenAI / Azure-style settings |
| `OPENAI_EMBEDDING_MODEL`, `OPENAI_CHAT_MODEL` | Model overrides |
| `CHROMA_PERSIST_DIR`, `CHROMA_COLLECTION` | Chroma path (relative to `rag-api/`) and collection |
| `DEFAULT_HANDBOOK_PATH`, `EMPLOYEES_JSON_PATH`, `ALLOWED_INGEST_ROOT` | Paths relative to monorepo root unless absolute |
| `RAG_CORS_ORIGINS` | Comma-separated browser origins for FastAPI CORS |
| `RAG_API_URL` | Vite **dev server** proxy target for `/api` |
| `VITE_API_BASE_URL` | Optional: browser calls this origin + `/api/...` directly (needs CORS) |
| `RAG_API_PORT` | Uvicorn port in `scripts/uvicorn-rag.cjs` (default `3001`) |
| `SEMANTIC_ROUTER_MODEL`, `SEMANTIC_ROUTER_THRESHOLD`, `SEMANTIC_ROUTER_LOCAL_ONLY` | Optional semantic rescue tuning (`semantic_router.py`) |

## MCP HR server (optional)

From `mcp-hr-server/`:

```bash
npm install
npm start
```

Registers **`get_leave_balance`** using **`fixtures/employees.json`** (override with **`EMPLOYEES_JSON_PATH`**).

## Documentation

- [CLAUDE.md](CLAUDE.md) — commands, env vars, graph and API details for agents
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — components and data flow
- [docs/BUILD_LOG.md](docs/BUILD_LOG.md) — progress log template
- [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) — demo outline

## Notes

- Purge semantic cache from **Settings** (or `DELETE /api/cache/purge`) if you change routing behavior and want to avoid stale cached answers during QA.
- Chat history and selected employee are persisted client-side (see `client/src/state/ChatHistoryContext.tsx`).

## License

Private / evaluation use unless otherwise stated.
