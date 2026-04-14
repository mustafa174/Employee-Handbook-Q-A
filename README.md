# Employee Handbook Q&A

AI-powered handbook assistant for employees and HR teams. It ingests policy documents into Chroma, runs a LangGraph-style RAG pipeline with guardrails and optional employee context, and returns grounded answers with citations, escalation signals, cache metadata, and live pipeline visualization.

## Architecture

- **Backend:** FastAPI (Python) + LangChain/LangGraph orchestration
- **Frontend:** React + Vite + Tailwind + TanStack Query
- **Shared Contracts:** Zod schemas in `shared/`
- **LLM + Embeddings:** OpenAI chat + embedding models
- **Vector DB:** ChromaDB
- **Cache:** Semantic answer cache with knowledge-base signature invalidation
- **Streaming:** SSE (`/api/ask/stream`) via FastAPI `StreamingResponse`
- **MCP Demo:** `mcp-hr-server` (`get_leave_balance`) over `fixtures/employees.json`

## Core Capabilities

- Ingest `.md`/`.txt` handbook content into one vector collection
- Ask policy and employee-context questions with citations
- Sensitive-topic guardrails with escalation path
- Semantic cache hit/miss with stale-answer prevention when handbook changes
- System status checks and runtime visual diagnostics
- Live RAG pipeline node flow in UI

## Quick Start

### Prerequisites

- Node.js 20+
- Python 3.11+ (recommended)
- OpenAI API key

### 1) Configure environment

Create root `.env` from `.env.example` and set at minimum:

```bash
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### 2) Install dependencies

```bash
npm install
cd rag-api
python -m venv .venv
# Windows
.venv\Scripts\pip install -r requirements.txt
# macOS/Linux
# .venv/bin/pip install -r requirements.txt
```

### 3) Run app (frontend + rag-api)

From repo root:

```bash
npm run dev
```

- Frontend: `http://localhost:5173`
- API: `http://127.0.0.1:3001`

## Local Development

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

## Tests & Build

```bash
# JS tests (shared + client)
npm run test:js

# Full tests (includes pytest when rag-api/.venv exists)
npm test

# Build shared + client
npm run build
```

## API Endpoints

- `GET /api/health` - API liveness and model metadata
- `POST /api/ingest` - re-index knowledge base from allowed path
- `POST /api/ingest/upload` - upload and ingest `.md`/`.txt`
- `POST /api/bootstrap` - ingest default fixture set
- `POST /api/ask` - standard non-streamed answer
- `POST /api/ask/stream` - streamed node/text events + final response
- `GET /api/cache/stats` - semantic cache count
- `DELETE /api/cache/purge` - purge cache
- `GET /api/cache/viz` - vector-space points for visualizer

## RAG Pipeline (Conceptual)

1. `query` - normalize/prepare request
2. `guardrail` - detect sensitive or restricted topics
3. `router` - intent routing (policy/personal/general)
4. `chroma` - retrieve handbook chunks when needed
5. `mcp` - optional employee context/tool path
6. `synthesis` - build grounded answer draft
7. `judge` - verify quality/grounding
8. `output` - final structured response

## Streaming (SSE)

`POST /api/ask/stream` emits SSE events consumed by the client:

- `run_start`
- `node_start`
- `node_end`
- `text`
- `done`
- `error`

This powers real-time response rendering and live node highlighting in the pipeline visualizer.

## Project Structure

```text
client/
  src/
    components/      # Chat UI, cache panel, pipeline visualizer
    pages/           # Settings and route-level pages
    state/           # Chat history persistence/context

shared/
  src/               # Shared Zod schemas and TS types

rag-api/
  app/
    main.py          # FastAPI routes
    rag_graph.py     # Orchestration graph logic
    semantic_cache.py# cache + viz point generation
    vectorstore.py   # Chroma initialization/access
    handbook_ingest.py

fixtures/            # handbook + employee demo data
mcp-hr-server/       # MCP server demo for leave-balance tool
docs/                # architecture docs
```

## Environment Variables

Common variables:

- `OPENAI_API_KEY`
- `OPENAI_CHAT_MODEL`
- `OPENAI_EMBEDDING_MODEL`
- `CHROMA_PERSIST_DIR`
- `CHROMA_COLLECTION`
- `DEFAULT_HANDBOOK_PATH`
- `ALLOWED_INGEST_ROOT`
- `EMPLOYEES_JSON_PATH`
- `RAG_CORS_ORIGINS`
- `RAG_API_PORT`
- `RAG_API_URL`
- `VITE_API_BASE_URL`

## Notes

- Cache invalidation is tied to fixture content signature; handbook changes naturally prevent stale cache hits.
- Chat history is persisted client-side (24h window), and can be cleared from Settings.
- See `docs/ARCHITECTURE.md` and `CLAUDE.md` for deeper implementation details.
# Employee Handbook Q&A

Demo monorepo: **RAG** over a bilingual employee handbook (**Chroma** + **OpenAI** embeddings), **LangGraph** orchestration in **FastAPI** (`rag-api/`), a **Vite/React** client with ingest, chat, and citations, **shared Zod** contracts, and an optional **`mcp-hr-server`** MCP tool (`get_leave_balance`) backed by `fixtures/employees.json`.

## Prerequisites

- Node.js **20+**, npm **9+**
- Python **3.12+** (3.14 works locally; CI uses 3.12)
- `OPENAI_API_KEY` in the **repository root** `.env` (see [`.env.example`](.env.example))

## Quick start

```bash
copy .env.example .env   # Windows; use cp on macOS/Linux — repository root, then set OPENAI_API_KEY

cd rag-api
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cd ..

npm install
npm run dev
```

Open **http://localhost:5173**. The UI uses Vite’s **`/api` proxy**; the target comes from the root **`.env`** variable **`RAG_API_URL`** (default **http://127.0.0.1:3001**).

Use **Ingest** (path or upload) to index `fixtures/handbook.md` (or your `.md`/`.txt`), then ask questions in **Chat**. Responses include **citations** and may **escalate** when policy is unclear.

## Scripts (repository root)

| Script | Description |
|--------|-------------|
| `npm run dev` | Vite client + **uvicorn** on `:3001` (`scripts/uvicorn-rag.cjs`) |
| `npm run build` | `shared` → `client` |
| `npm test` | Vitest in **shared** + **client**, then **pytest** in `rag-api` if `rag-api/.venv` exists |
| `npm run test:js` | JS workspaces only |

## MCP HR server (optional)

From `mcp-hr-server/`:

```bash
npm install
npm start
```

Registers **`get_leave_balance`** reading **`fixtures/employees.json`** (override with **`EMPLOYEES_JSON_PATH`**).

## Documentation

- [CLAUDE.md](CLAUDE.md) — commands, env vars, API summary, agent context.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — components and data flow.
- [docs/BUILD_LOG.md](docs/BUILD_LOG.md) — progress log template.
- [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) — demo outline.

## Environment

Use a **single file** at the **repository root**: **`.env`**. Copy from [`.env.example`](.env.example). It is loaded by **rag-api** (`app/config.py`), **Vite** (`client/vite.config.ts` `envDir`), **`scripts/uvicorn-rag.cjs`** / **`scripts/pytest-rag.cjs`** (Node `dotenv`), and **`mcp-hr-server`** on startup.

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Embeddings + chat (required for ingest/ask) |
| `OPENAI_BASE_URL`, `OPENAI_ORGANIZATION` | Optional OpenAI / Azure-style settings |
| `OPENAI_EMBEDDING_MODEL`, `OPENAI_CHAT_MODEL` | Model overrides |
| `CHROMA_PERSIST_DIR`, `CHROMA_COLLECTION` | Chroma path (relative to `rag-api/`) and collection |
| `DEFAULT_HANDBOOK_PATH`, `EMPLOYEES_JSON_PATH`, `ALLOWED_INGEST_ROOT` | Paths relative to monorepo root unless absolute |
| `RAG_CORS_ORIGINS` | Comma-separated browser origins for FastAPI CORS |
| `RAG_API_URL` | Vite **dev server** proxy target for `/api` (not sent to the browser) |
| `VITE_API_BASE_URL` | Optional: browser calls this origin + `/api/...` directly (skips proxy); needs CORS |
| `RAG_API_PORT` | Uvicorn port in `scripts/uvicorn-rag.cjs` (default `3001`) |

## License

Private / evaluation use unless otherwise stated.
