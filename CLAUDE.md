# CLAUDE.md

Guidance for Claude Code in this repo. **Product name:** Employee Handbook Q&A System — hybrid RAG (Chroma + structured employee data + LangGraph routing + post-graph contracts). Human-oriented overview: [README.md](README.md). **Pipeline and routing detail:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Project overview

- Ingest handbook/fixtures into **Chroma**; ask flows through **`rag-api/app/rag_graph.py`** (`StateGraph`: refiner → guardrail → **router** → retrieve/grade and/or **balance** → **generate**).
- **`intent_policy.classify_query`** + **`router_node`** heuristics/safety nets choose **policy** / **personal** / **mixed** / **general** / **clarify**; optional **`semantic_router.semantic_rescue_route`** when route would stay **general** with RAG on.
- **`main._enforce_response_contract`** after `build_ask_response_from_state`: no personal leakage on policy-only answers; PROFILE missing-profile hard error **only if** request had **`employee_id`** (preserves “select profile” copy when ID omitted).
- **Semantic cache** (`semantic_cache.py`): key includes normalized question, **classifier route bucket**, `employee_id`, `use_rag`, `kb_signature` — purge or bump `_CACHE_KEY_VERSION` when changing routing/contracts during dev.
- **Client:** `HandbookQA` uses **`POST /api/ask/stream`**; passes `employee_id` from `App.tsx` picker (`GET /api/employees`).

## Development commands

**Monorepo**

```bash
npm install && npm run dev
# UI :5173 | API :3001 (Vite proxies /api via RAG_API_URL in root .env)
```

**Backend only**

```bash
cd rag-api
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt   # Windows; .venv/bin/ on macOS/Linux
.venv\Scripts\uvicorn.exe app.main:app --reload --port 3001
```

**Frontend only:** `cd client && npm install && npm run dev`

**Tests:** `npm run test:js` · `npm test` (pytest in `rag-api` if `.venv` exists)

**Build:** `npm run build -w client` · `npm run build -w shared`

**Bootstrap KB:** `curl -X POST http://127.0.0.1:3001/api/bootstrap`

## Architecture (short)

```
LangGraph: query_refiner -> guardrail -> router
  -> POLICY/MIXED: retrieve -> grade_documents -> generate -> END
  -> PERSONAL: balance -> generate -> END
  -> GENERAL / clarify: generate or clarify -> END
FastAPI post-invoke: build_ask_response_from_state -> _enforce_response_contract -> [cache read/write]
```

**Ask / ingest / cache:** `POST /api/ask`, `/api/ask/stream`, `/api/ingest`, `/api/ingest/upload`, `/api/bootstrap`, `GET /api/employees`, `GET /api/health`, cache `stats` / `purge` / `viz` — see `rag-api/app/main.py`.

**SSE events:** `run_start`, `node_start`, `node_end`, `text`, `done` (final payload), `error`.

## Environment

Single root **`.env`** from `.env.example` — loaded by rag-api, Vite `envDir`, scripts, MCP. Important: `OPENAI_API_KEY`, `OPENAI_EMBEDDING_MODEL`, `OPENAI_CHAT_MODEL`, `CHROMA_PERSIST_DIR`, `CHROMA_COLLECTION`, `DEFAULT_HANDBOOK_PATH`, `ALLOWED_INGEST_ROOT`, `EMPLOYEES_JSON_PATH`, `RAG_CORS_ORIGINS`, `RAG_API_PORT`, `RAG_API_URL`, `VITE_API_BASE_URL`. Optional semantic rescue: `SEMANTIC_ROUTER_MODEL`, `SEMANTIC_ROUTER_THRESHOLD`, `SEMANTIC_ROUTER_LOCAL_ONLY`.

**`RAG_API_URL`** = Vite dev proxy target for `/api`. **`VITE_API_BASE_URL`** = optional browser-direct API origin (CORS).

## Code standards

TypeScript strict (no `any`); React hooks; thin FastAPI handlers; orchestration in graph/modules; Zod in `shared/` aligned with API models where used; prefer additive changes.

## Repository layout

`client/` · `shared/` · `rag-api/` (`main.py`, `rag_graph.py`, `intent_policy.py`, `semantic_router.py`, `semantic_cache.py`, ingest, vectorstore) · `fixtures/` · `mcp-hr-server/` · `docs/`.
