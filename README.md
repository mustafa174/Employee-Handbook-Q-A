# Employee Handbook Q&A System

**Hybrid RAG:** Chroma handbook retrieval + structured employee data (fixtures / MCP demo) + **LangGraph** routing (policy · personal · mixed) + **`main.py` response contracts**. One request can mix **policy + personal**. Pipeline diagram and routing detail: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

## Quick start

```bash
copy .env.example .env   # set OPENAI_API_KEY
cd rag-api && python -m venv .venv && .venv\Scripts\pip install -r requirements.txt  # Windows; use bin/ on Unix
cd .. && npm install && npm run dev
```

- UI: http://localhost:5173 (`/`, `/assistant`, `/settings`) · API: http://127.0.0.1:3001 · Vite proxies `/api` via **`RAG_API_URL`** in root `.env`.

Ingest via **Settings** or `POST /api/bootstrap`. For leave **balances**, pick an employee in the header (`employee_id` on ask).

## Scripts

| Script | Does |
|--------|------|
| `npm run dev` | Client + API `:3001` |
| `npm test` | JS + pytest if `rag-api/.venv` exists |
| `npm run build` | shared → client |

Backend only: `cd rag-api` → `.venv\Scripts\uvicorn.exe app.main:app --reload --port 3001`

## API

| Method | Path |
|--------|------|
| `GET` | `/api/health`, `/api/employees`, `/api/cache/stats`, `/api/cache/viz` |
| `POST` | `/api/ingest`, `/api/ingest/upload`, `/api/bootstrap`, `/api/ask`, `/api/ask/stream` |
| `DELETE` | `/api/cache/purge` |

Ask body: `question`, optional `employee_id`, `chat_history`, `use_rag`, `skip_cache`. Stream: `run_start` · `node_*` · `text` · `done` (final JSON).

## Stack

FastAPI · LangGraph (`rag_graph.py`) · OpenAI · Chroma · React/Vite · Zod (`shared/`) · optional `mcp-hr-server` (`get_leave_balance`).

## Project layout

`client/src` (HandbookQA, viz, cache) · `rag-api/app` (`main`, `rag_graph`, `intent_policy`, `semantic_router`, `semantic_cache`) · `fixtures/` · `docs/`.

## Environment

Single root **`.env`** — see [`.env.example`](.env.example). Main vars: `OPENAI_API_KEY`, `OPENAI_*_MODEL`, `CHROMA_*`, `DEFAULT_HANDBOOK_PATH`, `EMPLOYEES_JSON_PATH`, `ALLOWED_INGEST_ROOT`, `RAG_CORS_ORIGINS`, `RAG_API_URL`, `VITE_API_BASE_URL`, `RAG_API_PORT`, optional `SEMANTIC_ROUTER_*`.

## Docs

[CLAUDE.md](CLAUDE.md) · [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md)

## Limitations

Ambiguous phrasing (e.g. loan “type”) needs heuristics + tests; retrieval quality depends on ingest; semantic **cache** can show old answers after routing changes — **purge** when iterating.

## License

Private / evaluation unless stated otherwise.
