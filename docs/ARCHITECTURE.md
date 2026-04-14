# Architecture — Employee Handbook Q&A

## Problem

Employees need **accurate, cited** answers from the **handbook** (including bilingual content), with a clear path to **HR** when policy is ambiguous. The stack demonstrates **RAG**, **structured outputs**, and an optional **MCP** bridge to simulated HR data.

## High-level flow

1. **Ingest** — Read `.md`/`.txt` from an allowed path (or upload). Split into chunks, embed with **OpenAI**, store in **Chroma** under `rag-api/data/chroma` (configurable).
2. **Ask** — **LangGraph** pipeline: input guardrails → **similarity search** → optional **employee/leave** context (from fixtures or future MCP wiring) → **LLM** answer with citations → **escalation** flag when confidence or policy boundaries warrant it.
3. **Client** — Calls `/api/*` via Vite proxy to **FastAPI** on **3001**; displays citations and escalation messaging.

## Components

| Area | Responsibility |
|------|----------------|
| `client` | Handbook QA UI: health, ingest (path + file), chat, citations |
| `rag-api` | FastAPI REST, Chroma, OpenAI (LangChain), LangGraph `run_ask` |
| `shared` | Zod contracts shared with the client (`AskRequest`, `AskResponse`, etc.) |
| `fixtures` | `handbook.md` (EN + Urdu sections), `employees.json` (demo balances) |
| `mcp-hr-server` | MCP **`get_leave_balance`**; reads `employees.json` (or `EMPLOYEES_JSON_PATH`) |

## Security — ingest paths

Ingest resolves paths under an **allowed repo root** (see `rag-api/app/config.py`). Uploads land under `rag-api/fixtures/_uploads` and are ingested from there. Do not point production deployments at arbitrary filesystem trees without hardening.

## Failure modes

- Missing **`OPENAI_API_KEY`** → ask/ingest fail or degrade depending on code paths; set locally for demos.
- Empty vector store → retrieve returns little context; answers may be generic or escalate.
- Chroma persistence → ensure `CHROMA_PERSIST_DIR` is writable; ignored in git via `.gitignore`.

## CI

GitHub Actions: create **`rag-api/.venv`**, `pip install -r requirements.txt`, **pytest**; then **`npm ci`**, **`npm test`** (includes pytest when venv exists), **`npm run build`**.
