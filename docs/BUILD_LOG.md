# Build log — Employee Handbook Q&A

Daily bullets for evaluation **Part B — Development process**. Add dated entries as you work.

## Template (copy per day)

- **Date:**
- **Built:**
- **Tests / verification:**
- **Blockers:**
- **Learned / would do differently:**

---

## Foundation

- Monorepo pivot from API-doc generator: **FastAPI RAG** (`rag-api`), **Vite/React** client, **shared** Zod types, **`fixtures/`** handbook + employees, optional **`mcp-hr-server`**.

## Handbook RAG

- **RAG API:** Chroma persistence, OpenAI embeddings + chat, LangGraph ask flow; routes `/api/health`, `/api/ingest`, `/api/ingest/upload`, `/api/ask`, `/api/bootstrap`.
- **Client:** `HandbookQA` — ingest, chat, citations, escalation UI; `/api` → `:3001`.
- **CI:** Python venv + pytest; npm workspaces build/test.
- **Docs:** `README.md`, `CLAUDE.md`, `ARCHITECTURE.md`, Cursor rules aligned with handbook product.
