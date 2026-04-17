# Employee Handbook Q&A System

**Hybrid RAG:** Chroma handbook retrieval + structured employee data (fixtures / MCP demo) + **LangGraph** routing (policy · personal · mixed) + **`main.py` response contracts**. One request can mix **policy + personal**. Pipeline diagram and routing detail: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

## Architecture overview

This system is designed to answer two different classes of questions in one assistant:

- **Policy questions** ("what is the leave carry-over rule?") are answered from handbook retrieval over Chroma + citations.
- **Personal questions** ("how many leave days do I have?") are answered from employee profile data (fixtures today, MCP-compatible shape).
- **Mixed questions** ("how many leave days do I have and what is the carry-over policy?") combine both routes in a single graph run.

At runtime, `POST /api/ask` flows through LangGraph in `rag-api/app/rag_graph.py`:

1. `query_refiner` normalizes wording and optional history context.
2. `guardrail` rejects unsafe or out-of-domain content.
3. `router` chooses `policy`, `personal`, `mixed`, `general`, or `clarify`.
4. Retrieval/balance nodes execute based on route:
   - `policy` -> handbook retrieval + grading + generation
   - `personal` -> profile/balance resolution
   - `mixed` -> retrieval + balance, then merged generation
5. `main.py` applies response contracts (citations/escalation/personal-leak protection).
6. Semantic cache read/write wraps the graph for repeated prompts.

The semantic rescue step (`semantic_router.py`) can recover weakly phrased questions by using SentenceTransformer similarity (`query:` vs `passage:` style labels) before the request is finalized as general.

## Quick start

```bash
copy .env.example .env   # set OPENAI_API_KEY
cd rag-api && python -m venv .venv && .venv\Scripts\pip install -r requirements.txt  # Windows; use bin/ on Unix
cd .. && npm install && npm run dev
```

- UI: http://localhost:5173 (`/`, `/assistant`, `/settings`) · API: http://127.0.0.1:3001 · Vite proxies `/api` via **`RAG_API_URL`** in root `.env`.

Ingest via **Settings** or `POST /api/bootstrap`. For leave **balances**, pick an employee in the header (`employee_id` on ask).

## End-to-end request lifecycle

When a user asks a question in the UI, the system follows this lifecycle:

1. Frontend posts `AskRequest` to `/api/ask` (or opens `/api/ask/stream` for SSE).
2. API normalizes inputs and loads graph state (`question`, optional `employee_id`, history, flags).
3. Semantic cache is checked first (unless `skip_cache=true`).
4. LangGraph runs nodes (`query_refiner` -> `guardrail` -> `router` -> route-specific nodes).
5. Route-specific execution:
   - **policy**: handbook retrieval + document grading + generation with citations
   - **personal**: employee profile/balance tools + profile-aware answer rendering
   - **mixed**: both policy retrieval and personal context, then merged answer
   - **clarify/general**: clarification or safe fallback response path
6. Response contracts sanitize shape/content (citation guarantees, personal-data boundaries).
7. Final response is returned and may be written to semantic cache.

This separation is intentional: retrieval relevance, personal grounding, and response safety are all validated in different steps rather than one opaque generation call.

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

### Example ask payloads

Policy-only:

```json
{
  "question": "What is the leave carry-over policy?",
  "use_rag": true
}
```

Personal-only:

```json
{
  "question": "How many leave days do I have left?",
  "employee_id": "E001",
  "use_rag": false
}
```

Mixed:

```json
{
  "question": "How many leaves I have and what is leave policy?",
  "employee_id": "E001",
  "use_rag": true
}
```

### Streaming events (`/api/ask/stream`)

- `run_start`: graph run metadata
- `node_start` / `node_end`: node-level execution traces
- `text`: incremental model output chunks
- `done`: final full answer payload
- `error`: terminal failure event (if any)

## Stack

| Layer | Technologies | Why it is used |
|------|------|------|
| Frontend UI | React + Vite + TypeScript | Fast local UX for chat, ingest controls, pipeline visualization, and cache tools |
| API surface | FastAPI + Pydantic | Typed request/response contracts for `/api/ask`, ingest, cache, and health endpoints |
| Orchestration | LangGraph (`rag_graph.py`) | Deterministic multi-route flow: policy, personal, mixed, clarify |
| LLM + embeddings | OpenAI chat/embedding models | Generation, grading, and handbook semantic retrieval |
| Vector retrieval | Chroma (`rag-api/data/chroma`) | Persistent handbook chunk retrieval with metadata |
| Semantic fallback router | `sentence-transformers` (`semantic_router.py`) | Rescue weak/general queries into personal or policy when lexical rules are not enough |
| Shared contracts | `shared/` (Zod types) | Keeps API and client payload shapes synchronized |
| Optional HR integration | `mcp-hr-server` demo tool (`get_leave_balance`) | Example path for external employee data providers |

## Project layout

```text
Ai engr course Evaluation/
|- client/                         # Frontend app (React + Vite)
|  |- src/
|  |  |- components/               # UI building blocks (chat, modals, QA widgets)
|  |  |- pages/                    # Route-level screens
|  |  |- lib/                      # API client helpers / utilities
|  |  |- App.tsx                   # Root app shell
|  |  `- main.tsx                  # Vite bootstrap
|  `- package.json
|
|- rag-api/                        # Backend app (FastAPI + LangGraph)
|  |- app/
|  |  |- main.py                   # FastAPI endpoints + response contracts
|  |  |- rag_graph.py              # Core ask graph and node orchestration
|  |  |- intent_policy.py          # Rule/score based intent classification
|  |  |- semantic_router.py        # SentenceTransformer rescue routing
|  |  |- vectorstore.py            # Chroma wiring
|  |  |- semantic_cache.py         # Cache keying and persistence
|  |  `- ...                        # Guardrails, tools, profile resolvers
|  |- tests/                       # Pytest coverage for routing and behavior
|  |- data/                        # Chroma persistence + semantic cache JSON
|  `- requirements.txt
|
|- shared/                         # Shared Zod schemas/types for client+server
|- fixtures/                       # Demo handbook and employee fixture data
|- mcp-hr-server/                  # Optional MCP demo service
`- docs/                           # Architecture, demo script, and guides
```

## Environment

Single root **`.env`** — see [`.env.example`](.env.example). Main vars: `OPENAI_API_KEY`, `OPENAI_*_MODEL`, `CHROMA_*`, `DEFAULT_HANDBOOK_PATH`, `EMPLOYEES_JSON_PATH`, `ALLOWED_INGEST_ROOT`, `RAG_CORS_ORIGINS`, `RAG_API_URL`, `VITE_API_BASE_URL`, `RAG_API_PORT`, optional `SEMANTIC_ROUTER_*` (`MODEL`, `THRESHOLD`, `LOCAL_ONLY`) for the SentenceTransformer semantic rescue router.

### Environment groups

- `OPENAI_*`: generation and embedding model configuration
- `CHROMA_*`: vector database persistence/config
- `DEFAULT_HANDBOOK_PATH`, `ALLOWED_INGEST_ROOT`: ingestion source and file safety boundary
- `EMPLOYEES_JSON_PATH`: fixture employee profile data path
- `RAG_API_URL`, `VITE_API_BASE_URL`, `RAG_API_PORT`: API wiring between frontend and backend
- `SEMANTIC_ROUTER_*`: semantic rescue model name, threshold, and offline/local loading behavior

## Routing behavior guide

The router chooses one of five execution routes:

- `policy`: handbook/process/rule questions that need retrieval and citations
- `personal`: profile-linked user-specific questions ("my leave", "my status")
- `mixed`: explicit blend of personal state + policy guidance in one query
- `clarify`: asks for disambiguation when intent is under-specified
- `general`: safe generic answer path for out-of-domain or unsupported asks

The semantic rescue router (`semantic_router.py`) is a fallback used only when lexical routing lands on `general`; it tries to recover to `policy`/`personal` using embedding similarity.

## Development workflow

Typical local workflow:

1. Start the stack with `npm run dev`.
2. Confirm health via `GET /api/health`.
3. Ingest handbook content (`/api/bootstrap` or Settings page).
4. Run policy, personal, and mixed prompts from UI.
5. If router behavior changed, purge cache (`DELETE /api/cache/purge`) before re-testing.

For backend-focused work:

- run API only: `cd rag-api` then `uvicorn app.main:app --reload --port 3001`
- run tests: `npm test` (includes pytest if backend venv exists)

## Troubleshooting

- **API key errors**: verify `OPENAI_API_KEY` in root `.env`.
- **No retrieval citations**: confirm handbook was ingested and Chroma path is writable.
- **Old/incorrect answers after code changes**: purge semantic cache.
- **Personal answers missing data**: ensure `employee_id` is set and exists in `fixtures/employees.json` (or MCP source).
- **Semantic rescue not triggering**: check `SEMANTIC_ROUTER_MODEL`, `SEMANTIC_ROUTER_THRESHOLD`, and local model availability when `LOCAL_ONLY=true`.

## Docs

[CLAUDE.md](CLAUDE.md) · [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md)

## Limitations

Ambiguous phrasing (e.g. loan “type”) needs heuristics + tests; retrieval quality depends on ingest; semantic **cache** can show old answers after routing changes — **purge** when iterating.

## License

Private / evaluation unless stated otherwise.
