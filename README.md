# Hybrid RAG HR Assistant — Handbook Q&A

**Intent routing and structured employee context** on **FastAPI**, **LangGraph**, and **Chroma**: policy answers stay grounded in retrieval, personal balances come from fixture/MCP-aligned data, and **response contracts** in `main.py` enforce invariants after the graph runs.

## Overview

Classic **“retrieve chunks → one LLM call”** setups break down as soon as users mix **handbook policy** with **personal facts** (“How many PTO days do *I* have?” vs “What is the PTO policy?”). Plain RAG also has no first-class notion of **structured** HR data (balances, loan flags, tenure), cannot **prove** it separated policy text from profile numbers, and struggles with **ambiguous** phrasing that should map to tools instead of documents.

This repository implements a **LangGraph** orchestration layer on top of **Chroma** and **OpenAI**: questions are **classified and routed**, optionally **rescued** by an embedding similarity router, executed on **policy**, **personal**, or **mixed** paths, then **post-processed** under explicit API **contracts** so policy answers do not silently leak personal numbers and profile flows behave correctly when no employee is selected.

## Why not plain RAG alone?

| Limitation of “RAG only” | What this project adds |
|--------------------------|-------------------------|
| One retrieval + one prompt for everything | **Intent policy** + heuristics choose **policy** vs **personal** vs **mixed** execution |
| No authoritative source for “my” balances | **`balance`** node loads **structured** fixture/MCP-aligned profile data |
| Ambiguous queries fall through to wrong retrieval | Optional **semantic rescue** (`semantic_router.py`) nudges borderline routes |
| LLM can blend handbook prose with invented personal detail | **Response contract** in `main.py` after the graph strips unsafe mixes |
| No visibility into failures | **Pipeline steps**, **SSE** stream, cache **stats/viz**, escalation paths |

## What makes this system different

- **Deterministic intent policy** (`intent_policy.py`) with confidence and domain labels, layered with **router safety nets** (e.g. leave/PTO balance + `employee_id` must not be overwritten by a bare “policy” classification).
- **Semantic rescue routing** for weak “general” classifications when RAG is enabled — embedding similarity vs fixed label passages, configurable model and threshold.
- **Hybrid execution**: **Chroma** retrieval and grading for policy/mixed; **fixture-backed profile** path for personal (demo **MCP** server mirrors the same employee JSON).
- **Mixed-intent** decomposition: one user turn can drive both profile-grounded numbers and handbook-grounded policy in a controlled way.
- **Semantic cache** with keys that include **route bucket**, **`employee_id`**, and **`use_rag`**, plus **fixture signature** invalidation when the knowledge base changes.
- **Contract enforcement** after the graph — see [Response contracts](#response-contracts) below.

## Response contracts

API-level **invariants** after the graph: `build_ask_response_from_state` then **`_enforce_response_contract`** in `rag-api/app/main.py` before JSON/SSE responses:

- **No personal balance leakage on policy-only answers** — if the model echoes PTO/sick numbers on a question that did not ask for a personal balance, the contract strips or replaces with a deterministic policy-safe fallback.
- **No silent “profile” without data** — when the route is **PROFILE** but there is no `employee_profile` payload, the generic “couldn’t retrieve personal data” replacement runs **only if the client sent a non-empty `employee_id`** (so “select an employee profile” guidance for balance questions without an ID is not clobbered).
- **Grounding pressure on policy** — policy routes without usable citations are forced to handbook-safe fallbacks instead of invented section text.
- **Cache hygiene** — `_sanitize_cached_policy_leak` reduces stale personal phrases in cached policy rows when the question is policy-scoped.

Together with router invariants inside `rag_graph.py`, this is what makes the stack behave like a **product** rather than a single unstructured LLM call.

## Request path (high level)

```text
                         User query (+ optional employee_id, history)
                                        |
                                        v
+--------------------------- query_refiner (normalize, optional splits) ----+
|                                                                          |
+-------------------------------- guardrail -------------------------------+
|                         (sensitive / escalate?)                          |
+-------------------------------- router ----------------------------------+
|   intent_policy (deterministic)  +  heuristics / safety nets             |
|   optional: semantic_router (embedding rescue)                           |
+--------------------------+---------------------------+------------------+
                           |                           |
              +------------+------------+   +---------+---------+
              |                         |   |                   |
              v                         v   v                   v
        route: POLICY            route: PERSONAL          route: MIXED
              |                         |                   |
              v                         v                   v
     retrieve + grade_documents    balance (profile)   retrieve + balance
     (Chroma vectors)              (structured data)   (both paths)
              |                         |                   |
              +------------+------------+-------------------+
                           |
                           v
                    generate (+ clarify if routed)
                           |
                           v
              FastAPI: _enforce_response_contract, cache, response
```

## Example walkthrough: mixed policy + personal balance

**Query (with `employee_id: "E001"`):**  
`"how many leaves i have and what is leave policy"` — regression-tested in `test_run_ask_mixed_leave_and_policy_combines_personal_and_policy`.

1. **`query_refiner`** — Normalizes phrasing; may split into sub-questions for retrieval and generation.
2. **`router`** — Classifies as a **mixed** style turn: personal balance intent plus explicit policy ask (see `router_node` and the test above).
3. **`retrieve` + `grade_documents`** — Chroma returns handbook chunks (e.g. PTO request lead time, rollover language); grader accepts or triggers a bounded re-search.
4. **`balance`** — Loads structured profile (e.g. **14** PTO days, **6** sick days) from `fixtures/employees.json` / `EMPLOYEES_JSON_PATH`.
5. **`generate`** — Emits a **combined** answer: profile-grounded numbers plus policy-grounded prose with citations where applicable.
6. **`_enforce_response_contract`** — Ensures policy sub-answers do not incorrectly carry personal numbers, and mixed payloads remain internally consistent.

**What you should see in the UI:** pipeline steps show **router → retrieve → balance → generate** (and grading), final text cites handbook-style content for the policy portion and states balances for the personal portion.

## Example walkthrough: failure mode and mitigation

**Query:** `"What type of loans are there?"` (ambiguous: HR **services loan** product vs generic “employment type”.)

**What went wrong historically:** the personal route’s field resolver could match **`employment_type`** (“full-time”, etc.) before loan-specific logic, producing an answer that looked like a **loan catalog** but was actually **employment** — with little or no policy retrieval.

**Mitigation in this repo (regression-tested):**

- **Composite loan handling** runs **before** single-field profile resolution on the personal path (`_composite_services_loan_answer` ordering in `rag_graph.py`).
- **Router / catalog guards** (`profile_field_resolver`, intent heuristics) steer loan+“type” style asks toward the **services loan** interpretation when context matches.
- **Tests** — `rag-api/tests/test_personal_route_behavior.py` and related suites lock the intended behavior.

This is the kind of edge case called out in [Known limitations](#known-limitations); the README documents it to show **how** the system fails closed and **what** was changed to address it.

## Observability (what you can inspect without reading code)

| Signal | Where |
|--------|--------|
| **Semantic cache hit / miss** | API fields `cache_hit`, `cache_reason`; **Settings** / `GET /api/cache/stats`, `GET /api/cache/viz` |
| **Router decision** | `pipeline_steps` entry for **Semantic Intent Router** — intent, domain, confidence, free-text `intent_reason` |
| **Retrieval quality** | `retrieval_attempts[]`: query, **top_score**, **verdict** (`answerable` / `re-search`), grader reason; retrieve step `status` `ok` vs `empty` in steps |
| **Profile / MCP path** | Pipeline step for employee tool — active vs skipped, detail string (e.g. no `employee_id`) |
| **Streaming progress** | `POST /api/ask/stream` — `node_start` / `node_end` per logical node; client visualizer mirrors graph activity |
| **Stdout trace** | `rag_graph.trace()` emits JSON lines (`trace_id`, `step`, `route`, …) when running the API from a terminal |

There is no separate “accuracy dashboard” product — but the **same** signals you would use in an incident review are already exposed on the wire and in the UI.

## Stack at a glance

- **Backend:** FastAPI + LangGraph `StateGraph` (`rag_graph.py`)
- **Frontend:** React + Vite + Tailwind + TanStack Query (SSE consumer for `/api/ask/stream`)
- **Shared contracts:** Zod in `shared/`
- **Models:** OpenAI chat + embeddings
- **Vectors:** ChromaDB
- **Streaming:** `StreamingResponse` SSE
- **MCP demo:** `mcp-hr-server` — `get_leave_balance` over `fixtures/employees.json`

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

## Known limitations

- **Ambiguous or overloaded phrases** (e.g. “loan” + “type”) can still stress the router and field resolvers; disambiguation often depends on **catalog ordering** and **tests** rather than a single universal rule.
- **Profile field resolvers** can over-prioritize a structured field when the user’s wording matches multiple catalog entries; composite answers (e.g. loan package) are handled with explicit ordering in the graph.
- **Retrieval quality** depends on handbook coverage, chunking, and grading thresholds; empty or weak chunks produce policy fallbacks or escalation, not fabricated policy text.
- **Semantic rescue** quality depends on the **embedding model** and threshold; it is a safety net, not a guarantee of correct business intent.
- **Semantic cache** can mask routing changes until purged — use **Settings** or `DELETE /api/cache/purge` during development.

## Notes

- Purge semantic cache from **Settings** (or `DELETE /api/cache/purge`) if you change routing behavior and want to avoid stale cached answers during QA.
- Chat history and selected employee are persisted client-side (see `client/src/state/ChatHistoryContext.tsx`).

## License

Private / evaluation use unless otherwise stated.
