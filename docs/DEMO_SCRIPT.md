# Demo script (15 min) — Employee Handbook Q&A

Use for capstone demo: Problem (2) → Architecture (3) → Live demo (5) → Reflections (3) → Build process (2).

## Segment 1 — Problem (2 min)

Teams need **self-serve answers** from the employee handbook with **trust** (citations) and a path to **HR** when policy is unclear or user-specific data is needed (e.g. leave balances). Static search is brittle; **RAG** plus **structured responses** improves clarity.

**Assumptions:** `OPENAI_API_KEY` is set for the RAG API; handbook content lives in `fixtures/handbook.md` (bilingual demo).

## Segment 2 — Architecture (3 min)

- **Monorepo:** `client` (React + Vite + Tailwind + TanStack Query), `rag-api` (FastAPI + Chroma + LangChain/LangGraph), `shared` (Zod), `fixtures`, `mcp-hr-server` (MCP demo).
- **Flow:** ingest → embed/store → ask → graph (guard → retrieve → generate) → answer + citations + optional escalation.
- **Integration point:** MCP **`get_leave_balance`** reads **`fixtures/employees.json`**; RAG can reference employee context in demos (see code for current wiring).

## Segment 3 — Live demo (5 min)

1. `npm run dev` from repo root; open the UI (**:5173**).
2. Confirm **`/api/health`** (green) — RAG API on **:3001**.
3. **Ingest** default handbook path or upload a short `.md` snippet.
4. **Chat:** ask a policy question; show **citations** and a **neutral** answer.
5. Ask something **ambiguous** or **user-specific**; show **escalation** (or explain guardrails).
6. (Optional) Show **`mcp-hr-server`** with **`get_leave_balance`** for a sample `employee_id`.

## Segment 4 — Reflections (3 min)

- What worked: shared contracts, Chroma persistence, LangGraph single pipeline.
- What was hard: chunking quality, bilingual prompts, judging when to escalate.
- With more time: tighter MCP↔RAG integration, auth, real HR APIs.

## Segment 5 — Build process (2 min)

- **Cursor:** `.cursor/rules/*.mdc` + **`CLAUDE.md`** as agent context.
- **Tests:** Vitest (shared/client), pytest (`rag-api`).
- **CI:** `.github/workflows/ci.yml` — Python + pytest, then npm test/build.

## Q&A prep

- **Why RAG?** Ground answers in the handbook and show **sources**; reduces hallucination risk vs raw chat.
- **Why LangGraph?** Explicit steps (guard, retrieve, generate) are easier to extend and test than one opaque prompt.
- **Production gaps:** Auth, PII handling, real ticketing for escalation, SLA on ingest.
