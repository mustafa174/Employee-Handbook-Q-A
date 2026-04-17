# Self-assessment — Employee Handbook Q&A (HR RAG capstone)

Complete **before** your lead conversation (Part A) and **before** Demo Day (Part B). Replace empty cells and bracketed notes with your honest ratings and evidence.

## Capstone architecture (context)

This project is an **HR / employee handbook Q&A** stack, not a generic doc tool:

- **`rag-api/`** — FastAPI on port **3001**: Chroma persistence, OpenAI embeddings + chat (LangChain), **LangGraph** pipeline (guardrail → retrieve → optional balance context → generate), REST endpoints (`/api/health`, `/api/ingest`, `/api/ask`, etc.).
- **`client/`** — Vite + React: ingest, chat, and a **Visible RAG** trace panel (**recharts**) showing per-chunk retrieval scores and raw source text.
- **`shared/`** — Zod contracts (`AskRequest`, `AskResponse`, citations with `text` + `score`, `isEscalated`, etc.).
- **`fixtures/`** — Bilingual `handbook.md` and `employees.json` for demos.
- **`mcp-hr-server/`** (optional) — MCP tool **`get_leave_balance`** over employee fixtures.

Use this section when you cite “where in the repo” your evidence lives.

---

## Exceeds expectations — evidence to highlight

Use the bullets below if you rate **Exceeds expectations** in the matching rows. They tie directly to this codebase.

### RAG & retrieval systems

**Exceeds expectations:** We used **ChromaDB** with **semantic overlap chunking** (handbook ingest + vector store) rather than basic regex or purely deterministic text parsing. Retrieval is embedding-driven similarity search with scores surfaced to the UI.

*Evidence hooks:* `rag-api/app/handbook_ingest.py`, `rag-api/app/vectorstore.py`, Chroma data under `rag-api/data/chroma` (configurable).

### LangGraph & orchestration

**Exceeds expectations:** We implemented a **Python LangGraph** pipeline with an explicit **guardrail node** (`node_guardrail` in `rag-api/app/rag_graph.py`) that runs **before** retrieval and generation. Sensitive topics (e.g. harassment, legal threats) short-circuit the graph so they do not go through normal RAG to the LLM in the same way as policy questions.

*Evidence hooks:* `rag-api/app/rag_graph.py` — `SENSITIVE_PATTERNS`, `route_after_guardrail`, conditional edge to `END` vs `retrieve`.

### Product thinking

**Exceeds expectations:** We built a **Visible RAG** dashboard: the client uses **recharts** (`BarChart`) to show **vector similarity / relevance scores** for each retrieved chunk, with the **raw passage text** listed underneath. That makes grounding and model behavior inspectable instead of hiding retrieval in a black box.

*Evidence hooks:* `client/src/components/HandbookQA.tsx` — split chat vs **RAG trace** panel, guardrail alert when `isEscalated` is true.

---

## Part A — Learning (Days 1–20)

Rate each category: **Needs improvement** | **Meets expectations** | **Exceeds expectations**

| Category | Your rating | Evidence (links, repos, notes) |
|----------|-------------|----------------------------------|
| LLM fundamentals & API integration | | e.g. OpenAI via LangChain in `rag-api`, structured outputs, error handling |
| Prompt engineering | | e.g. system prompts in `rag_graph.py` for grounded, JSON-shaped answers |
| RAG & retrieval systems | | See **RAG & retrieval** “Exceeds expectations” block above if applicable |
| Agents & function calling | | e.g. MCP `get_leave_balance`, or future tool-calling extensions |
| LangGraph & orchestration | | See **LangGraph & orchestration** “Exceeds expectations” block above if applicable |
| Multi-agent systems | | Optional: not required for core MVP; note if you extended beyond single graph |

**Reflection (3–5 sentences):** What was hardest? What from Part A did you apply directly to this handbook RAG capstone?

---

## Part B — Capstone project (Handbook RAG)

| Dimension | Your rating | Evidence |
|-----------|-------------|----------|
| Technical quality | | FastAPI + Chroma + LangGraph + shared Zod + tests (`pytest`, Vitest), CI |
| Product thinking | | See **Product thinking** “Exceeds expectations” block above; escalation UX, bilingual handbook |
| Development process | | `CLAUDE.md`, Cursor rules, `docs/BUILD_LOG.md`, branching/PRs as applicable |
| Communication & demo | | Live walkthrough of ingest → ask → RAG trace chart |

**Reflection:** Tradeoffs (e.g. heuristic guardrail vs classifier), known limitations (empty index, API keys), and a short **production roadmap** (auth, real HR systems, observability).
