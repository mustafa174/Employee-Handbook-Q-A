# Learning artifacts (Part A evidence)

Use this checklist to tie **course topics** to **capstone-relevant** work. Replace bullets with links to repos, gists, or internal docs.

## Week 1 — LLMs, APIs, Cursor, Claude Code, MCP

- [ ] Working LLM API call (streaming or structured output) with error handling — *optional:* small CLI in a branch or separate folder.
- [ ] This repo: [.cursor/rules](../.cursor/rules), [CLAUDE.md](../CLAUDE.md) — show iteration history in git.
- [ ] **MCP:** List 3+ servers installed; one screenshot or note of a real task completed with MCP.

## Week 2 — RAG, tools, agents

- [ ] Mini RAG (chunk → embed → retrieve) over **your** docs — e.g. `CLAUDE.md` + architecture notes.
- [ ] Agent or tool-calling script with **3+ tools** (search, file read, calculator, etc.).

## Week 3 — LangGraph / multi-agent

- [ ] LangGraph notebook or repo **or** Claude Code Agent Teams run — include **one paragraph** on how it maps to this HR assistant's route graph and safety checks.

## Process evidence checklist (recommended for lead review)

- [ ] `docs/BUILD_LOG.md` has dated entries with: built, tests, blockers, and learnings.
- [ ] `docs/TDD_TRACES.md` has at least 3 concrete red-green-refactor traces.
- [ ] `docs/PARALLEL_WORKSTREAMS.md` is updated for the latest cycle.
- [ ] Commit history is mapped to milestones (MVP target and polish phase).

## Capstone crosswalk

| Course technique | Where it appears in this project |
|------------------|-----------------------------------|
| Structured outputs / Zod | `shared/src/index.ts` request/response schemas shared by client and API |
| RAG pipeline | `rag-api/app/handbook_ingest.py`, `rag-api/app/vectorstore.py`, Chroma-backed retrieval |
| Tool-style personal data access | `rag-api/app/mcp_tools.py`, `mcp-hr-server/src/index.ts` |
| LangGraph workflow | `rag-api/app/rag_graph.py` (refiner -> guardrail -> router -> route branches -> generate) |
| Safety contracts | `rag-api/app/main.py` response-contract enforcement and sanitization |
| Evaluation/process evidence | `docs/BUILD_LOG.md`, `docs/TDD_TRACES.md`, `docs/PARALLEL_WORKSTREAMS.md` |
