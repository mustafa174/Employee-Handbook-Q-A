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

- [ ] LangGraph notebook or repo **or** Claude Code Agent Teams run — include **one paragraph** on how it maps to `parse → enrich → OpenAPI → drift` for this product.

## Capstone crosswalk

| Course technique | Where it appears in this project |
|------------------|-----------------------------------|
| Structured outputs / Zod | Shared schemas, LLM enrichment response validation |
| Tool-style API | `POST /api/scan` orchestrates parsers + drift + optional LLM |
| (Optional) LangGraph | Document in ARCHITECTURE if you add a graph wrapper |
