# Architecture — Employee Handbook Q&A System

Hybrid **RAG**: Chroma (handbook) + fixture/MCP-shaped **structured** employee rows + **LangGraph** routing (policy / personal / mixed / …) + **`main._enforce_response_contract`** after the graph. Handles **mixed** policy + personal in one turn. Setup and demos: [README.md](../README.md).

## Problem (one paragraph)

Plain retrieve-then-generate RAG mixes policy text with “my” numbers, cannot ground balances in structured data, and has no enforceable split between **vector** evidence and **profile** fields. This stack routes first, runs different branches, then applies **contracts** so outputs stay consistent with the chosen path.

## Ask pipeline

```text
LangGraph:  [cache read] -> query_refiner -> guardrail -> router
                  |
                  +-- POLICY  --> retrieve -> grade_documents --+
                  +-- MIXED   --> retrieve -> grade_documents --+--> generate -> END
                  +-- PERSONAL --> balance ---------------------+
                  +-- GENERAL / CLARIFY --> generate or clarify -> END

  (mixed: profile merged in generate when employee_id set)

FastAPI after invoke:  build_ask_response_from_state -> _enforce_response_contract -> [cache write]
```

**Mixed example** (see `test_run_ask_mixed_leave_and_policy_combines_personal_and_policy`): `how many leaves i have and what is leave policy` + `E001` → router **mixed** → retrieve + balance → generate → contract → optional cache.

## Routing (compressed)

1. **`intent_policy.classify_query`** — Token/rule scores for PROFILE / POLICY / IT / OOS; `confidence = primary / sum(scores)`; overrides (loan phrases, IT dominance, PROFILE needs pronoun+alias, …); `needs_clarification` for weak PROFILE.
2. **`router_node`** — Maps to execution route; loan/PTO safety nets; **does not** let late “force policy” overwrite **personal/mixed**; sets `use_rag` (off for pure personal).
3. **`semantic_rescue_route`** — Only if route still **`general`**, RAG on, not hard-OOS: embedding vs label passages; threshold via `SEMANTIC_ROUTER_*` env.

## Design trade-offs (short)

| Choice | Trade-off |
|--------|-----------|
| Deterministic routing first | Testable / rigid on novel phrasing |
| Semantic rescue | Helps weak scope / less explainable |
| Mixed path | Correct / more moving parts |
| Post-graph contracts | Safer / must track new answer shapes |
| Semantic cache | Fast / stale until purge or `_CACHE_KEY_VERSION` bump |

## Contracts & cache (short)

**Contracts** (`main.py`): strip personal lines from policy-only answers; PROFILE missing profile only hard-errors if **`employee_id` was sent**; policy without citations → safe fallback; `_sanitize_cached_policy_leak` for cache rows.

**Cache** (`semantic_cache.py`): key = normalized question + **`classify_query` route bucket** + `employee_id` + `use_rag` + `kb_signature`. Purge after router/contract changes during dev.

## LangGraph nodes

`query_refiner` → `guardrail` → `router` → (`retrieve`→`grade_documents` \| `balance` \| `generate` \| `clarify`) — see `build_graph` in `rag_graph.py`.

## API & packages

| Piece | Role |
|-------|------|
| `GET /api/employees`, `POST /api/ask`, `/api/ask/stream` | Picker + graph + SSE |
| `client/` | UI, SSE, pipeline viz, cache panel |
| `rag-api/app/` | `main`, `rag_graph`, `intent_policy`, `semantic_router`, `semantic_cache`, ingest, Chroma |
| `fixtures/`, `mcp-hr-server/` | Handbook + employees; MCP demo |

## Failure modes (merged)

Ambiguous “loan type” phrasing; classifier vs personal overwrite (mitigated by nets); retrieval miss / grader fail → fallbacks; **cache hit hides routing fixes** → purge; missing API key / empty Chroma / rescue model load fail.

## Security & CI

Ingest only under **`ALLOWED_INGEST_ROOT`**; uploads under `fixtures/_uploads`. CI: `rag-api` venv + pytest, then `npm ci` + `npm test` + `npm run build`.
