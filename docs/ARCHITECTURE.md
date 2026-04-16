# Architecture — Hybrid RAG HR Assistant (Handbook Q&A)

This document describes **how** the system is wired and **why** key choices were made. For operator setup, scripts, and narrative walkthroughs, see the root [README.md](../README.md).

## Problem

**Traditional RAG** (retrieve chunks → generate) fails in real HR settings when:

- A single user turn **mixes** unstructured **policy** knowledge (“What is the leave policy?”) with **structured** **personal** facts (“How many PTO days do *I* have?”).
- The model **hallucinates** balances or eligibility from handbook prose that only describes rules, not individual entitlements.
- **Ambiguous** wording (“type of loans”) maps to the **wrong** execution path (profile field vs product policy) without explicit routing and ordering rules.
- There is **no post-hoc guarantee** that a “policy” response did not leak personal numbers, or that a “profile” response was grounded in tool/fixture data.

This repository treats those as **first-class** concerns: **deterministic and semantic routing**, **separate execution paths** for vectors vs structured employee data, **mixed** orchestration when both are required, and **response contracts** after the graph.

## Ask pipeline (end-to-end)

```text
Client: POST /api/ask | POST /api/ask/stream  (question, optional employee_id, use_rag, …)
    |
    v
Optional: semantic_cache read (skip when skip_cache / sensitive bypass)
    |
    v
LangGraph (rag_graph.py) — compiled StateGraph
    |
    +-- query_refiner ............... Normalize (e.g. vacation -> PTO), optional sub_questions / retrieval queries
    |
    +-- guardrail ................... Sensitive topics; may short-circuit to escalation
    |
    +-- router ...................... intent_policy.classify_query + heuristics + safety nets
    |       |                         optional: semantic_router.semantic_rescue_route
    |       |
    |       +-- route: POLICY or MIXED (needs handbook) ---> retrieve (Chroma) -> grade_documents -> generate
    |       +-- route: PERSONAL ........................ -> balance (fixtures / MCP-aligned profile) -> generate
    |       +-- route: GENERAL ......................... -> generate
    |       +-- route: CLARIFY ......................... -> clarify -> END
    |
    v
build_ask_response_from_state  (pipeline_steps, citations, route label, context_presence, …)
    |
    v
FastAPI: _enforce_response_contract  (invariants; policy/personal leakage; profile-missing vs employee_id)
    |
    v
_sanitize_cached_policy_leak (when applicable) -> JSON / SSE payload; optional semantic_cache write
```

**Important:** the system is **not** “similarity search, optional employee blob, LLM.” It is **branching execution** with different data sources and different invariants per branch, then **contract enforcement** on the wire.

## Example execution trace (one mixed request)

This makes the pipeline **concrete** (same class of behavior as `test_run_ask_mixed_leave_and_policy_combines_personal_and_policy`).

**Inputs:** `POST /api/ask` with `question`: `"how many leaves i have and what is leave policy"`, `employee_id`: `"E001"`, `use_rag`: `true`, semantic cache **miss** (or `skip_cache: true`).

| Step | What runs | Outcome |
|------|-----------|---------|
| 1 | **Cache read** | Miss → continue to graph. |
| 2 | **`query_refiner`** | Normalized query; may carry multi-part hints for retrieval / generation. |
| 3 | **`guardrail`** | No sensitive hit → continue. |
| 4 | **`router`** | Explicit mixed shape and/or decomposition → **`route` = `mixed`**, `use_rag` true for policy leg, profile path enabled for balances. |
| 5 | **`retrieve`** | Chroma returns handbook chunks (e.g. PTO request timing, rollover). |
| 6 | **`grade_documents`** | Scores chunks; may accept or trigger bounded **re-search** before generate. |
| 7 | **`balance`** | Loads structured row for **E001** (e.g. PTO/sick balances) from fixtures. |
| 8 | **`generate`** | Composes **balance line + policy section** with citations where applicable. |
| 9 | **`build_ask_response_from_state`** | Fills `pipeline_steps`, `retrieval_attempts`, `route` label (`MIXED` / `PROFILE` / `POLICY` per public mapping), `context_presence`. |
| 10 | **`_enforce_response_contract`** | Strips unsafe policy/personal mixing; enforces citation / fallback rules. |
| 11 | **`put_cached_answer`** (unless `skip_cache`) | Stores response under cache key that includes **route bucket**, **`employee_id`**, **`use_rag`**, and current **`kb_signature`**. |

A closely related natural phrasing (“How many PTO days do I have and what is the policy?”) follows the **same** branch pattern: mixed personal + policy retrieval, then contract pass.

## Design decisions

| Decision | Rationale |
|----------|-----------|
| **Deterministic intent policy first** (`intent_policy.py`) | Reproducible routing, testable thresholds, and auditable `intent_reason` strings — routing is not delegated to an LLM “guess” for the primary path. |
| **Router heuristics + safety nets** (`router_node`) | Classifiers mislabel borderline utterances; explicit overrides (e.g. loan limit with `employee_id`, leave/PTO balance + employee) prevent **policy** from overwriting **personal** when the user clearly asked for numbers. |
| **Semantic rescue second** (`semantic_router.py`) | Embedding similarity against fixed label passages nudges weak **general** classifications toward **policy** or **personal** when lexical scope is thin — a **fallback**, not the source of truth. |
| **Structured employee data outside the vector index** | Balances and loan flags are **authoritative** in JSON (and MCP demo); they must not be “retrieved” as if they were handbook chunks. |
| **Hybrid paths for mixed intent** | Single-turn questions that need **both** handbook evidence and profile fields run **retrieve** and **balance** before a combined **generate**. |
| **Response contracts after the graph** (`main.py`) | The LLM and cache can still violate product invariants; a final deterministic layer enforces **no personal leakage on policy**, safe **PROFILE** behavior when `employee_id` is absent vs present, and grounded fallbacks. |
| **Semantic cache keyed by route bucket + employee + rag flag** | Same English question can be **policy** vs **profile** depending on classifier and `employee_id`; keys must not collide. **`kb_signature`** invalidates answers when fixtures/handbook change. |

## Trade-offs

| Choice | Benefit | Cost |
|--------|---------|------|
| **Deterministic routing first** | Reproducible, testable, explainable from `intent_reason` and token scores | **Rigid**: rare utterances need new heuristics or tests; edge cases can misfire until patched. |
| **Semantic rescue** (`semantic_router.py`) | Recovers weak “general” / OOS-borderline asks when lexical scope is thin | **Less explainable** to end users (embedding similarity vs natural-language rationale); depends on model + threshold; extra cold-start or CPU cost. |
| **Hybrid execution (mixed)** | Correct answers when one sentence needs **both** numbers and policy text | **Higher complexity**: two data sources, more failure surfaces, more tests required. |
| **Post-hoc response contracts** | Hard guarantees on leakage and PROFILE edge cases after LLM/cache | **Extra CPU** (string rules + sanitizers) on every response; must stay in sync with new answer shapes. |
| **Semantic cache** | Faster repeat asks; KB-signature invalidation | **Stale entries** can mask improved routing until purge — operational discipline required after graph/contract changes. |

## Mixed intent

**Mixed** queries (e.g. `how many leaves i have and what is leave policy` with `employee_id`) intentionally trigger **both**:

1. **Vector retrieval** — Chroma + grading for handbook-backed policy text and citations.
2. **Structured profile load** — `balance` node reads `employees.json` (same shape as MCP demo) for PTO/sick balances and related fields.

`node_generate` composes answers under constraints that keep policy sections citation-grounded and personal sections tool-grounded. Regression coverage lives in `rag-api/tests/test_personal_route_behavior.py` (`test_run_ask_mixed_leave_and_policy_combines_personal_and_policy`).

## Response contracts (post-graph)

`_enforce_response_contract` and related helpers in **`rag-api/app/main.py`** are not cosmetic; they encode **product invariants**:

- **Policy answers must not carry personal balance lines** when the question did not ask for personal data; sanitizer patterns strip common leak shapes.
- **PROFILE without `employee_profile`** — the generic “could not retrieve personal data” replacement applies **only when the request included `employee_id`**, so intentional “select a profile” guidance for balance asks without an ID is preserved.
- **Policy without citations** — forced to handbook-safe messaging instead of invented section text.
- **Cached policy rows** — `_sanitize_cached_policy_leak` reduces stale personal phrases when the current question is policy-scoped.

Together with in-graph checks in `node_generate` (grounding gates, route-scoped profile visibility), this is what makes the stack **hybrid agentic RAG**: orchestration plus **enforceable** output rules.

## LangGraph stages (reference)

| Stage | Module | Role |
|--------|--------|------|
| Query preparation | `query_refiner_node` | `normalize_query`, optional multi-part `sub_questions` / retrieval queries |
| Guardrail | `node_guardrail` | Sensitive-topic detection; may short-circuit with escalation |
| Router | `router_node` | `classify_query` + heuristics + safety nets + optional `semantic_rescue_route` |
| Retrieval | `node_retrieve` | Chroma similarity search for **policy** / **mixed** |
| Grading | `grade_documents` | Relevance / bounded re-search |
| Profile | `node_balance` | Fixture-backed employee snippet; skipped or minimal when no `employee_id` |
| Generation | `node_generate` | Policy: LLM + grounding; personal: deterministic resolvers where applicable; mixed: combined |
| Clarify | `node_clarify` | Routed clarification; then END |

Conditional edges (`build_graph`): **router** → **retrieve** \| **balance** \| **generate** \| **clarify** (see `rag_graph.py`).

## Routing mechanics (how a label is chosen)

Routing is **not** a single LLM call. It is layered:

### 1. Deterministic classifier (`intent_policy.classify_query`)

- Tokenizes normalized text, scores **PROFILE**, **POLICY**, **IT**, and **OOS** using overlapping vocabularies plus regex bonuses (e.g. “what is … policy”, loan policy vs loan eligibility phrases, IT incident cues).
- Picks a **primary** and **secondary** domain with a long list of **override rules** (e.g. PROFILE requires personal pronoun + profile-alias tokens else demotion to POLICY; IT can dominate generic “my”; explicit loan-policy vs loan-personal shortcuts).
- **Confidence** is derived as `primary_score / sum(all_domain_scores)` (bounded 0–1), not a neural softmax — it is **transparent** but coarse.
- **`needs_clarification`** can be set for weak PROFILE cases (low confidence with pronoun + profile hints).
- **`query_scope_signal`** in `router_node` can downgrade borderline POLICY to **OOS** when lexical HR scope is weak **and** confidence is below **~0.85** (see `rag_graph.router_node`).

### 2. Graph router (`router_node`)

Maps classifier output to an **execution route** (`policy`, `personal`, `mixed`, `general`, `clarify`, …) and sets `use_rag`:

- **Heuristic overrides** before/after domain mapping (explicit mixed detector, loan-limit-with-employee → personal, strong PTO *policy process* phrases → policy, etc.).
- **Safety nets** so a late “force policy if `INTENT_POLICY`” line does **not** overwrite an already chosen **personal** or **mixed** route (critical for leave-balance + `employee_id` cases).
- **`forced_use_rag`**: personal routes turn off handbook retrieval; policy/mixed force retrieval on when the client asked for RAG.

### 3. Semantic rescue (`semantic_router.semantic_rescue_route`)

- Runs **only** when the execution route is still **`general`**, **`use_rag`** is on, and the query is not a hard out-of-domain pattern — i.e. it is a **fallback** for “classifier said general but we might still be in HR scope.”
- Compares the query embedding to fixed **label passages** for personal / policy / general; if the best score exceeds **`SEMANTIC_ROUTER_THRESHOLD`**, promotes to **personal** or **policy** and updates `intent_raw` / `domain` / `reason` trace text.
- Tunable via **`SEMANTIC_ROUTER_MODEL`**, **`SEMANTIC_ROUTER_THRESHOLD`**, **`SEMANTIC_ROUTER_LOCAL_ONLY`**.

### 4. Clarification path

- Low-confidence **PROFILE** (from `needs_clarification`) or mixed-clarification flags can route to **`clarify`** instead of retrieval.

| Module | File | Role |
|--------|------|------|
| Intent policy | `intent_policy.py` | Token + rule scores, `confidence`, `secondary_class`, `needs_clarification` |
| Semantic rescue | `semantic_router.py` | Embedding similarity rescue from **general** |
| Execution glue | `rag_graph.router_node` | Overrides, `use_rag`, safety nets, rescue invocation |

## API surface

- **`GET /api/employees`** — Picker data.
- **`POST /api/ask`**, **`POST /api/ask/stream`** — Run graph; stream merges node updates then builds response + contracts.

## Semantic cache design

Implemented in **`rag-api/app/semantic_cache.py`**:

- **Cache key** — `v{version}::{normalized_question}::route={route_bucket}::emp={employee_id or '-'}::rag={0|1}` where **`route_bucket`** comes from **`classify_query(question)`** (not from the post-router execution route). That separation matters: the bucket approximates “what kind of answer” for deduplication while **`employee_id`** still scopes personal rows so **Alice’s balance is not served for Bob’s key** when the question text is identical.
- **KB signature** — Hash of fixture files’ mtimes/sizes; on mismatch, stale entries are bypassed and refreshed so handbook or employee edits do not silently serve old text.
- **Risk: stale behavior after logic changes** — If you change **`router_node`** or **`_enforce_response_contract`** but keep the same classifier bucket + question + employee, a **hit** can still return a **pre-fix** payload. **Mitigation:** bump `_CACHE_KEY_VERSION` when changing key semantics, or run **`DELETE /api/cache/purge`** during development (this matches real incidents seen when iterating routing).

## Components

| Area | Responsibility |
|------|----------------|
| `client` | Assistant UI, SSE consumer, pipeline visualizer, cache panel, Settings; passes `employee_id` from header selection |
| `rag-api` | FastAPI, LangGraph graph, intent policy, semantic router, semantic cache, Chroma, ingest |
| `shared` | Zod contracts for request/response shapes consumed by the client |
| `fixtures` | Handbook + employee demo data |
| `mcp-hr-server` | MCP `get_leave_balance` demo aligned with `EMPLOYEES_JSON_PATH` |

## Observability (architecture-relevant)

Signals already exposed without a separate metrics product:

- **`pipeline_steps`** — router detail (intent, domain, confidence), retrieve empty vs ok, MCP skipped vs active, cache hit step.
- **`retrieval_attempts`** — per-attempt query, **top_score**, grader **verdict** / reason.
- **SSE** — `node_start` / `node_end` for logical nodes during `/api/ask/stream`.
- **`rag_graph.trace()`** — JSON lines to stdout for step-level debugging when running uvicorn in a terminal.

## System-level failure modes

| Failure | Behavior / mitigation |
|---------|-------------------------|
| **Ambiguous query** (e.g. loan “type” vs employment type) | Wrong resolver field risk; mitigated by **composite loan answers before** single-field resolution, catalog guards, and **tests** — see README “failure walkthrough”. |
| **Classifier + policy intent overwrites personal** | Safety net when `employee_id` and leave/PTO shape match; do not force **policy** route if execution is already **personal** or **mixed**. |
| **Retriever miss or grader FAIL** | Policy fallbacks or escalation paths; no fabricated handbook quotes as “facts.” |
| **Semantic cache masks new routing** | Purge (`DELETE /api/cache/purge`) when changing graph or contract logic during development. |
| **No `employee_id` on balance question** | Routed to **personal** with explicit UX copy — not handbook day counts; contracts must not replace that with a generic error. |
| **`employee_id` present but no fixture row** | Contract returns personal-data retrieval failure — expected hard failure for bad IDs. |

## Infrastructure-level failure modes

- Missing **`OPENAI_API_KEY`** — ingest and ask fail on model calls.
- Empty or corrupt **Chroma** directory — weak or empty retrieval; check `CHROMA_PERSIST_DIR` writable and bootstrapped.
- **Embedding / sentence-transformer** load failures — semantic rescue may no-op; primary routing still lexical + policy.

## Security — ingest paths

Ingest resolves paths under **`ALLOWED_INGEST_ROOT`** (`rag-api/app/config.py`). Uploads land under `fixtures/_uploads` (relative to that root) and are indexed from there. Do not point production deployments at arbitrary filesystem trees without hardening.

## CI

GitHub Actions: create **`rag-api/.venv`**, `pip install -r requirements.txt`, **pytest**; then **`npm ci`**, **`npm test`** (includes pytest when venv exists), **`npm run build`**.
