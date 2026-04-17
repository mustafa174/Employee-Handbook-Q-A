# Architecture — Employee Handbook Q&A System

This document explains how the application answers handbook and employee-specific questions with a hybrid architecture:

- **RAG for policy content** (Chroma over handbook chunks)
- **Structured grounding for personal data** (fixture or MCP-shaped employee records)
- **Deterministic routing** (LangGraph nodes decide policy/personal/mixed/clarify/general)
- **Post-graph response contracts** (final safety and shape enforcement in `main.py`)
- **Optional semantic cache** (faster repeated answers with route-aware keys)

For setup and usage commands, see [README.md](../README.md).

## 1) System goals

The architecture is optimized for these requirements:

1. Keep policy answers grounded in handbook citations.
2. Keep personal answers grounded in employee-specific records.
3. Support mixed questions in one turn ("my balance" + "policy rule").
4. Avoid accidental personal-data leakage in policy-only responses.
5. Make routing behavior testable and observable instead of purely prompt-driven.

## 2) High-level architecture

```text
Frontend (React/Vite)
   |
   | HTTP / SSE
   v
FastAPI (main.py)
   |
   |  AskRequest -> cache read -> LangGraph execution -> contracts -> cache write
   v
LangGraph (rag_graph.py)
   |
   +-- query_refiner
   +-- guardrail
   +-- router ------------------------------+
   |                                        |
   | policy/mixed -> retrieve -> grade_docs +--> generate -> final state
   | personal/mixed -> balance/profile -----+
   | clarify/general -> clarify or safe generation
   |
   +-- trace events for stream + debugging
```

Core data stores/services:

- `rag-api/data/chroma`: policy retrieval vector store
- `rag-api/data/semantic_cache.json`: semantic response cache
- `fixtures/employees.json` (or MCP-compatible source): personal/profile data

### Visual RAG logic flow

![RAG logic flow screenshot](../../Users/mustafa.hameed/.cursor/projects/c-Projects-Ai-engr-course-Evaluation/assets/c__Users_mustafa.hameed_AppData_Roaming_Cursor_User_workspaceStorage_ccf9c6b7a0d26f24f4a251123972962b_images_image-e8904845-3b53-4821-9507-5895c46e3737.png)

How to read this diagram:

1. **User Query -> Safety Guardrail (hard override)**
   - Every question first passes safety/guardrail checks.
   - Critical/sensitive patterns can hard-override normal routing behavior.
2. **Intent Router (hard route lock)**
   - The router determines whether the question is policy, personal, mixed, or clarify/general.
   - After lock-in, downstream nodes follow that route-specific execution plan.
3. **Policy branch: Chroma retrieval -> retrieval confidence check**
   - Handbook chunks are retrieved from Chroma with citations.
   - Retrieval quality/confidence is checked before answer composition.
4. **Mixed branch: Personal + policy resolver**
   - Personal context and policy evidence are merged for dual-intent questions.
   - This branch supports prompts such as "my leave balance and leave rules."
5. **Personal branch: Employee data only**
   - Personal route uses employee/profile tools only.
   - It intentionally bypasses policy retrieval when the ask is purely personal.
6. **Clarification branch**
   - Ambiguous requests are routed to clarification rather than forcing a risky answer.
7. **Answer Builder -> Atomic Judge -> Verified response**
   - Answer builder composes grounded output from the chosen branch.
   - A final verification step checks response quality/safety before returning the final answer.

Legend alignment to implementation:

- **Guardrail** -> `guardrail` node in `rag_graph.py`
- **Semantic Intent Router** -> `intent_policy.py` + `router_node` in `rag_graph.py`
- **Chroma retrieval** -> `vectorstore.py` + retrieval nodes in `rag_graph.py`
- **Employee tool (MCP-style)** -> profile/balance/tool nodes in `rag_graph.py` and `mcp_tools.py`
- **OpenAI chat model** -> generation and judge prompts configured in backend runtime/env

## 3) Request lifecycle (`POST /api/ask`)

### 3.1 API entry

`main.py` receives `AskRequest` (`question`, optional `employee_id`, `chat_history`, `use_rag`, `skip_cache`) and initializes graph state.

### 3.2 Cache lookup

If caching is enabled and `skip_cache` is false:

- build a route-aware semantic key
- read cached response candidate
- sanitize cached policy leaks before returning

If no cache hit, continue into graph execution.

### 3.3 LangGraph execution

1. **`query_refiner`**
   - normalizes user phrasing
   - may incorporate short history context
2. **`guardrail`**
   - blocks unsafe/out-of-scope content patterns
   - marks escalation/fallback requirements
3. **`router`**
   - classifies intent and selects execution route:
     - `policy`
     - `personal`
     - `mixed`
     - `clarify`
     - `general`
4. **route-specific nodes**
   - policy path: retrieval + grading + generation
   - personal path: balance/profile resolution
   - mixed path: both policy retrieval and personal grounding
   - clarify/general: clarification prompt or safe generic completion

### 3.4 Post-graph contract enforcement

`main._enforce_response_contract` validates/sanitizes the final answer shape:

- policy-only responses should include citation-safe behavior
- personal leaks are stripped where policy-only output is required
- escalation metadata remains consistent

### 3.5 Cache write

If eligible, the finalized contract-compliant response is written to cache.

## 4) Routing architecture (why this is not pure prompt routing)

Routing intentionally combines deterministic logic with semantic fallback:

1. **Primary classifier: `intent_policy.classify_query`**
   - rule/token scores for PROFILE / POLICY / IT / OOS
   - confidence from score distribution
   - targeted overrides for known ambiguity classes
2. **Execution mapper: `router_node`**
   - maps domain + context into graph route
   - applies leave/PTO and "my"-query safety nets
   - controls `use_rag` on/off by route intent
3. **Semantic rescue: `semantic_rescue_route`**
   - used only when route is still `general`
   - lazy-loads SentenceTransformer model
   - compares query embedding to route label passages
   - may rescue to `personal`/`policy` if above threshold

This layered strategy gives testable defaults while still recovering weak or unusual phrasing.

## 5) SentenceTransformer integration details

Semantic rescue is implemented in `semantic_router.py`:

- Model from `SEMANTIC_ROUTER_MODEL` (default `intfloat/e5-base-v2`)
- Threshold from `SEMANTIC_ROUTER_THRESHOLD`
- Local/offline load behavior from `SEMANTIC_ROUTER_LOCAL_ONLY`
- E5-style text formatting:
  - query text encoded as `query: ...`
  - route labels encoded as `passage: ...`
- Similarity computed via dot product on normalized embeddings (cosine equivalent)
- Any model load or encode failure is non-fatal; router continues without rescue

## 6) Retrieval architecture (policy path)

Policy answers use handbook retrieval over Chroma:

1. handbook content is chunked/embedded at ingest time
2. `retrieve` fetches top candidate chunks
3. `grade_documents` filters weak/off-topic chunks
4. `generate` answers from grounded context and prepares citations

Design effect:

- better explainability than free-form policy generation
- citation-aware behavior for user trust and contract checks

## 7) Personal data architecture (personal path)

Personal answers avoid handbook-only hallucination by grounding on structured records:

- employee context keyed by `employee_id`
- profile/balance resolution via dedicated resolver/tool modules
- answer renderer can compose user-specific values safely
- mixed route combines personal context with policy retrieval in one response

The same architecture supports moving from fixtures to MCP/backed systems with minimal API shape changes.

## 8) Cache architecture

`semantic_cache.py` stores route-aware results to reduce repeat latency.

Cache key inputs include:

- normalized user question
- classifier route bucket
- `employee_id` context
- `use_rag` mode
- knowledge-base signature/version signals

Operational note:

- after routing or contract logic changes, purge cache during development to avoid stale behavior masking fixes.

## 9) Ingest architecture

Ingestion endpoints (`/api/ingest`, `/api/ingest/upload`, `/api/bootstrap`) populate policy knowledge:

1. read handbook source (default path or uploaded file)
2. split into chunks
3. embed chunks
4. persist vectors + metadata into Chroma

Security boundary:

- ingestion constrained by `ALLOWED_INGEST_ROOT`
- uploaded files written under `fixtures/_uploads`

## 10) Streaming and observability

`/api/ask/stream` provides real-time visibility:

- graph lifecycle events (`run_start`, node events)
- incremental text events
- terminal `done` payload
- explicit `error` event on failure

Internally, trace hooks in graph state capture route/retrieval/rescue decisions for debugging and tests.

## 11) Contracts and shared types

The system keeps API/client alignment via shared schema types (`shared/`).

Contract enforcement in `main.py` is the final guardrail layer, ensuring:

- response shape consistency
- policy/personal boundary hygiene
- safer behavior on missing citations or partial context

## 12) Failure modes and mitigation

Common failure classes:

- ambiguous intent wording (mitigated by rules + rescue + clarify path)
- retrieval miss or low document quality (mitigated by grading and fallback behavior)
- stale cache after logic changes (mitigated by purge/versioning)
- missing API/model credentials
- local semantic model unavailable when `LOCAL_ONLY` is enabled

The design favors graceful degradation: failed rescue or weak retrieval should fall back safely, not crash the request.

## 13) Test strategy alignment

Architecture is validated through focused tests:

- intent/routing behavior tests (`policy`, `personal`, `mixed`, `clarify`)
- personal routing guard tests (employee-id-sensitive behavior)
- mixed-answer composition tests
- contract enforcement checks
- cache behavior checks around route and signature keys

## 14) Extension points

Recommended future evolution points:

- swap/augment retrievers (reranking, hybrid lexical+vector)
- enrich personal tool layer (benefits, payroll, approvals)
- add per-tenant policy corpora and cache partitioning
- tune semantic rescue labels/threshold from eval sets
- add more explicit policy section-level citations and confidence reporting

## 15) Component map

| Area | Primary files |
|------|------|
| API + contracts | `rag-api/app/main.py` |
| Graph orchestration | `rag-api/app/rag_graph.py` |
| Deterministic intent rules | `rag-api/app/intent_policy.py` |
| Semantic rescue routing | `rag-api/app/semantic_router.py` |
| Vectorstore/chroma access | `rag-api/app/vectorstore.py` |
| Semantic cache | `rag-api/app/semantic_cache.py` |
| Shared request/response types | `shared/src/index.ts` |
| Frontend experience | `client/src/` |

---

If this document and implementation diverge, treat code as source of truth and update this page alongside routing or contract changes.
