# Parallel workstreams - frontend / backend / tests

Purpose: document how work is split into parallel streams and recombined safely.

This file is workflow evidence for development-process review and also a practical delivery playbook.

## Standard split

- **Stream A - Frontend**
  - UI behavior, chat UX, settings, pipeline visibility.
  - Primary area: `client/src`.
- **Stream B - Backend**
  - API behavior, routing, retrieval, contracts, cache.
  - Primary area: `rag-api/app`.
- **Stream C - Tests & verification**
  - Regression coverage for routes, contracts, cache, stream parity, and UI assumptions.
  - Primary area: `rag-api/tests`, `client/src/**/*.test.tsx`, `shared/src/index.test.ts`.

## Handoffs and integration gates

- **A -> C handoff:** frontend change requires either updated UI tests or explicit manual verification notes.
- **B -> C handoff:** backend behavior change requires updated/added pytest coverage.
- **A/B -> Integration gate:** shared contract compatibility validated against `shared` schemas and runtime behavior.

Integration gate checklist:

- `npm test` passes (JS + pytest when backend venv is present).
- App runs with `npm run dev`.
- Core scenarios validated:
  - policy ask
  - personal ask (with employee context)
  - mixed ask
  - sensitive/escalation path

---

## Current project mapping

### Stream A - Frontend highlights

- `HandbookQA` dual-pane comparison and stream rendering.
- Pipeline visualizer and cache panel support inspection and debugging UX.
- Settings page flow for ingest/bootstrap/health/cache actions.

### Stream B - Backend highlights

- LangGraph orchestration for policy/personal/mixed/clarify/general routes.
- Response contracts for policy/personal boundary safety.
- Semantic cache with route-aware keying and KB signature checks.
- Ingest and retrieval infrastructure (handbook chunking + Chroma).

### Stream C - Tests highlights

- Routing and classifier regression coverage.
- Contract and sanitization coverage.
- Cache behavior and stream parity coverage.
- UI and schema-level tests.

---

## Example parallel cycle (repeatable)

1. **Plan split**
   - A: update UI behavior for new pipeline state.
   - B: implement route/contract change.
   - C: add failing tests for new expectations.
2. **Parallel execution**
   - A and B proceed independently against agreed request/response shape.
   - C keeps red tests visible until behavior is implemented.
3. **Merge + verify**
   - Run test suite and scenario checklist.
   - If mismatch appears, resolve in smallest responsible stream first.
4. **Log outcome**
   - Update `docs/BUILD_LOG.md` and `docs/TDD_TRACES.md`.

---

## Roles and ownership model (single-developer adaptable)

Even when one developer is executing all work, keep these roles explicit:

- **Feature owner (A/B):** implements behavior.
- **Quality owner (C):** validates expectations and regression safety.
- **Integrator:** confirms end-to-end scenario behavior before marking done.

This role separation prevents "code complete" from being mistaken as "release ready."

---

## Evidence checklist

- [ ] Each significant feature lists impacted streams (A/B/C).
- [ ] Test updates are logged for backend-affecting changes.
- [ ] Integration gate results are recorded in `docs/BUILD_LOG.md`.
- [ ] TDD traces are linked for non-trivial changes.
