# TDD traces - Employee Handbook Q&A

Purpose: capture **red -> green -> refactor** evidence for key features.  
Rule: each significant behavior change should have one trace entry, linked to tests.

## Template

- **Feature / behavior:**
- **Date:**
- **Scope:** frontend | backend | shared
- **Red (failing test first):**
  - Test file(s):
  - Failure expectation:
- **Green (minimal implementation):**
  - Change summary:
  - Verification:
- **Refactor (cleanup without behavior change):**
  - Refactor summary:
  - Regression checks:
- **Outcome / notes:**

---

## Trace 1 - Routing classification confidence and guard behavior

- **Feature / behavior:** route policy/personal/mixed/clarify reliably for handbook asks.
- **Date:** 2026-04 (ongoing hardening)
- **Scope:** backend
- **Red (failing test first):**
  - Test file(s): `rag-api/tests/test_routing.py`, `rag-api/tests/test_intent_policy_regression.py`, `rag-api/tests/test_clarification_guard.py`
  - Failure expectation: ambiguous or regression prompts route incorrectly.
- **Green (minimal implementation):**
  - Change summary: tighten classifier/routing logic and intent guardrails.
  - Verification: routing tests pass for policy/personal/mixed/clarify scenarios.
- **Refactor (cleanup without behavior change):**
  - Refactor summary: isolate decision logic in intent/router modules and improve reason fields.
  - Regression checks: run full routing regression suite.
- **Outcome / notes:** deterministic-first routing is now testable and explainable.

## Trace 2 - Policy/personal boundary contract

- **Feature / behavior:** prevent personal data leakage into policy-only answers.
- **Date:** 2026-04 (ongoing hardening)
- **Scope:** backend
- **Red (failing test first):**
  - Test file(s): `rag-api/tests/test_context_contracts.py`, `rag-api/tests/test_policy_answer_sanitizer.py`, `rag-api/tests/test_route_context_isolation.py`
  - Failure expectation: policy answers include personal-balance lines in edge cases.
- **Green (minimal implementation):**
  - Change summary: enforce post-graph response contracts and sanitize policy leaks.
  - Verification: contract and sanitizer tests pass.
- **Refactor (cleanup without behavior change):**
  - Refactor summary: centralize final-contract checks to keep API behavior consistent.
  - Regression checks: rerun contract + stream parity tests.
- **Outcome / notes:** safer response boundary for HR policy use cases.

## Trace 3 - Cache key integrity and stale-answer control

- **Feature / behavior:** cache answers by semantic intent context and KB signature.
- **Date:** 2026-04 (ongoing hardening)
- **Scope:** backend
- **Red (failing test first):**
  - Test file(s): `rag-api/tests/test_cache_integrity.py`, `rag-api/tests/test_api_cache_sanitizer.py`
  - Failure expectation: stale or cross-context cache hits appear after route/KB changes.
- **Green (minimal implementation):**
  - Change summary: route-aware keying + KB signature validation + sanitization before return.
  - Verification: cache integrity tests and cache-related API tests pass.
- **Refactor (cleanup without behavior change):**
  - Refactor summary: simplify cache helper responsibilities and key construction.
  - Regression checks: rerun cache + ask endpoint tests.
- **Outcome / notes:** reduced risk of stale or context-inappropriate answers.

## Trace 4 - Stream and non-stream parity

- **Feature / behavior:** `/api/ask` and `/api/ask/stream` should converge on consistent final payload behavior.
- **Date:** 2026-04 (ongoing hardening)
- **Scope:** backend + client integration
- **Red (failing test first):**
  - Test file(s): `rag-api/tests/test_stream_parity.py`
  - Failure expectation: streamed final state differs from non-stream answer semantics.
- **Green (minimal implementation):**
  - Change summary: align post-processing contract/sanitization across both endpoints.
  - Verification: stream parity tests pass.
- **Refactor (cleanup without behavior change):**
  - Refactor summary: remove duplicate logic drift risk through consistent final-state handling.
  - Regression checks: rerun stream parity + context contract tests.
- **Outcome / notes:** better UX consistency and safer debugging.

## Trace 5 - Frontend query flow and chat behavior stability

- **Feature / behavior:** stable chat send/render flow with RAG and baseline panes.
- **Date:** 2026-04 (ongoing hardening)
- **Scope:** frontend
- **Red (failing test first):**
  - Test file(s): `client/src/components/HandbookQA.test.tsx`, `shared/src/index.test.ts`
  - Failure expectation: message flow or typed contract assumptions break in UI behavior.
- **Green (minimal implementation):**
  - Change summary: stabilize ask request/response handling and pane behavior.
  - Verification: frontend and shared schema tests pass.
- **Refactor (cleanup without behavior change):**
  - Refactor summary: improve type safety and keep API contract usage explicit.
  - Regression checks: rerun JS tests and local UI walkthrough.
- **Outcome / notes:** clearer reliability between typed contracts and user-visible behavior.

---

## Completion checklist for new traces

- Add at least one new trace for each non-trivial feature.
- Include exact test file names.
- Keep refactor step explicit (not just "added code").
- Link to commit/PR when available.
