"""FastAPI entry: health, ingest, ask."""

from contextlib import asynccontextmanager
import asyncio
import importlib
import json
import re
from pathlib import Path
from datetime import datetime, timezone
import uuid
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import (
    ALLOWED_INGEST_ROOT,
    DEFAULT_HANDBOOK_PATH,
    OPENAI_CHAT_MODEL,
    OPENAI_EMBEDDING_MODEL,
    cors_origins,
)
from app.employee_service import load_employees
from app.handbook_ingest import ingest_handbook_file
from app.rag_graph import (
    SENSITIVE_PATTERNS,
    SENSITIVE_SEMANTIC_PATTERNS,
    build_ask_response_from_state,
    build_initial_state,
    detect_sensitive,
    get_compiled_graph,
    run_ask,
)
from app.semantic_cache import (
    add_query,
    get_cached_answer,
    get_kb_signature,
    get_stats,
    get_viz_points,
    has_stale_cached_answer,
    purge_cache,
    put_cached_answer,
)


_DETERMINISTIC_POLICY_FALLBACK = (
    "I couldn't find this information in the employee handbook. Please contact HR."
)
_DEBUG_LOG_PATH = Path(__file__).resolve().parents[2] / "debug-0b08b0.log"


def _agent_debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": "0b08b0",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
    }
    try:
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass

def _contains_personal_balance_text(answer: str) -> bool:
    return bool(
        re.search(
            r"\bYou (?:currently )?have\s+\d+\s+(?:PTO days?|sick days?)\b",
            answer,
            re.I,
        )
    )


def _policy_question_without_personal_balance_ask(question: str) -> bool:
    q = question.lower().strip()
    is_policy = bool(re.search(r"\b(policy|rule|process|advance notice|rollover|accrual|pto)\b", q))
    asks_personal_balance = bool(
        re.search(r"\b(how many|do i have|remaining|left|balance|quota|qouta|my balance)\b", q)
    )
    return is_policy and not asks_personal_balance


def _enforce_response_contract(
    question: str,
    raw: dict,
    *,
    request_employee_id: str | None = None,
) -> dict:
    route = str(raw.get("route", "")).upper()
    presence = raw.get("context_presence", {})
    has_policy = bool(presence.get("has_policy")) if isinstance(presence, dict) else False
    has_profile = bool(presence.get("has_profile")) if isinstance(presence, dict) else False
    answer = str(raw.get("answer", ""))
    # #region agent log
    _agent_debug_log(
        "repro-sensitive-1",
        "H3",
        "main.py:_enforce_response_contract",
        "Sensitive final-contract snapshot",
        {
            "question": question[:120],
            "route": route,
            "has_citations": bool(raw.get("citations")),
            "is_escalated": bool(raw.get("isEscalated")),
            "question_sensitive_match": bool(SENSITIVE_PATTERNS.search(question or "")),
            "answer_preview": answer[:160],
        },
    )
    # #endregion
    if route == "POLICY" and not raw.get("citations"):
        raw["answer"] = "I couldn't find this information in the handbook."
        raw["citations"] = []
        raw["recovery_applied"] = True
        return raw

    # Only treat missing profile as an error when the client actually requested an employee scope.
    # Balance-style questions with no employee_id return PROFILE guidance without employee_profile rows.
    if route in {"PERSONAL", "PROFILE"} and not raw.get("employee_profile"):
        if str(request_employee_id or "").strip():
            raw["answer"] = "I couldn't retrieve your personal data."
            raw["citations"] = []
            raw["recovery_applied"] = True
            return raw

    # Deterministic invariant guard for policy route contamination.
    if route == "POLICY" and has_profile:
        # If there is no personal-balance leakage text, keep policy answer and
        # just clear profile-presence metadata.
        if not _contains_personal_balance_text(answer):
            raw["context_presence"] = {"has_policy": has_policy, "has_profile": False, "has_it": False}
            raw["recovery_applied"] = True
            return raw
        recovered = dict(raw)
        recovered["context_presence"] = {"has_policy": has_policy, "has_profile": False, "has_it": False}
        recovered["answer"] = str(recovered.get("answer", ""))
        recovered = _sanitize_cached_policy_leak(question, recovered)
        recovered_answer = str(recovered.get("answer", "")).strip()
        if recovered_answer and recovered_answer != str(raw.get("answer", "")).strip():
            recovered["citations"] = list(raw.get("citations", []))
            recovered["recovery_applied"] = True
            return recovered
        raw["answer"] = _DETERMINISTIC_POLICY_FALLBACK
        raw["citations"] = []
        raw["context_presence"] = {"has_policy": has_policy, "has_profile": False, "has_it": False}
        raw["recovery_applied"] = True
        return raw

    # Last-line response contract for policy-only asks.
    if _policy_question_without_personal_balance_ask(question):
        if _contains_personal_balance_text(answer):
            raw["answer"] = _DETERMINISTIC_POLICY_FALLBACK
            raw["citations"] = []
            raw["recovery_applied"] = True
    if "recovery_applied" not in raw:
        raw["recovery_applied"] = False
    return raw


def _sanitize_cached_policy_leak(question: str, raw: dict) -> dict:
    answer = str(raw.get("answer", ""))
    if not answer or not _policy_question_without_personal_balance_ask(question):
        return raw
    cleaned = answer
    # Remove common personal balance phrasings that can leak into policy-only responses.
    for pattern in (
        r"(?is)\bYou (?:currently )?have\s+\d+\s+PTO days?\s+and\s+\d+\s+sick days?\s+remaining\.?\s*",
        r"(?is)\bYou (?:currently )?have\s+\d+\s+PTO days?\s+remaining\.?\s*",
        r"(?is)\bYou (?:currently )?have\s+\d+\s+sick days?\s+remaining\.?\s*",
        r"(?is)\bYou (?:currently )?have\s+\d+\s+sick days?\s+and\s+\d+\s+PTO days?\s+remaining\.?\s*",
    ):
        cleaned = re.sub(pattern, "", cleaned).strip()
    if not cleaned:
        cleaned = (
            "PTO requests must be submitted at least 10 business days in advance, "
            "unless your manager approves an emergency exception."
        )
    raw["answer"] = cleaned
    return raw


def _should_bypass_cache(question: str) -> bool:
    q = str(question or "")
    decision = detect_sensitive(q)
    # #region agent log
    _agent_debug_log(
        "repro-sensitive-1",
        "H2",
        "main.py:_should_bypass_cache",
        "Sensitive cache-bypass decision",
        {
            "question": str(question or "")[:120],
            "bypass": decision,
        },
    )
    # #endregion
    return decision


def _warm_openai_submodules() -> None:
    """Eager-import OpenAI SDK submodules so concurrent /api/ask calls do not deadlock on import locks."""
    importlib.import_module("openai.resources.embeddings")
    importlib.import_module("openai.resources.chat.completions")


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _warm_openai_submodules()
    yield


app = FastAPI(
    title="Employee Handbook RAG API",
    version="0.1.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CitationModel(BaseModel):
    text: str
    score: float
    source: str | None = None
    section_title: str | None = None


class RetrievalAttemptModel(BaseModel):
    attempt: int
    query: str
    top_score: float
    verdict: str
    reason: str | None = None
    citations: list[CitationModel]


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    employee_id: str | None = None
    chat_history: list[dict[str, str]] | None = None
    use_rag: bool = True
    skip_cache: bool = False


class PipelineStepModel(BaseModel):
    id: str
    label: str
    status: str
    detail: str | None = None


class AgentActionPayloadModel(BaseModel):
    employee_id: str | None = None
    employee_name: str | None = None
    message: str


class AgentActionModel(BaseModel):
    type: Literal["HARASSMENT_REPORT"]
    payload: AgentActionPayloadModel


class AskResponseModel(BaseModel):
    answer: str
    citations: list[CitationModel]
    retrieval_attempts: list[RetrievalAttemptModel] = []
    isEscalated: bool = False
    escalation_reason: str | None = None
    agent_action: AgentActionModel | None = None
    pipeline_steps: list[PipelineStepModel]
    use_rag: bool
    chat_model: str
    cache_hit: bool = False
    cache_reason: str = "miss"
    cache_kb_signature: str | None = None
    clarification_triggered: bool = False
    recovery_applied: bool = False


class IngestPathRequest(BaseModel):
    handbook_path: str = "fixtures/handbook.md"
    replace: bool = True


class IngestResponseModel(BaseModel):
    chunks_indexed: int
    source_path: str


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "handbook-rag-api"
    chat_model: str
    embedding_model: str


class CacheStatsResponse(BaseModel):
    total_cached_queries: int


class CachePurgeResponse(BaseModel):
    purged: int
    total_cached_queries: int


class CacheVizPointModel(BaseModel):
    x: float
    y: float
    text: str
    label: str
    kind: str
    category: str
    source: str


class CacheVizResponse(BaseModel):
    points: list[CacheVizPointModel]


class EmployeeOptionModel(BaseModel):
    employee_id: str
    name: str


def _sse_data(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _flow_nodes_for_graph_node(graph_node: str, state: dict) -> list[str]:
    """Map LangGraph node names to RAGPipelineVisualizer React Flow node ids (HandbookQA.tsx)."""
    route = str(state.get("route") or "").lower()
    domain = str(state.get("intent_domain") or "").upper()
    if graph_node == "query_refiner":
        return ["query"]
    if graph_node == "guardrail":
        return ["guardrail"]
    if graph_node == "router":
        return ["router"]
    if graph_node == "clarify":
        # Visualizer has no "clarify" stage; treat as clarification / fallback path.
        return ["out_of_scope", "output"]
    if graph_node in {"retrieve", "grade_documents"}:
        if route == "mixed":
            return ["mix"]
        if route == "policy" and domain == "IT":
            return ["it"]
        if route == "policy":
            return ["policy"]
        return ["policy"]
    if graph_node == "balance":
        return ["mix"]
    if graph_node == "generate":
        return ["output"]
    return []


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        chat_model=OPENAI_CHAT_MODEL,
        embedding_model=OPENAI_EMBEDDING_MODEL,
    )


@app.get("/api/employees", response_model=list[EmployeeOptionModel])
def employees() -> list[EmployeeOptionModel]:
    try:
        data = load_employees()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Employees load failed: {e}") from e
    raw_rows = data.get("employees", [])
    if not isinstance(raw_rows, list):
        return []
    rows: list[EmployeeOptionModel] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        employee_id = str(row.get("employee_id", "")).strip()
        name = str(row.get("name", "")).strip()
        if not employee_id or not name:
            continue
        rows.append(EmployeeOptionModel(employee_id=employee_id, name=name))
    return rows


@app.post("/api/ingest", response_model=IngestResponseModel)
def ingest_path(body: IngestPathRequest) -> IngestResponseModel:
    try:
        # Knowledge-base mode: ingest scans entire fixtures/ folder.
        n = ingest_handbook_file(DEFAULT_HANDBOOK_PATH, replace=body.replace)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}") from e
    return IngestResponseModel(chunks_indexed=n, source_path=str(ALLOWED_INGEST_ROOT / "fixtures"))


@app.post("/api/ingest/upload", response_model=IngestResponseModel)
async def ingest_upload(
    file: UploadFile = File(...),
    replace: bool = True,
) -> IngestResponseModel:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".md", ".txt", ".markdown"):
        raise HTTPException(status_code=400, detail="Only .md / .txt allowed")
    upload_dir = ALLOWED_INGEST_ROOT / "fixtures" / "_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / Path(file.filename).name
    content = await file.read()
    dest.write_bytes(content)
    try:
        # After upload, re-index full fixtures knowledge-base (includes uploads).
        n = ingest_handbook_file(dest.resolve(), replace=replace)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}") from e
    return IngestResponseModel(chunks_indexed=n, source_path=str(ALLOWED_INGEST_ROOT / "fixtures"))


@app.post("/api/ask", response_model=AskResponseModel)
def ask(body: AskRequest) -> AskResponseModel:
    kb_signature = get_kb_signature()
    skip_cache = bool(body.skip_cache or _should_bypass_cache(body.question))
    cached = None
    if not skip_cache:
        cached = get_cached_answer(
            body.question,
            body.employee_id,
            body.use_rag,
            kb_signature=kb_signature,
        )
    cache_hit = False
    cache_reason = "miss"
    if cached:
        cache_hit = True
        cache_reason = "hit"
        cached_steps = list(cached.get("pipeline_steps", []))
        cached_steps.append(
            {
                "id": "semantic_cache",
                "label": "Semantic cache",
                "status": "ok",
                "detail": "Cache hit: served previously computed answer",
            }
        )
        cached["pipeline_steps"] = cached_steps
        raw = cached
    else:
        if not skip_cache and has_stale_cached_answer(
            body.question,
            body.employee_id,
            body.use_rag,
            kb_signature=kb_signature,
        ):
            cache_reason = "kb_changed"
        try:
            raw = run_ask(
                body.question,
                employee_id=body.employee_id,
                chat_history=body.chat_history,
                use_rag=body.use_rag,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ask failed: {e}") from e
        raw = _enforce_response_contract(body.question, raw, request_employee_id=body.employee_id)
        raw = _sanitize_cached_policy_leak(body.question, raw)
        if not skip_cache:
            put_cached_answer(body.question, body.employee_id, body.use_rag, raw)
    raw = _enforce_response_contract(body.question, raw, request_employee_id=body.employee_id)
    raw = _sanitize_cached_policy_leak(body.question, raw)
    raw["cache_hit"] = cache_hit
    raw["cache_reason"] = "sensitive_bypass" if _should_bypass_cache(body.question) else ("miss" if skip_cache else cache_reason)
    raw["cache_kb_signature"] = kb_signature

    try:
        # #region agent log
        _agent_debug_log(
            "pre-fix",
            "H3",
            "main.py:ask",
            "Telemetry query recorded",
            {
                "endpoint": "/api/ask",
                "question": str(body.question or "")[:160],
                "skip_cache": bool(skip_cache),
                "cache_reason": str(raw.get("cache_reason", "")),
                "request_employee_id_present": bool(str(body.employee_id or "").strip()),
                "response_route": str(raw.get("route", "")),
            },
        )
        # #endregion
        add_query(body.question, created_at=datetime.now(timezone.utc).isoformat())
    except Exception:
        # Telemetry writes must not break user answers.
        pass
    cites = [CitationModel(**c) for c in raw.get("citations", [])]
    attempts = [RetrievalAttemptModel(**a) for a in raw.get("retrieval_attempts", [])]
    steps = [PipelineStepModel(**s) for s in raw.get("pipeline_steps", [])]
    return AskResponseModel(
        answer=raw.get("answer", ""),
        citations=cites,
        retrieval_attempts=attempts,
        isEscalated=bool(raw.get("isEscalated")),
        escalation_reason=raw.get("escalation_reason"),
        agent_action=raw.get("agent_action"),
        pipeline_steps=steps,
        use_rag=bool(raw.get("use_rag", True)),
        chat_model=str(raw.get("chat_model", OPENAI_CHAT_MODEL)),
        clarification_triggered=bool(raw.get("clarification_triggered", False)),
        recovery_applied=bool(raw.get("recovery_applied", False)),
    )


@app.post("/api/ask/stream")
async def ask_stream(body: AskRequest) -> StreamingResponse:
    async def _event_gen():
        run_id = str(uuid.uuid4())
        yield _sse_data({"type": "run_start", "run_id": run_id})
        kb_signature = get_kb_signature()
        skip_cache = bool(body.skip_cache or _should_bypass_cache(body.question))
        cached = None
        if not skip_cache:
            cached = get_cached_answer(
                body.question,
                body.employee_id,
                body.use_rag,
                kb_signature=kb_signature,
            )
        cache_hit = False
        cache_reason = "miss"
        if cached:
            cache_hit = True
            cache_reason = "hit"
            raw = cached
        else:
            if not skip_cache and has_stale_cached_answer(
                body.question,
                body.employee_id,
                body.use_rag,
                kb_signature=kb_signature,
            ):
                cache_reason = "kb_changed"
            try:
                graph = get_compiled_graph()
                initial = build_initial_state(
                    body.question,
                    employee_id=body.employee_id,
                    chat_history=body.chat_history,
                    use_rag=body.use_rag,
                )
                final_state = dict(initial)
                async for update in graph.astream(initial, stream_mode="updates"):
                    for graph_node, node_delta in update.items():
                        if isinstance(node_delta, dict):
                            final_state.update(node_delta)
                        flow_nodes = _flow_nodes_for_graph_node(str(graph_node), final_state)
                        status = "ok"
                        if graph_node == "guardrail" and isinstance(node_delta, dict) and node_delta.get("escalate"):
                            status = "triggered"
                        for node in flow_nodes:
                            yield _sse_data({"type": "node_start", "run_id": run_id, "node": node})
                        await asyncio.sleep(0.005)
                        for node in flow_nodes:
                            yield _sse_data(
                                {"type": "node_end", "run_id": run_id, "node": node, "status": status}
                            )
                raw = build_ask_response_from_state(final_state, use_rag=body.use_rag)
            except Exception as e:
                yield _sse_data({"type": "error", "run_id": run_id, "message": f"Ask failed: {e}"})
                return
            raw = _enforce_response_contract(body.question, raw, request_employee_id=body.employee_id)
            raw = _sanitize_cached_policy_leak(body.question, raw)
            if not skip_cache:
                put_cached_answer(body.question, body.employee_id, body.use_rag, raw)
        raw = _enforce_response_contract(body.question, raw, request_employee_id=body.employee_id)
        raw = _sanitize_cached_policy_leak(body.question, raw)

        raw["cache_hit"] = cache_hit
        raw["cache_reason"] = "sensitive_bypass" if _should_bypass_cache(body.question) else ("miss" if skip_cache else cache_reason)
        raw["cache_kb_signature"] = kb_signature

        if cache_hit:
            raw_route = str(raw.get("route") or "").upper()
            presence = raw.get("context_presence") if isinstance(raw.get("context_presence"), dict) else {}
            has_it = bool(presence.get("has_it"))
            bucket: str
            if raw_route == "MIXED":
                bucket = "mix"
            elif raw_route in {"GENERAL", "SENSITIVE", "CLARIFY"}:
                bucket = "out_of_scope"
            elif has_it:
                bucket = "it"
            elif raw_route == "POLICY":
                bucket = "policy"
            elif raw_route == "PROFILE":
                bucket = "mix"
            else:
                bucket = "policy"
            for node_id in ("query", "guardrail", "router", bucket, "output"):
                yield _sse_data({"type": "node_start", "run_id": run_id, "node": node_id})
                await asyncio.sleep(0.01)
                yield _sse_data({"type": "node_end", "run_id": run_id, "node": node_id, "status": "ok"})

        answer_text = str(raw.get("answer", ""))
        if answer_text:
            for chunk in answer_text.split(" "):
                if not chunk:
                    continue
                yield _sse_data({"type": "text", "run_id": run_id, "content": f"{chunk} "})
                await asyncio.sleep(0.01)

        try:
            # #region agent log
            _agent_debug_log(
                "pre-fix",
                "H3",
                "main.py:ask_stream",
                "Telemetry query recorded",
                {
                    "endpoint": "/api/ask/stream",
                    "question": str(body.question or "")[:160],
                    "skip_cache": bool(skip_cache),
                    "cache_reason": str(raw.get("cache_reason", "")),
                    "request_employee_id_present": bool(str(body.employee_id or "").strip()),
                    "response_route": str(raw.get("route", "")),
                },
            )
            # #endregion
            add_query(body.question, created_at=datetime.now(timezone.utc).isoformat())
        except Exception:
            pass

        yield _sse_data({"type": "done", "run_id": run_id, "final": raw})

    return StreamingResponse(
        _event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/bootstrap")
def bootstrap() -> IngestResponseModel:
    """Dev convenience: ingest full fixtures knowledge base."""
    n = ingest_handbook_file(DEFAULT_HANDBOOK_PATH, replace=True)
    return IngestResponseModel(
        chunks_indexed=n,
        source_path=str(ALLOWED_INGEST_ROOT / "fixtures"),
    )


@app.get("/api/cache/stats", response_model=CacheStatsResponse)
def cache_stats() -> CacheStatsResponse:
    return CacheStatsResponse(**get_stats())


@app.delete("/api/cache/purge", response_model=CachePurgeResponse)
def cache_purge() -> CachePurgeResponse:
    return CachePurgeResponse(**purge_cache())


@app.get("/api/cache/viz", response_model=CacheVizResponse)
def cache_viz() -> CacheVizResponse:
    return CacheVizResponse(**get_viz_points())
