"""FastAPI entry: health, ingest, ask."""

from contextlib import asynccontextmanager
import asyncio
import importlib
import json
from pathlib import Path
from datetime import datetime, timezone
import uuid

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
from app.rag_graph import build_ask_response_from_state, build_initial_state, get_compiled_graph, run_ask
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


class AskResponseModel(BaseModel):
    answer: str
    citations: list[CitationModel]
    retrieval_attempts: list[RetrievalAttemptModel] = []
    isEscalated: bool = False
    escalation_reason: str | None = None
    pipeline_steps: list[PipelineStepModel]
    use_rag: bool
    chat_model: str
    cache_hit: bool = False
    cache_reason: str = "miss"
    cache_kb_signature: str | None = None


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


def _flow_nodes_for_graph_node(graph_node: str) -> list[str]:
    if graph_node == "query_refiner":
        return ["query"]
    if graph_node == "guardrail":
        return ["guardrail"]
    if graph_node == "router":
        return ["router"]
    if graph_node == "clarify":
        return ["router", "output"]
    if graph_node in {"retrieve", "grade_documents"}:
        return ["chroma"]
    if graph_node == "balance":
        return ["mcp"]
    if graph_node == "generate":
        return ["synthesis", "judge", "output"]
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
    cached = None
    if not body.skip_cache:
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
        if not body.skip_cache and has_stale_cached_answer(
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
        if not body.skip_cache:
            put_cached_answer(body.question, body.employee_id, body.use_rag, raw)
    raw["cache_hit"] = cache_hit
    raw["cache_reason"] = "miss" if body.skip_cache else cache_reason
    raw["cache_kb_signature"] = kb_signature

    if not body.skip_cache:
        try:
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
        pipeline_steps=steps,
        use_rag=bool(raw.get("use_rag", True)),
        chat_model=str(raw.get("chat_model", OPENAI_CHAT_MODEL)),
    )


@app.post("/api/ask/stream")
async def ask_stream(body: AskRequest) -> StreamingResponse:
    async def _event_gen():
        run_id = str(uuid.uuid4())
        yield _sse_data({"type": "run_start", "run_id": run_id})
        kb_signature = get_kb_signature()
        cached = None
        if not body.skip_cache:
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
            if not body.skip_cache and has_stale_cached_answer(
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
                        flow_nodes = _flow_nodes_for_graph_node(str(graph_node))
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
                        if isinstance(node_delta, dict):
                            final_state.update(node_delta)
                raw = build_ask_response_from_state(final_state, use_rag=body.use_rag)
            except Exception as e:
                yield _sse_data({"type": "error", "run_id": run_id, "message": f"Ask failed: {e}"})
                return
            if not body.skip_cache:
                put_cached_answer(body.question, body.employee_id, body.use_rag, raw)

        raw["cache_hit"] = cache_hit
        raw["cache_reason"] = "miss" if body.skip_cache else cache_reason
        raw["cache_kb_signature"] = kb_signature

        if cache_hit:
            for node_id in ("query", "output"):
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

        if not body.skip_cache:
            try:
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
