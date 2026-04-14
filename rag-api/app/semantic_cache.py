"""Semantic cache + query telemetry + vector-space projection."""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from app.config import FIXTURES_DIR, RAG_API_ROOT
from app.vectorstore import get_vectorstore


class CacheEntry(TypedDict):
    query: str
    category: str
    created_at: str


class CachedAskEntry(TypedDict):
    key: str
    question_norm: str
    employee_id: str | None
    use_rag: bool
    kb_signature: str
    response: dict
    created_at: str


class VizPoint(TypedDict):
    x: float
    y: float
    text: str
    label: str
    kind: str
    category: str
    source: str


_CACHE_PATH = RAG_API_ROOT / "data" / "semantic_cache.json"
_LOCK = threading.Lock()


def _ensure_store() -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _CACHE_PATH.exists():
        _CACHE_PATH.write_text('{"entries":[],"answers":[]}', encoding="utf-8")


def _normalize_question(question: str) -> str:
    return " ".join(question.lower().strip().split())


def _ask_key(question: str, employee_id: str | None, use_rag: bool) -> str:
    norm_q = _normalize_question(question)
    emp = (employee_id or "").strip().upper()
    return f"{norm_q}::emp={emp or '-'}::rag={int(use_rag)}"


def _fixtures_signature() -> str:
    parts: list[str] = []
    if FIXTURES_DIR.exists():
        for p in sorted(FIXTURES_DIR.rglob("*")):
            if not p.is_file():
                continue
            try:
                stat = p.stat()
            except OSError:
                continue
            rel = p.relative_to(FIXTURES_DIR)
            parts.append(f"{rel.as_posix()}:{int(stat.st_mtime)}:{stat.st_size}")
    joined = "|".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def get_kb_signature() -> str:
    """Public signature for current handbook/fixtures content state."""
    return _fixtures_signature()


def _read_entries() -> list[CacheEntry]:
    _ensure_store()
    try:
        body = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    raw_entries = body.get("entries", [])
    if not isinstance(raw_entries, list):
        return []
    entries: list[CacheEntry] = []
    for item in raw_entries:
        if not isinstance(item, dict):
            continue
        q = str(item.get("query", "")).strip()
        if not q:
            continue
        entries.append(
            {
                "query": q,
                "category": str(item.get("category", "General")),
                "created_at": str(item.get("created_at", "")),
            }
        )
    return entries


def _read_answer_entries() -> list[CachedAskEntry]:
    _ensure_store()
    try:
        body = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    raw_entries = body.get("answers", [])
    if not isinstance(raw_entries, list):
        return []
    entries: list[CachedAskEntry] = []
    for item in raw_entries:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", "")).strip()
        response = item.get("response")
        kb_signature = str(item.get("kb_signature", "")).strip()
        if not key or not isinstance(response, dict) or not kb_signature:
            continue
        entries.append(
            {
                "key": key,
                "question_norm": str(item.get("question_norm", "")),
                "employee_id": str(item.get("employee_id", "")) or None,
                "use_rag": bool(item.get("use_rag", True)),
                "kb_signature": kb_signature,
                "response": response,
                "created_at": str(item.get("created_at", "")),
            }
        )
    return entries


def _write_store(entries: list[CacheEntry], answers: list[CachedAskEntry]) -> None:
    _ensure_store()
    payload = {"entries": entries, "answers": answers}
    _CACHE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _infer_category(text: str) -> str:
    q = text.lower()
    if any(t in q for t in ("pto", "vacation", "leave", "sick day")):
        return "PTO"
    if any(t in q for t in ("vpn", "globalprotect", "gateway", "it")):
        return "VPN"
    if any(t in q for t in ("my", "employee", "balance", "remaining", "days do i have")):
        return "Personal"
    return "General"


def _project_text(text: str) -> tuple[float, float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    x_int = int.from_bytes(digest[:8], "big", signed=False)
    y_int = int.from_bytes(digest[8:16], "big", signed=False)
    max_u64 = float(2**64 - 1)
    x = (x_int / max_u64) * 2.0 - 1.0
    y = (y_int / max_u64) * 2.0 - 1.0
    return round(x, 6), round(y, 6)


def add_query(query: str, *, created_at: str) -> None:
    cleaned = query.strip()
    if not cleaned:
        return
    entry: CacheEntry = {
        "query": cleaned,
        "category": _infer_category(cleaned),
        "created_at": created_at,
    }
    with _LOCK:
        entries = _read_entries()
        answers = _read_answer_entries()
        entries.append(entry)
        # Keep recent window bounded.
        _write_store(entries[-1000:], answers[-1000:])


def get_cached_answer(
    question: str,
    employee_id: str | None,
    use_rag: bool,
    *,
    kb_signature: str | None = None,
) -> dict | None:
    key = _ask_key(question, employee_id, use_rag)
    signature = kb_signature or _fixtures_signature()
    with _LOCK:
        answers = _read_answer_entries()
    for item in reversed(answers):
        if item["key"] != key:
            continue
        if item["kb_signature"] != signature:
            continue
        return dict(item["response"])
    return None


def has_stale_cached_answer(
    question: str,
    employee_id: str | None,
    use_rag: bool,
    *,
    kb_signature: str | None = None,
) -> bool:
    key = _ask_key(question, employee_id, use_rag)
    signature = kb_signature or _fixtures_signature()
    with _LOCK:
        answers = _read_answer_entries()
    for item in reversed(answers):
        if item["key"] != key:
            continue
        return item["kb_signature"] != signature
    return False


def put_cached_answer(
    question: str,
    employee_id: str | None,
    use_rag: bool,
    response: dict,
) -> None:
    key = _ask_key(question, employee_id, use_rag)
    now = datetime.now(timezone.utc).isoformat()
    kb_signature = _fixtures_signature()
    entry: CachedAskEntry = {
        "key": key,
        "question_norm": _normalize_question(question),
        "employee_id": employee_id,
        "use_rag": use_rag,
        "kb_signature": kb_signature,
        "response": response,
        "created_at": now,
    }
    with _LOCK:
        entries = _read_entries()
        answers = _read_answer_entries()
        answers = [a for a in answers if a["key"] != key]
        answers.append(entry)
        _write_store(entries[-1000:], answers[-1000:])


def get_stats() -> dict[str, int]:
    with _LOCK:
        entries = _read_entries()
        answers = _read_answer_entries()
    return {
        "total_cached_queries": len(answers),
        "total_queries_seen": len(entries),
    }


def purge_cache() -> dict[str, int]:
    with _LOCK:
        before = len(_read_answer_entries())
        _write_store([], [])
    return {"purged": before, "total_cached_queries": 0}


def _chunk_points(limit: int = 120) -> list[VizPoint]:
    out: list[VizPoint] = []
    try:
        vs = get_vectorstore()
        collection = vs._collection  # noqa: SLF001 - needed for raw docs used in viz
        data = collection.get(include=["documents", "metadatas"])
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        for idx, doc in enumerate(docs[:limit]):
            text = str(doc or "").strip()
            if not text:
                continue
            meta = metas[idx] if idx < len(metas) and isinstance(metas[idx], dict) else {}
            source_name = str(meta.get("source_name", "chunk"))
            section = str(meta.get("section_title", "")).strip()
            label = section or f"Chunk {idx + 1}"
            category = _infer_category(f"{label} {source_name} {text[:80]}")
            x, y = _project_text(f"chunk::{source_name}::{label}::{text}")
            out.append(
                {
                    "x": x,
                    "y": y,
                    "text": text[:400],
                    "label": label[:120],
                    "kind": "chunk",
                    "category": category,
                    "source": Path(source_name).name,
                }
            )
    except Exception:
        return []
    return out


def get_viz_points() -> dict[str, list[VizPoint]]:
    with _LOCK:
        entries = _read_entries()
    query_points: list[VizPoint] = []
    for idx, entry in enumerate(entries[-200:]):
        x, y = _project_text(f"query::{entry['query']}::{entry['created_at']}::{idx}")
        query_points.append(
            {
                "x": x,
                "y": y,
                "text": entry["query"],
                "label": f"Query {idx + 1}",
                "kind": "query",
                "category": entry["category"],
                "source": "semantic_cache",
            }
        )
    return {"points": [*query_points, *_chunk_points(limit=120)]}
