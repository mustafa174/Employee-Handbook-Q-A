"""Semantic rescue routing for weak/general queries."""

from __future__ import annotations

import os
from typing import TypedDict

_MODEL_NAME = os.getenv("SEMANTIC_ROUTER_MODEL", "intfloat/e5-base-v2")
_THRESHOLD = float(os.getenv("SEMANTIC_ROUTER_THRESHOLD", "0.58"))
_LOCAL_ONLY = os.getenv("SEMANTIC_ROUTER_LOCAL_ONLY", "1").strip().lower() in {"1", "true", "yes", "on"}

_MODEL = None
_LABELS = (
    "personal",
    "policy",
    "general",
)
_LABEL_PASSAGES = (
    "passage: Employee asks about their own profile or balances, such as my leave, my PTO, my sick days, my loan eligibility, or how many days I have.",
    "passage: Employee asks about company handbook policy, process, requirements, IT support rules, PTO policy, leave rules, hardware replacement, VPN, or onboarding process.",
    "passage: Query is outside company HR, IT, personal employee profile, or handbook policy scope.",
)


class SemanticRouteDecision(TypedDict):
    route: str
    score: float
    label: str
    model: str


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        _MODEL = SentenceTransformer(_MODEL_NAME, local_files_only=_LOCAL_ONLY)
    except Exception:
        _MODEL = False
    return _MODEL


def semantic_rescue_route(question: str, *, has_employee_id: bool) -> SemanticRouteDecision | None:
    q = (question or "").strip()
    if not q:
        return None
    model = _load_model()
    if not model:
        return None
    query_text = f"query: {q}"
    try:
        vectors = model.encode(
            [query_text, *_LABEL_PASSAGES],
            normalize_embeddings=True,
            convert_to_numpy=False,
        )
    except Exception:
        return None
    q_vec = list(vectors[0])
    best_idx = -1
    best_score = -1.0
    for idx, label_vec in enumerate(vectors[1:]):
        score = sum(float(a) * float(b) for a, b in zip(q_vec, list(label_vec)))
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_idx < 0:
        return None
    best_label = _LABELS[best_idx]
    if best_score < _THRESHOLD:
        return None
    if best_label == "personal":
        return {
            "route": "personal" if has_employee_id else "policy",
            "score": round(best_score, 3),
            "label": best_label,
            "model": _MODEL_NAME,
        }
    if best_label == "policy":
        return {
            "route": "policy",
            "score": round(best_score, 3),
            "label": best_label,
            "model": _MODEL_NAME,
        }
    return None
