"""Low-latency in-scope term index for handbook/IT routing."""

from __future__ import annotations

import re
from typing import TypedDict

from app.vectorstore import get_vectorstore


class ScopeSignal(TypedDict):
    in_scope: bool
    match_ratio: float
    hits: list[str]
    tokens: list[str]


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "for",
    "of",
    "in",
    "on",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "can",
    "could",
    "should",
    "would",
    "i",
    "my",
    "me",
    "we",
    "our",
    "you",
    "your",
    "what",
    "which",
    "who",
    "where",
    "when",
    "why",
    "how",
    "about",
    "regarding",
    "policy",
    "company",
    "please",
    "want",
    "know",
    "tell",
    "need",
    "help",
    "get",
    "give",
    "show",
    "explain",
    "understand",
    "need",
}
_BASE_SCOPE_TERMS = {
    "pto",
    "sick",
    "leave",
    "rollover",
    "vacation",
    "approval",
    "notice",
    "benefit",
    "loan",
    "eligible",
    "eligibility",
    "department",
    "location",
    "language",
    "salary",
    "vpn",
    "globalprotect",
    "gateway",
    "laptop",
    "hardware",
    "peripherals",
    "monitor",
    "keyboard",
    "mouse",
    "jira",
    "ticket",
    "servicedesk",
    "helpdesk",
    "email",
    "password",
    "login",
    "account",
    "access",
    "remote",
    "work",
}
_STRONG_SCOPE_TERMS = {
    "pto",
    "sick",
    "leave",
    "loan",
    "eligibility",
    "vpn",
    "globalprotect",
    "laptop",
    "hardware",
    "jira",
    "ticket",
    "helpdesk",
    "servicedesk",
    "hr",
    "policy",
    "handbook",
}
_CACHE: dict[str, set[str]] = {"terms": set()}


def _normalized_tokens(text: str) -> list[str]:
    raw = _TOKEN_RE.findall((text or "").lower())
    return [t for t in raw if len(t) >= 3 and t not in _STOPWORDS]


def _build_terms_from_collection() -> set[str]:
    terms: set[str] = set(_BASE_SCOPE_TERMS)
    try:
        vs = get_vectorstore()
        collection = vs._collection  # noqa: SLF001
        data = collection.get(include=["metadatas"], limit=500)
        metas = data.get("metadatas") or []
        for meta in metas:
            if not isinstance(meta, dict):
                continue
            section = str(meta.get("section_title") or "")
            source_name = str(meta.get("source_name") or "")
            for token in _normalized_tokens(section):
                terms.add(token)
            for token in _normalized_tokens(source_name):
                terms.add(token)
    except Exception:
        # If collection is unavailable (cold start/test), base terms still provide useful gating.
        return terms
    return terms


def _scope_terms() -> set[str]:
    cached = _CACHE.get("terms") or set()
    if cached:
        return cached
    built = _build_terms_from_collection()
    _CACHE["terms"] = built
    return built


def invalidate_scope_index() -> None:
    _CACHE["terms"] = set()


def query_scope_signal(question: str) -> ScopeSignal:
    tokens = _normalized_tokens(question)
    if not tokens:
        return {"in_scope": True, "match_ratio": 1.0, "hits": [], "tokens": []}
    terms = _scope_terms()
    hits = [t for t in tokens if t in terms]
    ratio = len(hits) / max(1, len(tokens))
    # Require majority lexical overlap for non-trivial queries to avoid topic drift.
    strong_overlap = bool(set(tokens) & _STRONG_SCOPE_TERMS)
    in_scope = ratio >= 0.6 or (len(tokens) == 1 and ratio >= 0.999) or strong_overlap
    return {
        "in_scope": in_scope,
        "match_ratio": round(ratio, 3),
        "hits": sorted(set(hits)),
        "tokens": tokens,
    }
