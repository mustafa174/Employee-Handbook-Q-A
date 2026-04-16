"""Generic resolver that maps profile queries to profile keys."""

from __future__ import annotations

import re
from typing import TypedDict

from app.profile_field_catalog import PROFILE_FIELD_CATALOG, ProfileFieldMeta


class ResolveResult(TypedDict):
    resolved_key: str | None
    confidence: float
    reason: str
    needs_clarification: bool


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _canonical(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


def _meta_for_key(key: str) -> ProfileFieldMeta:
    if key in PROFILE_FIELD_CATALOG:
        return PROFILE_FIELD_CATALOG[key]
    # Dynamic fallback for unknown fields in employee JSON.
    human = key.replace("_", " ")
    return {
        "aliases": [human],
        "description": f"Employee profile field: {human}.",
        "field_type": "string",
    }


def _candidate_score(query: str, key: str, meta: ProfileFieldMeta) -> float:
    q = _canonical(query)
    q_tokens = _tokens(q)
    if not q_tokens:
        return 0.0
    aliases = list(meta.get("aliases", []))
    aliases.append(key.replace("_", " "))
    best = 0.0
    for alias in aliases:
        a = _canonical(alias)
        if not a:
            continue
        a_tokens = _tokens(a)
        if not a_tokens:
            continue
        if len(a_tokens) == 1:
            # Avoid substring false positives like "laptop" -> "pto".
            token = next(iter(a_tokens))
            if token in q_tokens:
                best = max(best, 1.0)
                continue
        elif re.search(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])", q):
            best = max(best, 1.0)
            continue
        overlap = len(q_tokens & a_tokens) / max(1, len(a_tokens))
        best = max(best, overlap)
    return best


def resolve_profile_field(
    query: str,
    profile: dict[str, str],
    *,
    last_resolved_key: str | None = None,
) -> ResolveResult:
    if not profile:
        return {
            "resolved_key": None,
            "confidence": 0.0,
            "reason": "no_profile",
            "needs_clarification": False,
        }

    q = _canonical(query)
    asks_eligibility = bool(re.search(r"\bam i eligible\b|\beligib(?:le|ility)\b", q))
    asks_amount = bool(re.search(r"\bhow much\b|\blimit\b|\bamount\b|\bmaximum\b", q))

    if asks_eligibility and last_resolved_key:
        last_meta = _meta_for_key(last_resolved_key)
        elig_key = str(last_meta.get("eligibility_key", "")).strip()
        if elig_key and elig_key in profile:
            return {
                "resolved_key": elig_key,
                "confidence": 0.95,
                "reason": "followup_eligibility_from_last_key",
                "needs_clarification": False,
            }

    # Catalog questions ("types of loans") are handled by loan composite in rag_graph;
    # do not let token "type" resolve to employment_type.
    if re.search(r"\b(loans?)\b", q) and re.search(r"\b(type|types|kind|kinds)\b", q):
        return {
            "resolved_key": None,
            "confidence": 0.0,
            "reason": "loan_type_catalog_defer_to_composite",
            "needs_clarification": False,
        }

    # Explicit loan-topic fallbacks.
    if asks_eligibility and "services_loan_available" in profile and "loan" in q:
        return {
            "resolved_key": "services_loan_available",
            "confidence": 0.95,
            "reason": "loan_eligibility_direct",
            "needs_clarification": False,
        }
    if asks_amount and "services_loan_limit_pkr" in profile and "loan" in q:
        return {
            "resolved_key": "services_loan_limit_pkr",
            "confidence": 0.95,
            "reason": "loan_amount_direct",
            "needs_clarification": False,
        }

    best_key: str | None = None
    best_score = 0.0
    second_score = 0.0
    for key in profile:
        meta = _meta_for_key(key)
        score = _candidate_score(q, key, meta)
        if score > best_score:
            second_score = best_score
            best_score = score
            best_key = key
        elif score > second_score:
            second_score = score

    if not best_key:
        return {
            "resolved_key": None,
            "confidence": 0.0,
            "reason": "no_match",
            "needs_clarification": True,
        }

    ambiguous = best_score < 0.45 or (best_score - second_score) < 0.1
    return {
        "resolved_key": best_key,
        "confidence": round(best_score, 3),
        "reason": "alias_overlap",
        "needs_clarification": ambiguous,
    }
