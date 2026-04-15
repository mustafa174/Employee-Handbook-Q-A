"""Deterministic query intent policy with confidence scoring."""

from __future__ import annotations

import re
from typing import Literal, TypedDict

DomainClass = Literal["PROFILE", "POLICY", "IT", "OOS"]


class IntentPolicyResult(TypedDict):
    domain_class: DomainClass
    confidence: float
    reasons: list[str]
    secondary_class: DomainClass | None
    needs_clarification: bool


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_PROFILE_TOKENS = {
    "employee",
    "department",
    "location",
    "language",
    "salary",
    "loan",
    "eligible",
    "eligibility",
    "limit",
    "balance",
    "pto",
    "sick",
    "remaining",
    "leaves",
    "quota",
    "qouta",
    "employment",
    "type",
    "profile",
    "details",
}
_POLICY_TOKENS = {
    "policy",
    "policies",
    "rule",
    "rules",
    "required",
    "requirement",
    "requirements",
    "request",
    "requests",
    "process",
    "approval",
    "notice",
    "rollover",
    "allowance",
    "benefit",
    "pto",
    "leave",
    "leaves",
    "handbook",
    "carry",
    "unused",
    "time",
    "off",
    "holiday",
    "hr",
    "contact",
    "human",
    "resources",
    "onboarding",
}
_IT_TOKENS = {
    "laptop",
    "computer",
    "device",
    "hardware",
    "keyboard",
    "mouse",
    "monitor",
    "screen",
    "battery",
    "charger",
    "vpn",
    "globalprotect",
    "gateway",
    "network",
    "wifi",
    "internet",
    "email",
    "outlook",
    "password",
    "login",
    "signin",
    "access",
    "account",
    "jira",
    "ticket",
    "helpdesk",
    "servicedesk",
    "npm",
    "build",
    "code",
}
_PROFILE_ALIAS_TOKENS = {
    "employee",
    "department",
    "location",
    "language",
    "salary",
    "loan",
    "eligible",
    "eligibility",
    "limit",
    "balance",
    "pto",
    "sick",
    "remaining",
    "leave",
    "leaves",
    "quota",
    "qouta",
    "manager",
    "title",
    "employment",
    "type",
    "profile",
    "details",
}
_IN_SCOPE_HR_TOKENS = {
    "pto",
    "time",
    "off",
    "request",
    "hr",
    "human",
    "resources",
    "leave",
    "leaves",
    "contact",
    "holiday",
    "policy",
    "handbook",
    "onboarding",
}
_GENERIC_SCOPE_HINTS = {"policy", "company", "rule", "rules", "process", "request", "requests"}
_IN_SCOPE_IT_TOKENS = {
    "npm",
    "build",
    "code",
    "vpn",
    "access",
    "laptop",
    "password",
    "login",
    "email",
    "ticket",
    "jira",
    "helpdesk",
    "servicedesk",
}
_OOS_TOKENS = {
    "mars",
    "planet",
    "weather",
    "bitcoin",
    "cricket",
    "football",
    "movie",
    "song",
    "recipe",
    "stock",
    "crypto",
    "astrology",
    "horoscope",
    "poem",
    "rain",
    "black",
    "holes",
}
_OOS_PATTERN = re.compile(
    r"\b("
    r"mars|jupiter|saturn|uranus|neptune|pluto|planet|galaxy|solar system|"
    r"astrophysics|astronomy|zodiac|horoscope|recipe|poem|lyrics|movie|bitcoin|crypto"
    r")\b",
    re.I,
)


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _token_hits(tokens: set[str], vocab: set[str]) -> int:
    return sum(1 for t in tokens if t in vocab)


def _profile_score(question: str, tokens: set[str]) -> float:
    score = float(_token_hits(tokens, _PROFILE_TOKENS))
    if re.search(r"\b(?:what|which|show|tell|give)\s+(?:is\s+)?my\s+[a-z0-9 _-]{2,40}\b", question):
        score += 2.0
    if re.search(r"\bmy\s+profile\b|\bprofile\s+details\b", question):
        score += 1.5
    if re.search(r"\bam i eligible for\b", question):
        score += 2.0
    if re.search(r"\bhow much\b.*\bcan i\b", question):
        score += 1.5
    return score


def _policy_score(question: str, tokens: set[str]) -> float:
    score = float(_token_hits(tokens, _POLICY_TOKENS))
    if re.search(r"\bwhat is .*policy\b|\bpolicy for\b|\bcompany .*policy\b|\bwhat policy\b|\bpolicy covers\b", question):
        score += 2.0
    if re.search(r"\bhow far in advance\b|\badvance notice\b", question):
        score += 1.5
    if re.search(r"\bcarry over\b|\broll over\b", question):
        score += 1.5
    return score


def _it_score(question: str, tokens: set[str]) -> float:
    base_hits = float(_token_hits(tokens, _IT_TOKENS))
    score = base_hits * 0.8
    has_issue_signal = bool(re.search(r"\b(not working|failing|error|issue|problem|can't|cannot|unable)\b", question))
    if has_issue_signal:
        score += 1.5
    if re.search(r"\b(login|password|vpn|email|laptop|device|access)\b", question):
        score += 1.0
    if not has_issue_signal and "policy" in tokens:
        score = max(0.0, score - 0.8)
    return score


def _oos_score(question: str, tokens: set[str]) -> float:
    score = float(_token_hits(tokens, _OOS_TOKENS))
    if _OOS_PATTERN.search(question):
        score += 1.5
    return score


def _has_personal_pronoun(question: str) -> bool:
    return bool(re.search(r"\b(my|i|me|mine)\b", question))


def classify_query(question: str) -> IntentPolicyResult:
    q = question.lower().strip()
    tokens = _tokens(q)
    if not tokens:
        return {
            "domain_class": "OOS",
            "confidence": 0.2,
            "reasons": ["empty_query"],
            "secondary_class": None,
            "needs_clarification": True,
        }

    scores: dict[DomainClass, float] = {
        "PROFILE": _profile_score(q, tokens),
        "POLICY": _policy_score(q, tokens),
        "IT": _it_score(q, tokens),
        "OOS": _oos_score(q, tokens),
    }
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary, primary_score = ranked[0]
    secondary, _ = ranked[1]
    has_pronoun = _has_personal_pronoun(q)
    has_profile_alias = bool(tokens & _PROFILE_ALIAS_TOKENS)
    has_hr_hint = bool(tokens & _IN_SCOPE_HR_TOKENS)
    has_it_hint = bool(tokens & _IN_SCOPE_IT_TOKENS)
    has_in_scope_hint = has_hr_hint or has_it_hint
    has_oos_pattern = bool(_OOS_PATTERN.search(q))

    # Strong IT indicators should dominate generic "my" personal cues.
    if primary != "IT" and scores["IT"] >= 2.0 and scores["IT"] >= primary_score:
        primary, primary_score = "IT", scores["IT"]
        secondary = ranked[0][0] if ranked[0][0] != "IT" else ranked[1][0]

    # Restrict personal intent to true personal/profile phrasing.
    if primary == "PROFILE" and not (has_pronoun and has_profile_alias):
        if scores["IT"] >= scores["POLICY"] and (scores["IT"] > 0 or has_it_hint):
            primary, primary_score = "IT", scores["IT"]
            secondary = "POLICY"
        else:
            primary, primary_score = "POLICY", scores["POLICY"]
            secondary = "IT"

    # Out-of-scope entities should override policy wording (e.g., "policy about Mars").
    if (
        scores["OOS"] >= 1.0
        and scores["PROFILE"] < 2.0
        and scores["IT"] < 2.0
        and (not has_in_scope_hint or has_oos_pattern)
    ):
        primary, primary_score = "OOS", scores["OOS"]
        ranked_non_oos = sorted(
            [(k, v) for k, v in scores.items() if k != "OOS"],
            key=lambda x: x[1],
            reverse=True,
        )
        secondary, _ = ranked_non_oos[0]

    # If HR/IT scope hints exist, never force OOS.
    only_generic_scope_hints = bool(tokens & _GENERIC_SCOPE_HINTS) and not bool(
        (tokens & _IN_SCOPE_HR_TOKENS) - _GENERIC_SCOPE_HINTS
    )
    if primary == "OOS" and has_in_scope_hint and not (has_oos_pattern and only_generic_scope_hints):
        if has_it_hint and (scores["IT"] >= scores["POLICY"]):
            primary, primary_score = "IT", max(scores["IT"], 0.6)
            secondary = "POLICY"
        else:
            primary, primary_score = "POLICY", max(scores["POLICY"], 0.6)
            secondary = "IT"

    total = sum(scores.values())
    confidence = 0.0 if total <= 0 else primary_score / total
    confidence = round(max(0.0, min(1.0, confidence)), 3)

    needs_clarification = (
        primary == "PROFILE"
        and has_pronoun
        and has_profile_alias
        and confidence < 0.45
    )

    score_summary = {
        "PROFILE": round(scores["PROFILE"], 1),
        "POLICY": round(scores["POLICY"], 1),
        "IT": round(scores["IT"], 1),
        "OOS": round(scores["OOS"], 1),
    }
    reasons = [
        f"scores={score_summary}",
        f"primary={primary}",
        f"secondary={secondary}",
        f"hints(hr={has_hr_hint}, it={has_it_hint}, pronoun={has_pronoun}, profile_alias={has_profile_alias})",
    ]
    return {
        "domain_class": primary,
        "confidence": confidence,
        "reasons": reasons,
        "secondary_class": secondary,
        "needs_clarification": needs_clarification,
    }


def map_domain_to_intent(domain_class: DomainClass) -> str:
    if domain_class == "PROFILE":
        return "INTENT_PERSONAL"
    if domain_class in {"POLICY", "IT"}:
        return "INTENT_POLICY"
    return "INTENT_OUT_OF_DOMAIN"
