"""Deterministic query intent policy with confidence scoring."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
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
    "name"
}
_POLICY_TOKENS = {
    "policy",
    "polocy",
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
    "vacation",
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
_LOAN_PERSONAL_SIGNALS = (
    "am i",
    "can i",
    "eligible",
    "for me",
    "how many months",
    "after how many months",
    "when can i",
    "when am i",
)
_LOAN_POLICY_SIGNALS = (
    "loan policy",
    "loan rules",
    "company loan policy",
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
        "timestamp": int(time.time() * 1000),
    }
    try:
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass


def _normalize_classifier_text(text: str) -> str:
    q = (text or "").lower().strip()
    replacements = {
        "laptap": "laptop",
        "lptop": "laptop",
        "labtop": "laptop",
        "damge": "damage",
        "dammage": "damage",
        "damagede": "damaged",
        "vacation days": "pto days",
        "vacation day": "pto day",
        "days off": "pto",
    }
    for src, dest in replacements.items():
        q = q.replace(src, dest)
    q = re.sub(r"\bis damage\b", "is damaged", q)
    q = re.sub(r"\bgot damage\b", "got damaged", q)
    return q

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
    if re.search(
        r"\bwhat is .*(?:policy|polocy)\b|\b(?:policy|polocy) for\b|\bcompany .*(?:policy|polocy)\b|\bwhat (?:policy|polocy)\b|\b(?:policy|polocy) covers\b",
        question,
    ):
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
    return bool(re.search(r"\b(my|mine|i have|do i have|am i|for me)\b", question))


def classify_query(question: str) -> IntentPolicyResult:
    q = _normalize_classifier_text(question)
    if "vacation" in (question or "").lower() or "days off" in (question or "").lower() or "holiday" in (question or "").lower():
        # #region agent log
        _agent_debug_log(
            "repro-vocab-1",
            "H1",
            "intent_policy.py:classify_query",
            "Classifier normalization snapshot",
            {
                "raw_question": (question or "")[:160],
                "normalized_question": q[:160],
            },
        )
        # #endregion
    tokens = _tokens(q)
    if not tokens:
        return {
            "domain_class": "OOS",
            "confidence": 0.2,
            "reasons": ["empty_query"],
            "secondary_class": None,
            "needs_clarification": True,
        }

    if "loan" in q and any(signal in q for signal in _LOAN_POLICY_SIGNALS):
        return {
            "domain_class": "POLICY",
            "confidence": 0.95,
            "reasons": ["loan_policy_phrase"],
            "secondary_class": "PROFILE",
            "needs_clarification": False,
        }

    if "loan" in q and any(signal in q for signal in _LOAN_PERSONAL_SIGNALS):
        return {
            "domain_class": "PROFILE",
            "confidence": 0.95,
            "reasons": ["loan_personal_phrase"],
            "secondary_class": "POLICY",
            "needs_clarification": False,
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

    # "Employee handbook" refers to policy docs, not personal profile fields.
    if "handbook" in tokens and not has_pronoun:
        primary, primary_score = "POLICY", max(scores["POLICY"], 0.7)
        secondary = "PROFILE" if scores["PROFILE"] >= scores["IT"] else "IT"

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
    if "vacation" in q or ("leave" in q and ("days" in q or "how many" in q)):
        # #region agent log
        _agent_debug_log(
            "repro-vacation-1",
            "H1",
            "intent_policy.py:classify_query",
            "Vacation/leave classification snapshot",
            {
                "question": q[:160],
                "tokens": sorted(list(tokens))[:20],
                "scores": score_summary,
                "primary": primary,
                "secondary": secondary,
                "has_hr_hint": has_hr_hint,
                "has_in_scope_hint": has_in_scope_hint,
                "has_oos_pattern": has_oos_pattern,
                "only_generic_scope_hints": only_generic_scope_hints,
            },
        )
        # #endregion
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
