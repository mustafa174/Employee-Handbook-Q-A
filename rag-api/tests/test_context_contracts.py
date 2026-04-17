from app.main import _enforce_response_contract, _should_bypass_cache
from app.rag_graph import run_ask
from app.rag_graph import (
    _is_explicit_mixed_query,
    _needs_mixed_clarification,
    build_ask_response_from_state,
)
from app.semantic_cache import _ask_key


def test_policy_response_context_presence_excludes_profile() -> None:
    state = {
        "intent_domain": "POLICY",
        "retrieved_context": "chunk text",
        "retrieval_citations": [{"text": "policy", "score": 0.9}],
        "employee_profile": None,
        "balance_snippet": None,
        "answer": "Policy answer",
        "retrieval_attempts": [],
    }
    out = build_ask_response_from_state(state, use_rag=True)
    presence = out.get("context_presence", {})
    assert bool(presence.get("has_policy")) is True
    assert bool(presence.get("has_profile")) is False


def test_personal_response_context_presence_excludes_policy() -> None:
    state = {
        "intent_domain": "PROFILE",
        "retrieved_context": "",
        "retrieval_citations": [],
        "employee_profile": {"employee_id": "E001"},
        "balance_snippet": "User Profile (secure HR context): ...",
        "answer": "Personal answer",
        "retrieval_attempts": [],
    }
    out = build_ask_response_from_state(state, use_rag=True)
    presence = out.get("context_presence", {})
    assert bool(presence.get("has_profile")) is True
    assert bool(presence.get("has_policy")) is False


def test_enforce_response_contract_skips_profile_missing_guard_without_request_employee() -> None:
    raw = {
        "route": "PROFILE",
        "context_presence": {"has_policy": False, "has_profile": False, "has_it": False},
        "answer": (
            "I do not have an employee profile selected, so I cannot look up your personal PTO balance. "
            "Please select your employee profile in the assistant and ask again."
        ),
        "citations": [],
    }
    out = _enforce_response_contract("How many vacation days do I get?", raw, request_employee_id=None)
    assert "employee profile" in str(out.get("answer", "")).lower()
    assert bool(out.get("recovery_applied")) is False


def test_enforce_response_contract_profile_missing_with_employee_still_resets() -> None:
    raw = {
        "route": "PROFILE",
        "context_presence": {"has_policy": False, "has_profile": False, "has_it": False},
        "answer": "Some partial answer",
        "citations": [{"text": "x", "score": 0.9}],
    }
    out = _enforce_response_contract("How many PTO days do I have?", raw, request_employee_id="E001")
    assert "couldn't retrieve your personal data" in str(out.get("answer", "")).lower()
    assert out.get("citations") == []


def test_run_ask_vacation_no_employee_survives_contract_like_http_ask() -> None:
    raw = run_ask("How many vacation days do I get?", employee_id=None, use_rag=True)
    out = _enforce_response_contract("How many vacation days do I get?", raw, request_employee_id=None)
    assert "14" not in str(out.get("answer", "")).lower()
    assert "employee profile" in str(out.get("answer", "")).lower()


def test_enforce_response_contract_replaces_policy_leak() -> None:
    raw = {
        "route": "POLICY",
        "context_presence": {"has_policy": True, "has_profile": True, "has_it": False},
        "answer": "You currently have 14 PTO days and 6 sick days remaining.",
        "citations": [{"text": "x", "score": 0.8}],
    }
    cleaned = _enforce_response_contract("How far in advance should I request PTO?", raw)
    answer = str(cleaned.get("answer", ""))
    assert "You currently have 14 PTO days" not in answer
    assert ("I couldn't find the specific policy text in the handbook." in answer) or ("10 business days" in answer)


def test_mixed_detection_requires_explicit_personal_and_policy_operation() -> None:
    assert _is_explicit_mixed_query("How many PTO days do I have and how many can I carry over?")
    assert _is_explicit_mixed_query("What is the loan policy and how much loan can I get?")
    assert _is_explicit_mixed_query("Am I eligible for leave and what is the policy?")
    assert not _is_explicit_mixed_query("What is my company policy?")


def test_cache_key_includes_route_bucket() -> None:
    policy_key = _ask_key("How far in advance should I request PTO?", "E001", True)
    profile_key = _ask_key("How many PTO days do I have remaining?", "E001", True)
    assert "route=POLICY" in policy_key
    assert "route=PROFILE" in profile_key
    assert policy_key != profile_key


def test_cache_key_keeps_employee_scope_for_personal_route() -> None:
    e1_key = _ask_key("How many PTO days do I have remaining?", "E001", True)
    e2_key = _ask_key("How many PTO days do I have remaining?", "E002", True)
    assert e1_key != e2_key


def test_needs_mixed_clarification_for_weak_personal_plus_policy() -> None:
    assert _needs_mixed_clarification("What is my company policy?")
    assert not _needs_mixed_clarification("How many PTO days do I have and how many can I carry over?")


def test_enforce_response_contract_recovers_policy_text_before_fallback() -> None:
    raw = {
        "route": "POLICY",
        "context_presence": {"has_policy": True, "has_profile": True, "has_it": False},
        "answer": "You currently have 14 PTO days and 6 sick days remaining. PTO requests must be submitted at least 10 business days in advance.",
        "citations": [{"text": "x", "score": 0.8}],
    }
    cleaned = _enforce_response_contract("How far in advance should I request PTO?", raw)
    answer = str(cleaned.get("answer", ""))
    assert "14 PTO days" not in answer
    assert "10 business days" in answer


def test_enforce_response_contract_preserves_policy_constants() -> None:
    raw = {
        "route": "POLICY",
        "context_presence": {"has_policy": True, "has_profile": True, "has_it": False},
        "answer": "Unused PTO may roll over up to 5 days into the next calendar year.",
        "citations": [{"text": "x", "score": 0.8}],
    }
    cleaned = _enforce_response_contract("Can I carry over unused PTO?", raw)
    answer = str(cleaned.get("answer", ""))
    assert "5 days" in answer
    assert "Please contact HR." not in answer
    assert bool(cleaned.get("recovery_applied")) is True


def test_build_response_sets_clarification_triggered_flag() -> None:
    state = {
        "intent_domain": "POLICY",
        "mixed_clarification_needed": True,
        "retrieval_citations": [],
        "answer": "clarify",
        "retrieval_attempts": [],
    }
    out = build_ask_response_from_state(state, use_rag=True)
    assert bool(out.get("clarification_triggered")) is True


def test_sensitive_questions_bypass_cache() -> None:
    assert _should_bypass_cache("I want to report harassment by my manager.")
    assert _should_bypass_cache("i am harrassed")
    assert _should_bypass_cache("i am being bullied")
    assert _should_bypass_cache("my manager is abusing me")
    assert _should_bypass_cache("i feel unsafe at work")
    assert _should_bypass_cache("i got fired unfairly")
    assert not _should_bypass_cache("i need accommodation for medical issue")
    assert not _should_bypass_cache("How many PTO days do I have?")
