from app.rag_graph import (
    CRISIS_PATTERNS,
    SENSITIVE_PATTERNS,
    _build_harassment_agent_action,
    detect_sensitive,
    query_refiner_node,
)


def test_sensitive_patterns_match() -> None:
    assert SENSITIVE_PATTERNS.search("I want to file a lawsuit")
    assert SENSITIVE_PATTERNS.search("harassment complaint")
    assert SENSITIVE_PATTERNS.search("harrasment complaint")
    assert SENSITIVE_PATTERNS.search("harrasmsent complaint")
    assert SENSITIVE_PATTERNS.search("harasment concern")
    assert SENSITIVE_PATTERNS.search("i am harassed")
    assert SENSITIVE_PATTERNS.search("i am harrassed")
    assert SENSITIVE_PATTERNS.search("wrongful termination")


def test_guardrail_semantic_sensitive_phrases_match() -> None:
    from app.rag_graph import SENSITIVE_SEMANTIC_PATTERNS
    import re

    queries = [
        "i am being bullied",
        "my manager is abusing me",
        "my manager is treating me badle",
        "my manager is treating me badly",
        "i feel unsafe at work",
        "i got fired unfairly",
        "i am harrased by my manager",
    ]
    for q in queries:
        assert any(re.search(pattern, q, re.I) for pattern in SENSITIVE_SEMANTIC_PATTERNS)


def test_detect_sensitive_handles_real_user_sensitive_phrases() -> None:
    assert detect_sensitive("i am being bullied")
    assert detect_sensitive("my manager is abusing me")
    assert detect_sensitive("my manager is treating me badly")
    assert detect_sensitive("my manager is treating me badle")
    assert detect_sensitive("i feel unsafe at work")
    assert detect_sensitive("i got fired unfairly")
    assert detect_sensitive("i am harrased by my manager")


def test_detect_sensitive_does_not_flag_accommodation_requests() -> None:
    assert not detect_sensitive("i want to do accommodation requests")
    assert not detect_sensitive("i need disability accommodation support")


def test_query_refiner_short_circuits_sensitive_prompt_before_router() -> None:
    out = query_refiner_node({"question": "my manager is abusing me", "employee_id": "E001", "use_rag": True})
    assert out.get("route") == "sensitive"
    assert bool(out.get("escalate")) is True
    assert out.get("use_rag") is False


def test_harassment_action_is_built_for_abuse_unsafe_and_treating_badly() -> None:
    state = {
        "question": "My manager is treating me badly and I feel unsafe at work",
        "employee_id": "E001",
        "employee_profile": {"name": "Test Employee"},
    }
    action = _build_harassment_agent_action(state)
    assert action is not None
    assert action.get("type") == "HARASSMENT_REPORT"


def test_sensitive_patterns_no_match() -> None:
    assert not SENSITIVE_PATTERNS.search("How many PTO days do I get?")


def test_crisis_patterns_match() -> None:
    assert CRISIS_PATTERNS.search("I am dying")
    assert CRISIS_PATTERNS.search("I want to kill myself")
