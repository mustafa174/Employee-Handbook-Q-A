from app.rag_graph import CRISIS_PATTERNS, SENSITIVE_PATTERNS


def test_sensitive_patterns_match() -> None:
    assert SENSITIVE_PATTERNS.search("I want to file a lawsuit")
    assert SENSITIVE_PATTERNS.search("harassment complaint")
    assert SENSITIVE_PATTERNS.search("wrongful termination")


def test_sensitive_patterns_no_match() -> None:
    assert not SENSITIVE_PATTERNS.search("How many PTO days do I get?")


def test_crisis_patterns_match() -> None:
    assert CRISIS_PATTERNS.search("I am dying")
    assert CRISIS_PATTERNS.search("I want to kill myself")
