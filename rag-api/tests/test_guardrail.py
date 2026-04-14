from app.rag_graph import SENSITIVE_PATTERNS


def test_sensitive_patterns_match() -> None:
    assert SENSITIVE_PATTERNS.search("I want to file a lawsuit")
    assert SENSITIVE_PATTERNS.search("harassment complaint")
    assert SENSITIVE_PATTERNS.search("wrongful termination")


def test_sensitive_patterns_no_match() -> None:
    assert not SENSITIVE_PATTERNS.search("How many PTO days do I get?")
