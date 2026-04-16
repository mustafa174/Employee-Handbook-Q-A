from app.main import _sanitize_cached_policy_leak


def test_sanitize_cached_policy_leak_removes_balance_line() -> None:
    raw = {"answer": "You currently have 14 PTO days and 6 sick days remaining. PTO requests must be submitted 10 business days in advance."}
    cleaned = _sanitize_cached_policy_leak("How far in advance should I request PTO?", raw)
    assert "14 PTO days" not in str(cleaned.get("answer", ""))
    assert "10 business days in advance" in str(cleaned.get("answer", ""))


def test_sanitize_cached_policy_leak_keeps_balance_when_explicitly_asked() -> None:
    raw = {"answer": "You currently have 14 PTO days and 6 sick days remaining."}
    cleaned = _sanitize_cached_policy_leak("How many PTO days do I have remaining?", raw)
    assert "14 PTO days" in str(cleaned.get("answer", ""))


def test_sanitize_cached_policy_leak_replaces_balance_only_answer() -> None:
    raw = {"answer": "You currently have 14 PTO days and 6 sick days remaining."}
    cleaned = _sanitize_cached_policy_leak("How far in advance should I request PTO?", raw)
    answer = str(cleaned.get("answer", ""))
    assert "14 PTO days" not in answer
    assert "10 business days" in answer
