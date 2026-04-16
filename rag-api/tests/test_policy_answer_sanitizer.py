from app.rag_graph import _strip_unrequested_personal_balance


def test_strip_unrequested_balance_from_policy_answer() -> None:
    answer = (
        "You currently have 14 PTO days and 6 sick days remaining. "
        "PTO requests must be submitted at least 10 business days in advance."
    )
    cleaned = _strip_unrequested_personal_balance(answer, allow_personal_summary=False)
    assert "14 PTO days" not in cleaned
    assert "10 business days in advance" in cleaned


def test_keep_balance_when_personal_summary_allowed() -> None:
    answer = "You currently have 14 PTO days and 6 sick days remaining."
    cleaned = _strip_unrequested_personal_balance(answer, allow_personal_summary=True)
    assert cleaned == answer
