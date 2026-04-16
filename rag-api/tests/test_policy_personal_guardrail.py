from app.rag_graph import _explicit_personal_data_request, _strict_grounding_fallback


def test_personal_data_request_detection() -> None:
    assert not _explicit_personal_data_request("How far in advance should I request PTO?")
    assert not _explicit_personal_data_request("What is my PTO policy?")
    assert _explicit_personal_data_request("How many PTO and sick days do I have remaining?")


def test_strict_grounding_fallback_hides_balance_for_policy_only_question() -> None:
    snippet = (
        "User Profile (secure HR context):\n"
        "- Employee: Alex Chen (E001)\n"
        "- PTO days remaining: 14\n"
        "- Sick days remaining: 6\n"
    )
    answer = _strict_grounding_fallback(
        snippet,
        with_people_partner_prompt=True,
        allow_personal_summary=False,
    )
    assert "You currently have 14 PTO days and 6 sick days remaining." not in answer
