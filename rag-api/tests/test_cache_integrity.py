from app.semantic_cache import get_cached_answer, put_cached_answer, purge_cache


def test_cross_route_cache_poisoning_is_blocked() -> None:
    purge_cache()
    mixed_query = "How many PTO days do I have and how many can I carry over?"
    policy_query = "How many PTO days can carry over?"
    put_cached_answer(
        mixed_query,
        "E001",
        True,
        {
            "answer": "You currently have 14 PTO days and 6 sick days remaining. Unused PTO may roll over up to 5 days.",
            "route": "PROFILE",
        },
    )
    reused = get_cached_answer(policy_query, "E001", True)
    assert reused is None


def test_personal_cache_isolated_per_employee() -> None:
    purge_cache()
    query = "How many PTO days do I have remaining?"
    put_cached_answer(
        query,
        "E001",
        True,
        {"answer": "You currently have 14 PTO days remaining.", "route": "PROFILE"},
    )
    other = get_cached_answer(query, "E002", True)
    assert other is None


def test_mixed_not_reused_for_policy() -> None:
    purge_cache()
    mixed_query = "How many PTO days do I have and what is rollover?"
    policy_query = "What is PTO rollover?"
    put_cached_answer(
        mixed_query,
        "E001",
        True,
        {
            "answer": "You currently have 14 PTO days. Unused PTO may roll over up to 5 days.",
            "route": "PROFILE",
        },
    )
    reused = get_cached_answer(policy_query, None, True)
    assert reused is None
