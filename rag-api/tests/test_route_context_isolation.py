from app.rag_graph import INTENT_PERSONAL, INTENT_POLICY, node_balance, node_generate, route_after_grade


def test_node_balance_skips_non_personal_route() -> None:
    state = {
        "intent": INTENT_POLICY,
        "use_rag": True,
        "employee_id": "E001",
    }
    out = node_balance(state)
    assert out.get("balance_snippet") is None
    assert out.get("employee_profile") is None
    assert "only for personal route" in str(out.get("mcp_status_detail", "")).lower()


def test_node_balance_loads_profile_for_personal_route() -> None:
    state = {
        "intent": INTENT_PERSONAL,
        "use_rag": True,
        "employee_id": "E001",
    }
    out = node_balance(state)
    assert out.get("balance_snippet")
    assert isinstance(out.get("employee_profile"), dict)
    assert out.get("employee_profile")


def test_route_after_grade_goes_straight_to_generate() -> None:
    state = {"retrieval_verdict": "ANSWERABLE"}
    assert route_after_grade(state) == "generate"


def test_node_generate_it_invariant_blocks_profile_context() -> None:
    state = {
        "intent": INTENT_POLICY,
        "intent_domain": "IT",
        "route": "policy",
        "use_rag": True,
        "question": "VPN is not connecting",
        "employee_profile": {"employee_id": "E001"},
        "balance_snippet": "User Profile (secure HR context): ...",
        "retrieval_citations": [],
        "sub_questions": [],
        "sub_results": [],
        "messages": [],
    }
    out = node_generate(state)
    assert "Jira Service Desk" in str(out.get("answer", ""))
    assert out.get("retrieval_citations") == []
