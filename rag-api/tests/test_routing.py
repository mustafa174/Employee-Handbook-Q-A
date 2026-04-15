from app.rag_graph import INTENT_PERSONAL, INTENT_POLICY, route_after_router


def test_route_after_router_personal_goes_policy_for_process_terms() -> None:
    state = {
        "use_rag": True,
        "intent": INTENT_PERSONAL,
        "question": "How far in advance should I request PTO and how many days do I have left?",
        "needs_clarification": False,
    }
    assert route_after_router(state) == "policy"


def test_route_after_router_policy_stays_policy_for_policy_terms() -> None:
    state = {
        "use_rag": True,
        "intent": INTENT_POLICY,
        "question": "What is the policy when I request leave in advance?",
        "needs_clarification": False,
    }
    assert route_after_router(state) == "policy"
