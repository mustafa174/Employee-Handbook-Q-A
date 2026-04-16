from app.rag_graph import INTENT_PERSONAL, INTENT_POLICY, route_after_router


def test_route_after_router_personal_goes_policy_for_process_terms() -> None:
    state = {
        "use_rag": True,
        "intent": INTENT_PERSONAL,
        "question": "How far in advance should I request PTO and how many days do I have left?",
        "needs_clarification": False,
    }
    assert route_after_router(state) == "mixed"


def test_route_after_router_policy_stays_policy_for_policy_terms() -> None:
    state = {
        "use_rag": True,
        "intent": INTENT_POLICY,
        "question": "What is the policy when I request leave in advance?",
        "needs_clarification": False,
    }
    assert route_after_router(state) == "policy"


def test_route_after_router_asks_clarification_for_ambiguous_mixed_intent() -> None:
    state = {
        "use_rag": True,
        "intent": INTENT_POLICY,
        "question": "What is my company policy?",
        "needs_clarification": False,
        "mixed_clarification_needed": True,
    }
    assert route_after_router(state) == "clarify"


def test_route_after_router_uses_original_question_not_retrieval_query() -> None:
    state = {
        "use_rag": True,
        "intent": INTENT_POLICY,
        "question": "What is the PTO policy?",
        "retrieval_query": "How many PTO days do I have?",
        "needs_clarification": False,
        "mixed_clarification_needed": False,
    }
    assert route_after_router(state) == "policy"


def test_route_after_router_personal_how_many_balance_stays_personal() -> None:
    state = {
        "use_rag": True,
        "intent": INTENT_PERSONAL,
        "question": "How many PTO and sick days do I have remaining in my profile?",
        "needs_clarification": False,
        "mixed_clarification_needed": False,
    }
    assert route_after_router(state) == "personal"


def test_route_after_router_personal_override_for_domain_terms() -> None:
    state = {
        "use_rag": True,
        "intent": INTENT_POLICY,
        "question": "Show my sick days",
        "needs_clarification": False,
        "mixed_clarification_needed": False,
    }
    assert route_after_router(state) == "personal"


def test_route_after_router_type_of_leaves_is_personal() -> None:
    state = {
        "use_rag": True,
        "intent": INTENT_POLICY,
        "question": "What type of leaves do I have?",
        "needs_clarification": False,
        "mixed_clarification_needed": False,
    }
    assert route_after_router(state) == "personal"


def test_route_after_router_prefers_explicit_mixed_flag() -> None:
    state = {
        "use_rag": True,
        "intent": INTENT_POLICY,
        "question": "Show my sick days and sick leave rules",
        "needs_clarification": False,
        "mixed_clarification_needed": False,
        "is_mixed_intent": True,
    }
    assert route_after_router(state) == "mixed"


def test_route_after_router_personal_for_how_many_days_left() -> None:
    state = {
        "use_rag": True,
        "intent": INTENT_POLICY,
        "question": "How many days do I have left?",
        "needs_clarification": False,
        "mixed_clarification_needed": False,
    }
    assert route_after_router(state) == "personal"


def test_route_after_router_personal_for_still_have_in_pto() -> None:
    state = {
        "use_rag": True,
        "intent": INTENT_POLICY,
        "question": "What do I still have in PTO?",
        "needs_clarification": False,
        "mixed_clarification_needed": False,
    }
    assert route_after_router(state) == "personal"


def test_route_after_router_personal_for_my_leave_vague() -> None:
    state = {
        "use_rag": True,
        "intent": INTENT_POLICY,
        "question": "What about my leave?",
        "needs_clarification": False,
        "mixed_clarification_needed": False,
    }
    assert route_after_router(state) == "personal"


def test_route_after_router_personal_for_leave_situation() -> None:
    state = {
        "use_rag": True,
        "intent": INTENT_POLICY,
        "question": "Explain my leave situation",
        "needs_clarification": False,
        "mixed_clarification_needed": False,
    }
    assert route_after_router(state) == "personal"


def test_route_after_router_mixed_for_for_me_phrase() -> None:
    state = {
        "use_rag": True,
        "intent": INTENT_POLICY,
        "question": "Tell me about PTO for me",
        "needs_clarification": False,
        "mixed_clarification_needed": False,
    }
    assert route_after_router(state) == "mixed"
