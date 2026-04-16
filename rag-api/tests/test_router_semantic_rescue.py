from app import rag_graph
from app.rag_graph import INTENT_OUT_OF_DOMAIN, router_node


def test_router_semantic_rescues_general_to_personal(monkeypatch) -> None:
    def fake_classify(_q: str) -> dict:
        return {
            "domain_class": "OOS",
            "confidence": 0.2,
            "reasons": ["forced_oos_for_test"],
            "secondary_class": "POLICY",
            "needs_clarification": False,
        }

    def fake_semantic(_q: str, *, has_employee_id: bool) -> dict | None:
        assert has_employee_id
        return {
            "route": "personal",
            "score": 0.81,
            "label": "personal",
            "model": "intfloat/e5-base-v2",
        }

    monkeypatch.setattr(rag_graph, "classify_query", fake_classify)
    monkeypatch.setattr(rag_graph, "semantic_rescue_route", fake_semantic)
    out = router_node(
        {
            "question": "what benefits balance remains",
            "employee_id": "E001",
            "use_rag": True,
        }
    )
    assert str(out.get("route")) == "personal"
    assert str(out.get("intent")) != INTENT_OUT_OF_DOMAIN
    assert str(out.get("intent_domain")) == "PROFILE"
    assert bool(out.get("use_rag")) is False


def test_router_does_not_semantic_rescue_hard_oos_terms(monkeypatch) -> None:
    def fake_classify(_q: str) -> dict:
        return {
            "domain_class": "OOS",
            "confidence": 0.2,
            "reasons": ["forced_oos_for_test"],
            "secondary_class": "POLICY",
            "needs_clarification": False,
        }

    def fake_semantic(_q: str, *, has_employee_id: bool) -> dict | None:
        return {
            "route": "policy",
            "score": 0.92,
            "label": "policy",
            "model": "intfloat/e5-base-v2",
        }

    monkeypatch.setattr(rag_graph, "classify_query", fake_classify)
    monkeypatch.setattr(rag_graph, "semantic_rescue_route", fake_semantic)
    out = router_node(
        {
            "question": "What is policy for traveling to Mars for vacation?",
            "employee_id": "E001",
            "use_rag": True,
        }
    )
    assert str(out.get("route")) == "general"
    assert str(out.get("intent")) == INTENT_OUT_OF_DOMAIN


def test_router_skips_semantic_rescue_when_use_rag_disabled(monkeypatch) -> None:
    called = {"semantic": False}

    def fake_classify(_q: str) -> dict:
        return {
            "domain_class": "OOS",
            "confidence": 0.2,
            "reasons": ["forced_oos_for_test"],
            "secondary_class": "POLICY",
            "needs_clarification": False,
        }

    def fake_semantic(_q: str, *, has_employee_id: bool) -> dict | None:
        called["semantic"] = True
        return {
            "route": "personal",
            "score": 0.81,
            "label": "personal",
            "model": "intfloat/e5-base-v2",
        }

    monkeypatch.setattr(rag_graph, "classify_query", fake_classify)
    monkeypatch.setattr(rag_graph, "semantic_rescue_route", fake_semantic)
    out = router_node(
        {
            "question": "what type of loans are there",
            "employee_id": "E001",
            "use_rag": False,
        }
    )
    assert called["semantic"] is False
    assert str(out.get("route")) == "general"
    assert str(out.get("intent")) == INTENT_OUT_OF_DOMAIN
