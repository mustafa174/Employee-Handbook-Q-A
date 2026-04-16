from langchain_core.messages import AIMessage

from app.rag_graph import MIXED_CLARIFICATION_PROMPT, node_clarify


def test_clarification_loop_guard_defaults_to_general_policy_after_repeat() -> None:
    state = {
        "mixed_clarification_needed": True,
        "messages": [AIMessage(content=MIXED_CLARIFICATION_PROMPT)],
    }
    out = node_clarify(state)
    assert "general policy answer only" in str(out.get("answer", "")).lower()
    assert bool(out.get("escalate")) is True


def test_node_clarify_marks_ambiguous_policy_for_hr_escalation() -> None:
    state = {
        "intent_domain": "POLICY",
        "mixed_clarification_needed": False,
    }
    out = node_clarify(state)
    assert bool(out.get("escalate")) is True
    assert "hr" in str(out.get("answer", "")).lower()
    assert "ambiguous" in str(out.get("escalation_reason", "")).lower()
