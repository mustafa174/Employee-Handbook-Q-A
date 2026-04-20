import pytest

from app.intent_policy import classify_query
from app.rag_graph import INTENT_PERSONAL, INTENT_POLICY, _is_personal_query, node_generate, run_ask


def test_node_generate_personal_missing_profile_returns_employee_records_fallback() -> None:
    state = {
        "intent": INTENT_PERSONAL,
        "intent_domain": "PROFILE",
        "route": "personal",
        "use_rag": True,
        "question": "How many PTO days do I have?",
        "employee_profile": None,
        "balance_snippet": None,
        "retrieval_citations": [],
        "sub_questions": [],
        "sub_results": [],
        "messages": [],
    }
    out = node_generate(state)
    assert str(out.get("answer", "")) == "I couldn't find your personal data."


def test_node_generate_personal_unknown_leave_type_lists_supported_types() -> None:
    state = {
        "intent": INTENT_PERSONAL,
        "intent_domain": "PROFILE",
        "route": "personal",
        "use_rag": True,
        "question": "Do I have casual leaves?",
        "employee_profile": {"pto_days": "14", "sick_days": "6"},
        "balance_snippet": None,
        "retrieval_citations": [],
        "sub_questions": [],
        "sub_results": [],
        "messages": [],
    }
    out = node_generate(state)
    answer = str(out.get("answer", "")).lower()
    assert "not listed in your employee records" in answer
    assert "pto" in answer
    assert "sick leave" in answer


def test_route_and_execution_alignment() -> None:
    resp = run_ask("How many PTO and sick days do I have?", employee_id="E001", use_rag=True)
    answer = str(resp.get("answer", ""))
    assert "14" in answer
    assert "6" in answer


def test_personal_query_detector_covers_status_and_balance_phrasing() -> None:
    assert _is_personal_query("What is my leave balance?")
    assert _is_personal_query("Do I still have any days left?")
    assert _is_personal_query("What's my current PTO and sick leave status?")


def test_personal_query_detector_does_not_treat_policy_days_followup_as_personal() -> None:
    assert not _is_personal_query("How many days in advance do I need to request it?")


def test_node_generate_generic_leave_balance_does_not_trigger_unknown_leave_type() -> None:
    state = {
        "intent": INTENT_PERSONAL,
        "intent_domain": "PROFILE",
        "route": "personal",
        "use_rag": True,
        "question": "What is my leave balance?",
        "employee_profile": {"pto_days": "14", "sick_days": "6"},
        "balance_snippet": None,
        "retrieval_citations": [],
        "sub_questions": [],
        "sub_results": [],
        "messages": [],
    }
    out = node_generate(state)
    answer = str(out.get("answer", "")).lower()
    assert "not listed in your employee records" not in answer
    assert "14 pto days" in answer
    assert "6 sick days" in answer


def test_node_generate_general_route_does_not_return_it_fallback() -> None:
    state = {
        "intent": INTENT_POLICY,
        "intent_domain": "GENERAL",
        "route": "general",
        "use_rag": True,
        "question": "Best laptop under $1000",
        "retrieval_citations": [],
        "sub_questions": [],
        "sub_results": [],
        "messages": [],
        "employee_profile": None,
        "balance_snippet": None,
    }
    out = node_generate(state)
    answer = str(out.get("answer", ""))
    assert answer == "I can only help with HR, IT, and company policy questions."
    assert "Jira Service Desk" not in answer


def test_classify_query_treats_loan_tenure_as_profile() -> None:
    out = classify_query("after how many months can i get loan")
    assert out["domain_class"] == "PROFILE"


@pytest.mark.parametrize(
    "q",
    [
        "i want to avail loan",
        "i want to get loan",
        "i want to get company loan",
        "i want to know about company loan",
    ],
)
def test_classify_query_treats_services_loan_intent_as_profile(q: str) -> None:
    out = classify_query(q)
    assert out["domain_class"] == "PROFILE"
    assert "loan_personal_phrase" in out["reasons"]


def test_run_ask_avail_loan_uses_profile_not_policy_fallback() -> None:
    resp = run_ask("i want to avail loan", employee_id="E001", use_rag=True)
    answer = str(resp.get("answer", "")).lower()
    assert "couldn't find the specific policy text" not in answer
    assert "services loan" in answer or "pkr" in answer


def test_node_generate_loan_types_question_uses_composite_not_employment_type() -> None:
    """Personal route used to resolve 'type' to employment_type before loan composite ran."""
    state = {
        "intent": INTENT_PERSONAL,
        "intent_domain": "PROFILE",
        "route": "personal",
        "use_rag": False,
        "question": "what type of loans are there",
        "employee_id": "E001",
        "employee_profile": {
            "employment_type": "full-time",
            "services_loan_available": "True",
            "services_loan_limit_pkr": "250000",
            "services_loan_eligible_after_months": "12",
        },
        "balance_snippet": None,
        "retrieval_citations": [],
        "sub_questions": [],
        "sub_results": [],
        "messages": [],
    }
    out = node_generate(state)
    answer = str(out.get("answer", "")).lower()
    assert "services loan" in answer
    assert "full-time" not in answer


def test_run_ask_loan_tenure_uses_personal_profile() -> None:
    resp = run_ask("after how many months can i get loan", employee_id="E001", use_rag=True)
    answer = str(resp.get("answer", "")).lower()
    assert "12" in answer
    assert "month" in answer


def test_run_ask_vacation_days_with_employee_uses_profile_not_policy_fallback() -> None:
    resp = run_ask("How many vacation days do I get?", employee_id="E001", use_rag=True)
    answer = str(resp.get("answer", "")).lower()
    assert "14" in answer
    assert "couldn't find the specific policy text" not in answer


def test_run_ask_leaveeeee_balance_normalizes_to_personal_balance() -> None:
    resp = run_ask("i want to know leaveeee balance", employee_id="E001", use_rag=True)
    answer = str(resp.get("answer", "")).lower()
    assert "couldn't find your personal data" not in answer
    assert ("14" in answer) or ("pto" in answer)


def test_run_ask_vacation_days_without_employee_prompts_profile_not_policy_fallback() -> None:
    resp = run_ask("How many vacation days do I get?", employee_id=None, use_rag=True)
    answer = str(resp.get("answer", "")).lower()
    assert "employee profile" in answer or "profile selected" in answer
    assert "couldn't find the specific policy text" not in answer
    assert str(resp.get("route", "")).upper() == "PROFILE"


def test_run_ask_mixed_leave_and_policy_combines_personal_and_policy() -> None:
    resp = run_ask("how many leaves i have and what is leave policy", employee_id="E001", use_rag=True)
    answer = str(resp.get("answer", "")).lower()
    assert "14 pto days" in answer
    assert "6 sick days" in answer
    assert "1. how many leaves i have?" in answer
    assert "2. what is leave policy?" in answer


def test_run_ask_manager_treating_badly_routes_sensitive() -> None:
    resp = run_ask("My manager is treating me badly, what should I do?", employee_id="E001", use_rag=True)
    answer = str(resp.get("answer", "")).lower()
    assert "sensitive matter" in answer
    assert "prepare an hr report summary" in answer
    assert "your manager is" not in answer


def test_run_ask_harassed_routes_sensitive_with_hr_report_offer() -> None:
    resp = run_ask("i am being harrassed", employee_id="E001", use_rag=True)
    answer = str(resp.get("answer", "")).lower()
    assert "sensitive matter" in answer
    assert "prepare an hr report summary" in answer
    action = resp.get("agent_action") or {}
    assert action.get("type") == "HARASSMENT_REPORT"


def test_run_ask_manager_harrasing_routes_sensitive_with_hr_report_offer() -> None:
    resp = run_ask("my manager is harrasing me", employee_id="E001", use_rag=True)
    answer = str(resp.get("answer", "")).lower()
    assert "sensitive matter" in answer
    assert "prepare an hr report summary" in answer
    assert "your manager is" not in answer


def test_run_ask_manager_harrassssing_routes_sensitive_with_hr_report_offer() -> None:
    resp = run_ask("my manager is harrassssing me", employee_id="E001", use_rag=True)
    answer = str(resp.get("answer", "")).lower()
    assert "sensitive matter" in answer
    assert "prepare an hr report summary" in answer
    assert "your manager is" not in answer


def test_run_ask_manager_harraasssing_routes_sensitive_with_hr_report_offer() -> None:
    resp = run_ask("my manager is harraasssing me", employee_id="E001", use_rag=True)
    answer = str(resp.get("answer", "")).lower()
    assert "sensitive matter" in answer
    assert "prepare an hr report summary" in answer
    assert "your manager is" not in answer


def test_run_ask_report_harassment_by_manager_has_report_action() -> None:
    resp = run_ask("I want to report harassment by my manager.", employee_id="E001", use_rag=True)
    answer = str(resp.get("answer", "")).lower()
    assert "sensitive matter" in answer
    action = resp.get("agent_action") or {}
    assert action.get("type") == "HARASSMENT_REPORT"
    payload = action.get("payload") or {}
    msg = str(payload.get("message", "")).lower()
    assert "workplace harassment" in msg
    assert "manager" in msg


def test_run_ask_wrongful_termination_has_report_action() -> None:
    resp = run_ask("I was wrongfully terminated", employee_id="E001", use_rag=True)
    answer = str(resp.get("answer", "")).lower()
    assert "sensitive matter" in answer
    action = resp.get("agent_action") or {}
    assert action.get("type") == "HARASSMENT_REPORT"


def test_run_ask_unlimited_unpaid_leave_stays_policy_even_with_employee_scope() -> None:
    resp = run_ask("Can I take unlimited unpaid leave?", employee_id="E001", use_rag=True)
    answer = str(resp.get("answer", "")).lower()
    assert "couldn't find your personal data" not in answer
    assert str(resp.get("route", "")).upper() in {"POLICY", "GENERAL"}


def test_run_ask_loan_policy_and_amount_routes_mixed() -> None:
    resp = run_ask("What is the loan policy and how much loan can I get?", employee_id="E001", use_rag=True)
    assert str(resp.get("route", "")).upper() == "MIXED"
    answer = str(resp.get("answer", "")).lower()
    assert "policy" in answer
    assert ("pkr" in answer) or ("loan" in answer)


def test_run_ask_eligibility_and_policy_routes_mixed() -> None:
    resp = run_ask("Am I eligible for leave and what is the policy?", employee_id="E001", use_rag=True)
    assert str(resp.get("route", "")).upper() == "MIXED"


def test_run_ask_mixed_pto_balance_and_carryover_includes_carryover_policy() -> None:
    resp = run_ask(
        "How many PTO days do I have left and what is the policy for carry-over into next year?",
        employee_id="E001",
        use_rag=True,
    )
    answer = str(resp.get("answer", "")).lower()
    assert str(resp.get("route", "")).upper() == "MIXED"
    assert "14 pto" in answer
    assert ("roll over" in answer) or ("carry over" in answer)


def test_run_ask_mixed_loan_and_broken_laptop_request_returns_both_parts() -> None:
    resp = run_ask(
        "Am I eligible for a services loan and how do I request a replacement for a broken laptop?",
        employee_id="E001",
        use_rag=True,
    )
    answer = str(resp.get("answer", "")).lower()
    assert str(resp.get("route", "")).upper() == "MIXED"
    assert ("yes, you are eligible for services loan eligibility" in answer) or ("eligible for services loan" in answer)
    assert ("submit a ticket" in answer) or ("jira service desk" in answer)
    assert "couldn't find the specific policy text in the handbook" not in answer


def test_run_ask_employee_id_and_medical_certificate_routes_mixed() -> None:
    resp = run_ask(
        "What is my employee ID and do I need a medical certificate if I take 3 sick days?",
        employee_id="E001",
        use_rag=True,
    )
    answer = str(resp.get("answer", "")).lower()
    assert str(resp.get("route", "")).upper() == "MIXED"
    assert "employee id" in answer
    assert ("medical certification" in answer) or ("medical certificate" in answer)


def test_run_ask_policy_followup_with_it_coreference_keeps_policy_context() -> None:
    history = [
        {"role": "user", "content": "What is the PTO policy?"},
        {
            "role": "assistant",
            "content": (
                "PTO requests must be submitted at least 10 business days in advance. "
                "Unused PTO may roll over up to 5 days."
            ),
        },
    ]
    resp = run_ask(
        "How many days in advance do I need to request it?",
        employee_id="E001",
        chat_history=history,
        use_rag=True,
    )
    answer = str(resp.get("answer", "")).lower()
    assert "couldn't find your personal data" not in answer
    assert ("10 business days" in answer) or ("advance" in answer)


def test_run_ask_rollover_followup_with_that_coreference_stays_grounded() -> None:
    history = [
        {"role": "user", "content": "What is the PTO policy?"},
        {
            "role": "assistant",
            "content": (
                "PTO requests must be submitted at least 10 business days in advance. "
                "Unused PTO may roll over up to 5 days."
            ),
        },
        {"role": "user", "content": "How many days in advance do I need to request it?"},
        {"role": "assistant", "content": "At least 10 business days in advance."},
    ]
    resp = run_ask(
        "And what about rollover for that?",
        employee_id="E001",
        chat_history=history,
        use_rag=True,
    )
    answer = str(resp.get("answer", "")).lower()
    assert "couldn't find the specific policy text in the handbook" not in answer
    assert "pto" in answer


def test_run_ask_what_about_sick_days_after_manager_context_prefers_personal_balance() -> None:
    history = [
        {"role": "user", "content": "How many PTO days do I have?"},
        {"role": "assistant", "content": "You currently have 14 PTO days remaining."},
        {"role": "user", "content": "And my manager?"},
        {"role": "assistant", "content": "Your manager is Nadia Rahman."},
    ]
    resp = run_ask("What about sick days?", employee_id="E001", chat_history=history, use_rag=True)
    answer = str(resp.get("answer", "")).lower()
    assert "6 sick days" in answer


def test_run_ask_ambiguous_financial_support_asks_clarification() -> None:
    resp = run_ask("What support does company give me financially?", employee_id="E001", use_rag=True)
    answer = str(resp.get("answer", "")).lower()
    assert "can you clarify whether you are asking about company financial support policies" in answer


def test_run_ask_accommodation_request_is_not_sensitive() -> None:
    resp = run_ask("i want to do accommodation requests", employee_id="E001", use_rag=True)
    answer = str(resp.get("answer", "")).lower()
    assert "sensitive matter" not in answer
    assert str(resp.get("route", "")).upper() in {"POLICY", "GENERAL"}


def test_node_generate_hardware_query_uses_direct_grounded_instruction() -> None:
    state = {
        "intent": INTENT_POLICY,
        "intent_domain": "IT",
        "route": "policy",
        "use_rag": True,
        "question": "my laptop is broken what should I do",
        "retrieval_citations": [
            {
                "text": (
                    "### Hardware Requests\n"
                    "Standard laptop upgrades occur every 3 years.\n"
                    "Broken Hardware: If your device is damaged, submit a ticket via the 'Jira Service Desk' immediately."
                ),
                "score": 0.72,
                "source": "it_guide.md",
                "section_title": "Hardware Requests",
            }
        ],
        "citations": [
            {
                "text": (
                    "### Hardware Requests\n"
                    "Standard laptop upgrades occur every 3 years.\n"
                    "Broken Hardware: If your device is damaged, submit a ticket via the 'Jira Service Desk' immediately."
                ),
                "score": 0.72,
                "source": "it_guide.md",
                "section_title": "Hardware Requests",
            }
        ],
        "retrieved_context": "Broken Hardware: If your device is damaged, submit a ticket via the 'Jira Service Desk' immediately.",
        "sub_questions": [],
        "sub_results": [],
        "messages": [],
        "employee_profile": None,
        "balance_snippet": None,
    }
    out = node_generate(state)
    answer = str(out.get("answer", "")).lower()
    assert "jira service desk" in answer
    assert "submit a ticket" in answer


def test_run_ask_loan_limit_without_my_uses_personal_profile() -> None:
    resp = run_ask("what is loan limit", employee_id="E001", use_rag=True)
    answer = str(resp.get("answer", "")).lower()
    assert "pkr 250000" in answer


def test_node_generate_hardware_issue_query_uses_direct_grounded_instruction() -> None:
    state = {
        "intent": INTENT_POLICY,
        "intent_domain": "IT",
        "route": "policy",
        "use_rag": True,
        "question": "my laptop is having issue",
        "retrieval_citations": [
            {
                "text": (
                    "### Hardware Requests\n"
                    "Broken Hardware: If your device is damaged, submit a ticket via the 'Jira Service Desk' immediately."
                ),
                "score": 0.58,
                "source": "it_guide.md",
                "section_title": "Hardware Requests",
            }
        ],
        "citations": [
            {
                "text": (
                    "### Hardware Requests\n"
                    "Broken Hardware: If your device is damaged, submit a ticket via the 'Jira Service Desk' immediately."
                ),
                "score": 0.58,
                "source": "it_guide.md",
                "section_title": "Hardware Requests",
            }
        ],
        "retrieved_context": "Broken Hardware: If your device is damaged, submit a ticket via the 'Jira Service Desk' immediately.",
        "sub_questions": [],
        "sub_results": [],
        "messages": [],
        "employee_profile": None,
        "balance_snippet": None,
    }
    out = node_generate(state)
    answer = str(out.get("answer", "")).lower()
    assert "jira service desk" in answer


def test_node_generate_generic_replacement_policy_asks_for_clarification() -> None:
    state = {
        "intent": INTENT_POLICY,
        "intent_domain": "POLICY",
        "route": "policy",
        "use_rag": True,
        "question": "what is company replacement policy",
        "retrieval_citations": [
            {
                "text": "PTO requests should be submitted at least 10 business days in advance.",
                "score": 0.42,
                "source": "handbook.md",
                "section_title": "Paid Time Off",
            }
        ],
        "citations": [
            {
                "text": "PTO requests should be submitted at least 10 business days in advance.",
                "score": 0.42,
                "source": "handbook.md",
                "section_title": "Paid Time Off",
            }
        ],
        "sub_questions": [],
        "sub_results": [],
        "messages": [],
        "employee_profile": None,
        "balance_snippet": None,
    }
    out = node_generate(state)
    answer = str(out.get("answer", "")).lower()
    assert "clarify" in answer
    assert "replacement policy" in answer
