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
    assert "policy:" in answer


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
