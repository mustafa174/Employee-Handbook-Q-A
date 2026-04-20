from app.rag_graph import INTENT_PERSONAL, INTENT_POLICY, node_generate, run_ask


def test_golden_flow() -> None:
    # PERSONAL
    r1 = run_ask("What’s my leave balance?", employee_id="E001", use_rag=True)
    assert "14" in str(r1.get("answer", ""))

    # POLICY
    policy_state = {
        "intent": INTENT_POLICY,
        "intent_domain": "POLICY",
        "route": "policy",
        "use_rag": True,
        "question": "How far in advance should I request PTO?",
        "retrieval_citations": [{"text": "PTO policy excerpt", "score": 0.9}],
        "sub_questions": [
            "What is the PTO request policy?",
            "What is the PTO advance notice policy requirement?",
        ],
        "sub_results": [
            {
                "question": "What is the PTO request policy?",
                "answer": "PTO requests must be submitted at least 10 business days in advance.",
                "citations": [
                    {
                        "text": "PTO requests must be submitted at least 10 business days in advance.",
                        "section_title": "Paid Time Off (PTO)",
                    }
                ],
            },
            {
                "question": "What is the PTO advance notice policy requirement?",
                "answer": "PTO requests must be submitted at least 10 business days in advance.",
                "citations": [
                    {
                        "text": "PTO requests must be submitted at least 10 business days in advance.",
                        "section_title": "Paid Time Off (PTO)",
                    }
                ],
            },
        ],
        "messages": [],
    }
    r2 = node_generate(policy_state)
    assert "10 business days" in str(r2.get("answer", ""))

    # MIXED
    mixed_state = {
        "intent": INTENT_PERSONAL,
        "intent_domain": "POLICY",
        "route": "mixed",
        "is_mixed_intent": True,
        "use_rag": True,
        "question": "Show my sick days and sick leave rules",
        "employee_id": "E001",
        "employee_profile": {"pto_days": "14", "sick_days": "6"},
        "retrieved_context": "Sick leave rules context",
        "retrieval_citations": [{"text": "Notify your manager as soon as practicable."}],
        "sub_questions": ["How many sick days do I have?", "What are sick leave rules?"],
        "sub_results": [
            {
                "question": "What are sick leave rules?",
                "answer": "Notify your manager as soon as practicable on the day you cannot work.",
                "citations": [
                    {
                        "text": "Notify your manager as soon as practicable.",
                        "section_title": "Sick Leave",
                    }
                ],
            }
        ],
        "messages": [],
    }
    r3 = node_generate(mixed_state)
    answer3 = str(r3.get("answer", "")).lower()
    assert "6" in answer3
    assert "notify your manager" in answer3
    assert "1. how many sick days do i have?" in answer3
    assert "2. what are sick leave rules?" in answer3


def test_chaos_query() -> None:
    state = {
        "intent": INTENT_PERSONAL,
        "intent_domain": "POLICY",
        "route": "mixed",
        "is_mixed_intent": True,
        "use_rag": True,
        "question": "hey so like how many days i got and also how does this leave thing work",
        "employee_id": "E001",
        "employee_profile": {"pto_days": "14", "sick_days": "6"},
        "retrieved_context": "PTO rules context",
        "retrieval_citations": [{"text": "PTO requests must be submitted at least 10 business days in advance."}],
        "sub_questions": ["How many PTO days do I have?", "How does PTO policy work?"],
        "sub_results": [
            {
                "question": "How does PTO policy work?",
                "answer": "PTO requests must be submitted at least 10 business days in advance.",
                "citations": [
                    {
                        "text": "PTO requests must be submitted at least 10 business days in advance.",
                        "section_title": "Paid Time Off (PTO)",
                    }
                ],
            }
        ],
        "messages": [],
    }
    resp = node_generate(state)
    answer = str(resp.get("answer", "")).lower()
    assert "14" in answer
    assert "policy" in answer


def test_mixed_rollover_uses_policy_fallback_not_general() -> None:
    state = {
        "intent": INTENT_PERSONAL,
        "intent_domain": "POLICY",
        "route": "mixed",
        "is_mixed_intent": True,
        "use_rag": True,
        "question": "How many PTO days do I have and what is the rollover policy?",
        "employee_id": "E001",
        "employee_profile": {"pto_days": "14", "sick_days": "6"},
        "retrieved_context": "",
        "retrieval_citations": [],
        "retrieval_verdict": "FAILED",
        "sub_questions": ["How many PTO days do I have?", "What is the rollover policy?"],
        "sub_results": [],
        "messages": [],
    }
    out = node_generate(state)
    answer = str(out.get("answer", "")).lower()
    assert "1. how many pto days do i have?" in answer
    assert "2. what is the rollover policy?" in answer
    assert "**not found in handbook**" in answer
    assert "i can only help with hr, it, and company policy questions" not in answer


def test_mixed_detected_from_personal_and_policy_subquestions() -> None:
    state = {
        "intent": INTENT_POLICY,
        "intent_domain": "POLICY",
        "route": "policy",
        "use_rag": True,
        "question": "Show my sick days. Also explain sick leave rules.",
        "employee_id": "E001",
        "employee_profile": {"pto_days": "14", "sick_days": "6"},
        "retrieved_context": "",
        "retrieval_citations": [],
        "retrieval_verdict": "FAILED",
        "sub_questions": ["Show my sick days", "Explain sick leave rules"],
        "sub_results": [],
        "messages": [],
    }
    out = node_generate(state)
    answer = str(out.get("answer", "")).lower()
    assert "1. show my sick days?" in answer
    assert "2. explain sick leave rules?" in answer


def test_node_generate_generic_leave_policy_asks_scope_clarification() -> None:
    state = {
        "intent": INTENT_POLICY,
        "intent_domain": "POLICY",
        "route": "policy",
        "use_rag": True,
        "question": "leave policy",
        "retrieved_context": "PTO requests must be submitted at least 10 business days in advance.",
        "retrieval_citations": [
            {
                "text": "PTO requests must be submitted at least 10 business days in advance.",
                "section_title": "Paid time off (PTO)",
                "source_name": "handbook.md",
            }
        ],
        "sub_questions": [],
        "sub_results": [],
        "messages": [],
    }
    out = node_generate(state)
    answer = str(out.get("answer", "")).lower()
    assert "do you want the policy for" in answer
    assert "pto" in answer
    assert "sick leave" in answer
    assert "holidays" in answer


def test_node_generate_specific_sick_leave_policy_is_not_forced_to_clarify() -> None:
    state = {
        "intent": INTENT_POLICY,
        "intent_domain": "POLICY",
        "route": "policy",
        "use_rag": True,
        "question": "leave policy regarding sick",
        "retrieved_context": "Sick leave is for illness or medical appointments.",
        "retrieval_citations": [
            {
                "text": "Sick leave is for illness or medical appointments.",
                "section_title": "Sick leave",
                "source_name": "handbook.md",
            }
        ],
        "sub_questions": [],
        "sub_results": [],
        "messages": [],
    }
    out = node_generate(state)
    answer = str(out.get("answer", "")).lower()
    assert "do you want the policy for" not in answer


def test_multi_question_groups_repeated_out_of_scope_fallbacks() -> None:
    state = {
        "intent": INTENT_PERSONAL,
        "intent_domain": "POLICY",
        "route": "mixed",
        "is_mixed_intent": True,
        "use_rag": True,
        "question": "How many sick days do I have? moon policy? mars policy?",
        "employee_id": "E001",
        "employee_profile": {"pto_days": "14", "sick_days": "6", "employee_id": "E001"},
        "retrieved_context": "",
        "retrieval_citations": [],
        "retrieval_verdict": "FAILED",
        "sub_questions": ["How many sick days do I have?", "moon policy", "mars policy"],
        "sub_results": [],
        "messages": [],
    }
    out = node_generate(state)
    answer = str(out.get("answer", "")).lower()
    assert "1. how many sick days do i have?" in answer
    assert "6 sick days" in answer
    assert "**outside supported scope**" in answer
    assert "2. moon policy?" in answer
    assert "3. mars policy?" in answer
    assert answer.count("i can only help with hr, it, and company policy questions.") == 1
