from app.rag_graph import _fallback_sub_questions, _is_multi_question_prompt, grade_documents, query_refiner_node


def test_query_refiner_splits_multi_question_prompt() -> None:
    state = {
        "question": "How many PTO days do I have? What is PTO policy? How many PTO days do I have?",
        "messages": [],
    }
    out = query_refiner_node(state)
    sub_questions = list(out.get("sub_questions") or [])
    assert len(sub_questions) >= 2
    assert any("how many pto days do i have" in q.lower() for q in sub_questions)
    assert any("what is pto policy" in q.lower() for q in sub_questions)


def test_query_refiner_adds_paid_time_off_alias_for_pto_policy() -> None:
    state = {
        "question": "What is PTO policy?",
        "messages": [],
    }
    out = query_refiner_node(state)
    retrieval_queries = list(out.get("retrieval_queries") or [])
    assert "what is pto policy?" in retrieval_queries
    assert "paid time off policy" in retrieval_queries


def test_grade_documents_relaxes_threshold_for_rollover_policy() -> None:
    state = {
        "route": "mixed",
        "question": "How many PTO days do I have and what is the rollover policy?",
        "retrieval_citations": [
            {
                "text": "Unused PTO may roll over up to 5 days into the next calendar year.",
                "score": 0.22,
                "source": "handbook.md",
                "section_title": "Paid Time Off (PTO)",
            }
        ],
        "retrieval_top_score": 0.22,
        "retrieval_attempt": 1,
        "retrieval_attempts": [
            {
                "attempt": 1,
                "query": "rollover policy",
                "top_score": 0.22,
                "verdict": "pending",
                "reason": "pending grade",
                "citations": [],
            }
        ],
        "sub_questions": [],
    }
    out = grade_documents(state)
    assert out.get("retrieval_verdict") == "ANSWERABLE"
    assert "Rollover policy query" in str(out.get("retrieval_reason", ""))


def test_query_refiner_adds_sick_leave_three_days_alias() -> None:
    state = {
        "question": "What happens after 3 sick days?",
        "messages": [],
    }
    out = query_refiner_node(state)
    retrieval_queries = [str(x).lower() for x in (out.get("retrieval_queries") or [])]
    assert any("3 consecutive days medical certification" in q for q in retrieval_queries)


def test_multi_question_detection_for_also_explain_sentence() -> None:
    q = "Show my sick days. Also explain sick leave rules."
    assert _is_multi_question_prompt(q)
    parts = _fallback_sub_questions(q)
    assert len(parts) >= 2
