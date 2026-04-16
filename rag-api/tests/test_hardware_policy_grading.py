from app.rag_graph import _looks_like_hardware_policy_query, normalize_query, grade_documents


def test_grade_documents_accepts_hardware_policy_evidence() -> None:
    state = {
        "question": "i want to know about laptop replacement policy",
        "route": "policy",
        "retrieval_citations": [
            {
                "text": (
                    "### Hardware Requests\n"
                    "Standard laptop upgrades occur every 3 years.\n"
                    "Broken Hardware: submit a ticket via Jira Service Desk."
                ),
                "score": 0.33,
                "source": "it_guide.md",
                "section_title": "Hardware Requests",
            }
        ],
        "retrieval_top_score": 0.33,
        "retrieval_attempt": 1,
        "retrieval_attempts": [
            {
                "attempt": 1,
                "query": "i want to know about laptop replacement policy",
                "top_score": 0.33,
                "verdict": "answerable",
                "reason": "pending grade",
                "citations": [],
            }
        ],
    }

    graded = grade_documents(state)
    assert graded.get("retrieval_verdict") == "ANSWERABLE"
    assert "Hardware policy fast-path" in str(graded.get("retrieval_reason", ""))


def test_grade_documents_accepts_it_vpn_policy_evidence() -> None:
    state = {
        "question": "which vpn company is using for servers",
        "route": "policy",
        "intent_domain": "IT",
        "retrieval_citations": [
            {
                "text": (
                    "### VPN Access\n"
                    "To access internal servers remotely, use GlobalProtect VPN.\n"
                    "Gateway: vpn.company.com\n"
                    "Credentials: standard SSO credentials."
                ),
                "score": 1.0244,
                "source": "it_guide.md",
                "section_title": "VPN Access",
            }
        ],
        "retrieval_top_score": 1.0244,
        "retrieval_attempt": 1,
        "retrieval_attempts": [
            {
                "attempt": 1,
                "query": "which vpn company is using for servers",
                "top_score": 1.0244,
                "verdict": "answerable",
                "reason": "pending grade",
                "citations": [],
            }
        ],
    }

    graded = grade_documents(state)
    assert graded.get("retrieval_verdict") == "ANSWERABLE"
    assert "IT policy fast-path" in str(graded.get("retrieval_reason", ""))


def test_hardware_query_normalizes_laptap_damage_typo() -> None:
    assert "laptop" in normalize_query("my laptap is damage")
    assert "damaged" in normalize_query("my laptap is damage")
    assert _looks_like_hardware_policy_query("my laptap is damage")
