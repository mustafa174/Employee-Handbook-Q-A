import json
from pathlib import Path

from app.intent_policy import classify_query


def _load_suite() -> list[dict[str, str]]:
    path = Path(__file__).with_name("regression_suite.json")
    rows = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(rows, list)
    return [r for r in rows if isinstance(r, dict)]


def test_intent_policy_regression_suite() -> None:
    rows = _load_suite()
    assert len(rows) >= 50
    mismatches: list[str] = []
    for row in rows:
        query = str(row.get("query", "")).strip()
        expected = str(row.get("expected_domain", "")).strip().upper()
        if not query or not expected:
            continue
        result = classify_query(query)
        actual = str(result.get("domain_class", "")).upper()
        if actual != expected:
            mismatches.append(f"{query} -> expected {expected}, got {actual}")
    assert not mismatches, "Intent policy mismatches:\n" + "\n".join(mismatches[:10])
