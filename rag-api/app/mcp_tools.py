"""MCP-style tool functions used by rag-api graph orchestration."""

from __future__ import annotations

from app.employee_service import get_employee_record


def fetch_user_balance(employee_id: str) -> dict | None:
    """
    Return normalized personal leave balances for a single employee.

    Shape:
    {
        "employee_id": "...",
        "name": "...",
        "pto_balance": <number>,
        "sick_balance": <number>,
    }
    """
    emp = get_employee_record(employee_id)
    if not emp:
        return None
    return {
        "employee_id": str(emp.get("employee_id", "")),
        "name": str(emp.get("name", "Unknown Employee")),
        "pto_balance": float(emp.get("pto_days_remaining", 0)),
        "sick_balance": float(emp.get("sick_days_remaining", 0)),
    }


def get_employee_details(employee_id: str) -> dict | None:
    """Return personal profile + leave balances for MCP-style personal lookup."""
    emp = get_employee_record(employee_id)
    if not emp:
        return None
    profile: dict[str, str] = {}
    for key, value in emp.items():
        if value is None:
            continue
        profile[str(key)] = str(value)
    return {
        "employee_id": str(emp.get("employee_id", "")),
        "name": str(emp.get("name", "Unknown Employee")),
        "pto_balance": float(emp.get("pto_days_remaining", 0)),
        "sick_balance": float(emp.get("sick_days_remaining", 0)),
        "language_pref": str(emp.get("language_pref", "")),
        "profile": profile,
    }

