"""Simulated employee DB — same data as mcp-hr-server."""

import json
from pathlib import Path

from app.config import EMPLOYEES_JSON_PATH


def load_employees(path: Path | None = None) -> dict:
    p = path or EMPLOYEES_JSON_PATH
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def get_employee_record(employee_id: str, path: Path | None = None) -> dict | None:
    data = load_employees(path)
    for emp in data.get("employees", []):
        if emp.get("employee_id") == employee_id:
            return emp
    return None


def format_balance_context(employee_id: str, path: Path | None = None) -> str | None:
    emp = get_employee_record(employee_id, path)
    if not emp:
        return None
    return (
        f"Employee {emp.get('employee_id')} ({emp.get('name')}): "
        f"PTO days remaining: {emp.get('pto_days_remaining')}, "
        f"sick days remaining: {emp.get('sick_days_remaining')}, "
        f"language preference: {emp.get('language_pref')}."
    )


def needs_balance_lookup(question: str) -> bool:
    q = question.lower()
    keys = (
        "pto",
        "balance",
        "vacation",
        "sick day",
        "sick leave",
        "days left",
        "how many days",
    )
    return any(k in q for k in keys)
