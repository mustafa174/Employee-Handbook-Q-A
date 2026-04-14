import json
from pathlib import Path

import pytest

from app.employee_service import (
    format_balance_context,
    get_employee_record,
    needs_balance_lookup,
)


@pytest.fixture
def emp_file(tmp_path: Path) -> Path:
    p = tmp_path / "employees.json"
    p.write_text(
        json.dumps(
            {
                "employees": [
                    {
                        "employee_id": "X1",
                        "name": "Test",
                        "pto_days_remaining": 3,
                        "sick_days_remaining": 2,
                        "language_pref": "en",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return p


def test_get_employee(emp_file: Path) -> None:
    assert get_employee_record("X1", emp_file)["pto_days_remaining"] == 3
    assert get_employee_record("missing", emp_file) is None


def test_format_balance(emp_file: Path) -> None:
    s = format_balance_context("X1", emp_file)
    assert s and "PTO" in s and "3" in s


def test_needs_balance_lookup() -> None:
    assert needs_balance_lookup("How many PTO days do I have?") is True
    assert needs_balance_lookup("What is the holiday policy?") is False
