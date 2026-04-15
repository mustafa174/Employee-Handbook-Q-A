"""Catalog for employee profile fields and query aliases."""

from __future__ import annotations

from typing import Literal, TypedDict


FieldType = Literal["string", "number", "boolean"]


class ProfileFieldMeta(TypedDict, total=False):
    aliases: list[str]
    description: str
    field_type: FieldType
    unit: str
    sensitivity: Literal["normal", "restricted"]
    eligibility_key: str


PROFILE_FIELD_CATALOG: dict[str, ProfileFieldMeta] = {
    "employee_id": {
        "aliases": ["employee id", "id", "my id"],
        "description": "Unique employee identifier.",
        "field_type": "string",
    },
    "name": {
        "aliases": ["name", "my name", "who am i"],
        "description": "Employee full name.",
        "field_type": "string",
    },
    "department": {
        "aliases": ["department", "team", "business unit"],
        "description": "Employee department or team name.",
        "field_type": "string",
    },
    "location": {
        "aliases": ["location", "office", "city", "work location"],
        "description": "Employee work location.",
        "field_type": "string",
    },
    "language_pref": {
        "aliases": ["language", "language preference", "preferred language"],
        "description": "Preferred communication language.",
        "field_type": "string",
    },
    "pto_days_remaining": {
        "aliases": ["pto", "pto days", "vacation days", "days left", "remaining pto"],
        "description": "Remaining paid time off days.",
        "field_type": "number",
    },
    "sick_days_remaining": {
        "aliases": ["sick days", "sick leave", "remaining sick days"],
        "description": "Remaining sick leave days.",
        "field_type": "number",
    },
    "services_loan_limit_pkr": {
        "aliases": ["loan limit", "services loan limit", "loan amount", "how much loan", "max loan"],
        "description": "Maximum eligible services loan amount in PKR.",
        "field_type": "number",
        "unit": "PKR",
        "eligibility_key": "services_loan_available",
    },
    "services_loan_available": {
        "aliases": ["loan eligible", "eligible for loan", "services loan eligibility", "am i eligible"],
        "description": "Whether employee is eligible for services loan.",
        "field_type": "boolean",
    },
    "services_loan_eligible_after_months": {
        "aliases": ["loan tenure", "eligible after months", "service months for loan"],
        "description": "Months of service required for loan eligibility.",
        "field_type": "number",
        "unit": "months",
    },
}
