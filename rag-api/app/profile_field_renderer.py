"""Deterministic renderer for resolved employee profile fields."""

from __future__ import annotations

from app.profile_field_catalog import PROFILE_FIELD_CATALOG, ProfileFieldMeta


def _meta_for_key(key: str) -> ProfileFieldMeta:
    if key in PROFILE_FIELD_CATALOG:
        return PROFILE_FIELD_CATALOG[key]
    return {
        "aliases": [key.replace("_", " ")],
        "description": f"Employee profile field: {key}.",
        "field_type": "string",
    }


def _display_key(key: str) -> str:
    explicit = {
        "language_pref": "language preference",
        "services_loan_limit_pkr": "services loan limit",
        "services_loan_available": "services loan eligibility",
        "services_loan_eligible_after_months": "services loan eligible after months",
    }
    if key in explicit:
        return explicit[key]
    return key.replace("_", " ").strip()


def render_profile_answer(query: str, key: str, value: str) -> str:
    q = query.lower()
    meta = _meta_for_key(key)
    field_type = meta.get("field_type", "string")
    unit = str(meta.get("unit", "")).strip()
    label = _display_key(key)

    if field_type == "boolean":
        v = value.strip().lower()
        is_yes = v in {"true", "yes", "1"}
        is_no = v in {"false", "no", "0"}
        if "eligible" in q:
            if is_yes:
                return f"Yes, you are eligible for {label.replace(' available', '')}."
            if is_no:
                return f"No, you are not eligible for {label.replace(' available', '')}."
        if is_yes:
            return f"Your {label} is Yes."
        if is_no:
            return f"Your {label} is No."
        return f"Your {label} is {value}."

    if field_type == "number":
        normalized_value = value.strip()
        if unit.lower() == "pkr":
            return f"Your {label} is PKR {normalized_value}."
        if unit:
            return f"Your {label} is {normalized_value} {unit}."
        return f"Your {label} is {normalized_value}."

    return f"Your {label} is {value.strip()}."
