"""LangGraph: guardrail -> retrieve -> balance hint -> grounded answer."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from app.config import OPENAI_CHAT_MODEL, OPENAI_EMBEDDING_MODEL
from app.mcp_tools import get_employee_details
from app.vectorstore import get_vectorstore


class HandbookAnswerOut(BaseModel):
    answer: str


class IntentRouteOut(BaseModel):
    intent: str
    reason: str


class RetrievalGradeOut(BaseModel):
    verdict: str
    reason: str
    needs_second_hop: bool = False


class QueryRewriteOut(BaseModel):
    query: str
    reason: str


class QueryRefinerOut(BaseModel):
    standalone_query: str
    reason: str
    alternatives: list[str] | None = None
    sub_questions: list[str] | None = None


SENSITIVE_TOPICS = [
    "harassment",
    "termination",
    "firing",
    "discrimination",
    "accommodation request",
    "sexual assault",
    "violence",
    "lawsuit",
    "legal action",
]
SENSITIVE_PATTERNS = re.compile(
    r"\b(" + "|".join(re.escape(t).replace(r"\ ", r"\s+") for t in SENSITIVE_TOPICS) + r")\b",
    re.I,
)


class RagState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    employee_id: str | None
    use_rag: bool
    intent: str
    intent_reason: str
    retrieval_query: str
    retrieval_queries: list[str]
    sub_questions: list[str]
    sub_results: list[dict]
    retrieval_attempt: int
    retrieval_top_score: float
    retrieval_verdict: str
    retrieval_reason: str
    retrieval_attempts: list[dict]
    escalate: bool
    escalation_reason: str | None
    retrieved_context: str
    retrieval_citations: list[dict]
    balance_snippet: str | None
    mcp_status_detail: str | None
    answer: str
    citations: list[dict]


INTENT_PERSONAL = "INTENT_PERSONAL"
INTENT_POLICY = "INTENT_POLICY"
INTENT_GENERAL = "INTENT_GENERAL"
RETRIEVE_MAX_ATTEMPTS = 3
RETRIEVE_SCORE_THRESHOLD = 0.4
MAX_SUB_QUESTIONS = 24
MEGA_PROMPT_SUBQUESTION_THRESHOLD = 8
RETRIEVE_MAX_ATTEMPTS_FAST = 2
RETRIEVE_K = 4
MAX_LOCAL_QUERIES_DEFAULT = 2
MAX_LOCAL_QUERIES_VPN = 3
GRADE_FASTPATH_SCORE = 0.62
_FRIENDLY_SOURCE_NAMES = {
    "it_guide.md": "IT Support & Infrastructure Guide",
    "handbook.md": "Employee Handbook",
}
_GENERIC_QUERY_WORDS = {
    "policy",
    "policies",
    "guide",
    "guideline",
    "guidelines",
    "info",
    "information",
    "details",
    "about",
    "tell",
    "me",
}
_GROUNDING_STOPWORDS = {
    "what",
    "which",
    "where",
    "when",
    "is",
    "are",
    "the",
    "for",
    "and",
    "with",
    "from",
    "about",
    "policy",
    "company",
    "days",
    "rule",
    "required",
    "requirement",
    "using",
    "there",
    "any",
}
_PROFILE_QUERY_HINTS = ("employee id", "name", "balance", "days do i have", "days left", "sick", "pto")
_POLICY_QUERY_HINTS = ("policy", "rule", "required", "must", "gateway", "vpn", "rollover", "advance notice")
_QUESTION_NORMALIZE_RE = r"[\W_]+"
_PROFILE_BALANCE_TERMS = ("how many", "balance", "days do i have", "days left", "remaining")
POLICY_CLAIM_PATTERN = re.compile(r"\b(must|required|days?)\b", re.I)
POLICY_SCOPE_PATTERN = re.compile(r"\b(policy|allowed|eligible|approval|remote work)\b", re.I)


def _is_broad_query(question: str) -> bool:
    words = re.findall(r"[a-zA-Z]+", question.lower())
    if not words:
        return False
    # Broad if it's effectively a short category/topic phrase.
    return len(words) <= 2 and not any(
        token in {"how", "what", "when", "where", "why", "which", "can", "do", "is", "are"}
        for token in words
    )


def _fallback_sub_questions(question: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n+|(?<=[?])\s+", question) if p.strip()]
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        if p.lower().startswith("expected:"):
            continue
        expanded = [
            x.strip()
            for x in re.split(
                r"\s+(?:and|also)\s+(?=(?:what|how|can|when|where|who)\b)",
                p,
                flags=re.I,
            )
            if x.strip()
        ]
        # Avoid over-fragmenting into contextless tails like "what's the rule?".
        normalized_expanded: list[str] = []
        for idx, qx in enumerate(expanded if expanded else [p]):
            token_count = len(re.findall(r"[a-zA-Z0-9]+", qx))
            is_generic_tail = token_count <= 4 and bool(
                re.search(r"\b(rule|policy|process|details?)\b", qx, re.I)
            )
            if idx > 0 and is_generic_tail:
                normalized_expanded = [p]
                break
            normalized_expanded.append(qx)
        for q in normalized_expanded:
            key = re.sub(_QUESTION_NORMALIZE_RE, " ", q.lower()).strip()
            if key in seen:
                continue
            seen.add(key)
            out.append(q)
    return out if out else [question.strip()]


def _is_multi_question_prompt(question: str) -> bool:
    lines = [ln.strip() for ln in question.splitlines() if ln.strip()]
    if len(lines) >= 2:
        return True
    return len(re.findall(r"\?", question)) >= 2


def _sub_question_grounded(sub_q: str, citation: dict) -> bool:
    norm_q = sub_q.lower().replace("von", "vpn").replace("globel", "global").replace("acess", "access").replace("severs", "servers")
    text = str(citation.get("text") or "").lower()
    section = str(citation.get("section_title") or "").lower()
    combined = f"{section} {text}"
    terms = [
        t
        for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", norm_q)
        if t not in _GROUNDING_STOPWORDS
    ]
    if not terms:
        return False
    hits = sum(1 for t in terms if t in combined)
    # Adaptive grounding for short valid asks (e.g., "VPN gateway").
    min_hits = 1 if len(terms) <= 2 else 2
    if hits >= min_hits:
        return True
    # Domain fallback: VPN questions can ground on explicit VPN provider text.
    if "vpn" in norm_q and ("vpn" in combined or "globalprotect" in combined):
        return True
    return False


def _extract_best_section(full_text: str, fallback: str) -> str:
    lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
    # Prefer the most specific visible subsection.
    for ln in lines:
        if ln.startswith("### "):
            return ln[4:].strip()
    for ln in lines:
        if ln.startswith("## "):
            return ln[3:].strip()
    for ln in lines:
        if ln.startswith("# "):
            return ln[2:].strip()
    return fallback


def _lexical_overlap_score(sub_q: str, text: str, section: str) -> float:
    norm_q = sub_q.lower().replace("von", "vpn").replace("globel", "global").replace("acess", "access").replace("severs", "servers")
    combined = f"{section} {text}".lower()
    terms = [
        t
        for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", norm_q)
        if t not in _GROUNDING_STOPWORDS
    ]
    if not terms:
        return 0.0
    hits = sum(1 for t in terms if t in combined)
    return hits / max(len(terms), 1)


def _friendly_source_name(source: str) -> str:
    key = source.strip().lower()
    if key in _FRIENDLY_SOURCE_NAMES:
        return _FRIENDLY_SOURCE_NAMES[key]
    if key.endswith(".md"):
        return source.replace(".md", "").replace("_", " ").title()
    if key.endswith(".txt"):
        return source.replace(".txt", "").replace("_", " ").title()
    if key.endswith(".pdf"):
        return source.replace(".pdf", "").replace("_", " ").title()
    return source


def _has_high_confidence_specific_match(question: str, citations: list[dict]) -> bool:
    if not citations:
        return False
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}", question)
    terms = [w for w in words if w.lower() not in _GENERIC_QUERY_WORDS]
    if not terms:
        return False
    best = max((float(c.get("score", 0.0)) for c in citations), default=0.0)
    if best < 0.9:
        return False
    texts = [str(c.get("text") or "").lower() for c in citations]
    for term in terms:
        hits = sum(1 for t in texts if term.lower() in t)
        if hits == 1:
            return True
    return False


def _related_sections_hint(citations: list[dict], answered_section: str | None = None) -> str | None:
    sections: list[str] = []
    seen: set[str] = set()
    for c in citations:
        sec = str(c.get("section_title") or "").strip()
        if not sec:
            continue
        if answered_section and sec.lower() == answered_section.lower():
            continue
        if sec in seen:
            continue
        seen.add(sec)
        sections.append(sec)
    if not sections:
        return None
    preview = ", ".join(sections[:2])
    return f"If helpful, I can also share related guidance on {preview}."


def _sanitize_answer_filenames(answer: str) -> str:
    out = re.sub(r"\bit_guide\.md\b", "our official IT policy guide", answer, flags=re.I)
    out = re.sub(r"\bhandbook\.md\b", "our employee handbook", out, flags=re.I)
    return re.sub(r"\b[a-z0-9_-]+\.(md|txt|pdf)\b", "our internal policy guide", out, flags=re.I)


def _build_source_line(citations: list[dict]) -> str | None:
    if not citations:
        return None
    labels: list[str] = []
    seen: set[tuple[str, str]] = set()
    for c in citations:
        raw = str(c.get("source") or "").strip()
        if not raw:
            continue
        friendly = _friendly_source_name(raw)
        section_title = str(c.get("section_title") or "").strip()
        section_label = f"Section: {section_title}" if section_title else "Section: General"
        key = (friendly, section_label)
        if key in seen:
            continue
        seen.add(key)
        labels.append(f"{friendly} > {section_label}")
    if not labels:
        return None
    return "Source: " + "; ".join(labels[:2])


def _has_section_citation(citations: list[dict]) -> bool:
    return any(str(c.get("section_title") or "").strip() for c in citations)


def _profile_balance_summary(balance_snippet: str | None) -> str:
    if not balance_snippet:
        return ""
    pto = re.search(r"PTO days remaining:\s*(\d+)", balance_snippet, re.I)
    sick = re.search(r"Sick days remaining:\s*(\d+)", balance_snippet, re.I)
    if pto and sick:
        return f"You currently have {pto.group(1)} PTO days and {sick.group(1)} sick days remaining."
    return ""


def _strict_grounding_fallback(balance_snippet: str | None, *, with_people_partner_prompt: bool) -> str:
    bal_line = _profile_balance_summary(balance_snippet)
    profile_hint = (
        "\n\nWould you like me to draft a quick summary of this request for your People Partner?"
        if with_people_partner_prompt
        else ""
    )
    return (
        f"{bal_line}\n\n" if bal_line else ""
    ) + (
        "I couldn't find the specific policy text in the handbook. Please contact HR "
        "or ask me to search for a different section."
        f"{profile_hint}\n\n"
        "Next Steps:\n"
        "- Confirm the exact policy area you need (for example, remote-work location eligibility).\n"
        "- Share role/location constraints if applicable.\n"
        "- I can format this into a concise message for your People Partner."
    )


def _extract_profile_fields(balance_snippet: str | None) -> dict[str, str]:
    if not balance_snippet:
        return {}
    fields: dict[str, str] = {}
    employee = re.search(r"Employee:\s*(.+?)\s*\(([^)]+)\)", balance_snippet)
    if employee:
        fields["name"] = employee.group(1).strip()
        fields["employee_id"] = employee.group(2).strip()
    pto = re.search(r"PTO days remaining:\s*(\d+)", balance_snippet, re.I)
    sick = re.search(r"Sick days remaining:\s*(\d+)", balance_snippet, re.I)
    if pto:
        fields["pto_days"] = pto.group(1)
    if sick:
        fields["sick_days"] = sick.group(1)
    return fields


def _profile_answer_for_subquestion(sub_q: str, profile: dict[str, str]) -> str | None:
    q = sub_q.lower()
    qpad = f" {q} "
    personal_cues = any(k in qpad for k in (" my ", " i ", " me ")) or any(
        k in q for k in ("do i have", "remaining", "left")
    )
    identity_cues = any(k in q for k in ("employee id", "who am i", "my id", "my name"))
    balance_cues = any(k in q for k in _PROFILE_BALANCE_TERMS) and any(
        k in q for k in ("sick", "pto", "leave", "time off", "days")
    )
    policy_cues = any(k in q for k in ("policy", "rule", "rollover", "advance notice", "required", "requirement"))
    wants_identity = identity_cues
    wants_balance = personal_cues and balance_cues and not policy_cues
    parts: list[str] = []
    if wants_identity and profile.get("name") and profile.get("employee_id"):
        parts.append(f"Your name is {profile['name']} and your employee ID is {profile['employee_id']}.")
    if wants_balance and (profile.get("sick_days") or profile.get("pto_days")):
        sick = profile.get("sick_days")
        pto = profile.get("pto_days")
        if sick and pto:
            parts.append(f"You currently have {sick} sick days and {pto} PTO days remaining.")
        elif sick:
            parts.append(f"You currently have {sick} sick days remaining.")
        elif pto:
            parts.append(f"You currently have {pto} PTO days remaining.")
    if not parts:
        return None
    return " ".join(parts)


def _is_profile_only_query(question: str) -> bool:
    q = question.lower()
    has_profile = any(h in q for h in _PROFILE_QUERY_HINTS)
    has_personal_cue = any(k in q for k in ("my ", " me ", " do i have", "remaining", "days left", "who am i"))
    has_policy = any(h in q for h in _POLICY_QUERY_HINTS)
    return has_profile and has_personal_cue and not has_policy


def _sanitize_history_for_retrieval(
    chat_history: list[dict[str, str]] | None,
    current_question: str,
) -> list[dict[str, str]]:
    """Prevent post-escalation retrieval poisoning by sensitive keywords."""
    if not chat_history:
        return []
    if SENSITIVE_PATTERNS.search(current_question):
        return chat_history
    has_sensitive = any(SENSITIVE_PATTERNS.search(str(item.get("content", ""))) for item in chat_history)
    has_escalation_card = any(
        "I have detected that this is a sensitive matter" in str(item.get("content", ""))
        for item in chat_history
    )
    if has_sensitive or has_escalation_card:
        return []
    return chat_history


def _build_multi_match_options(citations: list[dict]) -> str | None:
    """Return a friendly options list when multiple relevant sections are present."""
    if not citations:
        return None
    picked: list[str] = []
    seen: set[str] = set()
    for c in citations:
        title = str(c.get("section_title") or "").strip()
        if not title:
            continue
        norm = title.lower()
        if norm in seen:
            continue
        seen.add(norm)
        picked.append(title)
    if len(picked) < 2:
        return None
    listed = ", ".join(f"Section: {p}" for p in picked[:4])
    return (
        "I found a few policies that might help: "
        f"{listed}. Which one would you like to dive into?"
    )


def _citation_snippet(citation: dict) -> str:
    text = str(citation.get("text") or "").strip()
    if not text:
        return "No grounded excerpt available."
    first = re.split(r"(?<=[.!?])\s+", text)[0].strip()
    if len(first) > 220:
        first = first[:220].rstrip() + "..."
    return first




def node_guardrail(state: RagState) -> RagState:
    q = state.get("question", "")
    if SENSITIVE_PATTERNS.search(q):
        return {
            **state,
            "escalate": True,
            "escalation_reason": (
                "Sensitive HR/legal matter detected."
            ),
            "answer": (
                "I have detected that this is a sensitive matter. For your protection and to ensure "
                "proper handling, I cannot provide policy details on this topic. Please contact the "
                "HR Intake Team directly at hr-support@company.com or via the internal portal link below."
            ),
            "citations": [],
            "retrieval_citations": [],
        }
    return {**state, "escalate": False, "escalation_reason": None}


def route_after_guardrail(state: RagState) -> str:
    return "end" if state.get("escalate") else "router"


def router_node(state: RagState) -> RagState:
    """Semantic intent classifier for dynamic routing."""
    question = state.get("question", "")
    system = """Classify user intent into exactly one label:
- INTENT_PERSONAL: user asks about own data/profile/status/balances.
- INTENT_POLICY: user asks about company rules/process/policies.
- INTENT_GENERAL: greetings/chitchat/unrelated conversation.

Return STRICT JSON with keys:
- intent: one of INTENT_PERSONAL | INTENT_POLICY | INTENT_GENERAL
- reason: short rationale"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured = llm.with_structured_output(IntentRouteOut)
    try:
        out = structured.invoke(
            [
                SystemMessage(content=system),
                HumanMessage(content=f"User query:\n{question}"),
            ]
        )
        data = out.model_dump()
        intent_raw = str(data.get("intent", INTENT_POLICY)).strip().upper()
        if intent_raw not in {INTENT_PERSONAL, INTENT_POLICY, INTENT_GENERAL}:
            intent_raw = INTENT_POLICY
        reason = str(data.get("reason", "semantic router default"))
    except Exception:
        intent_raw = INTENT_POLICY
        reason = "semantic router fallback"
    return {**state, "intent": intent_raw, "intent_reason": reason}


def route_after_router(state: RagState) -> str:
    if not state.get("use_rag", True):
        return "general"
    intent = state.get("intent", INTENT_POLICY)
    if intent in {INTENT_PERSONAL, INTENT_POLICY}:
        return "policy"
    return "general"


def query_refiner_node(state: RagState) -> RagState:
    """Rewrite turn into standalone query using threaded history when needed."""
    q = state.get("question", "").strip()
    if not q:
        return {**state}
    msgs = state.get("messages") or []
    history_lines: list[str] = []
    # Exclude newest user query from history summary.
    for m in msgs[:-1]:
        role = "assistant" if isinstance(m, AIMessage) else "user"
        content = str(m.content).strip()
        if not content:
            continue
        history_lines.append(f"{role}: {content}")
    history = "\n".join(history_lines[-8:]) if history_lines else "<none>"
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured = llm.with_structured_output(QueryRefinerOut)
    try:
        out = structured.invoke(
            [
                SystemMessage(
                    content=(
                        "Rewrite the latest user query as a standalone search query when needed. "
                        "If already clear, keep it almost unchanged. Return STRICT JSON with keys "
                        "standalone_query, reason, alternatives (0-3 extra retrieval variants), "
                        "and sub_questions (list of independent atomic user asks). "
                        "Do not carry stale terms from unrelated prior turns."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Conversation history:\n{history}\n\n"
                        f"Latest query:\n{q}\n\n"
                        "If VPN/provider/setup appears in the query, include alternatives such as:\n"
                        "- VPN access and setup\n"
                        "- Company VPN provider\n"
                        "- Accessing internal servers remotely\n\n"
                        "Example ambiguous query: 'yes help with the process' should be rewritten "
                        "into a complete standalone enterprise policy query."
                    )
                ),
            ]
        )
        data = out.model_dump()
        refined = str(data.get("standalone_query", q)).strip() or q
        alt_raw = data.get("alternatives") or []
        alts = [str(x).strip() for x in alt_raw if str(x).strip()]
        sq_raw = data.get("sub_questions") or []
        sub_questions = [
            str(x).strip()
            for x in sq_raw
            if str(x).strip() and not str(x).strip().lower().startswith("expected:")
        ]
    except Exception:
        refined = q
        alts = []
        sub_questions = []

    # Deterministic fallback expansion for VPN-style asks.
    if "vpn" in q.lower():
        seeds = [
            "VPN access and setup",
            "Company VPN provider",
            "Accessing internal servers remotely",
        ]
        existing = {a.lower() for a in alts}
        for seed in seeds:
            if seed.lower() != refined.lower() and seed.lower() not in existing:
                alts.append(seed)
                existing.add(seed.lower())

    retrieval_queries = [refined]
    for alt in alts:
        if alt.lower() not in {rq.lower() for rq in retrieval_queries}:
            retrieval_queries.append(alt)
    fallback_sq = _fallback_sub_questions(q)
    is_multi_prompt = _is_multi_question_prompt(q)
    # Prefer literal user clauses for multi-part prompts to avoid LLM-invented paraphrases.
    if not is_multi_prompt:
        sub_questions = [q]
    elif len(fallback_sq) >= 2:
        sub_questions = fallback_sq
    elif not sub_questions or (len(sub_questions) <= 1 and len(fallback_sq) > 1):
        sub_questions = fallback_sq
    else:
        # Preserve explicitly asked items that LLM decomposition may have dropped.
        known = {re.sub(_QUESTION_NORMALIZE_RE, " ", sq.strip().lower()).strip() for sq in sub_questions}
        for sq in fallback_sq:
            key = re.sub(_QUESTION_NORMALIZE_RE, " ", sq.strip().lower()).strip()
            if key not in known:
                sub_questions.append(sq)
                known.add(key)
    return {
        **state,
        "retrieval_query": refined,
        "retrieval_queries": retrieval_queries[:4],
        "sub_questions": sub_questions[:MAX_SUB_QUESTIONS],
    }


def node_retrieve(state: RagState) -> RagState:
    if not state.get("use_rag", True):
        return {
            **state,
            "retrieved_context": "",
            "retrieval_citations": [],
        }
    q = state.get("retrieval_query") or state["question"]
    queries = list(state.get("retrieval_queries") or [q])
    sub_questions = list(state.get("sub_questions") or [q])
    attempt = int(state.get("retrieval_attempt", 1))
    vs = get_vectorstore()
    k = RETRIEVE_K
    citations: list[dict] = []
    parts: list[str] = []
    sub_results: list[dict] = []
    candidates: list[tuple[float, str, str, str]] = []
    seen_chunks: set[str] = set()
    for sub_q in sub_questions:
        local_queries = [sub_q]
        if "vpn" in sub_q.lower():
            local_queries.extend(
                [
                    "VPN access and setup",
                    "Company VPN provider",
                    "Accessing internal servers remotely",
                ]
            )
        for extra in queries:
            if extra.lower() not in {x.lower() for x in local_queries}:
                local_queries.append(extra)
        local_candidates: list[tuple[float, str, str, str]] = []
        query_cap = MAX_LOCAL_QUERIES_VPN if "vpn" in sub_q.lower() else MAX_LOCAL_QUERIES_DEFAULT
        for query in local_queries[:query_cap]:
            try:
                docs = vs.similarity_search_with_score(query, k=k)
            except Exception:
                docs = []
            for doc, dist in docs:
                score = 1.0 / (1.0 + float(dist))
                src = doc.metadata.get("source", "handbook")
                source_name = str(doc.metadata.get("source_name") or Path(str(src)).name)
                section_title = _extract_best_section(
                    doc.page_content.strip(),
                    str(doc.metadata.get("section_title") or "General"),
                )
                full_text = doc.page_content.strip()
                if "vpn" in sub_q.lower() and source_name.lower() == "it_guide.md":
                    score += 0.12
                local_score = score + 0.4 * _lexical_overlap_score(sub_q, full_text, section_title)
                local_candidates.append((local_score, source_name, section_title, full_text))
        local_candidates.sort(key=lambda x: x[0], reverse=True)
        best = local_candidates[:2]
        sub_citations: list[dict] = []
        for score, source_name, section_title, full_text in best:
            text = full_text if len(full_text) <= 2000 else full_text[:2000] + "…"
            sub_citations.append(
                {
                    "text": text,
                    "score": round(score, 4),
                    "source": source_name,
                    "section_title": section_title,
                }
            )
            chunk_key = full_text[:500].lower()
            if chunk_key and chunk_key not in seen_chunks:
                seen_chunks.add(chunk_key)
                candidates.append((score, source_name, section_title, full_text))
        # Keep only citations lexically grounded for this sub-question.
        grounded_sub_citations = [c for c in sub_citations if _sub_question_grounded(sub_q, c)]
        sub_results.append({"question": sub_q, "citations": grounded_sub_citations})

    candidates.sort(key=lambda x: x[0], reverse=True)
    top = candidates[:k]
    for i, (score, source_name, section_title, full_text) in enumerate(top):
        text = full_text if len(full_text) <= 2000 else full_text[:2000] + "…"
        citations.append(
            {
                "text": text,
                "score": round(score, 4),
                "source": source_name,
                "section_title": section_title,
            }
        )
        parts.append(
            f"[Chunk {i + 1} | source: {source_name} | section_title: {section_title}]\n{full_text}"
        )
    top_score = max((float(c["score"]) for c in citations), default=0.0)
    if citations:
        min_keep = max(0.42, top_score - 0.12)
        kept: list[dict] = []
        kept_parts: list[str] = []
        for c, p in zip(citations, parts):
            if float(c.get("score", 0.0)) >= min_keep:
                kept.append(c)
                kept_parts.append(p)
        if kept:
            citations = kept
            parts = kept_parts
        else:
            # Grounding gate should not erase all retrieval evidence; keep best chunk.
            citations = [citations[0]]
            parts = [parts[0]]
    ctx = "\n\n---\n\n".join(parts) if parts else ""
    history = list(state.get("retrieval_attempts") or [])
    history.append(
        {
            "attempt": attempt,
            "query": q,
            "top_score": round(top_score, 4),
            "verdict": "answerable",
            "reason": "pending grade",
            "citations": citations,
        }
    )
    return {
        **state,
        "retrieval_query": q,
        "retrieval_attempt": attempt,
        "retrieved_context": ctx,
        "retrieval_citations": citations,
        "retrieval_top_score": top_score,
        "retrieval_attempts": history,
        "sub_results": sub_results,
    }


def grade_documents(state: RagState) -> RagState:
    """Assess retrieval quality and optionally trigger a re-search query rewrite."""
    citations = state.get("retrieval_citations") or []
    top_score = float(state.get("retrieval_top_score", 0.0))
    attempt = int(state.get("retrieval_attempt", 1))
    sub_questions = list(state.get("sub_questions") or [])
    max_attempts = RETRIEVE_MAX_ATTEMPTS_FAST if len(sub_questions) >= MEGA_PROMPT_SUBQUESTION_THRESHOLD else RETRIEVE_MAX_ATTEMPTS
    if not citations:
        verdict = "re-search"
        reason = "No citations returned from vector search"
    elif top_score >= GRADE_FASTPATH_SCORE and len(citations) >= 2:
        verdict = "answerable"
        reason = f"Fast-path accept: top score {top_score:.2f} with {len(citations)} citation(s)"
    elif top_score < RETRIEVE_SCORE_THRESHOLD:
        high_quality_count = sum(1 for c in citations if float(c.get("score", 0.0)) >= 0.35)
        if high_quality_count >= 3:
            verdict = "answerable"
            reason = f"Multiple relevant chunks ({high_quality_count}) support option-summary answer"
        else:
            verdict = "re-search"
            reason = f"Top score {top_score:.2f} below threshold {RETRIEVE_SCORE_THRESHOLD:.2f}"
    else:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        structured = llm.with_structured_output(RetrievalGradeOut)
        preview = "\n\n".join(
            [
                f"[{i + 1}] score={c.get('score', 0):.4f} source={c.get('source', 'unknown')}\n{str(c.get('text', ''))[:500]}"
                for i, c in enumerate(citations[:3])
            ]
        )
        try:
            out = structured.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are a retrieval relevance grader. Decide if the retrieved chunks are relevant enough "
                            "to answer the question. Return STRICT JSON with keys: verdict, reason. "
                            "verdict must be exactly one of: answerable, re-search. "
                            "Set needs_second_hop=true when context is partial (e.g., technical facts without process steps)."
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"Question:\n{state.get('question', '')}\n\nRetrieved chunks:\n{preview}\n\n"
                            "If chunks are weak/off-topic, pick re-search."
                        )
                    ),
                ]
            )
            data = out.model_dump()
            verdict = str(data.get("verdict", "answerable")).strip().lower()
            reason = str(data.get("reason", "llm relevance check"))
            needs_second_hop = bool(data.get("needs_second_hop", False))
            if verdict not in {"answerable", "re-search"}:
                verdict = "answerable"
        except Exception:
            verdict = "answerable"
            reason = "grader fallback"
            needs_second_hop = False
    if "needs_second_hop" not in locals():
        needs_second_hop = False

    updated_attempts = list(state.get("retrieval_attempts") or [])
    if updated_attempts:
        updated_attempts[-1] = {
            **updated_attempts[-1],
            "verdict": verdict,
            "reason": reason,
        }

    if (verdict == "re-search" or needs_second_hop) and attempt < max_attempts:
        rewriter = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        structured_rewriter = rewriter.with_structured_output(QueryRewriteOut)
        current_query = state.get("retrieval_query") or state.get("question", "")
        try:
            out2 = structured_rewriter.invoke(
                [
                    SystemMessage(
                        content=(
                            "Rewrite the search query for enterprise policy retrieval. "
                            "Expand shorthand terms into policy/process language and explore missing intent "
                            "(e.g., process, eligibility, exceptions). Return STRICT JSON with keys query and reason."
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"Original question:\n{state.get('question', '')}\n\n"
                            f"Current retrieval query:\n{current_query}\n\n"
                            f"Failed reason:\n{reason}"
                        )
                    ),
                ]
            )
            rewritten = out2.model_dump()
            next_query = str(rewritten.get("query", current_query)).strip() or current_query
            rewrite_reason = str(rewritten.get("reason", reason))
        except Exception:
            next_query = current_query
            rewrite_reason = f"{reason}; rewrite fallback"
        return {
            **state,
            "retrieval_verdict": "RE-SEARCH",
            "retrieval_reason": rewrite_reason,
            "retrieval_query": next_query,
            "retrieval_attempt": attempt + 1,
            "retrieval_attempts": updated_attempts,
        }

    final_verdict = "FAILED" if verdict == "re-search" else "ANSWERABLE"
    return {
        **state,
        "retrieval_verdict": final_verdict,
        "retrieval_reason": reason,
        "retrieval_attempts": updated_attempts,
    }


def route_after_grade(state: RagState) -> str:
    return "retrieve" if state.get("retrieval_verdict") == "RE-SEARCH" else "generate"


def node_balance(state: RagState) -> RagState:
    if not state.get("use_rag", True):
        return {**state, "balance_snippet": None, "mcp_status_detail": "Intent routing disabled in direct LLM mode"}

    eid = state.get("employee_id")
    if not eid:
        return {
            **state,
            "balance_snippet": None,
            "mcp_status_detail": "No employee ID provided",
        }
    details = get_employee_details(eid)
    if not details:
        return {
            **state,
            "balance_snippet": None,
            "mcp_status_detail": f"No employee record found for {eid}",
        }
    snippet = (
        "User Profile (secure HR context):\n"
        f"- Employee: {details['name']} ({details['employee_id']})\n"
        f"- PTO days remaining: {details['pto_balance']:.0f}\n"
        f"- Sick days remaining: {details['sick_balance']:.0f}\n"
        "Always check this profile. For questions like 'how many', 'my status', or "
        "'can I take time off', answer with these personal numbers first, then explain "
        "the relevant handbook policy."
    )
    return {
        **state,
        "balance_snippet": snippet,
        "mcp_status_detail": f"✅ Active: Securely retrieved data for {details['employee_id']}",
    }


def node_generate(state: RagState) -> RagState:
    use_rag = state.get("use_rag", True)
    intent = state.get("intent", INTENT_POLICY)
    citations = list(state.get("retrieval_citations") or [])
    sub_questions = list(state.get("sub_questions") or [])
    sub_results = list(state.get("sub_results") or [])
    profile = _extract_profile_fields(state.get("balance_snippet"))
    if use_rag and intent == INTENT_GENERAL:
        return {
            **state,
            "answer": (
                "I can help with company handbook, HR, and IT policy questions only. "
                "I couldn't find a grounded policy source for this request.\n\n"
                "Next Steps:\n"
                "- Ask a company policy question (for example: PTO, sick leave, VPN, hardware).\n"
                "- Or share the policy area and I will search that section directly."
            ),
            "retrieval_citations": [],
        }
    if use_rag and profile and _is_profile_only_query(state.get("question", "")):
        profile_answer = _profile_answer_for_subquestion(state.get("question", ""), profile)
        if profile_answer:
            return {**state, "answer": profile_answer}
    if use_rag and (len(sub_questions) >= 2 or len(sub_results) >= 2):
        profile_items: list[str] = []
        grounded_items: list[str] = []
        grounded_citations: list[dict] = []
        not_found_items: list[str] = []
        seen_profile: set[str] = set()
        seen_grounded: set[str] = set()
        seen_not_found: set[str] = set()
        for item in sub_results:
            sub_q = str(item.get("question") or "").strip()
            sub_q_key = re.sub(_QUESTION_NORMALIZE_RE, " ", sub_q.lower()).strip()
            profile_answer = _profile_answer_for_subquestion(sub_q, profile)
            if profile_answer:
                if sub_q_key not in seen_profile:
                    profile_items.append(f"- **{sub_q}**\n  - {profile_answer}")
                    seen_profile.add(sub_q_key)
                continue
            sub_cites = list(item.get("citations") or [])
            good = [
                c
                for c in sub_cites
                if str(c.get("section_title") or "").strip()
                and _sub_question_grounded(sub_q, c)
            ]
            if good:
                c0 = good[0]
                section = str(c0.get("section_title") or "General")
                snippet = _citation_snippet(c0)
                if sub_q_key not in seen_grounded:
                    grounded_items.append(f"- **{sub_q}**\n  - {snippet}\n  - Section: {section}")
                    grounded_citations.extend(good[:1])
                    seen_grounded.add(sub_q_key)
            else:
                if sub_q_key not in seen_not_found:
                    not_found_items.append(f"- {sub_q}")
                    seen_not_found.add(sub_q_key)

        # Include any decomposed sub-questions that had no retrieval result entry.
        covered = {
            re.sub(_QUESTION_NORMALIZE_RE, " ", str(item.get("question") or "").strip().lower()).strip()
            for item in sub_results
        }
        for sq in sub_questions:
            sq_norm = re.sub(_QUESTION_NORMALIZE_RE, " ", sq.strip().lower()).strip()
            if sq_norm not in covered:
                sq_key = re.sub(_QUESTION_NORMALIZE_RE, " ", sq.strip().lower()).strip()
                profile_answer = _profile_answer_for_subquestion(sq.strip(), profile)
                if profile_answer:
                    if sq_key not in seen_profile:
                        profile_items.append(f"- **{sq.strip()}**\n  - {profile_answer}")
                        seen_profile.add(sq_key)
                    continue
                if sq_key not in seen_not_found:
                    not_found_items.append(f"- {sq.strip()}")
                    seen_not_found.add(sq_key)

        blocks: list[str] = []
        if profile_items:
            blocks.append("Grounded Answers (Profile):\n" + "\n".join(profile_items))
        if grounded_items:
            blocks.append("Grounded Answers (Policy):\n" + "\n".join(grounded_items))
        if not_found_items:
            blocks.append(
                "Not Found:\n"
                + "\n".join(not_found_items)
                + "\n\nNext Steps:\n"
                "- I couldn't find specific policy text for the items above in the current handbook.\n"
                "- Ask me to search another section or I can help draft a quick summary request for your People Partner."
            )
        if blocks:
            answer = "\n\n".join(blocks)
            src = _build_source_line(grounded_citations)
            if src:
                answer = f"{answer}\n\n{src}"
            return {**state, "answer": answer}
    if use_rag and intent == INTENT_POLICY and _is_broad_query(state.get("question", "")):
        options = _build_multi_match_options(citations)
        if options:
            src = _build_source_line(citations)
            answer = (
                "I found multiple relevant paths and included both so you can act immediately.\n\n"
                f"{options}\n\n{src}"
                if src
                else f"I found multiple relevant paths and included both so you can act immediately.\n\n{options}"
            )
            return {**state, "answer": answer}
    ctx = state.get("retrieved_context", "") or ""
    bal = state.get("balance_snippet")
    if use_rag and state.get("intent") == INTENT_POLICY and not ctx.strip():
        bal_line = _profile_balance_summary(bal)
        answer = (
            f"{bal_line}\n\n" if bal_line else ""
        ) + (
            "I couldn't find the specific policy text in the handbook. Please contact HR "
            "or ask me to search for a different section."
        )
        return {**state, "answer": answer}
    if (
        use_rag
        and state.get("intent") == INTENT_POLICY
        and state.get("retrieval_verdict") == "FAILED"
        and not state.get("balance_snippet")
    ):
        return {
            **state,
            "answer": _strict_grounding_fallback(
                state.get("balance_snippet"),
                with_people_partner_prompt=bool(state.get("employee_id")),
            ),
        }
    if bal and intent in {INTENT_PERSONAL, INTENT_POLICY}:
        ctx = f"{ctx}\n\n---\n{bal}\n"

    if use_rag and intent != INTENT_GENERAL:
        system = """You are a Support Colleague using a Universal Reasoning Engine.
For every query follow this exact sequence:
1) ANALYZE: infer user situation and intent from question + context.
2) MULTI-PATH: if context contains multiple valid processes/paths, provide both paths immediately.
   Never ask "which one?" when you can provide both.
3) PERSONALIZATION: always check User Profile context when present and merge it with policy.
   If user asks "how many"/"status"/eligibility and profile provides numbers, answer with those first,
   then explain the policy/process.
4) BOUNDARY: if answer is not grounded in context, do not hallucinate. Explain what was found and
   provide official HR/IT contact direction.

Additional rules:
- Stay grounded in handbook excerpts and User Profile.
- Keep tone warm, concise, and actionable.
- NEVER start a sentence with "Contact HR" or "I recommend reaching out to HR."
- If information is missing, use concierge language and offer proactive intake support
  (for example, drafting a summary request for the People Partner).
- If profile context includes a People Partner field, offer to surface that contact.
- Do not mention raw filenames in body text. Put sources in a final "Source: ..." line.
- Use "Source:" only for grounded facts from context.
- For missing facts, use a "Next Steps:" block with a collaborative plan.
- Respond in English only.
- Output STRICT JSON with a single key: answer (string). No other keys."""
    else:
        system = """You are a helpful assistant.
You do NOT have access to the company employee handbook, internal HR documents, or any employee database (including leave balances).
Answer using general knowledge only, in English.
For company-specific policy questions, clearly state that you cannot see internal policy and the user should contact HR.
Output STRICT JSON with a single key: answer (string). No other keys."""

    human = f"Question:\n{state['question']}\n\nHandbook excerpts:\n{ctx}"

    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)
    structured = llm.with_structured_output(HandbookAnswerOut)
    out = structured.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=human),
        ]
    )
    data = out.model_dump()
    answer = str(data.get("answer", "")).strip()
    if use_rag:
        answer = _sanitize_answer_filenames(answer)
        has_question_grounding = any(_sub_question_grounded(state.get("question", ""), c) for c in citations)
        if state.get("intent") == INTENT_POLICY and not has_question_grounding:
            answer = _strict_grounding_fallback(
                state.get("balance_snippet"),
                with_people_partner_prompt=bool(state.get("employee_id")),
            )
        if _has_high_confidence_specific_match(state.get("question", ""), citations):
            answered_section = None
            first = (citations or [None])[0]
            if isinstance(first, dict):
                answered_section = str(first.get("section_title") or "").strip() or None
            hint = _related_sections_hint(
                citations,
                answered_section=answered_section,
            )
            if hint and hint.lower() not in answer.lower():
                answer = f"{answer}\n\n{hint}"
        # Strict grounding gate: policy claims require section-level citation evidence.
        has_policy_claim = bool(POLICY_CLAIM_PATTERN.search(answer) or POLICY_SCOPE_PATTERN.search(answer))
        lacks_section_grounding = not _has_section_citation(citations)
        has_generic_fake_source = "source: handbook excerpts" in answer.lower()
        if state.get("intent") == INTENT_POLICY and (has_policy_claim and lacks_section_grounding or has_generic_fake_source):
            answer = _strict_grounding_fallback(
                state.get("balance_snippet"),
                with_people_partner_prompt=bool(state.get("employee_id")),
            )
        source_line = _build_source_line(citations)
        has_missing_fact_signal = "Next Steps:" in answer or "don't have the specific" in answer.lower()
        if source_line and "source:" not in answer.lower() and not has_missing_fact_signal and has_question_grounding:
            answer = f"{answer}\n\n{source_line}"
    return {
        **state,
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }


def build_graph():
    g = StateGraph(RagState)
    g.add_node("query_refiner", query_refiner_node)
    g.add_node("guardrail", node_guardrail)
    g.add_node("router", router_node)
    g.add_node("retrieve", node_retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("balance", node_balance)
    g.add_node("generate", node_generate)

    g.set_entry_point("query_refiner")
    g.add_edge("query_refiner", "guardrail")
    g.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"end": END, "router": "router"},
    )
    g.add_conditional_edges(
        "router",
        route_after_router,
        {"policy": "retrieve", "general": "balance"},
    )
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges(
        "grade_documents",
        route_after_grade,
        {"retrieve": "retrieve", "generate": "balance"},
    )
    g.add_edge("balance", "generate")
    g.add_edge("generate", END)
    return g.compile()


_compiled = None


def get_compiled_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled


def _build_pipeline_steps(result: RagState, use_rag: bool) -> list[dict]:
    if result.get("escalate"):
        return [
            {
                "id": "guardrail",
                "label": "Guardrail (sensitive topics)",
                "status": "triggered",
                "detail": (result.get("escalation_reason") or "")[:200],
            },
            {
                "id": "retrieve",
                "label": "Chroma vector retrieval",
                "status": "skipped",
                "detail": "Pipeline stopped after guardrail",
            },
            {
                "id": "mcp_hr",
                "label": "Employee profile data (MCP-style)",
                "status": "skipped",
                "detail": "Not run",
            },
            {
                "id": "llm",
                "label": f"OpenAI chat — {OPENAI_CHAT_MODEL}",
                "status": "skipped",
                "detail": "Escalation message returned",
            },
        ]

    steps: list[dict] = [
        {
            "id": "guardrail",
            "label": "Guardrail",
            "status": "ok",
            "detail": "No sensitive pattern matched",
        },
        {
            "id": "router",
            "label": "Semantic Intent Router",
            "status": "ok",
            "detail": f"{result.get('intent', INTENT_POLICY)} — {result.get('intent_reason', '')}",
        },
    ]
    if use_rag:
        intent = result.get("intent", INTENT_POLICY)
        n = len(result.get("retrieval_citations") or [])
        attempts = result.get("retrieval_attempts") or []
        if intent == INTENT_POLICY:
            st = "ok" if n else "empty"
            detail = (
                f"{n} chunk(s), {len(attempts)} retrieval attempt(s)"
                if n
                else "No vectors (ingest handbook or check API key)"
            )
            if result.get("retrieval_verdict") == "FAILED":
                st = "empty"
                detail = f"Re-search failed after {len(attempts)} attempt(s): {result.get('retrieval_reason', '')}"
        else:
            st = "skipped"
            detail = (
                "Intent is personal data lookup"
                if intent == INTENT_PERSONAL
                else "Intent is general conversation"
            )
        steps.append(
            {
                "id": "retrieve",
                "label": "Chroma retrieval (embeddings + similarity)",
                "status": st,
                "detail": detail,
            }
        )
        mcp_detail = result.get("mcp_status_detail")
        if result.get("balance_snippet"):
            steps.append(
                {
                    "id": "mcp_hr",
                    "label": "Employee tool (MCP-style)",
                    "status": "ok",
                    "detail": mcp_detail or "✅ Active: Employee profile merged into prompt",
                }
            )
        else:
            steps.append(
                {
                    "id": "mcp_hr",
                    "label": "Employee tool (MCP-style)",
                    "status": "skipped",
                    "detail": mcp_detail or "No employee profile context",
                }
            )
    else:
        steps.append(
            {
                "id": "retrieve",
                "label": "Chroma retrieval",
                "status": "skipped",
                "detail": "Direct LLM mode (no handbook vectors)",
            }
        )
        steps.append(
            {
                "id": "mcp_hr",
                "label": "Employee profile data (MCP-style)",
                "status": "skipped",
                "detail": "Disabled in direct LLM mode",
            }
        )
    steps.append(
        {
            "id": "llm",
            "label": f"OpenAI chat — {OPENAI_CHAT_MODEL}",
            "status": "ok",
            "detail": f"Embeddings: {OPENAI_EMBEDDING_MODEL}; structured JSON answer",
        }
    )
    return steps


def run_ask(
    question: str,
    *,
    employee_id: str | None = None,
    chat_history: list[dict[str, str]] | None = None,
    use_rag: bool = True,
) -> dict:
    graph = get_compiled_graph()
    safe_history = _sanitize_history_for_retrieval(chat_history, question)
    prior_messages: list[AnyMessage] = []
    for item in safe_history:
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        if role == "assistant":
            prior_messages.append(AIMessage(content=content))
        else:
            prior_messages.append(HumanMessage(content=content))
    initial: RagState = {
        "question": question,
        "employee_id": employee_id,
        "use_rag": use_rag,
        "retrieval_query": question,
        "retrieval_attempt": 1,
        "retrieval_attempts": [],
        "messages": [*prior_messages, HumanMessage(content=question)],
    }
    result = graph.invoke(initial)
    pipeline_steps = _build_pipeline_steps(result, use_rag)

    if result.get("escalate"):
        return {
            "answer": result.get("answer", ""),
            "citations": [],
            "retrieval_attempts": [],
            "isEscalated": True,
            "escalation_reason": result.get("escalation_reason"),
            "pipeline_steps": pipeline_steps,
            "use_rag": use_rag,
            "chat_model": OPENAI_CHAT_MODEL,
        }
    cites = result.get("retrieval_citations") or []
    return {
        "answer": result.get("answer", ""),
        "citations": cites,
        "retrieval_attempts": result.get("retrieval_attempts") or [],
        "isEscalated": False,
        "escalation_reason": None,
        "pipeline_steps": pipeline_steps,
        "use_rag": use_rag,
        "chat_model": OPENAI_CHAT_MODEL,
    }
