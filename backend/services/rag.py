"""
services/rag.py
===============
RAG (Retrieval-Augmented Generation) pipeline for the Legal Assistant.

Uses ChromaDB (already managed by CUADIndex) to:
  1. Index a newly uploaded contract's clauses so the assistant can answer
     questions about it specifically.
  2. Answer user questions by retrieving the top-N matching clauses and
     building a structured template response with citation badges.

No external LLM API is required. Answers are formed from retrieved clause
text — plug in OpenAI/Gemini by replacing `_compose_answer()`.
"""
from __future__ import annotations

import os
import re
import httpx
import logging
from typing import Dict, List

from cuad_index import CUADIndex

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Answer templates keyed on question intent                            #
# ------------------------------------------------------------------ #

_INTENT_PATTERNS = [
    ("risk",       re.compile(r"risk|concern|danger|problem|issue|flag", re.I)),
    ("liability",  re.compile(r"liabilit|indemnif|exposure|cap|ceiling", re.I)),
    ("termination",re.compile(r"terminat|cancel|exit|end|expir", re.I)),
    ("payment",    re.compile(r"payment|pay|fee|price|cost|escal|invoic", re.I)),
    ("ip",         re.compile(r"ip\b|intellectual property|license|patent|copyright|trademark", re.I)),
    ("data",       re.compile(r"data|privacy|gdpr|ccpa|hipaa|personal information", re.I)),
    ("noncompete", re.compile(r"non-?compete|compet|restrict|solicit|exclusiv", re.I)),
    ("governing",  re.compile(r"governing law|jurisdiction|arbitration|dispute|forum", re.I)),
]

_INTENT_INTRODUCTIONS: Dict[str, str] = {
    "risk": (
        "Here are the key risk areas I've identified in this contract based on "
        "deviation from CUAD market-standard language patterns:"
    ),
    "liability": (
        "The contract contains the following liability-related provisions:"
    ),
    "termination": (
        "The contract includes these termination and expiration provisions:"
    ),
    "payment": (
        "Here are the payment, fee, and pricing provisions I found:"
    ),
    "ip": (
        "The contract contains the following intellectual property provisions:"
    ),
    "data": (
        "Here are the data protection and privacy-related clauses:"
    ),
    "noncompete": (
        "The contract contains these competitive restriction provisions:"
    ),
    "governing": (
        "The governing law and dispute resolution provisions are as follows:"
    ),
    "default": (
        "Based on my analysis of this contract, here are the most relevant sections:"
    ),
}


def _detect_intent(question: str) -> str:
    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(question):
            return intent
    return "default"


def _compose_answer(
    question: str,
    retrieved: List[Dict],
    contract_name: str = "this contract",
) -> str:
    """
    Build a structured answer string from retrieved clause chunks.

    To plug in an LLM, replace this function with an API call and pass
    `retrieved` as context — the citation metadata stays the same.
    """
    intent = _detect_intent(question)
    intro = _INTENT_INTRODUCTIONS.get(intent, _INTENT_INTRODUCTIONS["default"])

    if not retrieved:
        return (
            f"I couldn't find specific clauses in {contract_name} that directly "
            f"address your question. Try rephrasing or ask about a specific clause "
            f"type (e.g. 'liability', 'termination', 'payment terms')."
        )

    lines = [intro, ""]
    for i, chunk in enumerate(retrieved, 1):
        section = chunk.get("section", f"§{i}")
        text = chunk.get("text", "")
        category = chunk.get("category", "")
        risk_level = chunk.get("risk_level", "")

        # Truncate long clause text for the chat response
        preview = text[:350].strip()
        if len(text) > 350:
            preview += "…"

        risk_note = ""
        if risk_level in ("critical", "high"):
            risk_note = f" ⚠️ **{risk_level.upper()} RISK**"

        if category and category != section:
            lines.append(f"**{section}** ({category}){risk_note}")
        else:
            lines.append(f"**{section}**{risk_note}")
        lines.append(preview)
        lines.append("")

    # Add a brief actionable note based on intent
    lines.append(_action_note(intent))

    return "\n".join(lines)


def _action_note(intent: str) -> str:
    notes = {
        "risk": (
            "💡 **Tip**: Focus on critical and high-risk clauses first. Use the "
            "Clause Analysis screen to see suggested market-standard rewrites."
        ),
        "liability": (
            "💡 **Tip**: Ensure the liability cap equals at least 12 months' fees "
            "and includes carve-outs for IP infringement and data breaches."
        ),
        "termination": (
            "💡 **Tip**: Verify that confidentiality obligations survive termination "
            "and that data return windows are sufficient for migration."
        ),
        "payment": (
            "💡 **Tip**: Watch for price escalation caps above 5% and ensure you "
            "have the right to offset against credits owed."
        ),
        "ip": (
            "💡 **Tip**: Check whether any IP license is irrevocable or perpetual — "
            "these provisions survive contract termination."
        ),
        "data": (
            "💡 **Tip**: Ensure a Data Processing Agreement (DPA) is in place and "
            "that processing purposes are explicitly limited."
        ),
    }
    return notes.get(intent, "")


# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #

def _call_llm_api(question: str, retrieved_clauses: List[Dict], contract_name: str) -> str:
    """
    Attempt to use the Gemini API to summarize and answer the question using the retrieved clauses.
    Times out after 8 seconds and falls back to template-based response on failure.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.info("GEMINI_API_KEY not set. Using template-based answer generation.")
        return _compose_answer(question, retrieved_clauses, contract_name)

    # Prepare context from retrieved clauses
    context_str = ""
    for c in retrieved_clauses:
        context_str += f"Section: {c.get('section', 'N/A')}\nText: {c.get('text', '')}\n\n"

    prompt = (
        f"You are a helpful legal assistant for ContractIQ. Answer the question about the contract '{contract_name}' "
        f"using only the following retrieved clauses from the contract. If the answer cannot be found in the context, "
        f"say so clearly.\n\n"
        f"Context:\n{context_str}\n"
        f"Question: {question}\n"
        f"Answer:"
    )

    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    # Standard Gemini v1beta API endpoint
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

    logger.info("Sending request to Gemini API (timeout=8s) for query: '%s'...", question)
    try:
        with httpx.Client(timeout=8.0) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            res_json = response.json()
            # Extract content from Gemini response structure
            answer = res_json['candidates'][0]['content']['parts'][0]['text']
            return answer.strip()
    except (httpx.TimeoutException, httpx.RequestError) as err:
        logger.error("Gemini API network call failed/timed out: %s. Falling back to template-based response.", err, exc_info=True)
        fallback_ans = _compose_answer(question, retrieved_clauses, contract_name)
        return (
            "[Note: The AI assistant is operating in degraded mode due to a temporary LLM API timeout/connection failure. "
            "Below is the direct contract clause extraction for your query.]\n\n" + fallback_ans
        )
    except Exception as err:
        logger.error("Gemini API error occurred: %s. Falling back to template-based response.", err, exc_info=True)
        fallback_ans = _compose_answer(question, retrieved_clauses, contract_name)
        return (
            "[Note: The AI assistant is operating in degraded mode due to a temporary LLM API error. "
            "Below is the direct contract clause extraction for your query.]\n\n" + fallback_ans
        )


# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #

def answer_question(
    question: str,
    contract_id: str,
    index: CUADIndex,
    contract_name: str = "this contract",
) -> Dict:
    """
    Main entry point for the Legal Assistant.

    Returns:
        {
          "content": str,
          "citations": [{"section": str, "text": str, "score": float}]
        }
    """
    # If ChromaDB collection is empty or uninitialized
    if index._collection is None:
        logger.warning("RAG Warning - ChromaDB collection is uninitialized.")
        return {
            "content": "No contract clauses have been indexed yet. Please upload a contract first before asking questions.",
            "citations": []
        }
        
    try:
        count = index._collection.count()
        if count == 0:
            logger.warning("RAG Warning - ChromaDB collection is empty.")
            return {
                "content": "No contract clauses have been indexed yet. Please upload a contract first before asking questions.",
                "citations": []
            }
    except Exception as exc:
        logger.error("RAG Failed - Failed to check collection count: %s", exc, exc_info=True)
        return {
            "content": "[Note: The AI assistant's search engine is temporarily unavailable due to a database error. Please try again later.]",
            "citations": []
        }

    # Retrieve relevant clauses — first try contract-specific, then CUAD-wide
    try:
        retrieved = index.query_rag(question, contract_id=contract_id, n=4)
        # If no contract-specific clauses found (e.g. mock contract), fall back
        # to searching the whole CUAD corpus
        if not retrieved:
            retrieved = index.query_rag(question, contract_id=None, n=4)
    except Exception as exc:
        logger.error("RAG Failed - Vector search query failed: %s", exc, exc_info=True)
        return {
            "content": "[Note: The AI assistant's search engine is temporarily unavailable due to a database query error. Please try again later.]",
            "citations": []
        }

    if not retrieved:
        answer = _compose_answer(question, retrieved, contract_name)
    else:
        answer = _call_llm_api(question, retrieved, contract_name)

    citations = [
        {
            "section": r.get("section", "§"),
            "text": r.get("text", "")[:300],
            "score": r.get("score", 0.0),
        }
        for r in retrieved
    ]

    return {"content": answer, "citations": citations}
