"""
routers/assistant.py
====================
POST /api/assistant/ask   – answer a question about a contract using RAG
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from cuad_index import get_index
from models import AskRequest, AskResponse, Citation
from routers.contracts import get_contract_store
from services.rag import answer_question

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/assistant", tags=["assistant"])


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """
    Answer a question about a specific contract using the RAG pipeline.

    The assistant:
    1. Retrieves the most relevant clause chunks from ChromaDB
       (filtered by contract_id if it's an uploaded contract)
    2. Returns a structured answer with clickable citation badges
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    index = get_index()
    store = get_contract_store()

    # Determine contract name for answer preamble
    contract_name = "this contract"
    if req.contractId in store:
        contract_name = store[req.contractId].name

    # For mock contract IDs (c1–c5), query the full CUAD corpus
    # (no uploaded clauses indexed for them)
    effective_id = req.contractId if req.contractId in store else None

    result = answer_question(
        question=req.question.strip(),
        contract_id=effective_id,
        index=index,
        contract_name=contract_name,
    )

    return AskResponse(
        content=result["content"],
        citations=[
            Citation(
                section=c["section"],
                text=c["text"],
                score=c.get("score"),
            )
            for c in result["citations"]
        ],
    )
