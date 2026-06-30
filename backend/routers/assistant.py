"""
routers/assistant.py
====================
POST /api/v1/assistant/ask  — answer a question about a contract using RAG

Security controls:
  ✅ JWT authentication required
  ✅ Rate limited: 30 requests/minute per user
  ✅ Input validation (question length)
  ✅ Ownership check for contract access
"""


import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

import cuad_index as cuad_idx
from auth.dependencies import TokenPayload, get_current_user
from database.connection import get_db
from database.repositories import ContractRepository
from middleware.rate_limiter import limiter
from models import AskRequest, AskResponse, Citation
from services.rag import answer_question

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/assistant", tags=["assistant"])


@router.post("/ask", response_model=AskResponse)
@limiter.limit("30/minute")
async def ask(
    request: Request,
    req: AskRequest,
    current_user: TokenPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Answer a question about a specific contract using the RAG pipeline.

    1. Retrieves the most relevant clause chunks from ChromaDB
       (filtered by contract_id if it's an uploaded contract)
    2. Performs ownership check for non-admin users
    3. Returns a structured answer with clickable citation badges
    """
    question = req.question.strip()
    logger.info("Received Legal Assistant query: '%s' (contract_id=%s) from user: %s", question, req.contractId, current_user.email)
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(question) > 2000:
        raise HTTPException(status_code=400, detail="Question too long (max 2000 chars).")

    index = cuad_idx.get_index()

    # Look up contract name + ownership check
    contract_name = "this contract"
    effective_id = None

    if req.contractId:
        logger.info("Performing ownership check for contract ID: %s", req.contractId)
        contract_repo = ContractRepository(db)
        db_contract = await contract_repo.get(req.contractId)

        if db_contract:
            # Ownership check: non-admins only access their own contracts
            if (
                current_user.role != "admin"
                and db_contract.owner_id != current_user.user_id
            ):
                raise HTTPException(status_code=403, detail="Access denied to this contract.")
            contract_name = db_contract.name
            effective_id = req.contractId

    result = answer_question(
        question=question,
        contract_id=effective_id,
        index=index,
        contract_name=contract_name,
    )
    logger.info("RAG search returned %d citations for query: '%s'", len(result.get("citations", [])), question)

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
