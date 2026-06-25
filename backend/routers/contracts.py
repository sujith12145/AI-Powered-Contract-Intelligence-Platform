"""
routers/contracts.py
====================
POST /api/contracts/analyze   – upload and analyze a contract file
GET  /api/contracts           – list all contracts (uploaded + mock fallback)
GET  /api/contracts/{id}      – get a single contract by ID
"""
from __future__ import annotations

import hashlib
import logging
import time
import uuid
from datetime import date, datetime
from typing import Dict, List, Optional
from pydantic import BaseModel

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from cuad_index import get_index
from models import (
    Clause,
    ComplianceEntry,
    Contract,
    ContractListItem,
    ContractSummary,
    DeadlineEntry,
    RiskLevel,
    TopRisk,
)
from services.classifier import (
    aggregate_contract_risk,
    analyse_clauses,
    build_compliance_entries,
    build_contract_summary,
)
from services.parser import extract_text, infer_contract_metadata, segment_clauses

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/contracts", tags=["contracts"])

# ------------------------------------------------------------------ #
# In-memory contract store                                             #
# (keyed by contract ID; survives for the lifetime of the process)    #
# ------------------------------------------------------------------ #
_CONTRACTS: Dict[str, Contract] = {}


def _store_contract(contract: Contract) -> None:
    _CONTRACTS[contract.id] = contract


def get_contract_store() -> Dict[str, Contract]:
    return _CONTRACTS


# ------------------------------------------------------------------ #
# Mock fallback contracts (from the original frontend mock data)       #
# Returned when no uploaded contracts exist yet.                       #
# ------------------------------------------------------------------ #

def _mock_contracts() -> List[ContractListItem]:
    return [
        ContractListItem(
            id="c1",
            name="Mutual Non-Disclosure Agreement",
            type="NDA",
            counterparty="Nexus Technologies Inc.",
            effectiveDate="2024-01-15",
            expiryDate="2026-01-15",
            value="N/A",
            status="active",
            overallRisk=RiskLevel.medium,
            riskScore=42,
            reviewedDate="2024-01-10",
            source="mock",
        ),
        ContractListItem(
            id="c2",
            name="Enterprise SaaS License Agreement",
            type="SaaS License",
            counterparty="CloudBase Solutions LLC",
            effectiveDate="2024-03-01",
            expiryDate="2027-03-01",
            value="$480,000/year",
            status="active",
            overallRisk=RiskLevel.high,
            riskScore=68,
            reviewedDate="2024-02-20",
            source="mock",
        ),
        ContractListItem(
            id="c3",
            name="IT Services Vendor Agreement",
            type="Vendor Agreement",
            counterparty="TechServ Global Partners",
            effectiveDate="2024-02-01",
            expiryDate="2025-02-01",
            value="$2.4M",
            status="active",
            overallRisk=RiskLevel.high,
            riskScore=63,
            reviewedDate="2024-01-25",
            source="mock",
        ),
        ContractListItem(
            id="c4",
            name="Senior Engineer Employment Agreement",
            type="Employment Contract",
            counterparty="Marcus J. Williams",
            effectiveDate="2024-04-01",
            expiryDate="2027-04-01",
            value="$320,000/year",
            status="active",
            overallRisk=RiskLevel.medium,
            riskScore=38,
            reviewedDate="2024-03-15",
            source="mock",
        ),
        ContractListItem(
            id="c5",
            name="Data Center Procurement Agreement",
            type="Procurement Contract",
            counterparty="Apex Infrastructure Holdings",
            effectiveDate="2024-06-01",
            expiryDate="2029-06-01",
            value="$8.5M",
            status="pending",
            overallRisk=RiskLevel.high,
            riskScore=71,
            reviewedDate="2024-05-20",
            source="mock",
        ),
    ]


# ------------------------------------------------------------------ #
# Routes                                                               #
# ------------------------------------------------------------------ #

@router.post("/analyze")
async def analyze_contract(file: UploadFile = File(...)):
    """
    Upload a PDF, DOCX, or TXT contract file.
    Returns a fully analyzed Contract object with CUAD-classified clauses.
    """
    t_start = time.time()
    allowed = {".pdf", ".docx", ".doc", ".txt"}
    filename = file.filename or "contract.pdf"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed)}",
        )

    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 50 MB).")

    # --- Parse text ---
    try:
        raw_text = extract_text(data, filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if len(raw_text.strip()) < 100:
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not extract enough text from the file. "
                "For scanned PDFs, OCR support requires Tesseract to be installed."
            ),
        )

    # --- Segment into clauses ---
    segments = segment_clauses(raw_text, max_clauses=30)
    if not segments:
        raise HTTPException(
            status_code=422,
            detail="Could not identify distinct clauses in this document.",
        )

    # --- Classify & score ---
    index = get_index()
    contract_id = "c_" + hashlib.md5(data[:512]).hexdigest()[:12]
    clauses = analyse_clauses(segments, contract_id, index)

    # --- Aggregate contract risk ---
    overall_score, overall_level = aggregate_contract_risk(clauses)

    # --- Extract metadata from text ---
    meta = infer_contract_metadata(raw_text, filename)
    today = date.today().isoformat()
    
    # Derive a readable contract name from filename
    name_without_ext = filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ")
    contract_name = name_without_ext[:80] if name_without_ext else "Uploaded Contract"

    # --- Build summary and compliance ---
    summary_dict = build_contract_summary(
        clauses, meta["type"], meta["counterparty"], overall_score, overall_level
    )
    compliance_list = build_compliance_entries(clauses)

    contract = Contract(
        id=contract_id,
        name=contract_name,
        type=meta["type"],
        counterparty=meta["counterparty"],
        effectiveDate=meta.get("effectiveDate") or today,
        expiryDate="",
        value=meta.get("value") or "N/A",
        status="active",
        overallRisk=overall_level,
        riskScore=overall_score,
        reviewedDate=today,
        clauses=clauses,
        summary=ContractSummary(
            overview=summary_dict["overview"],
            keyObligations=summary_dict["keyObligations"],
            deadlines=[DeadlineEntry(**d) for d in summary_dict["deadlines"]],
            financialCommitments=summary_dict["financialCommitments"],
            topRisks=[TopRisk(**r) for r in summary_dict["topRisks"]],
        ),
        compliance=[ComplianceEntry(**c) for c in compliance_list],
        version=1,
        source="uploaded",
    )

    # Persist contract and index its clauses for the Legal Assistant
    _store_contract(contract)
    index.index_contract_clauses(
        contract_id,
        [c.model_dump() for c in clauses],
    )

    ms = int((time.time() - t_start) * 1000)
    logger.info(
        "Analyzed '%s': %d clauses, risk=%s (%d) in %dms",
        filename,
        len(clauses),
        overall_level.value,
        overall_score,
        ms,
    )
    return {"contract": contract.model_dump(), "processingTimeMs": ms}


@router.get("")
async def list_contracts():
    """
    Return all contracts: uploaded ones first, then mock fallback contracts.
    """
    uploaded = list(_CONTRACTS.values())
    uploaded_ids = {c.id for c in uploaded}

    # Include mock contracts so existing frontend views keep working
    mocks = [m for m in _mock_contracts() if m.id not in uploaded_ids]

    result = [
        ContractListItem(
            id=c.id,
            name=c.name,
            type=c.type,
            counterparty=c.counterparty,
            effectiveDate=c.effectiveDate,
            expiryDate=c.expiryDate,
            value=c.value,
            status=c.status,
            overallRisk=c.overallRisk,
            riskScore=c.riskScore,
            reviewedDate=c.reviewedDate,
            source=c.source or "uploaded",
        )
        for c in uploaded
    ] + mocks

    return [r.model_dump() for r in result]


@router.get("/search")
async def search_contracts(q: str, limit: int = 20):
    """
    Search for clauses across all contracts semantically using ChromaDB.
    """
    if not q.strip():
        return []

    index = get_index()
    if index._collection is None:
        return []

    try:
        results = index._collection.query(
            query_texts=[q],
            n_results=min(limit, 50),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        logger.warning("Search query failed: %s", exc)
        return []

    out = []
    if not results or not results["documents"] or not results["documents"][0]:
        return out

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        contract_id = meta.get("contract_id", "")
        category = meta.get("category", "General")
        section = meta.get("section", "§")
        title = meta.get("title", category)
        risk_level = meta.get("risk_level", "medium")
        
        # Calculate similarity score from cosine distance
        score_val = round(1.0 - dist, 3)
        confidence = int(max(0.0, min(1.0, score_val)) * 100)

        # Look up additional contract details if present in store
        contract_name = meta.get("contract_name", "Uploaded Contract")
        counterparty = "N/A"
        risk_score = 30
        why_this_score = "Calculated by AI based on clause language."
        clause_id = f"cl_{hash(doc) & 0xffffffff}"

        if contract_id in _CONTRACTS:
            contract_obj = _CONTRACTS[contract_id]
            contract_name = contract_obj.name
            counterparty = contract_obj.counterparty
            # Find matching clause in contract_obj
            for clause_obj in contract_obj.clauses:
                if clause_obj.type == category or clause_obj.section == section:
                    clause_id = clause_obj.id
                    title = clause_obj.title
                    risk_level = clause_obj.riskLevel.value
                    risk_score = clause_obj.riskScore
                    why_this_score = clause_obj.whyThisScore or why_this_score
                    break

        out.append({
            "contractId": contract_id,
            "contractName": contract_name,
            "counterparty": counterparty,
            "clauseId": clause_id,
            "clauseTitle": title,
            "clauseSection": section,
            "clauseType": category,
            "riskLevel": risk_level,
            "riskScore": risk_score,
            "text": doc,
            "matchReason": f"Semantic Match ({confidence}% confidence)",
            "whyThisScore": why_this_score,
            "confidence": confidence,
        })

    return out


@router.get("/{contract_id}")
async def get_contract(contract_id: str):
    """
    Return a full contract by ID.
    Uploaded contracts come from the in-memory store;
    mock contract IDs (c1–c5) are served from the frontend's own mock data
    (no real data — the frontend handles those itself).
    """
    if contract_id in _CONTRACTS:
        return _CONTRACTS[contract_id].model_dump()

    # For mock IDs, return 404 — the frontend will fall back to its own mock data
    raise HTTPException(
        status_code=404,
        detail=f"Contract '{contract_id}' not found. Upload a real contract to analyze it.",
    )


class ClauseReviewRequest(BaseModel):
    action: str  # "confirm" | "override"
    reviewerName: str
    reviewerRole: str
    riskLevel: Optional[str] = None
    riskScore: Optional[int] = None
    category: Optional[str] = None
    reason: Optional[str] = None


@router.post("/{contract_id}/clauses/{clause_id}/review")
async def review_clause(contract_id: str, clause_id: str, req: ClauseReviewRequest):
    if contract_id not in _CONTRACTS:
        raise HTTPException(status_code=404, detail=f"Contract '{contract_id}' not found")

    contract = _CONTRACTS[contract_id]
    clause = next((c for c in contract.clauses if c.id == clause_id), None)
    if not clause:
        raise HTTPException(status_code=404, detail=f"Clause '{clause_id}' not found")

    original_cat = clause.type
    original_level = clause.riskLevel
    original_score = clause.riskScore

    # Use ISO format for timestamp
    timestamp = datetime.utcnow().isoformat() + "Z"

    from models import ReviewHistoryEntry, RiskLevel

    if req.action == "confirm":
        entry = ReviewHistoryEntry(
            reviewerName=req.reviewerName,
            reviewerRole=req.reviewerRole,
            timestamp=timestamp,
            action="confirm",
            originalCategory=original_cat,
            originalRiskLevel=original_level,
            originalRiskScore=original_score,
            finalCategory=original_cat,
            finalRiskLevel=original_level,
            finalRiskScore=original_score,
            reason=None
        )
        clause.status = "Confirmed"
        if not clause.reviewHistory:
            clause.reviewHistory = []
        clause.reviewHistory.append(entry)

    elif req.action == "override":
        if not req.reason:
            raise HTTPException(status_code=400, detail="Reason is required for overrides")
        if not req.riskLevel:
            raise HTTPException(status_code=400, detail="riskLevel is required for overrides")

        try:
            final_level = RiskLevel(req.riskLevel.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid riskLevel: '{req.riskLevel}'")

        final_score = req.riskScore
        if final_score is None:
            score_map = {
                RiskLevel.critical: 85,
                RiskLevel.high: 65,
                RiskLevel.medium: 45,
                RiskLevel.low: 15
            }
            final_score = score_map.get(final_level, 15)

        final_cat = req.category or original_cat

        entry = ReviewHistoryEntry(
            reviewerName=req.reviewerName,
            reviewerRole=req.reviewerRole,
            timestamp=timestamp,
            action="override",
            originalCategory=original_cat,
            originalRiskLevel=original_level,
            originalRiskScore=original_score,
            finalCategory=final_cat,
            finalRiskLevel=final_level,
            finalRiskScore=final_score,
            reason=req.reason
        )

        clause.status = "Overridden"
        clause.type = final_cat
        clause.riskLevel = final_level
        clause.riskScore = final_score

        if not clause.reviewHistory:
            clause.reviewHistory = []
        clause.reviewHistory.append(entry)

    else:
        raise HTTPException(status_code=400, detail="Action must be 'confirm' or 'override'")

    # Recalculate contract overall risk
    overall_score, overall_level = aggregate_contract_risk(contract.clauses)
    contract.riskScore = overall_score
    contract.overallRisk = overall_level

    _CONTRACTS[contract_id] = contract

    return contract.model_dump()
