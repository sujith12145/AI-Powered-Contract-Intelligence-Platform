"""
routers/contracts.py
====================
Contract management endpoints with full security controls:

  POST /api/v1/contracts/analyze              — upload & analyze (Legal Counsel+)
  GET  /api/v1/contracts                      — list (All authenticated)
  GET  /api/v1/contracts/search               — semantic search (All)
  GET  /api/v1/contracts/{id}                 — get contract (All, ownership check)
  POST /api/v1/contracts/{id}/clauses/{cid}/review — review clause (Legal Counsel+)
  DELETE /api/v1/contracts/{id}               — delete (Admin)

Security controls applied:
  ✅ JWT authentication (all routes)
  ✅ RBAC — Viewer=read-only, LegalCounsel=write, Admin=full
  ✅ Ownership check — non-admins see only their own contracts
  ✅ Rate limiting on analyze endpoint
  ✅ File validation (extension + size)
  ✅ Filename sanitisation (UUID rename)
  ✅ Audit log for all mutations
  ✅ PostgreSQL persistence (replaces in-memory dict)
"""


import hashlib
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

import cuad_index as cuad_idx
from auth.dependencies import CurrentUser, TokenPayload, get_current_user, require_role
from config import get_settings
from database.connection import get_db
from database.repositories import AuditRepository, ContractRepository
from middleware.rate_limiter import limiter
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
settings = get_settings()

router = APIRouter(prefix="/api/v1/contracts", tags=["contracts"])

# Allowed upload extensions
_ALLOWED_EXTS = {".pdf", ".docx", ".doc", ".txt"}
_MAX_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024

# Wall-clock timeout for text extraction (pdfminer / docx can be CPU-heavy).
# Patchable at module level for tests.
_EXTRACT_TIMEOUT_SECS: int = 15


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _sanitise_filename(original: str) -> str:
    """Return a UUID-based safe filename preserving the extension."""
    ext = ""
    if "." in original:
        ext = "." + original.rsplit(".", 1)[-1].lower()
    return str(uuid.uuid4()) + ext


def _ensure_upload_dir() -> Path:
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


# ------------------------------------------------------------------ #
# Mock fallback contracts (for demo — shown only to Viewer role)      #
# ------------------------------------------------------------------ #

def _mock_contracts() -> List[ContractListItem]:
    return [
        ContractListItem(
            id="c1", name="Mutual Non-Disclosure Agreement", type="NDA",
            counterparty="Nexus Technologies Inc.", effectiveDate="2024-01-15",
            expiryDate="2026-01-15", value="N/A", status="active",
            overallRisk=RiskLevel.medium, riskScore=42, reviewedDate="2024-01-10", source="mock",
        ),
        ContractListItem(
            id="c2", name="Enterprise SaaS License Agreement", type="SaaS License",
            counterparty="CloudBase Solutions LLC", effectiveDate="2024-03-01",
            expiryDate="2027-03-01", value="$480,000/year", status="active",
            overallRisk=RiskLevel.high, riskScore=68, reviewedDate="2024-02-20", source="mock",
        ),
        ContractListItem(
            id="c3", name="IT Services Vendor Agreement", type="Vendor Agreement",
            counterparty="TechServ Global Partners", effectiveDate="2024-02-01",
            expiryDate="2025-02-01", value="$2.4M", status="active",
            overallRisk=RiskLevel.high, riskScore=63, reviewedDate="2024-01-25", source="mock",
        ),
        ContractListItem(
            id="c4", name="Senior Engineer Employment Agreement", type="Employment Contract",
            counterparty="Marcus J. Williams", effectiveDate="2024-04-01",
            expiryDate="2027-04-01", value="$320,000/year", status="active",
            overallRisk=RiskLevel.medium, riskScore=38, reviewedDate="2024-03-15", source="mock",
        ),
        ContractListItem(
            id="c5", name="Data Center Procurement Agreement", type="Procurement Contract",
            counterparty="Apex Infrastructure Holdings", effectiveDate="2024-06-01",
            expiryDate="2029-06-01", value="$8.5M", status="pending",
            overallRisk=RiskLevel.high, riskScore=71, reviewedDate="2024-05-20", source="mock",
        ),
    ]


# ------------------------------------------------------------------ #
# Routes                                                               #
# ------------------------------------------------------------------ #

@router.post("/analyze")
@limiter.limit("20/hour")
async def analyze_contract(
    request: Request,
    file: UploadFile = File(...),
    current_user: TokenPayload = Depends(require_role("legal_counsel")),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a PDF, DOCX, or TXT contract file. Requires Legal Counsel or Admin role.
    Returns a fully analyzed Contract with CUAD-classified clauses.
    """
    t_start = time.time()
    filename = file.filename or "contract.pdf"
    logger.info("Pipeline Start - Received contract upload request: '%s' (%d bytes) from user: %s", filename, file.size or -1, current_user.email)

    # ---- Validate file ----
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if ext not in _ALLOWED_EXTS:
        logger.warning("Pipeline Failed - Upload validation: unsupported file type '%s' for filename: %s", ext, filename)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(_ALLOWED_EXTS)}",
        )

    data = await file.read()
    if len(data) == 0:
        logger.warning("Pipeline Failed - Upload validation: file %s is empty", filename)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > _MAX_BYTES:
        logger.warning("Pipeline Failed - Upload validation: file %s is too large (%d bytes)", filename, len(data))
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max {settings.MAX_FILE_SIZE_MB} MB.",
        )

    # Verify magic bytes
    if ext == ".pdf" and not data.startswith(b"%PDF"):
        logger.warning("Pipeline Failed - Upload validation: magic bytes mismatch for PDF file %s", filename)
        raise HTTPException(status_code=400, detail="Invalid PDF file (magic bytes mismatch).")
    if ext == ".docx" and not data.startswith(b"PK\x03\x04"):
        logger.warning("Pipeline Failed - Upload validation: magic bytes mismatch for DOCX file %s", filename)
        raise HTTPException(status_code=400, detail="Invalid DOCX file (magic bytes mismatch).")

    # Limit checks (T10)
    if ext == ".pdf":
        from pdfminer.pdfpage import PDFPage
        import io
        try:
            pages = list(PDFPage.get_pages(io.BytesIO(data)))
            if len(pages) > 50:
                logger.warning("Pipeline Failed - Upload validation: PDF file %s exceeds page limit (%d pages)", filename, len(pages))
                raise HTTPException(status_code=400, detail="PDF exceeds maximum page limit of 50 pages.")
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("Failed to count PDF pages: %s", exc)
            raise HTTPException(status_code=400, detail="Corrupted or invalid PDF structure.")

    if ext in (".docx", ".doc"):
        import io
        try:
            from docx import Document
            doc = Document(io.BytesIO(data))
            if len(doc.paragraphs) > 1000:
                logger.warning("Pipeline Failed - Upload validation: DOCX file %s exceeds paragraph limit (%d paragraphs)", filename, len(doc.paragraphs))
                raise HTTPException(status_code=400, detail="DOCX/DOC exceeds maximum paragraph limit of 1000 paragraphs.")
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("Failed to count DOCX paragraphs: %s", exc)
            raise HTTPException(status_code=400, detail="Corrupted or invalid DOCX structure.")

    # Calculate SHA-256 hash of the uploaded document bytes (T3)
    file_hash = hashlib.sha256(data).hexdigest()

    # ---- Save file securely (UUID rename) ----
    safe_name = _sanitise_filename(filename)
    upload_dir = _ensure_upload_dir()
    file_path = upload_dir / safe_name
    try:
        file_path.write_bytes(data)
        logger.info("Pipeline Step 1/5 - Saved raw file securely to disk: %s", file_path)
    except IOError as e:
        logger.error("Pipeline Failed - Could not save uploaded file %s to disk: %s", filename, e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to save uploaded contract file to server storage."
        )

    # ---- Parse text (with wall-clock timeout to prevent CPU lock-up) ----
    logger.info("Pipeline Step 2/5 - Extraction starting for %s...", filename)
    try:
        with ThreadPoolExecutor(max_workers=1) as _pool:
            future = _pool.submit(extract_text, data, filename)
            try:
                raw_text = future.result(timeout=_EXTRACT_TIMEOUT_SECS)
            except FuturesTimeoutError:
                logger.error(
                    "Pipeline Failed - Extraction timed out after %ds for %s",
                    _EXTRACT_TIMEOUT_SECS, filename,
                )
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Document too complex to process: text extraction exceeded "
                        f"{_EXTRACT_TIMEOUT_SECS}s. Try a smaller or less complex file."
                    ),
                )
        logger.info(
            "Pipeline Step 2/5 - Extraction successful: extracted %d characters from %s",
            len(raw_text), filename,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Pipeline Failed - Extraction stage failed for %s: %s", filename, exc, exc_info=True)
        raise HTTPException(status_code=422, detail=str(exc))

    if len(raw_text.strip()) < 100:
        logger.warning("Pipeline Failed - Extraction stage failed: extracted text too short (%d chars) for contract %s", len(raw_text), filename)
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not extract enough text from the file. "
                "For scanned PDFs, OCR support requires Tesseract to be installed."
            ),
        )

    # ---- Segment clauses ----
    logger.info("Pipeline Step 3/5 - Segmentation starting for %s...", filename)
    try:
        segments = segment_clauses(raw_text, max_clauses=30)
        logger.info("Pipeline Step 3/5 - Segmentation successful: identified %d clause segments in %s", len(segments), filename)
    except Exception as exc:
        logger.error("Pipeline Failed - Segmentation stage failed for %s: %s", filename, exc, exc_info=True)
        raise HTTPException(status_code=422, detail="Failed to segment contract text into clauses.")

    if not segments:
        logger.error("Pipeline Failed - Segmentation stage failed: could not identify distinct clauses in contract: %s", filename)
        raise HTTPException(status_code=422, detail="Could not identify distinct clauses.")

    # ---- Classify & score ----
    logger.info("Pipeline Step 4/5 - Classification starting on %d segments for %s...", len(segments), filename)
    try:
        index = cuad_idx.get_index()
        contract_id = "c_" + hashlib.md5(data[:512]).hexdigest()[:12]
        clauses = analyse_clauses(segments, contract_id, index)
        logger.info("Pipeline Step 4/5 - Classification successful: classified %d clauses", len(clauses))
    except Exception as exc:
        logger.error("Pipeline Failed - Classification stage failed for %s: %s", filename, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Clause classification failed.")

    # ---- Aggregate risk ----
    logger.info("Pipeline Step 5/5 - Risk scoring starting for %s...", filename)
    try:
        overall_score, overall_level = aggregate_contract_risk(clauses)
        logger.info("Pipeline Step 5/5 - Risk scoring successful: score=%d, level=%s", overall_score, overall_level.value)
    except Exception as exc:
        logger.error("Pipeline Failed - Risk scoring stage failed for %s: %s", filename, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Risk scoring aggregation failed.")

    # ---- Metadata ----
    meta = infer_contract_metadata(raw_text, filename)
    today = date.today().isoformat()
    name_without_ext = filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ")
    contract_name = name_without_ext[:80] if name_without_ext else "Uploaded Contract"

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

    # ---- Persist to DB ----
    try:
        contract_repo = ContractRepository(db)
        await contract_repo.upsert(
            contract.model_dump(),
            owner_id=current_user.user_id,
            file_path=str(file_path),
            file_hash=file_hash,
        )
        logger.info("Pipeline Step 5/5 - Contract successfully persisted to database.")
    except Exception as exc:
        logger.error("Pipeline Failed - Database persistence failed for %s: %s", filename, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to save analyzed contract to database.")

    # ---- Index clauses for RAG ----
    try:
        index.index_contract_clauses(contract_id, [c.model_dump() for c in clauses])
        logger.info("Pipeline Step 5/5 - Contract clauses indexed in vector store.")
    except Exception as exc:
        logger.error("Pipeline Failed - Vector store indexing failed for %s: %s", filename, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to index contract clauses in vector store.")

    # ---- Audit log ----
    audit = AuditRepository(db)
    await audit.log(
        action="contract_analyzed",
        user_id=current_user.user_id,
        user_email=current_user.email,
        user_role=current_user.role,
        resource_type="contract",
        resource_id=contract_id,
        extra_data={
            "filename": filename,
            "clauses": len(clauses),
            "risk_score": overall_score,
            "risk_level": overall_level.value,
        },
        response_status=200,
    )

    ms = int((time.time() - t_start) * 1000)
    logger.info(
        "Pipeline Complete - Analyzed '%s': %d clauses, risk=%s (%d) in %dms",
        filename, len(clauses), overall_level.value, overall_score, ms,
    )
    return {"contract": contract.model_dump(), "processingTimeMs": ms}


@router.get("")
async def list_contracts(
    current_user: TokenPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List contracts. Admins see all; others see only their own.
    Mock contracts are always appended as demo data.
    """
    contract_repo = ContractRepository(db)

    # Admins see everything; others see their own
    owner_filter = None if current_user.role == "admin" else current_user.user_id
    db_contracts = await contract_repo.list_all(owner_id=owner_filter)

    uploaded_ids = {c.id for c in db_contracts}

    result = [
        ContractListItem(
            id=c.id,
            name=c.name,
            type=c.contract_type or "Unknown",
            counterparty=c.counterparty or "N/A",
            effectiveDate=c.effective_date or "",
            expiryDate=c.expiry_date or "",
            value=c.value or "N/A",
            status=c.status or "active",
            overallRisk=RiskLevel(c.overall_risk) if c.overall_risk else RiskLevel.low,
            riskScore=c.risk_score or 0,
            reviewedDate=c.reviewed_date or "",
            source=c.source or "uploaded",
        ).model_dump()
        for c in db_contracts
    ] + [m.model_dump() for m in _mock_contracts() if m.id not in uploaded_ids]

    return result


@router.get("/search")
async def search_contracts(
    q: str = Query(..., min_length=1, max_length=500),
    limit: int = Query(20, ge=1, le=50),
    current_user: TokenPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Semantic clause search across all indexed contracts."""
    allowed_contract_ids = set()
    if current_user.role != "admin":
        contract_repo = ContractRepository(db)
        user_contracts = await contract_repo.list_all(owner_id=current_user.user_id)
        allowed_contract_ids = {c.id for c in user_contracts}
    index = cuad_idx.get_index()
    if index._collection is None:
        logger.error("Search Failed - ChromaDB collection is uninitialized.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The semantic search index is not available. Please ensure the CUAD index is loaded."
        )

    try:
        # Check if collection is empty
        count = index._collection.count()
        if count == 0:
            logger.warning("Search warning - ChromaDB collection is empty.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No contract clauses have been indexed yet. Please upload a contract first before searching."
            )
            
        results = index._collection.query(
            query_texts=[q],
            n_results=min(limit, 50),
            include=["documents", "metadatas", "distances"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Search Failed - ChromaDB search query failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search query failed due to a database error."
        )

    out = []
    if not results or not results["documents"] or not results["documents"][0]:
        return out

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        contract_id = meta.get("contract_id", "")
        if (
            current_user.role != "admin"
            and not contract_id.startswith("cuad_")
            and contract_id not in allowed_contract_ids
        ):
            continue
        category = meta.get("category", "General")
        section = meta.get("section", "§")
        title = meta.get("title", category)
        risk_level = meta.get("risk_level", "medium")
        score_val = round(1.0 - dist, 3)
        confidence = int(max(0.0, min(1.0, score_val)) * 100)

        out.append({
            "contractId": contract_id,
            "contractName": meta.get("contract_name", "Uploaded Contract"),
            "counterparty": "N/A",
            "clauseId": f"cl_{hash(doc) & 0xffffffff}",
            "clauseTitle": title,
            "clauseSection": section,
            "clauseType": category,
            "riskLevel": risk_level,
            "riskScore": 30,
            "text": doc,
            "matchReason": f"Semantic Match ({confidence}% confidence)",
            "confidence": confidence,
        })

    return out


@router.get("/{contract_id}/download")
async def download_contract(
    contract_id: str,
    current_user: TokenPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Download the contract file. Enforces ownership authorization checks (T7 BOLA).
    Verifies SHA-256 file integrity on disk (T3).
    """
    contract_repo = ContractRepository(db)
    db_contract = await contract_repo.get(contract_id)
    if not db_contract:
        raise HTTPException(status_code=404, detail="Contract not found")

    # Ownership check / BOLA (T7)
    if (
        current_user.role != "admin"
        and db_contract.owner_id != current_user.user_id
    ):
        raise HTTPException(status_code=403, detail="Access denied.")

    if not db_contract.file_path or not os.path.exists(db_contract.file_path):
        raise HTTPException(status_code=404, detail="Contract file not found on disk")

    # Read and verify hash (T3)
    try:
        with open(db_contract.file_path, "rb") as f:
            file_bytes = f.read()
    except IOError as e:
        logger.error("Failed to read contract file %s: %s", db_contract.file_path, e)
        raise HTTPException(status_code=500, detail="Failed to read contract file from storage.")

    calculated_hash = hashlib.sha256(file_bytes).hexdigest()
    if calculated_hash != db_contract.file_hash:
        logger.critical(
            "Security Alert: Contract %s file integrity verification failed! Expected %s, got %s",
            contract_id, db_contract.file_hash, calculated_hash
        )
        raise HTTPException(
            status_code=409,
            detail="Contract file integrity check failed (tampering detected)."
        )

    return FileResponse(
        db_contract.file_path,
        filename=db_contract.name,
        media_type="application/octet-stream"
    )


@router.get("/{contract_id}")
async def get_contract(
    contract_id: str,
    current_user: TokenPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Return a full contract by ID.
    Ownership check: non-admins can only access their own contracts.
    """
    contract_repo = ContractRepository(db)
    db_contract = await contract_repo.get(contract_id)

    if db_contract:
        # Ownership check
        if (
            current_user.role != "admin"
            and db_contract.owner_id != current_user.user_id
        ):
            raise HTTPException(status_code=403, detail="Access denied.")
        return db_contract.raw_data

    # For mock IDs, return 404 — frontend falls back to mock data
    raise HTTPException(
        status_code=404,
        detail=f"Contract '{contract_id}' not found.",
    )


# ------------------------------------------------------------------ #
# Clause Review                                                        #
# ------------------------------------------------------------------ #

class ClauseReviewRequest(BaseModel):
    action: str          # "confirm" | "override"
    reviewerName: str
    reviewerRole: str
    riskLevel: Optional[str] = None
    riskScore: Optional[int] = None
    category: Optional[str] = None
    reason: Optional[str] = None


@router.post("/{contract_id}/clauses/{clause_id}/review")
async def review_clause(
    contract_id: str,
    clause_id: str,
    req: ClauseReviewRequest,
    current_user: TokenPayload = Depends(require_role("legal_counsel")),
    db: AsyncSession = Depends(get_db),
):
    """
    Confirm or override a clause risk assessment. Legal Counsel or Admin only.
    """
    contract_repo = ContractRepository(db)
    db_contract = await contract_repo.get(contract_id)

    if not db_contract:
        raise HTTPException(status_code=404, detail=f"Contract '{contract_id}' not found")

    # Ownership check
    if current_user.role != "admin" and db_contract.owner_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Access denied.")

    # Load full contract data
    contract_data = db_contract.raw_data
    clauses = contract_data.get("clauses", [])
    clause = next((c for c in clauses if c["id"] == clause_id), None)

    if not clause:
        raise HTTPException(status_code=404, detail=f"Clause '{clause_id}' not found")

    original_cat = clause["type"]
    original_level = clause["riskLevel"]
    original_score = clause["riskScore"]
    timestamp = datetime.utcnow().isoformat() + "Z"

    if req.action == "confirm":
        entry = {
            "reviewerName": req.reviewerName,
            "reviewerRole": req.reviewerRole,
            "timestamp": timestamp,
            "action": "confirm",
            "originalCategory": original_cat,
            "originalRiskLevel": original_level,
            "originalRiskScore": original_score,
            "finalCategory": original_cat,
            "finalRiskLevel": original_level,
            "finalRiskScore": original_score,
            "reason": None,
        }
        clause["status"] = "Confirmed"
        if not clause.get("reviewHistory"):
            clause["reviewHistory"] = []
        clause["reviewHistory"].append(entry)

    elif req.action == "override":
        if not req.reason:
            raise HTTPException(status_code=400, detail="Reason is required for overrides.")
        if not req.riskLevel:
            raise HTTPException(status_code=400, detail="riskLevel is required for overrides.")

        valid_levels = {r.value for r in RiskLevel}
        if req.riskLevel.lower() not in valid_levels:
            raise HTTPException(status_code=400, detail=f"Invalid riskLevel: '{req.riskLevel}'")

        final_level = req.riskLevel.lower()
        final_score = req.riskScore or {"critical": 85, "high": 65, "medium": 45, "low": 15}.get(final_level, 15)
        final_cat = req.category or original_cat

        entry = {
            "reviewerName": req.reviewerName,
            "reviewerRole": req.reviewerRole,
            "timestamp": timestamp,
            "action": "override",
            "originalCategory": original_cat,
            "originalRiskLevel": original_level,
            "originalRiskScore": original_score,
            "finalCategory": final_cat,
            "finalRiskLevel": final_level,
            "finalRiskScore": final_score,
            "reason": req.reason,
        }
        clause["status"] = "Overridden"
        clause["type"] = final_cat
        clause["riskLevel"] = final_level
        clause["riskScore"] = final_score
        if not clause.get("reviewHistory"):
            clause["reviewHistory"] = []
        clause["reviewHistory"].append(entry)

    else:
        raise HTTPException(status_code=400, detail="Action must be 'confirm' or 'override'")

    # Recalculate contract risk
    from models import Clause as ClauseModel
    clause_objs = [ClauseModel(**c) for c in clauses]
    overall_score, overall_level = aggregate_contract_risk(clause_objs)
    contract_data["riskScore"] = overall_score
    contract_data["overallRisk"] = overall_level.value

    # Persist updated contract
    await contract_repo.upsert(contract_data, owner_id=db_contract.owner_id)

    # Audit log
    audit = AuditRepository(db)
    await audit.log(
        action=f"clause_{req.action}",
        user_id=current_user.user_id,
        user_email=current_user.email,
        user_role=current_user.role,
        resource_type="clause",
        resource_id=clause_id,
        extra_data={
            "contract_id": contract_id,
            "action": req.action,
            "reason": req.reason,
            "new_risk_level": clause.get("riskLevel"),
        },
        response_status=200,
    )

    return contract_data


@router.delete("/{contract_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_contract(
    contract_id: str,
    current_user: TokenPayload = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db),
):
    """Delete a contract (Admin only)."""
    contract_repo = ContractRepository(db)
    deleted = await contract_repo.delete(contract_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Contract not found.")

    audit = AuditRepository(db)
    await audit.log(
        action="contract_deleted",
        user_id=current_user.user_id,
        user_email=current_user.email,
        user_role=current_user.role,
        resource_type="contract",
        resource_id=contract_id,
        response_status=204,
    )
