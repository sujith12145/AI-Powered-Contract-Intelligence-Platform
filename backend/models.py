"""
Pydantic models that exactly mirror the TypeScript types in src/types/index.ts.
All API responses use these models so the frontend needs zero JSX changes.
"""
from __future__ import annotations
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel


class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class ComplianceStatus(str, Enum):
    pass_ = "pass"
    fail = "fail"
    review = "review"


class ReviewHistoryEntry(BaseModel):
    reviewerName: str
    reviewerRole: str
    timestamp: str
    action: str  # "confirm" | "override"
    originalCategory: str
    originalRiskLevel: RiskLevel
    originalRiskScore: int
    finalCategory: str
    finalRiskLevel: RiskLevel
    finalRiskScore: int
    reason: Optional[str] = None


class Clause(BaseModel):
    id: str
    type: str           # CUAD category mapped to friendly type name
    title: str
    text: str
    riskLevel: RiskLevel
    riskScore: int      # 0-100
    deviation: str
    suggestedText: str
    explanation: str
    section: str
    whyThisScore: Optional[str] = None
    # Extra fields the backend adds (ignored by frontend)
    cuadCategory: Optional[str] = None
    confidence: Optional[float] = None
    status: Optional[str] = "AI-Suggested"
    reviewHistory: Optional[List[ReviewHistoryEntry]] = []


class DeadlineEntry(BaseModel):
    date: str
    description: str


class TopRisk(BaseModel):
    rank: int
    title: str
    description: str
    severity: RiskLevel


class ContractSummary(BaseModel):
    overview: str
    keyObligations: List[str]
    deadlines: List[DeadlineEntry]
    financialCommitments: List[str]
    topRisks: List[TopRisk]


class ComplianceEntry(BaseModel):
    regulation: str
    status: str          # "pass" | "fail" | "review"
    details: str
    checkedDate: str


class Contract(BaseModel):
    id: str
    name: str
    type: str
    counterparty: str
    effectiveDate: str
    expiryDate: str
    value: str
    status: str          # "active" | "expired" | "pending" | "draft"
    overallRisk: RiskLevel
    riskScore: int
    reviewedDate: str
    clauses: List[Clause]
    summary: ContractSummary
    compliance: List[ComplianceEntry]
    version: int
    # Backend-only metadata
    source: Optional[str] = "uploaded"


class AnalyzeResponse(BaseModel):
    contract: Contract
    processingTimeMs: int


class AskRequest(BaseModel):
    contractId: str
    question: str


class Citation(BaseModel):
    section: str
    text: str
    score: Optional[float] = None


class AskResponse(BaseModel):
    content: str
    citations: List[Citation]


class ContractListItem(BaseModel):
    id: str
    name: str
    type: str
    counterparty: str
    effectiveDate: str
    expiryDate: str
    value: str
    status: str
    overallRisk: RiskLevel
    riskScore: int
    reviewedDate: str
    source: Optional[str] = "uploaded"
