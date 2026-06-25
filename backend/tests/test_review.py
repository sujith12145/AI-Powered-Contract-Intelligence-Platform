import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from main import app
from routers.contracts import _CONTRACTS
from models import Contract, Clause, ContractSummary, RiskLevel

client = TestClient(app)

@pytest.fixture
def mock_contract():
    contract_id = "test_contract_123"
    clause_id = "test_clause_456"
    
    # Setup in-memory mock contract
    clause = Clause(
        id=clause_id,
        type="Liability",
        title="Limitation of Liability",
        text="In no event shall vendor be liable...",
        riskLevel=RiskLevel.critical,
        riskScore=90,
        deviation="Uncapped liability deviation.",
        suggestedText="Mutual liability cap equal to 12 months fees.",
        explanation="Uncapped liability exposes the company to unlimited damages.",
        section="Section 11.2",
        cuadCategory="Uncapped Liability",
        confidence=95.0,
        status="AI-Suggested",
        reviewHistory=[]
    )
    
    contract = Contract(
        id=contract_id,
        name="Test Procurement Agreement",
        type="Procurement",
        counterparty="Vendor Corp",
        effectiveDate="2026-01-01",
        expiryDate="2027-01-01",
        value="$100,000",
        status="active",
        overallRisk=RiskLevel.critical,
        riskScore=90,
        reviewedDate="2026-01-01",
        clauses=[clause],
        summary=ContractSummary(
            overview="A test contract.",
            keyObligations=["Review liability terms."],
            deadlines=[],
            financialCommitments=[],
            topRisks=[]
        ),
        compliance=[],
        version=1,
        source="uploaded"
    )
    
    _CONTRACTS[contract_id] = contract
    yield contract_id, clause_id
    # Cleanup after test
    _CONTRACTS.pop(contract_id, None)


def test_confirm_clause(mock_contract):
    contract_id, clause_id = mock_contract
    
    # Confirm the clause
    payload = {
        "action": "confirm",
        "reviewerName": "Sarah Mitchell",
        "reviewerRole": "Senior Legal Counsel"
    }
    
    response = client.post(f"/api/contracts/{contract_id}/clauses/{clause_id}/review", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["id"] == contract_id
    
    # Check that the clause is updated to Confirmed
    clause = data["clauses"][0]
    assert clause["status"] == "Confirmed"
    assert len(clause["reviewHistory"]) == 1
    
    history_entry = clause["reviewHistory"][0]
    assert history_entry["action"] == "confirm"
    assert history_entry["reviewerName"] == "Sarah Mitchell"
    assert history_entry["reviewerRole"] == "Senior Legal Counsel"
    assert history_entry["originalRiskLevel"] == "critical"
    assert history_entry["finalRiskLevel"] == "critical"


def test_override_clause(mock_contract):
    contract_id, clause_id = mock_contract
    
    # Override the clause
    payload = {
        "action": "override",
        "reviewerName": "Sarah Mitchell",
        "reviewerRole": "Senior Legal Counsel",
        "riskLevel": "medium",
        "riskScore": 45,
        "category": "Liability",
        "reason": "Standard mutual cap is acceptable."
    }
    
    response = client.post(f"/api/contracts/{contract_id}/clauses/{clause_id}/review", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["id"] == contract_id
    
    # Check that the clause is updated to Overridden with new risk
    clause = data["clauses"][0]
    assert clause["status"] == "Overridden"
    assert clause["riskLevel"] == "medium"
    assert clause["riskScore"] == 45
    assert len(clause["reviewHistory"]) == 1
    
    history_entry = clause["reviewHistory"][0]
    assert history_entry["action"] == "override"
    assert history_entry["reason"] == "Standard mutual cap is acceptable."
    assert history_entry["originalRiskLevel"] == "critical"
    assert history_entry["finalRiskLevel"] == "medium"
    assert history_entry["finalRiskScore"] == 45


def test_override_requires_reason(mock_contract):
    contract_id, clause_id = mock_contract
    
    # Attempt override without reason
    payload = {
        "action": "override",
        "reviewerName": "Sarah Mitchell",
        "reviewerRole": "Senior Legal Counsel",
        "riskLevel": "medium"
    }
    
    response = client.post(f"/api/contracts/{contract_id}/clauses/{clause_id}/review", json=payload)
    assert response.status_code == 400
    assert "reason" in response.json()["detail"].lower()
