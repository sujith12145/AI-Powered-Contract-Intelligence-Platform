import os
import sys
import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from main import app
from database.connection import Base, get_db
from database.repositories import ContractRepository
from auth.dependencies import get_current_user, TokenPayload
from models import Contract, Clause, ContractSummary, RiskLevel

# Create in-memory SQLite engine for tests
engine = create_async_engine("sqlite+aiosqlite:///:memory:", connect_args={"check_same_thread": False})
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def override_get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# Mock TokenPayload
mock_user = TokenPayload(user_id="test_user_id_123", email="counsel@contractiq.local", role="legal_counsel")

async def override_get_current_user(request: Request):
    request.state.user_id = mock_user.user_id
    request.state.user_email = mock_user.email
    request.state.user_role = mock_user.role
    return mock_user

app.dependency_overrides[get_db] = override_get_db
app.dependency_overrides[get_current_user] = override_get_current_user

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_database():
    """Synchronous database table creation and cleanup for tests."""
    import asyncio
    
    async def create_tables():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            
    async def drop_tables():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            
    asyncio.run(create_tables())
    yield
    asyncio.run(drop_tables())


@pytest.fixture
def mock_contract():
    import asyncio
    contract_id = "test_contract_123"
    clause_id = "test_clause_456"
    
    # Setup mock contract data
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
    
    async def insert_contract():
        async with AsyncSessionLocal() as session:
            repo = ContractRepository(session)
            # owner_id matches mock_user.user_id to pass ownership checks
            await repo.upsert(contract.model_dump(), owner_id="test_user_id_123")
            await session.commit()
            
    asyncio.run(insert_contract())
    yield contract_id, clause_id


def test_confirm_clause(mock_contract):
    contract_id, clause_id = mock_contract
    
    # Confirm the clause
    payload = {
        "action": "confirm",
        "reviewerName": "Sarah Mitchell",
        "reviewerRole": "Senior Legal Counsel"
    }
    
    response = client.post(f"/api/v1/contracts/{contract_id}/clauses/{clause_id}/review", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    
    data = response.json()
    
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
    
    response = client.post(f"/api/v1/contracts/{contract_id}/clauses/{clause_id}/review", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    
    data = response.json()
    
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
    
    response = client.post(f"/api/v1/contracts/{contract_id}/clauses/{clause_id}/review", json=payload)
    assert response.status_code == 400
    assert "reason" in response.json()["detail"].lower()

