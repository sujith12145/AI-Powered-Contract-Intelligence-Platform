"""
tests/test_hardening.py
========================
Tests for critical security hardening, request binding, CSP/HSTS, audit chaining,
and contract download/tampering verification.
"""
import os
import sys
import pytest
import hashlib
import json
import asyncio
from unittest.mock import patch, MagicMock
from fastapi import Request
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from main import app
from database.connection import Base, get_db
from database.repositories import ContractRepository, AuditRepository
from auth.dependencies import get_current_user, TokenPayload
from database import models as _db_models  # Ensure models are registered with Base.metadata

from sqlalchemy.pool import StaticPool

# Create in-memory SQLite engine for tests with StaticPool to share connection state
engine = create_async_engine(
    "sqlite+aiosqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
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
    
    # Save previous overrides to prevent cross-test contamination
    old_get_db = app.dependency_overrides.get(get_db)
    old_get_current_user = app.dependency_overrides.get(get_current_user)
    
    # Apply our overrides
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user
    
    async def create_tables():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            
    async def drop_tables():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            
    asyncio.run(create_tables())
    yield
    asyncio.run(drop_tables())
    
    # Restore previous overrides
    if old_get_db is not None:
        app.dependency_overrides[get_db] = old_get_db
    else:
        app.dependency_overrides.pop(get_db, None)
        
    if old_get_current_user is not None:
        app.dependency_overrides[get_current_user] = old_get_current_user
    else:
        app.dependency_overrides.pop(get_current_user, None)


# ------------------------------------------------------------------ #
# Security Headers Tests                                              #
# ------------------------------------------------------------------ #

def test_security_headers_present():
    """Check that CSP is restricted and HSTS is set on endpoints."""
    response = client.get("/")
    assert response.status_code == 200
    
    csp = response.headers.get("Content-Security-Policy", "")
    assert "script-src 'self';" in csp
    assert "script-src 'self' 'unsafe-inline'" not in csp
    
    hsts = response.headers.get("Strict-Transport-Security", "")
    assert "max-age=31536000" in hsts


# ------------------------------------------------------------------ #
# Authentication Request State Binding Tests                         #
# ------------------------------------------------------------------ #

def test_get_current_user_binds_request_state():
    """Verify that get_current_user dependency binds claims to request.state."""
    request = MagicMock(spec=Request)
    request.state = MagicMock()
    
    with patch("auth.dependencies.decode_access_token") as mock_decode:
        mock_decode.return_value = {
            "sub": "user_999",
            "email": "counsel_test@contractiq.local",
            "role": "legal_counsel"
        }
        
        credentials = MagicMock()
        credentials.credentials = "mocked_jwt_token"
        
        payload = asyncio.run(get_current_user(request=request, credentials=credentials))
        
        assert payload.user_id == "user_999"
        assert request.state.user_id == "user_999"
        assert request.state.user_email == "counsel_test@contractiq.local"
        assert request.state.user_role == "legal_counsel"


# ------------------------------------------------------------------ #
# Audit Log SHA-256 Chaining Tests                                    #
# ------------------------------------------------------------------ #

def test_audit_log_chaining():
    """Verify that database audit logs compute hashes and chain them together."""
    async def _run():
        async with AsyncSessionLocal() as session:
            repo = AuditRepository(session)
            
            # Log first entry
            entry1 = await repo.log(
                action="action1",
                user_id="u1",
                user_email="u1@example.com",
                user_role="admin",
                extra_data={"key1": "val1"}
            )
            await session.commit()
            
            # Log second entry
            entry2 = await repo.log(
                action="action2",
                user_id="u2",
                user_email="u2@example.com",
                user_role="viewer",
                extra_data={"key2": "val2"}
            )
            await session.commit()
            
            assert entry1.prev_hash is None
            assert entry1.log_hash is not None
            
            assert entry2.prev_hash == entry1.log_hash
            assert entry2.log_hash is not None
            
            # Manually compute expected hash for entry2 and verify
            extra_data_str = json.dumps({"key2": "val2"}, sort_keys=True)
            payload = f"{entry1.log_hash}|u2|u2@example.com|action2|||{extra_data_str}"
            expected_hash = hashlib.sha256(payload.encode('utf-8')).hexdigest()
            assert entry2.log_hash == expected_hash

    asyncio.run(_run())


def test_audit_log_verification():
    """Verify that verify_chain detects tampering and validates a secure chain."""
    async def _run():
        async with AsyncSessionLocal() as session:
            repo = AuditRepository(session)
            # Chain should be intact initially (empty)
            assert await repo.verify_chain() is None
            
            # Log some entries
            e1 = await repo.log(action="act1", user_id="u1")
            e2 = await repo.log(action="act2", user_id="u2")
            await session.commit()
            
            # Verify chain is intact
            assert await repo.verify_chain() is None
            
            # Tamper with entry2 in DB
            e2.action = "tampered_action"
            await session.commit()
            
            # Verify chain detects tampering
            err = await repo.verify_chain()
            assert err is not None
            assert err["error"] == "log_hash_invalid"
            assert err["entry_id"] == e2.id

    asyncio.run(_run())


# ------------------------------------------------------------------ #
# Upload Limits (T10) Tests                                           #
# ------------------------------------------------------------------ #

def test_pdf_page_limits_exceeded():
    """Verify PDF upload is rejected if it has more than 50 pages."""
    with patch("pdfminer.pdfpage.PDFPage.get_pages") as mock_get_pages:
        mock_get_pages.return_value = [MagicMock()] * 51
        
        response = client.post(
            "/api/v1/contracts/analyze",
            files={"file": ("test.pdf", b"%PDF-1.4 dummy contents", "application/pdf")}
        )
        assert response.status_code == 400
        assert "exceeds maximum page limit" in response.json()["detail"]


def test_docx_paragraph_limits_exceeded():
    """Verify DOCX upload is rejected if it has more than 1000 paragraphs."""
    with patch("docx.Document") as mock_doc_class:
        mock_doc = MagicMock()
        mock_doc.paragraphs = [MagicMock()] * 1001
        mock_doc_class.return_value = mock_doc
        
        response = client.post(
            "/api/v1/contracts/analyze",
            files={"file": ("test.docx", b"PK\x03\x04 dummy contents", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
        )
        assert response.status_code == 400
        assert "exceeds maximum paragraph limit" in response.json()["detail"]


# ------------------------------------------------------------------ #
# Contract Download & Tamper Verification (T3/T7) Tests              #
# ------------------------------------------------------------------ #

def test_contract_download_tampering_and_bola(tmp_path):
    """Test downloading a contract, verifying disk-tamper checking (T3), and BOLA authorization (T7)."""
    # Create temporary contract file
    contract_file = tmp_path / "mock_contract.pdf"
    contract_file.write_bytes(b"original contract contents")
    
    file_path = str(contract_file)
    file_hash = hashlib.sha256(b"original contract contents").hexdigest()
    contract_id = "test_down_999"
    
    async def _setup():
        async with AsyncSessionLocal() as session:
            repo = ContractRepository(session)
            contract_data = {
                "id": contract_id,
                "name": "Test Download Contract",
                "type": "NDA",
                "counterparty": "Test Corp",
                "overallRisk": "low",
                "riskScore": 10,
                "status": "active"
            }
            await repo.upsert(
                contract_data,
                owner_id="test_user_id_123", # matches mock_user
                file_path=file_path,
                file_hash=file_hash
            )
            await session.commit()

    asyncio.run(_setup())
        
    # 1. Download file successfully
    response = client.get(f"/api/v1/contracts/{contract_id}/download")
    assert response.status_code == 200
    assert response.content == b"original contract contents"
    
    # 2. Modify file on disk to simulate tampering (T3)
    contract_file.write_bytes(b"tampered contract contents")
    response = client.get(f"/api/v1/contracts/{contract_id}/download")
    assert response.status_code == 409
    assert "tampering detected" in response.json()["detail"]
    
    # 3. Verify BOLA ownership check (T7)
    # Override current user to a different viewer user who doesn't own the contract
    different_viewer = TokenPayload(user_id="other_user_456", email="viewer@contractiq.local", role="viewer")
    
    async def override_other_user(request: Request):
        request.state.user_id = different_viewer.user_id
        request.state.user_email = different_viewer.email
        request.state.user_role = different_viewer.role
        return different_viewer
        
    app.dependency_overrides[get_current_user] = override_other_user
    
    response = client.get(f"/api/v1/contracts/{contract_id}/download")
    assert response.status_code == 403
    assert "Access denied" in response.json()["detail"]
    
    # Restore mock user dependency
    app.dependency_overrides[get_current_user] = override_get_current_user
