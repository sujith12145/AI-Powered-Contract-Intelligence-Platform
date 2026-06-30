"""
tests/test_audit_and_timeout.py
================================
Three concrete proof-of-requirement tests:

  [1] test_hmac_chain_intact  — writes N audit rows, verifies chain passes.
  [2] test_hmac_tamper_detected — writes rows, directly mutates one row's
      `action` field in SQLite, confirms verify_chain() catches it and
      returns the tampered row's id.
  [3] test_extraction_timeout — patches extract_text to sleep longer than
      the timeout, posts a real file upload, confirms the endpoint returns
      422 with "too complex" message instead of hanging.
"""
from __future__ import annotations

import asyncio
import io
import os
import sys
import time
import unittest.mock as mock

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

# ---- Make sure backend/ is on sys.path ----
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ---- Ensure a stable HMAC key for the whole test session ----
os.environ.setdefault("AUDIT_LOG_HMAC_KEY", "test_hmac_key_stable_for_test_runs_only_00000000")

from main import app
from database.connection import Base, get_db
from database.repositories import AuditRepository, _audit_hmac
from auth.dependencies import get_current_user, TokenPayload

# ------------------------------------------------------------------ #
# In-memory SQLite engine shared across all tests in this module       #
# ------------------------------------------------------------------ #
engine = create_async_engine(
    "sqlite+aiosqlite:///:memory:",
    connect_args={"check_same_thread": False},
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


mock_user = TokenPayload(
    user_id="audit_test_uid",
    email="auditor@contractiq.local",
    role="admin",
)


async def override_get_current_user(request: Request):
    request.state.user_id = mock_user.user_id
    request.state.user_email = mock_user.email
    request.state.user_role = mock_user.role
    return mock_user


app.dependency_overrides[get_db] = override_get_db
app.dependency_overrides[get_current_user] = override_get_current_user

client = TestClient(app)


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

@pytest.fixture(autouse=True)
def fresh_schema():
    """Drop + recreate all tables before every test (full isolation)."""
    async def _setup():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_setup())
    yield


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

async def _write_audit_rows(n: int = 5) -> list:
    """Insert n audit rows and return the list of AuditLogModel objects."""
    async with AsyncSessionLocal() as session:
        repo = AuditRepository(session)
        rows = []
        for i in range(n):
            row = await repo.log(
                action=f"test_action_{i}",
                user_id="u1",
                user_email="u@example.com",
                resource_type="contract",
                resource_id=f"c{i}",
                extra_data={"seq": i},
                response_status=200,
            )
            rows.append(row)
        await session.commit()
        return rows


async def _verify() -> dict | None:
    """Run verify_chain() and return None (intact) or the error dict."""
    async with AsyncSessionLocal() as session:
        repo = AuditRepository(session)
        return await repo.verify_chain()


# ================================================================== #
# Test 1 – HMAC chain is intact after normal writes                   #
# ================================================================== #

def test_hmac_chain_intact():
    """
    Write 5 audit rows through AuditRepository.log(), then call
    verify_chain().  The chain must be intact (returns None).
    """
    rows = asyncio.run(_write_audit_rows(5))
    assert len(rows) == 5, "Expected 5 rows to be inserted"

    # Each row's log_hash must be a valid 64-char hex HMAC-SHA256
    for row in rows:
        assert len(row.log_hash) == 64, f"log_hash for row {row.id} is not 64 hex chars"

    # Chain verification must pass
    result = asyncio.run(_verify())
    assert result is None, f"Chain should be intact but got: {result}"

    print("\n[PASS] test_hmac_chain_intact")
    print(f"  Wrote {len(rows)} rows, chain intact [OK]")
    print(f"  First row log_hash (HMAC-SHA256): {rows[0].log_hash}")


# ================================================================== #
# Test 2 – Tamper detection: direct DB mutation is caught             #
# ================================================================== #

def test_hmac_tamper_detected():
    """
    Write 3 audit rows, then directly UPDATE audit_logs.action for row 2
    using raw SQL (simulating an attacker with DB write access).
    verify_chain() must return an error dict whose entry_id matches row 2.
    """
    rows = asyncio.run(_write_audit_rows(3))
    tampered_id = rows[1].id  # middle row

    async def _tamper_and_verify():
        async with AsyncSessionLocal() as session:
            # Direct raw-SQL mutation — bypasses ORM, no HMAC recomputation
            await session.execute(
                text("UPDATE audit_logs SET action = 'TAMPERED_ACTION' WHERE id = :rid"),
                {"rid": tampered_id},
            )
            await session.commit()

        # Now verify
        return await _verify()

    error = asyncio.run(_tamper_and_verify())

    # Must not be None
    assert error is not None, (
        "verify_chain() returned None after tampering — chain is NOT detecting the mutation!"
    )

    # Must identify the correct row
    assert error["entry_id"] == tampered_id, (
        f"Expected tampered entry_id={tampered_id}, got {error['entry_id']}"
    )

    # Error type must be log_hash_invalid (HMAC mismatch on the tampered row)
    assert error["error"] == "log_hash_invalid", (
        f"Expected log_hash_invalid but got {error['error']}"
    )

    print("\n[PASS] test_hmac_tamper_detected")
    print(f"  Tampered row id={tampered_id}, action set to 'TAMPERED_ACTION'")
    print(f"  verify_chain() caught it: error={error['error']}, entry_id={error['entry_id']} [OK]")
    print("  Without the HMAC key, attacker cannot recompute a valid hash [OK]")


# ================================================================== #
# Test 3 – Extraction timeout returns clean 422                       #
# ================================================================== #

def test_extraction_timeout_returns_422():
    """
    Verify that a CPU-heavy extraction is aborted and returns HTTP 422.

    Strategy:
      - Patch _EXTRACT_TIMEOUT_SECS to 2s (avoids waiting 15s in tests).
      - Patch extract_text to sleep 6s (longer than the 2s timeout).
      - The ThreadPoolExecutor.result(timeout=2) raises FuturesTimeoutError,
        the endpoint returns 422 with 'too complex'.
      - The sleeping thread continues in the background but the test only
        blocks for ~2s + small overhead (we assert elapsed < 10s).

    Note: ThreadPoolExecutor cannot kill a running thread, so the sleep
    thread runs to completion in the background.  The important guarantee
    is that the HTTP *response* is returned promptly, not that the
    thread is killed.
    """

    def _slow_extractor(data: bytes, filename: str) -> str:
        time.sleep(6)   # longer than the patched 2s timeout
        return "should never reach here"

    MINIMAL_PDF = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
        b"xref\n0 4\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"trailer<</Size 4/Root 1 0 R>>\n"
        b"startxref\n173\n%%EOF\n"
    )

    with (
        mock.patch("routers.contracts.extract_text", side_effect=_slow_extractor),
        mock.patch(
            "pdfminer.pdfpage.PDFPage.get_pages",
            return_value=iter([object()]),   # pretend 1 page
        ),
        # Shorten the timeout from 15s to 2s so the test runs fast
        mock.patch("routers.contracts._EXTRACT_TIMEOUT_SECS", 2),
    ):
        t0 = time.monotonic()
        response = client.post(
            "/api/v1/contracts/analyze",
            files={"file": ("timeout_test.pdf", io.BytesIO(MINIMAL_PDF), "application/pdf")},
        )
        elapsed = time.monotonic() - t0

    # The HTTP response must be 422
    assert response.status_code == 422, (
        f"Expected 422, got {response.status_code}: {response.text}"
    )

    # The detail must mention 'too complex'/'exceeded'
    detail = response.json().get("detail", "")
    assert any(kw in detail.lower() for kw in ("too complex", "timed out", "exceeded")), (
        f"Expected timeout language in detail, got: {detail!r}"
    )

    # The response must arrive promptly (within 2s timeout + 8s buffer)
    assert elapsed < 10.0, (
        f"Request took {elapsed:.1f}s -- timeout is not firing, request is hanging!"
    )

    print("\n[PASS] test_extraction_timeout_returns_422")
    print(f"  Response returned in {elapsed:.2f}s (patched timeout=2s, buffer=10s) [OK]")
    print(f"  HTTP status: {response.status_code} [OK]")
    print(f"  Detail: {detail!r} [OK]")

