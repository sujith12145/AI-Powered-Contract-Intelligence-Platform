"""
database/repositories.py
========================
Data-access layer (Repository pattern).

All DB reads/writes go through these classes — routes never touch
SQLAlchemy directly. This keeps routes thin and makes testing easy.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy import select, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from database.models import AuditLogModel, ContractModel, UserModel

logger = logging.getLogger(__name__)
_settings = get_settings()


# ------------------------------------------------------------------ #
# HMAC helper                                                          #
# ------------------------------------------------------------------ #

def _audit_hmac(payload: str) -> str:
    """
    Compute HMAC-SHA256 of *payload* using the secret key from settings.
    The key is read from the AUDIT_LOG_HMAC_KEY environment variable and
    is NEVER stored in the database, making the chain tamper-resistant:
    an attacker with raw DB write access cannot recompute valid hashes
    without knowing the key.
    """
    key = _settings.AUDIT_LOG_HMAC_KEY.encode()
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


# ------------------------------------------------------------------ #
# UserRepository                                                       #
# ------------------------------------------------------------------ #

class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, user_id: str) -> Optional[UserModel]:
        result = await self.session.get(UserModel, user_id)
        return result

    async def get_by_email(self, email: str) -> Optional[UserModel]:
        stmt = select(UserModel).where(UserModel.email == email.lower())
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def create(
        self,
        email: str,
        hashed_password: str,
        full_name: str,
        role: str = "viewer",
    ) -> UserModel:
        user = UserModel(
            email=email.lower(),
            hashed_password=hashed_password,
            full_name=full_name,
            role=role,
        )
        self.session.add(user)
        await self.session.flush()
        return user

    async def list_all(self, skip: int = 0, limit: int = 100) -> List[UserModel]:
        stmt = select(UserModel).offset(skip).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update_role(self, user_id: str, role: str) -> Optional[UserModel]:
        user = await self.get_by_id(user_id)
        if not user:
            return None
        user.role = role
        user.updated_at = datetime.now(timezone.utc)
        await self.session.flush()
        return user

    async def record_failed_login(self, user: UserModel) -> None:
        user.failed_login_attempts = (user.failed_login_attempts or 0) + 1
        # Lock after 5 failures for 15 minutes
        if user.failed_login_attempts >= 5:
            from datetime import timedelta
            user.locked_until = datetime.now(timezone.utc) + timedelta(minutes=15)
        await self.session.flush()

    async def reset_failed_login(self, user: UserModel) -> None:
        user.failed_login_attempts = 0
        user.locked_until = None
        await self.session.flush()

    async def count(self) -> int:
        result = await self.session.execute(select(func.count()).select_from(UserModel))
        return result.scalar_one()

    async def update_settings(
        self,
        user: UserModel,
        *,
        full_name: Optional[str] = None,
        email: Optional[str] = None,
        hashed_password: Optional[str] = None,
        notify_needs_review: Optional[bool] = None,
        notify_high_risk: Optional[bool] = None,
        notify_overridden: Optional[bool] = None,
    ) -> UserModel:
        if full_name is not None:
            user.full_name = full_name
        if email is not None:
            user.email = email.lower()
        if hashed_password is not None:
            user.hashed_password = hashed_password
        if notify_needs_review is not None:
            user.notify_needs_review = notify_needs_review
        if notify_high_risk is not None:
            user.notify_high_risk = notify_high_risk
        if notify_overridden is not None:
            user.notify_overridden = notify_overridden
        user.updated_at = datetime.now(timezone.utc)
        await self.session.flush()
        return user

    async def mark_notifications_read(self, user: UserModel) -> None:
        user.last_notification_check = datetime.now(timezone.utc)
        await self.session.flush()


# ------------------------------------------------------------------ #
# ContractRepository                                                   #
# ------------------------------------------------------------------ #

class ContractRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, contract_id: str) -> Optional[ContractModel]:
        return await self.session.get(ContractModel, contract_id)

    async def list_all(
        self,
        owner_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[ContractModel]:
        stmt = select(ContractModel)
        if owner_id:
            stmt = stmt.where(ContractModel.owner_id == owner_id)
        stmt = stmt.order_by(desc(ContractModel.created_at)).offset(skip).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def upsert(
        self,
        contract_data: Dict,
        owner_id: Optional[str] = None,
        file_path: Optional[str] = None,
        file_hash: Optional[str] = None,
    ) -> ContractModel:
        """Insert or update a contract. Stores the full dict as raw_data."""
        existing = await self.get(contract_data["id"])
        if existing:
            existing.raw_data = contract_data
            existing.name = contract_data.get("name", existing.name)
            existing.overall_risk = contract_data.get("overallRisk", existing.overall_risk)
            existing.risk_score = contract_data.get("riskScore", existing.risk_score)
            existing.status = contract_data.get("status", existing.status)
            if file_path is not None:
                existing.file_path = file_path
            if file_hash is not None:
                existing.file_hash = file_hash
            existing.updated_at = datetime.now(timezone.utc)
            await self.session.flush()
            return existing
        else:
            obj = ContractModel(
                id=contract_data["id"],
                owner_id=owner_id,
                name=contract_data.get("name", "Unnamed"),
                contract_type=contract_data.get("type"),
                counterparty=contract_data.get("counterparty"),
                effective_date=contract_data.get("effectiveDate"),
                expiry_date=contract_data.get("expiryDate"),
                value=contract_data.get("value"),
                status=contract_data.get("status", "active"),
                overall_risk=contract_data.get("overallRisk"),
                risk_score=contract_data.get("riskScore", 0),
                reviewed_date=contract_data.get("reviewedDate"),
                version=contract_data.get("version", 1),
                source=contract_data.get("source", "uploaded"),
                file_path=file_path,
                file_hash=file_hash,
                raw_data=contract_data,
            )
            self.session.add(obj)
            await self.session.flush()
            return obj

    async def delete(self, contract_id: str) -> bool:
        obj = await self.get(contract_id)
        if not obj:
            return False
        await self.session.delete(obj)
        await self.session.flush()
        return True

    async def count(self) -> int:
        result = await self.session.execute(select(func.count()).select_from(ContractModel))
        return result.scalar_one()


# ------------------------------------------------------------------ #
# AuditRepository                                                      #
# ------------------------------------------------------------------ #

class AuditRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def log(
        self,
        action: str,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        user_role: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        extra_data: Optional[Dict] = None,
        response_status: Optional[int] = None,
    ) -> AuditLogModel:
        # Fetch the last audit log entry to chain hashes
        stmt = select(AuditLogModel).order_by(desc(AuditLogModel.id)).limit(1)
        result = await self.session.execute(stmt)
        last_entry = result.scalar_one_or_none()
        prev_hash = last_entry.log_hash if last_entry else ""

        # Compute log hash using HMAC-SHA256 (key never stored in DB)
        extra_data_str = json.dumps(extra_data or {}, sort_keys=True)
        log_payload = (
            f"{prev_hash}|{user_id or ''}|{user_email or ''}"
            f"|{action}|{resource_type or ''}|{resource_id or ''}|{extra_data_str}"
        )
        log_hash = _audit_hmac(log_payload)

        entry = AuditLogModel(
            user_id=user_id,
            user_email=user_email,
            user_role=user_role,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            user_agent=user_agent,
            extra_data=extra_data,
            response_status=response_status,
            prev_hash=prev_hash if prev_hash else None,
            log_hash=log_hash,
        )
        self.session.add(entry)
        await self.session.flush()
        return entry

    async def list_recent(self, skip: int = 0, limit: int = 100) -> List[AuditLogModel]:
        stmt = (
            select(AuditLogModel)
            .order_by(desc(AuditLogModel.created_at))
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count(self) -> int:
        result = await self.session.execute(select(func.count()).select_from(AuditLogModel))
        return result.scalar_one()

    async def verify_chain(self) -> Optional[dict]:
        """
        Verify the entire audit log chain using HMAC-SHA256.

        Walks audit_logs in ascending id order. For each row:
          1. Checks that prev_hash matches the log_hash of the previous row.
          2. Recomputes the HMAC-SHA256 of the row's payload and compares
             it to the stored log_hash.

        Returns None if the chain is fully intact.
        Returns a dict describing the FIRST row where verification fails,
        including its id and action, so the caller can pinpoint tampering.
        """
        stmt = select(AuditLogModel).order_by(AuditLogModel.id.asc())
        result = await self.session.execute(stmt)
        entries = result.scalars().all()

        prev_expected_hash = ""
        for entry in entries:
            # 1. Check prev_hash matches what we expected from the previous row
            actual_prev = entry.prev_hash or ""
            if actual_prev != prev_expected_hash:
                return {
                    "error": "prev_hash_mismatch",
                    "entry_id": entry.id,
                    "expected_prev_hash": prev_expected_hash,
                    "actual_prev_hash": actual_prev,
                    "action": entry.action,
                }

            # 2. Re-compute HMAC of this row's payload and compare
            extra_data_str = json.dumps(entry.extra_data or {}, sort_keys=True)
            payload = (
                f"{actual_prev}|{entry.user_id or ''}|{entry.user_email or ''}"
                f"|{entry.action}|{entry.resource_type or ''}|{entry.resource_id or ''}|{extra_data_str}"
            )
            expected_hash = _audit_hmac(payload)

            if not hmac.compare_digest(entry.log_hash, expected_hash):
                return {
                    "error": "log_hash_invalid",
                    "entry_id": entry.id,
                    "expected_log_hash": expected_hash,
                    "actual_log_hash": entry.log_hash,
                    "action": entry.action,
                }

            prev_expected_hash = entry.log_hash

        return None
