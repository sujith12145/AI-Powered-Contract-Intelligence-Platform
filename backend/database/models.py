"""
database/models.py
==================
SQLAlchemy ORM models: User, Contract, AuditLog.

These are the persistent counterparts of the Pydantic models in models.py.
The raw contract JSON is stored in the `raw_data` TEXT column (JSON-serialised)
so the existing analysis pipeline does not need to change.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from database.connection import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


# ------------------------------------------------------------------ #
# User                                                                 #
# ------------------------------------------------------------------ #

class UserModel(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    # roles: "admin" | "legal_counsel" | "viewer"
    role = Column(String(50), nullable=False, default="viewer")
    is_active = Column(Boolean, default=True)
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)
    last_notification_check = Column(DateTime(timezone=True), nullable=True)
    notify_needs_review = Column(Boolean, default=True)
    notify_high_risk = Column(Boolean, default=True)
    notify_overridden = Column(Boolean, default=True)

    contracts = relationship("ContractModel", back_populates="owner", lazy="select")
    audit_logs = relationship("AuditLogModel", back_populates="user", lazy="select")

    def __repr__(self):
        return f"<User {self.email} [{self.role}]>"


# ------------------------------------------------------------------ #
# Contract                                                             #
# ------------------------------------------------------------------ #

class ContractModel(Base):
    __tablename__ = "contracts"

    id = Column(String(50), primary_key=True)
    owner_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    name = Column(String(255), nullable=False)
    contract_type = Column(String(100), nullable=True)
    counterparty = Column(String(255), nullable=True)
    effective_date = Column(String(20), nullable=True)
    expiry_date = Column(String(20), nullable=True)
    value = Column(String(100), nullable=True)
    status = Column(String(50), default="active", index=True)
    overall_risk = Column(String(20), nullable=True)
    risk_score = Column(Integer, default=0)
    reviewed_date = Column(String(20), nullable=True)
    version = Column(Integer, default=1)
    source = Column(String(50), default="uploaded")
    file_hash = Column(String(64), nullable=True)
    file_path = Column(String(255), nullable=True)
    # Full contract JSON stored as text for fast retrieval
    raw_data = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)

    owner = relationship("UserModel", back_populates="contracts")

    def __repr__(self):
        return f"<Contract {self.id} [{self.name}]>"


# ------------------------------------------------------------------ #
# Audit Log                                                            #
# ------------------------------------------------------------------ #

class AuditLogModel(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    user_email = Column(String(255), nullable=True)
    user_role = Column(String(50), nullable=True)
    action = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(50), nullable=True)
    resource_id = Column(String(100), nullable=True)
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(Text, nullable=True)
    extra_data = Column(JSON, nullable=True)
    response_status = Column(Integer, nullable=True)
    prev_hash = Column(String(64), nullable=True)
    log_hash = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now, index=True)

    user = relationship("UserModel", back_populates="audit_logs")

    def __repr__(self):
        return f"<AuditLog {self.action} by {self.user_email}>"
