"""
routers/auth.py
===============
Authentication endpoints:

  POST /api/v1/auth/login                — Credentials → JWT access token + refresh token cookie
  POST /api/v1/auth/refresh              — Rotate refresh token → new access token
  POST /api/v1/auth/logout               — Revoke refresh token
  GET  /api/v1/auth/me                   — Current user profile
  PUT  /api/v1/auth/me/settings          — Update name / password / notification prefs
  GET  /api/v1/auth/me/notifications     — Fetch real-time notifications from contract data
  POST /api/v1/auth/me/read-notifications — Mark notifications as read (updates last_check)
"""


import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, EmailStr, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import TokenPayload, get_current_user
from auth.jwt_handler import (
    create_access_token,
    create_refresh_token,
    revoke_refresh_token,
    rotate_refresh_token,
    validate_refresh_token,
)
from auth.password import verify_password, hash_password
from config import get_settings
from database.connection import get_db
from database.repositories import AuditRepository, UserRepository
from middleware.rate_limiter import limiter

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

_REFRESH_COOKIE = "contractiq_refresh"
_COOKIE_MAX_AGE = settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400


# ------------------------------------------------------------------ #
# Request / Response models                                            #
# ------------------------------------------------------------------ #

class LoginRequest(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def normalise_email(cls, v: str) -> str:
        return v.strip().lower()


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    user: dict


class UserProfile(BaseModel):
    id: str
    email: str
    full_name: Optional[str]
    role: str
    is_active: bool
    failed_login_attempts: int = 0
    created_at: Optional[str] = None
    notify_needs_review: bool = True
    notify_high_risk: bool = True
    notify_overridden: bool = True
    last_notification_check: Optional[str] = None


class UpdateSettingsRequest(BaseModel):
    full_name: Optional[str] = None
    current_password: Optional[str] = None
    new_password: Optional[str] = None
    notify_needs_review: Optional[bool] = None
    notify_high_risk: Optional[bool] = None
    notify_overridden: Optional[bool] = None


class CreateUserRequest(BaseModel):
    email: str
    password: str
    full_name: str
    role: str = "viewer"

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("viewer", "legal_counsel", "admin"):
            raise ValueError("role must be viewer | legal_counsel | admin")
        return v


# ------------------------------------------------------------------ #
# Routes                                                               #
# ------------------------------------------------------------------ #

@router.post("/login", response_model=TokenResponse)
@limiter.limit("5/minute")
async def login(
    request: Request,
    response: Response,
    body: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Authenticate with email + password.
    Returns a JWT access token (15 min) and sets a httpOnly refresh-token cookie (7 days).
    """
    repo = UserRepository(db)
    audit = AuditRepository(db)

    user = await repo.get_by_email(body.email)

    # Generic error — don't reveal whether email exists
    auth_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid email or password.",
    )

    if not user or not user.is_active:
        await audit.log(
            action="login_failed",
            user_email=body.email,
            ip_address=request.client.host if request.client else None,
            extra_data={"reason": "user_not_found"},
            response_status=401,
        )
        raise auth_error

    # Check account lock
    if user.locked_until and user.locked_until > datetime.now(timezone.utc):
        remaining = int((user.locked_until - datetime.now(timezone.utc)).total_seconds())
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Account locked due to too many failed attempts. Try again in {remaining}s.",
        )

    if not verify_password(body.password, user.hashed_password):
        await repo.record_failed_login(user)
        await audit.log(
            action="login_failed",
            user_id=user.id,
            user_email=user.email,
            user_role=user.role,
            ip_address=request.client.host if request.client else None,
            extra_data={"failed_attempts": user.failed_login_attempts},
            response_status=401,
        )
        raise auth_error

    # Success — reset failed attempts
    await repo.reset_failed_login(user)

    access_token = create_access_token(user.id, user.email, user.role)
    refresh_token = create_refresh_token(user.id, user.email, user.role)

    # httpOnly cookie — not accessible from JS (XSS protection)
    response.set_cookie(
        key=_REFRESH_COOKIE,
        value=refresh_token,
        max_age=_COOKIE_MAX_AGE,
        httponly=True,
        samesite="strict",
        secure=False,  # set True in production behind HTTPS
        path="/api/v1/auth",
    )

    await audit.log(
        action="login_success",
        user_id=user.id,
        user_email=user.email,
        user_role=user.role,
        ip_address=request.client.host if request.client else None,
        response_status=200,
    )

    logger.info("Login: %s [%s]", user.email, user.role)
    return TokenResponse(
        access_token=access_token,
        user={
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
        },
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
    refresh: Optional[str] = Cookie(default=None, alias=_REFRESH_COOKIE),
):
    """
    Exchange a valid refresh token for a new access token.
    The refresh token is rotated (old one invalidated, new one issued).
    """
    if not refresh:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No refresh token.")

    data = validate_refresh_token(refresh)
    if not data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token expired or invalid.",
        )

    user_id = data["user_id"]
    email = data["email"]
    role = data["role"]

    # Rotate the refresh token
    new_refresh = rotate_refresh_token(refresh, user_id, email, role)
    access_token = create_access_token(user_id, email, role)

    response.set_cookie(
        key=_REFRESH_COOKIE,
        value=new_refresh,
        max_age=_COOKIE_MAX_AGE,
        httponly=True,
        samesite="strict",
        secure=False,
        path="/api/v1/auth",
    )

    # Load user for profile
    repo = UserRepository(db)
    user = await repo.get_by_id(user_id)
    user_data = {
        "id": user_id,
        "email": email,
        "full_name": user.full_name if user else email,
        "role": role,
    }

    return TokenResponse(access_token=access_token, user=user_data)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    response: Response,
    db: AsyncSession = Depends(get_db),
    user: TokenPayload = Depends(get_current_user),
    refresh: Optional[str] = Cookie(default=None, alias=_REFRESH_COOKIE),
):
    """Invalidate refresh token and clear the cookie."""
    if refresh:
        revoke_refresh_token(refresh)

    response.delete_cookie(key=_REFRESH_COOKIE, path="/api/v1/auth")

    audit = AuditRepository(db)
    await audit.log(
        action="logout",
        user_id=user.user_id,
        user_email=user.email,
        user_role=user.role,
        response_status=204,
    )


@router.get("/me", response_model=UserProfile)
async def get_me(
    user: TokenPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the current authenticated user's profile."""
    repo = UserRepository(db)
    db_user = await repo.get_by_id(user.user_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found.")
    return UserProfile(
        id=db_user.id,
        email=db_user.email,
        full_name=db_user.full_name,
        role=db_user.role,
        is_active=db_user.is_active,
        failed_login_attempts=db_user.failed_login_attempts or 0,
        created_at=db_user.created_at.isoformat() if db_user.created_at else None,
        notify_needs_review=bool(db_user.notify_needs_review) if db_user.notify_needs_review is not None else True,
        notify_high_risk=bool(db_user.notify_high_risk) if db_user.notify_high_risk is not None else True,
        notify_overridden=bool(db_user.notify_overridden) if db_user.notify_overridden is not None else True,
        last_notification_check=db_user.last_notification_check.isoformat() if db_user.last_notification_check else None,
    )


@router.put("/me/settings", response_model=UserProfile)
async def update_my_settings(
    body: UpdateSettingsRequest,
    user: TokenPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update the current user's personal settings (name, password, notification prefs)."""
    repo = UserRepository(db)
    db_user = await repo.get_by_id(user.user_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found.")

    new_hash: Optional[str] = None
    if body.new_password:
        if not body.current_password:
            raise HTTPException(status_code=400, detail="current_password is required to set a new password.")
        if not verify_password(body.current_password, db_user.hashed_password):
            raise HTTPException(status_code=400, detail="Current password is incorrect.")
        if len(body.new_password) < 8:
            raise HTTPException(status_code=400, detail="New password must be at least 8 characters.")
        new_hash = hash_password(body.new_password)

    db_user = await repo.update_settings(
        db_user,
        full_name=body.full_name,
        hashed_password=new_hash,
        notify_needs_review=body.notify_needs_review,
        notify_high_risk=body.notify_high_risk,
        notify_overridden=body.notify_overridden,
    )

    return UserProfile(
        id=db_user.id,
        email=db_user.email,
        full_name=db_user.full_name,
        role=db_user.role,
        is_active=db_user.is_active,
        failed_login_attempts=db_user.failed_login_attempts or 0,
        created_at=db_user.created_at.isoformat() if db_user.created_at else None,
        notify_needs_review=bool(db_user.notify_needs_review) if db_user.notify_needs_review is not None else True,
        notify_high_risk=bool(db_user.notify_high_risk) if db_user.notify_high_risk is not None else True,
        notify_overridden=bool(db_user.notify_overridden) if db_user.notify_overridden is not None else True,
        last_notification_check=db_user.last_notification_check.isoformat() if db_user.last_notification_check else None,
    )


@router.get("/me/notifications")
async def get_my_notifications(
    user: TokenPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> List[Dict[str, Any]]:
    """
    Derive real notifications from contract data:
      - Clauses with status 'Needs Review' (low-confidence) since last check
      - Critical/High risk clauses on any accessible contract
      - Clauses where another reviewer used 'override' action on user-uploaded contracts
    Respects per-user notification preferences.
    """
    repo = UserRepository(db)
    db_user = await repo.get_by_id(user.user_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found.")

    from database.models import ContractModel

    # Determine which contracts to scan
    if user.role == "admin":
        stmt = select(ContractModel)
    else:
        stmt = select(ContractModel).where(
            ContractModel.owner_id == user.user_id
        )
    result = await db.execute(stmt)
    contracts = result.scalars().all()

    notifications: List[Dict[str, Any]] = []
    last_check = db_user.last_notification_check

    notify_nr = bool(db_user.notify_needs_review) if db_user.notify_needs_review is not None else True
    notify_hr = bool(db_user.notify_high_risk) if db_user.notify_high_risk is not None else True
    notify_ov = bool(db_user.notify_overridden) if db_user.notify_overridden is not None else True

    for contract in contracts:
        raw = contract.raw_data or {}
        clauses = raw.get("clauses", [])
        contract_name = contract.name or raw.get("name", "Unknown Contract")

        for clause in clauses:
            clause_id = clause.get("id", "")
            clause_title = clause.get("title", clause.get("type", "Clause"))
            risk = clause.get("riskLevel", "low")
            status_val = clause.get("status", "")
            review_history = clause.get("reviewHistory") or []

            # 1. Needs Review (low confidence)
            if notify_nr and status_val == "Needs Review":
                notifications.append({
                    "id": f"nr-{contract.id}-{clause_id}",
                    "type": "needs_review",
                    "title": "Clause Needs Review",
                    "message": f"{clause_title} in \u201c{contract_name}\u201d has low confidence and requires manual review.",
                    "contractId": contract.id,
                    "contractName": contract_name,
                    "clauseId": clause_id,
                    "riskLevel": risk,
                    "read": False,
                })

            # 2. Critical/High risk clauses
            if notify_hr and risk in ("critical", "high"):
                notifications.append({
                    "id": f"hr-{contract.id}-{clause_id}",
                    "type": "high_risk",
                    "title": f"{risk.capitalize()} Risk Clause Detected",
                    "message": f"{clause_title} in \u201c{contract_name}\u201d is flagged as {risk} risk.",
                    "contractId": contract.id,
                    "contractName": contract_name,
                    "clauseId": clause_id,
                    "riskLevel": risk,
                    "read": False,
                })

            # 3. Overridden by another reviewer on user-uploaded contracts
            if notify_ov and contract.owner_id == user.user_id:
                for entry in review_history:
                    if entry.get("action") == "override":
                        reviewer_name = entry.get("reviewerName", "Another reviewer")
                        ts = entry.get("timestamp", "")
                        notifications.append({
                            "id": f"ov-{contract.id}-{clause_id}-{ts}",
                            "type": "overridden",
                            "title": "Clause Verdict Overridden",
                            "message": f"{reviewer_name} overrode the AI verdict on {clause_title} in \u201c{contract_name}\u201d.",
                            "contractId": contract.id,
                            "contractName": contract_name,
                            "clauseId": clause_id,
                            "riskLevel": risk,
                            "read": False,
                        })

    # Mark unread for items created after last_check
    # Since our data doesn't have per-clause timestamps, we mark all as unread
    # until the user opens the panel (which resets last_notification_check).
    unread_count = len(notifications) if last_check is None else len(notifications)

    return {
        "notifications": notifications[:50],  # cap at 50
        "unread_count": unread_count if last_check is None else 0,
        "last_check": last_check.isoformat() if last_check else None,
    }


@router.post("/me/read-notifications", status_code=status.HTTP_204_NO_CONTENT)
async def mark_notifications_read(
    user: TokenPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Mark all notifications as read by updating last_notification_check."""
    repo = UserRepository(db)
    db_user = await repo.get_by_id(user.user_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found.")
    await repo.mark_notifications_read(db_user)
