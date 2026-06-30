"""
routers/admin.py
================
Admin-only endpoints:

  GET  /api/v1/admin/users           — list all users
  POST /api/v1/admin/users           — create a new user
  PUT  /api/v1/admin/users/{id}/role — change a user's role
  GET  /api/v1/admin/audit-logs      — paginated audit log
  GET  /api/v1/admin/system-stats    — DB counts + index status

All routes are protected at the router level via require_role("admin").
Individual endpoints that need the acting-user for audit logging
call get_current_user() directly — it is safe to use because the
router-level guard already validated the token.
"""


import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import TokenPayload, get_current_user, require_role
from auth.password import hash_password
from config import get_settings
from database.connection import get_db
from database.repositories import AuditRepository, ContractRepository, UserRepository

logger = logging.getLogger(__name__)
settings = get_settings()

# ---- Router: admin-only, all routes enforce the role ----
router = APIRouter(
    prefix="/api/v1/admin",
    tags=["admin"],
    dependencies=[Depends(require_role("admin"))],
)


# ------------------------------------------------------------------ #
# Pydantic schemas                                                     #
# ------------------------------------------------------------------ #

class UserOut(BaseModel):
    id: str
    email: str
    full_name: Optional[str]
    role: str
    is_active: bool
    failed_login_attempts: int
    created_at: str

    model_config = {"from_attributes": True}


class CreateUserBody(BaseModel):
    email: str
    password: str
    full_name: str
    role: str = "viewer"


class UpdateRoleBody(BaseModel):
    role: str


class AuditLogOut(BaseModel):
    id: int
    user_email: Optional[str]
    user_role: Optional[str]
    action: str
    resource_type: Optional[str]
    resource_id: Optional[str]
    ip_address: Optional[str]
    response_status: Optional[int]
    created_at: str


class SystemStats(BaseModel):
    total_users: int
    total_contracts: int
    total_audit_logs: int
    cuad_index_built: bool
    cuad_categories: int
    database_type: str


# ------------------------------------------------------------------ #
# Routes                                                               #
# ------------------------------------------------------------------ #

@router.get("/users", response_model=List[UserOut])
async def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    """List all registered users (Admin only)."""
    repo = UserRepository(db)
    users = await repo.list_all(skip=skip, limit=limit)
    return [
        UserOut(
            id=u.id,
            email=u.email,
            full_name=u.full_name,
            role=u.role,
            is_active=u.is_active,
            failed_login_attempts=u.failed_login_attempts,
            created_at=u.created_at.isoformat() if u.created_at else "",
        )
        for u in users
    ]


@router.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def create_user(
    body: CreateUserBody,
    db: AsyncSession = Depends(get_db),
    # NOTE: Router-level guard already enforced admin auth.
    # We call get_current_user again only to log the acting user.
    actor: TokenPayload = Depends(get_current_user),
):
    """Create a new user account (Admin only)."""
    repo = UserRepository(db)
    existing = await repo.get_by_email(body.email)
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered.")

    if body.role not in ("viewer", "legal_counsel", "admin"):
        raise HTTPException(status_code=400, detail="Invalid role.")

    hashed = hash_password(body.password)
    user = await repo.create(
        email=body.email,
        hashed_password=hashed,
        full_name=body.full_name,
        role=body.role,
    )

    # Audit
    audit = AuditRepository(db)
    await audit.log(
        action="user_created",
        user_id=actor.user_id,
        user_email=actor.email,
        user_role=actor.role,
        resource_type="user",
        resource_id=user.id,
        extra_data={"new_user_email": body.email, "role": body.role},
        response_status=201,
    )

    return UserOut(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        is_active=user.is_active,
        failed_login_attempts=0,
        created_at=user.created_at.isoformat() if user.created_at else "",
    )


@router.put("/users/{user_id}/role")
async def update_user_role(
    user_id: str,
    body: UpdateRoleBody,
    db: AsyncSession = Depends(get_db),
    actor: TokenPayload = Depends(get_current_user),
):
    """Change a user's role (Admin only)."""
    if body.role not in ("viewer", "legal_counsel", "admin"):
        raise HTTPException(status_code=400, detail="Invalid role.")

    repo = UserRepository(db)
    user = await repo.update_role(user_id, body.role)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    audit = AuditRepository(db)
    await audit.log(
        action="user_role_changed",
        user_id=actor.user_id,
        user_email=actor.email,
        resource_type="user",
        resource_id=user_id,
        extra_data={"new_role": body.role},
        response_status=200,
    )
    return {"id": user.id, "role": user.role, "message": "Role updated."}


@router.get("/audit-logs", response_model=List[AuditLogOut])
async def get_audit_logs(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    """Retrieve paginated audit log (Admin only)."""
    repo = AuditRepository(db)
    logs = await repo.list_recent(skip=skip, limit=limit)
    return [
        AuditLogOut(
            id=log.id,
            user_email=log.user_email,
            user_role=log.user_role,
            action=log.action,
            resource_type=log.resource_type,
            resource_id=log.resource_id,
            ip_address=log.ip_address,
            response_status=log.response_status,
            created_at=log.created_at.isoformat() if log.created_at else "",
        )
        for log in logs
    ]


@router.get("/audit-logs/verify")
async def verify_audit_chain(db: AsyncSession = Depends(get_db)):
    """
    Cryptographically verify the integrity of the audit log chain (Admin only).
    """
    repo = AuditRepository(db)
    verification_error = await repo.verify_chain()
    if verification_error:
        return {
            "intact": False,
            "message": "Audit log tampering detected!",
            "details": verification_error
        }
    return {
        "intact": True,
        "message": "Audit log chain is cryptographically secure."
    }


@router.get("/system-stats", response_model=SystemStats)
async def system_stats(db: AsyncSession = Depends(get_db)):
    """System-wide metrics for the admin dashboard (Admin only)."""
    import cuad_index as cuad_idx

    user_repo = UserRepository(db)
    contract_repo = ContractRepository(db)
    audit_repo = AuditRepository(db)

    index = cuad_idx.get_index() if cuad_idx._instance else None

    db_type = settings.DATABASE_URL.split("+")[0].split(":")[0]

    return SystemStats(
        total_users=await user_repo.count(),
        total_contracts=await contract_repo.count(),
        total_audit_logs=await audit_repo.count(),
        cuad_index_built=bool(index and index.is_built),
        cuad_categories=len(index.category_centroids) if (index and index.is_built) else 0,
        database_type=db_type,
    )
