"""
auth/dependencies.py
====================
FastAPI dependency injection for authentication and authorisation.

Usage:
    @router.get("/protected")
    async def endpoint(user: CurrentUser):
        ...

    @router.post("/admin")
    async def admin_only(user: CurrentUser = Depends(require_role("admin"))):
        ...
"""
from __future__ import annotations

import logging
from typing import Annotated, Callable

from fastapi import Cookie, Depends, HTTPException, Security, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from auth.jwt_handler import decode_access_token
from database.repositories import UserRepository

logger = logging.getLogger(__name__)
bearer_scheme = HTTPBearer(auto_error=False)


# ------------------------------------------------------------------ #
# Current user model                                                   #
# ------------------------------------------------------------------ #

class TokenPayload:
    """Lightweight user context extracted from JWT — no DB hit needed."""
    def __init__(self, user_id: str, email: str, role: str):
        self.user_id = user_id
        self.email = email
        self.role = role

    def __repr__(self):
        return f"<TokenPayload user={self.email} role={self.role}>"


# ------------------------------------------------------------------ #
# Core dependency                                                       #
# ------------------------------------------------------------------ #

async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> TokenPayload:
    """
    FastAPI dependency: extract and validate JWT from Authorization header.
    Raises 401 if token is missing or invalid.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if credentials is None:
        raise credentials_exception

    try:
        from jose import JWTError
        payload = decode_access_token(credentials.credentials)
    except Exception:
        raise credentials_exception

    user_id = payload.get("sub")
    email = payload.get("email")
    role = payload.get("role")

    if not user_id or not email or not role:
        raise credentials_exception

    request.state.user_id = user_id
    request.state.user_email = email
    request.state.user_role = role

    return TokenPayload(user_id=user_id, email=email, role=role)


# Convenience type alias
CurrentUser = Annotated[TokenPayload, Depends(get_current_user)]


# ------------------------------------------------------------------ #
# Role-based access control                                            #
# ------------------------------------------------------------------ #

ROLE_HIERARCHY = {
    "viewer": 0,
    "legal_counsel": 1,
    "admin": 2,
}


def require_role(*allowed_roles: str) -> Callable:
    """
    Dependency factory: requires the current user to have one of the
    specified roles (or a higher role in the hierarchy).

    Example:
        Depends(require_role("legal_counsel"))  # allows legal_counsel + admin
        Depends(require_role("admin"))          # admin only
    """
    min_level = min(ROLE_HIERARCHY.get(r, 0) for r in allowed_roles)

    async def _check(user: CurrentUser) -> TokenPayload:
        user_level = ROLE_HIERARCHY.get(user.role, -1)
        if user_level < min_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {' or '.join(allowed_roles)}.",
            )
        return user

    return _check


def require_admin(user: CurrentUser = Depends(require_role("admin"))) -> TokenPayload:
    """Shorthand: admin-only dependency."""
    return user


def require_counsel(user: CurrentUser = Depends(require_role("legal_counsel"))) -> TokenPayload:
    """Shorthand: legal counsel or admin dependency."""
    return user
