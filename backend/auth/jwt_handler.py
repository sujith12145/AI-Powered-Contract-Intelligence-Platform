"""
auth/jwt_handler.py
===================
JWT access token creation, decoding, and refresh token management.

Access tokens:  JWT signed with HS256 (or RS256 if keys provided), 15-min TTL.
Refresh tokens: Opaque random token stored in Redis (or in-memory fallback),
                7-day TTL. Rotated on every use.
"""
from __future__ import annotations

import logging
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from jose import JWTError, jwt

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ------------------------------------------------------------------ #
# In-memory fallback token store (used when Redis is not configured) #
# ------------------------------------------------------------------ #
_mem_refresh_store: Dict[str, dict] = {}


# ------------------------------------------------------------------ #
# Access Token                                                         #
# ------------------------------------------------------------------ #

def create_access_token(
    user_id: str,
    email: str,
    role: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Return a signed JWT access token."""
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "iat": datetime.now(timezone.utc),
        "exp": expire,
        "type": "access",
    }
    key = settings.JWT_PRIVATE_KEY or settings.SECRET_KEY
    algorithm = "RS256" if settings.JWT_PRIVATE_KEY else settings.JWT_ALGORITHM
    return jwt.encode(payload, key, algorithm=algorithm)


def decode_access_token(token: str) -> Dict:
    """
    Decode and validate a JWT access token.
    Raises JWTError if invalid or expired.
    """
    key = settings.JWT_PUBLIC_KEY or settings.SECRET_KEY
    algorithm = "RS256" if settings.JWT_PUBLIC_KEY else settings.JWT_ALGORITHM
    payload = jwt.decode(token, key, algorithms=[algorithm])
    if payload.get("type") != "access":
        raise JWTError("Not an access token")
    return payload


# ------------------------------------------------------------------ #
# Refresh Token                                                        #
# ------------------------------------------------------------------ #

def _get_redis():
    """Return Redis client if configured, else None."""
    if not settings.REDIS_URL:
        return None
    try:
        import redis  # type: ignore
        r = redis.from_url(settings.REDIS_URL, decode_responses=True)
        r.ping()
        return r
    except Exception:
        return None


def create_refresh_token(user_id: str, email: str, role: str) -> str:
    """
    Generate an opaque refresh token and persist it.
    Returns the raw token string.
    """
    token = secrets.token_urlsafe(48)
    data = {"user_id": user_id, "email": email, "role": role}
    ttl = settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400

    r = _get_redis()
    if r:
        r.setex(f"refresh_token:{token}", ttl, str(data))
    else:
        _mem_refresh_store[token] = {
            **data,
            "expires_at": time.time() + ttl,
        }
    return token


def validate_refresh_token(token: str) -> Optional[Dict]:
    """
    Validate a refresh token. Returns the stored user data or None.
    Does NOT consume/rotate the token — call rotate_refresh_token for that.
    """
    r = _get_redis()
    if r:
        raw = r.get(f"refresh_token:{token}")
        if not raw:
            return None
        import ast
        return ast.literal_eval(raw)
    else:
        entry = _mem_refresh_store.get(token)
        if not entry:
            return None
        if time.time() > entry.get("expires_at", 0):
            _mem_refresh_store.pop(token, None)
            return None
        return {k: v for k, v in entry.items() if k != "expires_at"}


def rotate_refresh_token(old_token: str, user_id: str, email: str, role: str) -> str:
    """
    Invalidate the old refresh token and issue a new one (rotation).
    """
    revoke_refresh_token(old_token)
    return create_refresh_token(user_id, email, role)


def revoke_refresh_token(token: str) -> None:
    """Invalidate a refresh token (logout / rotation)."""
    r = _get_redis()
    if r:
        r.delete(f"refresh_token:{token}")
    else:
        _mem_refresh_store.pop(token, None)
