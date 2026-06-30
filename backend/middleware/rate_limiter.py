"""
middleware/rate_limiter.py
==========================
Rate limiting configuration using SlowAPI (a FastAPI-compatible port of Flask-Limiter).

Limits:
  - /api/v1/auth/login  → 5 requests/minute  (brute-force protection)
  - /api/v1/contracts/analyze → 20/hour      (expensive AI operation)
  - /api/v1/assistant/* → 30/minute
  - Everything else     → 200/minute
"""
from __future__ import annotations

import logging

from fastapi import Request, Response
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _get_key(request: Request) -> str:
    """Rate limit by authenticated user ID if available, else by IP."""
    user_id = getattr(request.state, "user_id", None)
    if user_id:
        return f"user:{user_id}"
    return get_remote_address(request)


# SlowAPI limiter instance — attach to app via app.state.limiter
limiter = Limiter(key_func=_get_key, default_limits=[settings.RATE_LIMIT_DEFAULT])


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """Custom 429 response with Retry-After header."""
    retry_after = getattr(exc, "retry_after", 60)
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Too many requests. Please slow down.",
            "retry_after_seconds": retry_after,
        },
        headers={"Retry-After": str(retry_after)},
    )
