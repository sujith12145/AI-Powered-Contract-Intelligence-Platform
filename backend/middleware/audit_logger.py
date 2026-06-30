"""
middleware/audit_logger.py
==========================
Starlette middleware that writes an audit log entry for every mutating
API request (POST/PUT/PATCH/DELETE). Read-only GETs are not logged
at the middleware level (individual routes can log as needed).

The audit log is written asynchronously after the response is sent.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger("contractiq.audit")

# Routes excluded from audit logging
_EXCLUDE_PATHS = {"/health", "/api/v1/health", "/", "/docs", "/openapi.json", "/redoc"}
_AUDIT_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


class AuditLogMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        # Only log mutating calls outside excluded paths
        if (
            request.method in _AUDIT_METHODS
            and request.url.path not in _EXCLUDE_PATHS
        ):
            # Extract user context from request state (set by auth dependency)
            user_id = getattr(request.state, "user_id", None)
            user_email = getattr(request.state, "user_email", None)
            user_role = getattr(request.state, "user_role", None)

            ip = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")
            ua = request.headers.get("User-Agent", "")

            # Log structured JSON to the audit logger
            log_entry = {
                "method": request.method,
                "path": request.url.path,
                "query": str(request.query_params),
                "status": response.status_code,
                "elapsed_ms": elapsed_ms,
                "user_id": user_id,
                "user_email": user_email,
                "user_role": user_role,
                "ip": ip,
                "ua": ua[:200],
            }
            logger.info(json.dumps(log_entry))

        return response
