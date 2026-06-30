"""
middleware/error_handler.py
===========================
Global exception handler — converts all unhandled exceptions into
safe, consistent JSON error responses. Never exposes stack traces
to clients in production.
"""
from __future__ import annotations

import logging
import traceback
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger("contractiq.errors")


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, debug: bool = False):
        super().__init__(app)
        self.debug = debug

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            logger.error(
                "Unhandled exception [%s] %s %s: %s",
                request_id,
                request.method,
                request.url.path,
                exc,
                exc_info=True,
            )
            body = {
                "detail": "An internal server error occurred.",
                "request_id": request_id,
            }
            if self.debug:
                body["debug"] = traceback.format_exc()
            return JSONResponse(status_code=500, content=body)


def add_request_id_header(request: Request, response: JSONResponse) -> JSONResponse:
    """Add X-Request-ID to every response for tracing."""
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        response.headers["X-Request-ID"] = request_id
    return response
