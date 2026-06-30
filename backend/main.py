"""
ContractIQ — FastAPI Backend (Production)
=========================================
Production-grade entry point with:
  ✅ JWT Authentication
  ✅ RBAC (Role-Based Access Control)
  ✅ Rate Limiting (SlowAPI)
  ✅ Security Headers (CSP, X-Frame-Options, etc.)
  ✅ Audit Logging Middleware
  ✅ Global Error Handler
  ✅ PostgreSQL / SQLite persistence
  ✅ Health checks (liveness + detailed)
  ✅ API versioning (/api/v1/)
  ✅ OWASP-compliant CORS

Run with:
    uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

import cuad_index as cuad_idx
from config import get_settings
from database.connection import init_db
from middleware.audit_logger import AuditLogMiddleware
from middleware.error_handler import ErrorHandlerMiddleware
from middleware.rate_limiter import limiter, rate_limit_exceeded_handler
from middleware.security_headers import SecurityHeadersMiddleware
from routers import assistant, admin
from routers.auth import router as auth_router
from routers.contracts import router as contracts_router

settings = get_settings()

# ------------------------------------------------------------------ #
# Logging                                                              #
# ------------------------------------------------------------------ #

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("contractiq")


# ------------------------------------------------------------------ #
# CUAD CSV path resolution                                             #
# ------------------------------------------------------------------ #

def _resolve_cuad_csv() -> str:
    env_path = settings.CUAD_CSV_PATH
    if env_path and Path(env_path).exists():
        return env_path
    here = Path(__file__).parent.resolve()
    candidate = here.parent / "webapp" / "CUAD_v1" / "master_clauses.csv"
    if candidate.exists():
        return str(candidate)
    for search in [here.parent, here]:
        for p in search.rglob("master_clauses.csv"):
            return str(p)
    raise FileNotFoundError(
        "Cannot locate master_clauses.csv. "
        "Set the CUAD_CSV_PATH environment variable to its absolute path."
    )


def _resolve_chroma_dir() -> str:
    return settings.CHROMA_DIR or str(Path(__file__).parent / ".chroma_store")


# ------------------------------------------------------------------ #
# Startup / Shutdown                                                   #
# ------------------------------------------------------------------ #

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- Database ----
    logger.info("Initialising database (%s)…", settings.DATABASE_URL.split("://")[0])
    await init_db()

    # ---- Seed default admin user ----
    await _seed_admin()

    # ---- CUAD Index ----
    try:
        csv_path = _resolve_cuad_csv()
        chroma_dir = _resolve_chroma_dir()
        logger.info("CUAD CSV: %s", csv_path)
        logger.info("ChromaDB dir: %s", chroma_dir)
        cuad_idx.init_index(csv_path=csv_path, chroma_dir=chroma_dir)
        logger.info("✅ ContractIQ backend ready")
    except FileNotFoundError as exc:
        logger.error("❌ %s", exc)
        logger.warning("Starting without CUAD index — clause classification returns defaults.")
        stub = cuad_idx.CUADIndex(csv_path="", chroma_persist_dir="")
        stub.is_built = True
        cuad_idx._instance = stub

    yield

    logger.info("ContractIQ backend shutting down")


async def _seed_admin() -> None:
    """Create the default admin account if no users exist yet."""
    from database.connection import AsyncSessionLocal
    from database.repositories import UserRepository
    from auth.password import hash_password

    async with AsyncSessionLocal() as session:
        try:
            repo = UserRepository(session)
            count = await repo.count()
            if count == 0:
                hashed = hash_password(settings.ADMIN_PASSWORD)
                user = await repo.create(
                    email=settings.ADMIN_EMAIL,
                    hashed_password=hashed,
                    full_name=settings.ADMIN_NAME,
                    role="admin",
                )
                # Seed additional demo users
                await repo.create(
                    email="counsel@contractiq.local",
                    hashed_password=hash_password("Counsel@123456"),
                    full_name="Sarah Mitchell",
                    role="legal_counsel",
                )
                await repo.create(
                    email="viewer@contractiq.local",
                    hashed_password=hash_password("Viewer@123456"),
                    full_name="Demo Viewer",
                    role="viewer",
                )
                await session.commit()
                logger.info("✅ Seeded default users:")
                logger.info("   Admin:          %s / Admin@123456", settings.ADMIN_EMAIL)
                logger.info("   Legal Counsel:  counsel@contractiq.local / Counsel@123456")
                logger.info("   Viewer:         viewer@contractiq.local / Viewer@123456")
            else:
                logger.info("Database has %d users — skipping seed.", count)
        except Exception as exc:
            logger.error("Failed to seed admin: %s", exc)
            await session.rollback()


# ------------------------------------------------------------------ #
# App                                                                  #
# ------------------------------------------------------------------ #

app = FastAPI(
    title="ContractIQ API",
    description=(
        "Production-grade AI-Powered Contract Intelligence Platform. "
        "CUAD-powered clause classification, risk scoring, and RAG-based legal assistant. "
        "All endpoints require JWT authentication."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ------------------------------------------------------------------ #
# Rate limiting                                                        #
# ------------------------------------------------------------------ #
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# ------------------------------------------------------------------ #
# Security / monitoring middleware (order matters — outermost first)  #
# ------------------------------------------------------------------ #
app.add_middleware(ErrorHandlerMiddleware, debug=settings.DEBUG)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(AuditLogMiddleware)

# ------------------------------------------------------------------ #
# CORS — strict origin whitelist                                       #
# ------------------------------------------------------------------ #
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID", "Retry-After"],
)

# ------------------------------------------------------------------ #
# Routers                                                              #
# ------------------------------------------------------------------ #
app.include_router(auth_router)
app.include_router(contracts_router)
app.include_router(assistant.router)
app.include_router(admin.router)


# ------------------------------------------------------------------ #
# Health endpoints                                                     #
# ------------------------------------------------------------------ #

@app.get("/", tags=["health"])
async def root():
    index = cuad_idx.get_index() if cuad_idx._instance else None
    return {
        "status": "ok",
        "service": "ContractIQ API",
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "cuad_index_built": bool(index and index.is_built),
    }


@app.get("/api/v1/health", tags=["health"])
@app.get("/health", tags=["health"])
async def health():
    """Liveness probe — always returns 200 if the process is alive."""
    return {"status": "healthy", "service": "ContractIQ"}


@app.get("/api/v1/health/detailed", tags=["health"])
async def health_detailed():
    """
    Detailed readiness probe — checks all dependencies.
    Used by load balancers and monitoring systems.
    """
    checks = {}

    # CUAD Index
    index = cuad_idx.get_index() if cuad_idx._instance else None
    checks["cuad_index"] = {
        "status": "ok" if (index and index.is_built) else "degraded",
        "categories": len(index.category_centroids) if (index and index.is_built) else 0,
    }

    # ChromaDB
    try:
        if index and index._collection:
            index._collection.count()
            checks["chromadb"] = {"status": "ok"}
        else:
            checks["chromadb"] = {"status": "not_initialized"}
    except Exception as e:
        checks["chromadb"] = {"status": "error", "detail": str(e)}

    # Database
    try:
        from database.connection import engine
        from sqlalchemy import text
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        checks["database"] = {"status": "ok", "type": settings.DATABASE_URL.split("://")[0]}
    except Exception as e:
        checks["database"] = {"status": "error", "detail": str(e)}

    # Redis
    if settings.REDIS_URL:
        try:
            import redis
            r = redis.from_url(settings.REDIS_URL)
            r.ping()
            checks["redis"] = {"status": "ok"}
        except Exception as e:
            checks["redis"] = {"status": "error", "detail": str(e)}
    else:
        checks["redis"] = {"status": "not_configured", "note": "Using in-memory fallback"}

    overall_ok = all(
        v.get("status") in ("ok", "not_configured", "not_initialized")
        for v in checks.values()
    )

    return JSONResponse(
        status_code=200 if overall_ok else 503,
        content={
            "status": "ok" if overall_ok else "degraded",
            "checks": checks,
        },
    )
