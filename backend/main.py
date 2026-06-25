"""
ContractIQ — FastAPI Backend
============================
Entry point. Builds the CUAD index at startup then mounts all routers.

Run with:
    uvicorn main:app --reload --port 8000

The CUAD dataset is expected at:
    ../webapp/CUAD_v1/master_clauses.csv
(relative to this file's directory)

Override with env var:
    CUAD_CSV_PATH=C:/path/to/master_clauses.csv
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import cuad_index as cuad_idx
from routers import assistant, contracts

# ------------------------------------------------------------------ #
# Logging                                                              #
# ------------------------------------------------------------------ #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("contractiq")

# ------------------------------------------------------------------ #
# CUAD CSV path resolution                                             #
# ------------------------------------------------------------------ #

def _resolve_cuad_csv() -> str:
    # 1. Environment variable override
    env_path = os.environ.get("CUAD_CSV_PATH", "")
    if env_path and Path(env_path).exists():
        return env_path

    # 2. Relative to this file: ../webapp/CUAD_v1/master_clauses.csv
    here = Path(__file__).parent.resolve()
    candidate = here.parent / "webapp" / "CUAD_v1" / "master_clauses.csv"
    if candidate.exists():
        return str(candidate)

    # 3. Sibling directory search
    for search in [here.parent, here]:
        for p in search.rglob("master_clauses.csv"):
            return str(p)

    raise FileNotFoundError(
        "Cannot locate master_clauses.csv. "
        "Set the CUAD_CSV_PATH environment variable to its absolute path."
    )


def _resolve_chroma_dir() -> str:
    env_path = os.environ.get("CHROMA_DIR", "")
    if env_path:
        return env_path
    here = Path(__file__).parent.resolve()
    return str(here / ".chroma_store")


# ------------------------------------------------------------------ #
# Lifespan (startup / shutdown)                                        #
# ------------------------------------------------------------------ #

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        csv_path = _resolve_cuad_csv()
        chroma_dir = _resolve_chroma_dir()
        logger.info("CUAD CSV: %s", csv_path)
        logger.info("ChromaDB dir: %s", chroma_dir)
        cuad_idx.init_index(csv_path=csv_path, chroma_dir=chroma_dir)
        logger.info("✅ ContractIQ backend ready")
    except FileNotFoundError as exc:
        logger.error("❌ %s", exc)
        logger.warning(
            "Starting without CUAD index — clause classification will return defaults."
        )
        # Create a stub index that returns safe defaults
        stub = cuad_idx.CUADIndex(csv_path="", chroma_persist_dir="")
        stub.is_built = True
        cuad_idx._instance = stub

    yield  # Application runs here

    # Shutdown (nothing to clean up for in-memory store)
    logger.info("ContractIQ backend shutting down")


# ------------------------------------------------------------------ #
# App                                                                  #
# ------------------------------------------------------------------ #

app = FastAPI(
    title="ContractIQ API",
    description=(
        "CUAD-powered contract analysis backend. "
        "Classifies clauses against 41 legal categories, "
        "scores risk, and answers questions via RAG."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the Vite dev server and any localhost origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(contracts.router)
app.include_router(assistant.router)


@app.get("/", tags=["health"])
async def root():
    index = cuad_idx.get_index() if cuad_idx._instance else None
    return {
        "status": "ok",
        "service": "ContractIQ API",
        "cuad_index_built": bool(index and index.is_built),
        "categories_indexed": len(index.category_centroids) if (index and index.is_built) else 0,
    }


@app.get("/api/health", tags=["health"])
@app.get("/health", tags=["health"])
async def health():
    return {"status": "healthy"}
