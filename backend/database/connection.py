"""
database/connection.py
======================
SQLAlchemy async engine + session factory.

Supports:
  - SQLite (default, zero config): sqlite+aiosqlite:///./contractiq.db
  - PostgreSQL (production):       postgresql+asyncpg://user:pass@host/db

Set DATABASE_URL env var to switch.
"""
from __future__ import annotations

import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Build engine kwargs — SQLite needs check_same_thread=False
connect_args = {}
if "sqlite" in settings.DATABASE_URL:
    connect_args["check_same_thread"] = False

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DB_ECHO,
    connect_args=connect_args,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


async def init_db() -> None:
    """Create all tables (idempotent — safe to call on every startup)."""
    from database import models as _  # noqa: F401 — ensure models are imported
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✅ Database tables ready (%s)", settings.DATABASE_URL.split("://")[0])

    # ---- SQLite column migration (idempotent) ----
    # Add any new columns to existing databases that pre-date these features.
    _migrations = [
        "ALTER TABLE users ADD COLUMN last_notification_check DATETIME",
        "ALTER TABLE users ADD COLUMN notify_needs_review BOOLEAN DEFAULT 1",
        "ALTER TABLE users ADD COLUMN notify_high_risk BOOLEAN DEFAULT 1",
        "ALTER TABLE users ADD COLUMN notify_overridden BOOLEAN DEFAULT 1",
    ]
    async with engine.begin() as conn:
        for stmt in _migrations:
            try:
                await conn.execute(__import__('sqlalchemy').text(stmt))
            except Exception:
                pass  # column already exists — safe to ignore


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yield an async DB session, auto-close on exit."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
