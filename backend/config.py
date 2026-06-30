"""
config.py
=========
Centralised configuration loaded from environment variables / .env file.
All secrets live here — never hardcode credentials in other modules.
"""
from __future__ import annotations

import secrets
from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # App                                                                  #
    # ------------------------------------------------------------------ #
    APP_NAME: str = "ContractIQ"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"  # development | staging | production

    # ------------------------------------------------------------------ #
    # Security                                                             #
    # ------------------------------------------------------------------ #
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_hex(32))

    # HMAC key used to sign every audit-log row hash.
    # Without this key an attacker with raw DB write access cannot recompute
    # valid hashes after tampering, even if they know the payload format.
    # Set AUDIT_LOG_HMAC_KEY=<64-char hex> in your .env / environment.
    # If not set a per-process random key is used (chain breaks on restart —
    # acceptable for development, not for production).
    AUDIT_LOG_HMAC_KEY: str = Field(default_factory=lambda: secrets.token_hex(32))

    # JWT — HS256 (simple, uses SECRET_KEY).
    # For production, switch to RS256 by supplying JWT_PRIVATE_KEY / JWT_PUBLIC_KEY.
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Optional RS256 keys (PEM strings). If set, overrides HS256.
    JWT_PRIVATE_KEY: str = ""
    JWT_PUBLIC_KEY: str = ""

    # ------------------------------------------------------------------ #
    # Database                                                             #
    # ------------------------------------------------------------------ #
    # SQLite by default (works without Docker).
    # Set DATABASE_URL=postgresql+asyncpg://user:pass@host/db for production.
    DATABASE_URL: str = "sqlite+aiosqlite:///./contractiq.db"
    DB_ECHO: bool = False  # Set True to log SQL

    # ------------------------------------------------------------------ #
    # Redis                                                                #
    # ------------------------------------------------------------------ #
    # Used for: refresh token store, rate-limit counters, contract cache.
    # Falls back gracefully to in-memory if REDIS_URL is empty.
    REDIS_URL: str = ""
    REDIS_TTL_SECONDS: int = 3600  # default cache TTL

    # ------------------------------------------------------------------ #
    # CORS                                                                 #
    # ------------------------------------------------------------------ #
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]

    # ------------------------------------------------------------------ #
    # Rate Limiting                                                        #
    # ------------------------------------------------------------------ #
    RATE_LIMIT_LOGIN: str = "5/minute"           # brute-force protection
    RATE_LIMIT_ANALYZE: str = "20/hour"          # expensive GPU operation
    RATE_LIMIT_ASSISTANT: str = "30/minute"      # AI assistant
    RATE_LIMIT_DEFAULT: str = "200/minute"       # general API

    # ------------------------------------------------------------------ #
    # File Uploads                                                         #
    # ------------------------------------------------------------------ #
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE_MB: int = 50

    # ------------------------------------------------------------------ #
    # CUAD / ChromaDB                                                      #
    # ------------------------------------------------------------------ #
    CUAD_CSV_PATH: str = ""
    CHROMA_DIR: str = "./.chroma_store"

    # ------------------------------------------------------------------ #
    # Admin bootstrap                                                      #
    # ------------------------------------------------------------------ #
    ADMIN_EMAIL: str = "admin@contractiq.local"
    ADMIN_PASSWORD: str = "Admin@123456"   # change after first login!
    ADMIN_NAME: str = "System Administrator"


@lru_cache()
def get_settings() -> Settings:
    """Return the cached settings singleton."""
    return Settings()
