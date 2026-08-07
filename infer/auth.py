# -*- coding: utf-8 -*-
# Written by GD Studio / Antigravity AI
# Date: 2026-08-07
#
# Authentication & Security Module for Embeat Admin Dashboard
# Supports ADMIN_PASSWORD verification, Session Token generation, and FastAPI auth middleware.

import os
import secrets
import time
from typing import Optional
from fastapi import Request, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials

# Default admin password can be customized via ADMIN_PASSWORD environment variable
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

# Session token storage: {token: expire_timestamp}
_VALID_TOKENS: dict[str, float] = {}

# Security schemes
api_key_header = APIKeyHeader(name="X-Admin-Token", auto_error=False)
http_bearer = HTTPBearer(auto_error=False)


def verify_password(password: str) -> bool:
    """Check if provided password matches ADMIN_PASSWORD."""
    if not password:
        return False
    return secrets.compare_digest(password.strip(), ADMIN_PASSWORD.strip())


def create_admin_token() -> str:
    """Generate a new secure admin token valid for 7 days."""
    token = secrets.token_hex(32)
    expire_at = time.time() + (7 * 86400)  # 7 days
    _VALID_TOKENS[token] = expire_at
    return token


def is_valid_token(token: str) -> bool:
    """Validate token and check expiration."""
    if not token or token not in _VALID_TOKENS:
        return False
    expire_at = _VALID_TOKENS[token]
    if time.time() > expire_at:
        del _VALID_TOKENS[token]
        return False
    return True


async def require_admin_auth(
    request: Request,
    token_header: Optional[str] = Depends(api_key_header),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(http_bearer)
) -> str:
    """
    FastAPI Dependency to intercept administrative endpoints.
    Checks X-Admin-Token header, Bearer token, or admin_token cookie.
    Raises HTTP 401 if unauthorized.
    """
    token = None
    if token_header:
        token = token_header
    elif credentials and credentials.credentials:
        token = credentials.credentials
    elif request and request.cookies.get("admin_token"):
        token = request.cookies.get("admin_token")

    if not token or not is_valid_token(token):
        raise HTTPException(
            status_code=401,
            detail="管理后台未授权访问。请输入正确管理员密码。"
        )
    return token
