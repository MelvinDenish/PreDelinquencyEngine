"""
Authentication & Authorization module for PDI Engine.
Provides JWT bearer token auth, API key auth, and RBAC role enforcement.
"""
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta
from typing import Optional, Union

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from pydantic import BaseModel

# ──────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
JWT_ISSUER = "pdi-engine"

# ──────────────────────────────────────────────────
# Roles
# ──────────────────────────────────────────────────
ROLES = {
    "analyst":         {"score_single", "score_get", "explain", "health"},
    "risk_officer":    {"score_single", "score_batch", "score_get", "notify", "explain", "health"},
    "admin":           {"score_single", "score_batch", "score_get", "notify", "explain", "health", "metrics", "admin"},
    "read_only":       {"score_get", "health"},
    "service_account": {"score_single", "score_batch", "score_get", "notify", "health"},
}


class TokenPayload(BaseModel):
    sub: str          # username or service name
    role: str
    jti: Optional[str] = None  # JWT ID for revocation tracking
    exp: Optional[int] = None


# ──────────────────────────────────────────────────
# JWT (try to import jose; graceful fallback)
# ──────────────────────────────────────────────────
try:
    from jose import JWTError, jwt as jose_jwt
    _JOSE_AVAILABLE = True
except ImportError:
    _JOSE_AVAILABLE = False


def _check_jwt_available():
    if not _JOSE_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="JWT library not installed. Run: pip install python-jose[cryptography]",
        )


def create_access_token(subject: str, role: str, expires_delta: Optional[timedelta] = None) -> str:
    """Create a signed JWT access token."""
    _check_jwt_available()
    if not JWT_SECRET_KEY:
        raise RuntimeError("JWT_SECRET_KEY environment variable is not set.")
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES))
    payload = {
        "sub": subject,
        "role": role,
        "iss": JWT_ISSUER,
        "jti": secrets.token_hex(16),
        "exp": expire,
    }
    return jose_jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> TokenPayload:
    """Verify and decode a JWT token."""
    _check_jwt_available()
    if not JWT_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service is not configured.",
        )
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jose_jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        sub: str = payload.get("sub")
        role: str = payload.get("role")
        if sub is None or role is None:
            raise credentials_exception
        if payload.get("iss") != JWT_ISSUER:
            raise credentials_exception
        return TokenPayload(sub=sub, role=role, jti=payload.get("jti"), exp=payload.get("exp"))
    except Exception:
        raise credentials_exception


# ──────────────────────────────────────────────────
# API Key auth
# ──────────────────────────────────────────────────
_api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


def hash_api_key(raw_key: str) -> str:
    """SHA-256 hash of an API key for storage comparison."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


def verify_api_key_against_db(raw_key: str):
    """
    Verify an API key by comparing its hash against the database.
    Returns (service_name, role) if valid, raises HTTPException otherwise.
    Falls back gracefully if DB is unavailable.
    """
    import psycopg2
    from config.settings import PostgresConfig

    key_hash = hash_api_key(raw_key)
    try:
        conn = psycopg2.connect(
            host=PostgresConfig.HOST, port=PostgresConfig.PORT,
            user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
            dbname=PostgresConfig.DB, connect_timeout=3,
        )
        cursor = conn.cursor()
        cursor.execute(
            """SELECT service_name, role FROM api_keys
               WHERE key_hash = %s AND revoked = FALSE
               AND (expires_at IS NULL OR expires_at > NOW())""",
            (key_hash,),
        )
        row = cursor.fetchone()
        # Update last_used_at
        if row:
            cursor.execute(
                "UPDATE api_keys SET last_used_at = NOW() WHERE key_hash = %s",
                (key_hash,),
            )
            conn.commit()
        cursor.close()
        conn.close()
        if not row:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or revoked API key.",
                headers={"WWW-Authenticate": "X-API-Key"},
            )
        return TokenPayload(sub=row[0], role=row[1])
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service temporarily unavailable.",
        )


# ──────────────────────────────────────────────────
# FastAPI dependency factories
# ──────────────────────────────────────────────────
_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)


async def get_current_user(
    bearer_token: Optional[str] = Depends(_oauth2_scheme),
    api_key: Optional[str] = Depends(_api_key_scheme),
) -> TokenPayload:
    """
    Accepts either a JWT bearer token OR an X-API-Key header.
    At least one must be provided.
    """
    if bearer_token:
        return verify_token(bearer_token)
    if api_key:
        return verify_api_key_against_db(api_key)
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Provide Bearer token or X-API-Key header.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_role(*required_roles: str):
    """
    RBAC dependency factory.
    Usage: Depends(require_role("analyst", "risk_officer", "admin"))
    """
    async def _check(current_user: TokenPayload = Depends(get_current_user)) -> TokenPayload:
        if current_user.role not in required_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{current_user.role}' is not permitted to perform this action.",
            )
        return current_user
    return _check


# ──────────────────────────────────────────────────
# Password hashing (for /auth/token login)
# ──────────────────────────────────────────────────
try:
    from passlib.context import CryptContext
    _pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    _PASSLIB_AVAILABLE = True
except ImportError:
    _PASSLIB_AVAILABLE = False


def verify_password(plain_password: str, hashed_password: str) -> bool:
    if _PASSLIB_AVAILABLE:
        return _pwd_context.verify(plain_password, hashed_password)
    # Fallback: compare SHA-256 (less secure, but avoids hard crash if passlib missing)
    return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password


def get_password_hash(password: str) -> str:
    if _PASSLIB_AVAILABLE:
        return _pwd_context.hash(password)
    return hashlib.sha256(password.encode()).hexdigest()


def authenticate_user_db(username: str, password: str) -> Optional[TokenPayload]:
    """
    Verify username/password against the users table in PostgreSQL.
    Returns TokenPayload on success, None on failure.
    """
    import psycopg2
    from config.settings import PostgresConfig

    try:
        conn = psycopg2.connect(
            host=PostgresConfig.HOST, port=PostgresConfig.PORT,
            user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
            dbname=PostgresConfig.DB, connect_timeout=3,
        )
        cursor = conn.cursor()
        cursor.execute(
            "SELECT username, password_hash, role FROM users WHERE username = %s AND is_active = TRUE",
            (username,),
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if not row:
            return None
        stored_hash = row[1]
        role = row[2]
        if not verify_password(password, stored_hash):
            return None
        return TokenPayload(sub=username, role=role)
    except Exception:
        return None


def generate_api_key() -> tuple[str, str]:
    """
    Generate a new API key.
    Returns (raw_key, key_hash) — store only the hash in the DB.
    """
    raw_key = "pdi_" + secrets.token_urlsafe(32)
    return raw_key, hash_api_key(raw_key)
