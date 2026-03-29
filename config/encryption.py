"""
Field-level PII encryption for PDI Engine.
Uses Fernet symmetric encryption (AES-128-CBC + HMAC-SHA256).
Key is loaded from environment variable PII_ENCRYPTION_KEY (Azure Key Vault in production).
"""
import base64
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_fernet = None


def _get_fernet():
    """Lazy-initialize Fernet with the key from environment."""
    global _fernet
    if _fernet is not None:
        return _fernet

    try:
        from cryptography.fernet import Fernet
    except ImportError:
        raise RuntimeError(
            "cryptography package is required for PII encryption. "
            "Run: pip install cryptography"
        )

    key = os.getenv("PII_ENCRYPTION_KEY")
    if not key:
        # In development/demo mode without a key, log a warning and return None
        logger.warning(
            "[Encryption] PII_ENCRYPTION_KEY is not set. "
            "PII fields will be stored unencrypted. "
            "This is NOT acceptable in production."
        )
        return None

    # Accept both raw base64 keys and URL-safe base64 keys
    try:
        _fernet = Fernet(key.encode() if isinstance(key, str) else key)
    except Exception:
        # Try to pad and fix the key
        try:
            padded = key + "=" * (4 - len(key) % 4)
            raw = base64.urlsafe_b64decode(padded)
            _fernet = Fernet(base64.urlsafe_b64encode(raw))
        except Exception as e:
            raise RuntimeError(f"Invalid PII_ENCRYPTION_KEY format: {e}")

    return _fernet


def encrypt_pii(value: Optional[str]) -> Optional[str]:
    """
    Encrypt a PII string value.
    Returns the encrypted ciphertext as a string, or None if value is None.
    Falls back to returning the original value if encryption key is not configured
    (with a warning — only acceptable in development).
    """
    if value is None:
        return None
    f = _get_fernet()
    if f is None:
        return value  # development fallback
    return f.encrypt(str(value).encode()).decode()


def decrypt_pii(ciphertext: Optional[str]) -> Optional[str]:
    """
    Decrypt a PII ciphertext.
    Returns the original plaintext string, or None if ciphertext is None.
    """
    if ciphertext is None:
        return None
    f = _get_fernet()
    if f is None:
        return ciphertext  # development fallback
    try:
        return f.decrypt(ciphertext.encode()).decode()
    except Exception:
        # Could be an unencrypted legacy value — return as-is
        logger.warning("[Encryption] Failed to decrypt value — may be unencrypted legacy data.")
        return ciphertext


def generate_encryption_key() -> str:
    """
    Generate a new Fernet key for use as PII_ENCRYPTION_KEY.
    Run this once during setup: python -c "from config.encryption import generate_encryption_key; print(generate_encryption_key())"
    Store the result in Azure Key Vault as secret 'pdi-pii-encryption-key'.
    """
    from cryptography.fernet import Fernet
    return Fernet.generate_key().decode()
