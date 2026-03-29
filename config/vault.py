# pyre-ignore-all-errors
"""
Pre-Delinquency Intervention Engine - Azure Key Vault Integration

Retrieves secrets from Azure Key Vault in production, with local env-var
fallback for development.
"""
import os
import logging
from functools import lru_cache
from typing import Optional

from config.settings import _require_env

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vault client (lazy-initialised)
# ---------------------------------------------------------------------------
_client = None


def _get_client():
    """Return a cached SecretClient, creating it on first call."""
    global _client
    if _client is not None:
        return _client

    vault_url = os.getenv("AZURE_VAULT_URL")
    if not vault_url:
        logger.info("AZURE_VAULT_URL not set — vault integration disabled (using env vars).")
        return None

    from azure.keyvault.secrets import SecretClient
    from azure.identity import ManagedIdentityCredential, DefaultAzureCredential

    try:
        # Prefer Managed Identity in production (AKS workload identity)
        credential = ManagedIdentityCredential()
        _client = SecretClient(vault_url=vault_url, credential=credential)
        # Quick connectivity check
        _client.list_properties_of_secrets(max_page_size=1)
        logger.info("Connected to Azure Key Vault via ManagedIdentityCredential.")
    except Exception:
        logger.warning("ManagedIdentity unavailable — falling back to DefaultAzureCredential.")
        credential = DefaultAzureCredential()
        _client = SecretClient(vault_url=vault_url, credential=credential)

    return _client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@lru_cache(maxsize=64)
def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Retrieve a secret by name.

    Resolution order:
      1. Azure Key Vault (if AZURE_VAULT_URL is set)
      2. Environment variable
      3. Provided default
    """
    client = _get_client()
    if client is not None:
        try:
            secret = client.get_secret(name)
            return secret.value
        except Exception as exc:
            logger.warning("Vault lookup failed for '%s': %s — falling back to env.", name, exc)

    return os.getenv(name, default)


# Mapping of logical secret names to their env-var / vault key names
_PDI_SECRET_KEYS = {
    "POSTGRES_PASSWORD": "POSTGRES-PASSWORD",
    "REDIS_PASSWORD": "REDIS-PASSWORD",
    "JWT_SECRET_KEY": "JWT-SECRET-KEY",
    "PII_ENCRYPTION_KEY": "PII-ENCRYPTION-KEY",
    "AUDIT_HASH_SALT": "AUDIT-HASH-SALT",
    "REDIS_KEY_SECRET": "REDIS-KEY-SECRET",
    "GROQ_API_KEY": "GROQ-API-KEY",
    "TWILIO_AUTH_TOKEN": "TWILIO-AUTH-TOKEN",
}


def get_all_pdi_secrets() -> dict[str, Optional[str]]:
    """Retrieve all PDI secrets as a dict.

    Keys in the returned dict use the env-var style names (underscores).
    Vault keys use hyphens per Azure naming convention.
    """
    secrets: dict[str, Optional[str]] = {}
    client = _get_client()

    for env_name, vault_name in _PDI_SECRET_KEYS.items():
        if client is not None:
            try:
                secret = client.get_secret(vault_name)
                secrets[env_name] = secret.value
                continue
            except Exception as exc:
                logger.warning("Vault lookup failed for '%s': %s", vault_name, exc)

        secrets[env_name] = _require_env(env_name)

    return secrets
