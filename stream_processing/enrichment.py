"""
Merchant Risk Enrichment
Provides merchant risk score lookups from Redis for transaction enrichment.
"""
import redis as redis_lib
import logging
from typing import Dict, Optional, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import RedisConfig

logger = logging.getLogger(__name__)

# In-memory cache for merchant risk scores (reduces Redis calls)
_merchant_cache: Dict[str, Dict] = {}


def get_redis_client():
    """Get Redis client connection."""
    return redis_lib.Redis(
        host=RedisConfig.HOST,
        port=RedisConfig.PORT,
        db=RedisConfig.DB,
        decode_responses=True,
    )


def get_merchant_risk(merchant_category: str, redis_client=None) -> Tuple[float, str]:
    """
    Look up merchant risk score from Redis.
    Returns (risk_score, risk_category) tuple.
    Falls back to medium risk if not found.
    """
    # Check in-memory cache first
    if merchant_category in _merchant_cache:
        data = _merchant_cache[merchant_category]
        return float(data["risk_score"]), data["risk_category"]

    # Look up in Redis
    own_client = False
    if redis_client is None:
        redis_client = get_redis_client()
        own_client = True

    try:
        data = redis_client.hgetall(f"merchant_risk:{merchant_category}")
        if data:
            _merchant_cache[merchant_category] = data
            return float(data["risk_score"]), data["risk_category"]
        else:
            # Default: medium risk for unknown categories
            default = {"risk_score": "0.30", "risk_category": "medium"}
            _merchant_cache[merchant_category] = default
            return 0.30, "medium"
    finally:
        if own_client:
            redis_client.close()


def enrich_transaction(transaction: Dict, redis_client=None) -> Dict:
    """
    Enrich a transaction with merchant risk score and category.
    Returns the enriched transaction dict.
    """
    merchant_category = transaction.get("merchant_category", "unknown")
    risk_score, risk_category = get_merchant_risk(merchant_category, redis_client)

    enriched = transaction.copy()
    enriched["merchant_risk_score"] = risk_score
    enriched["risk_category"] = risk_category

    return enriched


def enrich_batch(transactions: list, redis_client=None) -> list:
    """Enrich a batch of transactions."""
    own_client = False
    if redis_client is None:
        redis_client = get_redis_client()
        own_client = True

    try:
        return [enrich_transaction(t, redis_client) for t in transactions]
    finally:
        if own_client:
            redis_client.close()
