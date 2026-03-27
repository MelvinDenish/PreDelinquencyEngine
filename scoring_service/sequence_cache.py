"""
Sequence Cache — TFT + LSTM Inference Optimisation (M7)
Precomputes and caches encoded sequence representations in Redis
so real-time scoring only processes the latest timestep.
"""
import json
import logging
import pickle
from typing import Optional, Tuple

import numpy as np
import redis

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import RedisConfig

logger = logging.getLogger(__name__)

SEQUENCE_PREFIX = "seq_cache:"
ATTENTION_PREFIX = "tft_attn:"
HIDDEN_PREFIX = "lstm_hidden:"
TTL_SECONDS = 23 * 3600  # 23 hours — refreshed by daily batch job


class SequenceCache:
    """
    Caches precomputed sequences for TFT and LSTM models.
    - TFT: caches the full 29-day encoded sequence tensor
    - LSTM: caches (hidden_state, cell_state) tuples
    - Also caches TFT attention weights for dashboard display
    """

    def __init__(self):
        self.redis_client = redis.Redis(
            host=RedisConfig.HOST,
            port=RedisConfig.PORT,
            db=RedisConfig.DB,
        )

    # ─────────────────────────────────────────────
    # TFT Sequence Cache
    # ─────────────────────────────────────────────
    def cache_tft_sequence(self, customer_id: str, sequence: np.ndarray,
                            static_features: np.ndarray) -> None:
        """Cache 29-day TFT sequence for a customer."""
        key = f"{SEQUENCE_PREFIX}{customer_id}"
        payload = pickle.dumps({
            "sequence": sequence,        # (29, n_temporal)
            "static": static_features,   # (n_static,)
        })
        self.redis_client.setex(key, TTL_SECONDS, payload)

    def get_tft_sequence(self, customer_id: str) -> Optional[dict]:
        """Retrieve cached TFT sequence."""
        key = f"{SEQUENCE_PREFIX}{customer_id}"
        data = self.redis_client.get(key)
        if data:
            return pickle.loads(data)
        return None

    # ─────────────────────────────────────────────
    # LSTM Hidden State Cache
    # ─────────────────────────────────────────────
    def cache_lstm_hidden_state(self, customer_id: str,
                                 hidden_state: np.ndarray,
                                 cell_state: np.ndarray) -> None:
        """Cache LSTM hidden state for a customer."""
        key = f"{HIDDEN_PREFIX}{customer_id}"
        payload = pickle.dumps({
            "hidden": hidden_state,
            "cell": cell_state,
        })
        self.redis_client.setex(key, TTL_SECONDS, payload)

    def get_lstm_hidden_state(self, customer_id: str) -> Optional[dict]:
        """Retrieve cached LSTM hidden state."""
        key = f"{HIDDEN_PREFIX}{customer_id}"
        data = self.redis_client.get(key)
        if data:
            return pickle.loads(data)
        return None

    # ─────────────────────────────────────────────
    # TFT Attention Weights (for dashboard)
    # ─────────────────────────────────────────────
    def cache_attention_weights(self, customer_id: str,
                                  weights: np.ndarray) -> None:
        """Cache TFT attention weights for dashboard display."""
        key = f"{ATTENTION_PREFIX}{customer_id}"
        payload = json.dumps(weights.tolist())
        self.redis_client.setex(key, TTL_SECONDS, payload)

    def get_attention_weights(self, customer_id: str) -> Optional[list]:
        """Retrieve cached TFT attention weights."""
        key = f"{ATTENTION_PREFIX}{customer_id}"
        data = self.redis_client.get(key)
        if data:
            return json.loads(data)
        return None

    # ─────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────
    def clear_customer_cache(self, customer_id: str) -> None:
        """Clear all cached data for a customer."""
        for prefix in [SEQUENCE_PREFIX, HIDDEN_PREFIX, ATTENTION_PREFIX]:
            self.redis_client.delete(f"{prefix}{customer_id}")

    def get_cached_customer_count(self) -> int:
        """Count how many customers have cached sequences."""
        keys = self.redis_client.keys(f"{SEQUENCE_PREFIX}*")
        return len(keys) if keys else 0
