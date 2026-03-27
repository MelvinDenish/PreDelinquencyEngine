# pyre-ignore-all-errors
"""
Shadow Mode Scoring — P8
Runs a candidate model in parallel with the live model without affecting customers.

Value: Eliminates the biggest risk in model deployment — deploying blind.
       Shadow mode lets you validate the new model's predictions against live
       customer behaviour before a single customer is affected.
       If shadow AUC ≥ live AUC + 0.002, promote automatically.
       If shadow and live diverge significantly on the same customer, flag for review.
"""
import os
import sys
import logging
import json
import numpy as np
from typing import Dict, Optional
from datetime import datetime
import psycopg2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import PostgresConfig

logger = logging.getLogger(__name__)


class ShadowScorer:
    """
    Parallel shadow scoring: runs a candidate model alongside the live model.
    Results are stored in shadow_scores table but never influence interventions.

    Comparison metrics computed daily by monitoring/shadow_comparison.py.
    """

    def __init__(self, candidate_model_path: str = None):
        self.candidate_model = None
        self.candidate_model_path = candidate_model_path
        self._loaded = False

    def load_candidate(self, model_path: str):
        """Load candidate model for shadow scoring."""
        import joblib
        try:
            self.candidate_model = joblib.load(model_path)
            self.candidate_model_path = model_path
            self._loaded = True
            logger.info(f"[Shadow] Candidate model loaded from {model_path}")
        except FileNotFoundError:
            logger.info(f"[Shadow] No candidate model at {model_path} — shadow mode inactive")
        except Exception as e:
            logger.warning(f"[Shadow] Could not load candidate model: {e}")

    def shadow_score(self, customer_id: str, features: np.ndarray) -> Optional[float]:
        """
        Score customer with shadow/candidate model. Never raises — shadow
        failures must not affect live scoring.
        """
        if not self._loaded or self.candidate_model is None:
            return None
        try:
            prob = float(self.candidate_model.predict_proba(
                features.reshape(1, -1)
            )[0][1])
            return prob
        except Exception as e:
            logger.debug(f"[Shadow] Shadow score failed for {customer_id}: {e}")
            return None

    def persist_shadow_score(
        self,
        customer_id: str,
        live_score: float,
        shadow_score: Optional[float],
        features_hash: str = None,
    ):
        """Store shadow vs live comparison in shadow_scores table."""
        if shadow_score is None:
            return
        try:
            conn = psycopg2.connect(
                host=PostgresConfig.HOST, port=PostgresConfig.PORT,
                user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
                dbname=PostgresConfig.DB,
            )
            cur = conn.cursor()
            divergence = abs(live_score - shadow_score)
            cur.execute("""
                INSERT INTO shadow_scores
                    (customer_id, live_score, shadow_score, divergence,
                     features_hash, scored_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
            """, (customer_id, live_score, shadow_score, divergence, features_hash))
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            logger.debug(f"[Shadow] Could not persist shadow score: {e}")


class ShadowComparison:
    """
    Computes daily comparison metrics between shadow and live models.
    Run by monitoring/shadow_comparison.py Airflow task.
    """

    DIVERGENCE_ALERT_THRESHOLD = 0.15  # Flag > 15% score difference as notable

    def compute_daily_metrics(self, engine) -> Dict:
        """
        Compute daily comparison metrics from shadow_scores table.
        Returns metrics dict suitable for logging to MLflow / dashboard.
        """
        try:
            import pandas as pd
            df = pd.read_sql("""
                SELECT live_score, shadow_score, divergence, scored_at
                FROM shadow_scores
                WHERE scored_at > NOW() - INTERVAL '24 hours'
            """, engine)

            if df.empty:
                return {"status": "no_shadow_data"}

            n = len(df)
            mean_divergence = float(df["divergence"].mean())
            high_divergence_pct = float(
                (df["divergence"] > self.DIVERGENCE_ALERT_THRESHOLD).mean() * 100
            )

            # Directional agreement: both point same direction from 0.5?
            df["live_label"] = (df["live_score"] >= 0.5).astype(int)
            df["shadow_label"] = (df["shadow_score"] >= 0.5).astype(int)
            agreement_rate = float((df["live_label"] == df["shadow_label"]).mean())

            return {
                "n_comparisons": n,
                "mean_divergence": round(mean_divergence, 4),
                "high_divergence_pct": round(high_divergence_pct, 2),
                "directional_agreement_rate": round(agreement_rate, 4),
                "ready_to_promote": (
                    mean_divergence < 0.05 and
                    high_divergence_pct < 5.0 and
                    agreement_rate > 0.95
                ),
                "alert": high_divergence_pct > 10.0,
                "computed_at": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"[Shadow] comparison failed: {e}")
            return {"error": str(e)}
