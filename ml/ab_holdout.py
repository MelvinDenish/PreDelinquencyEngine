# pyre-ignore-all-errors
"""
A/B Holdout Framework — P4
Assigns a control group to measure true intervention lift.

Value: Without a holdout group you cannot prove ROI.
       "82% of intervened customers paid" is meaningless if 79% would have paid anyway.
       This gives you: lift = treated_payment_rate - holdout_payment_rate
       and the statistical confidence to act on it.
"""
import os
import sys
import hashlib
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional
import psycopg2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import PostgresConfig

logger = logging.getLogger(__name__)


class ABHoldout:
    """
    Deterministic holdout assignment using customer_id hash.
    10% of watch/critical customers are silently held out — no intervention sent.
    Outcome-resolution DAG tracks both groups.

    Deterministic: same customer_id always gets same assignment (stable partition).
    """

    HOLDOUT_FRACTION = 0.10  # 10% control group

    def __init__(self, holdout_fraction: float = None):
        self.holdout_fraction = holdout_fraction or self.HOLDOUT_FRACTION

    def is_holdout(self, customer_id: str, experiment_id: str = "default") -> bool:
        """
        Deterministically assign customer to holdout based on hash.
        Stable across runs — same customer always in same cohort.
        """
        # Hash customer_id + experiment_id to get stable random-like assignment
        hash_input = f"{customer_id}:{experiment_id}"
        hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = (hash_val % 100) / 100.0
        return bucket < self.holdout_fraction

    def get_assignment(self, customer_id: str, experiment_id: str = "default") -> Dict:
        """Returns assignment dict for scoring service to use."""
        holdout = self.is_holdout(customer_id, experiment_id)
        return {
            "experiment_id": experiment_id,
            "group": "control" if holdout else "treated",
            "suppress_intervention": holdout,
        }

    def save_assignment(self, customer_id: str, risk_tier: str,
                        experiment_id: str = "default"):
        """Persist holdout assignment to PostgreSQL for later lift calculation."""
        try:
            conn = psycopg2.connect(
                host=PostgresConfig.HOST, port=PostgresConfig.PORT,
                user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
                dbname=PostgresConfig.DB,
            )
            cur = conn.cursor()
            group = "control" if self.is_holdout(customer_id, experiment_id) else "treated"
            cur.execute("""
                INSERT INTO ab_holdout_assignments
                    (customer_id, experiment_id, group_name, risk_tier, assigned_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (customer_id, experiment_id) DO NOTHING
            """, (customer_id, experiment_id, group, risk_tier))
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            logger.warning(f"[ABHoldout] Could not save assignment: {e}")

    @staticmethod
    def compute_lift(
        treated_outcomes: pd.Series,
        control_outcomes: pd.Series,
    ) -> Dict:
        """
        Compute intervention lift with two-proportion z-test.

        Args:
            treated_outcomes: Binary series (1=paid/recovered, 0=defaulted)
            control_outcomes: Binary series for holdout group

        Returns:
            dict with lift, p_value, confidence_interval, is_significant
        """
        from scipy import stats as scipy_stats

        n_t = len(treated_outcomes)
        n_c = len(control_outcomes)

        if n_t < 10 or n_c < 10:
            return {
                "lift": None,
                "treated_rate": treated_outcomes.mean() if n_t > 0 else None,
                "control_rate": control_outcomes.mean() if n_c > 0 else None,
                "p_value": None,
                "is_significant": False,
                "note": f"Insufficient data (treated={n_t}, control={n_c})",
            }

        treated_rate = treated_outcomes.mean()
        control_rate = control_outcomes.mean()
        lift = treated_rate - control_rate

        # Two-proportion z-test
        count = np.array([treated_outcomes.sum(), control_outcomes.sum()])
        nobs = np.array([n_t, n_c])
        _, p_value = scipy_stats.proportions_ztest(count, nobs)

        # 95% confidence interval on lift
        se = np.sqrt(
            treated_rate * (1 - treated_rate) / n_t +
            control_rate * (1 - control_rate) / n_c
        )
        ci_lower = lift - 1.96 * se
        ci_upper = lift + 1.96 * se

        return {
            "lift": round(float(lift), 4),
            "treated_rate": round(float(treated_rate), 4),
            "control_rate": round(float(control_rate), 4),
            "n_treated": n_t,
            "n_control": n_c,
            "p_value": round(float(p_value), 4),
            "ci_lower": round(float(ci_lower), 4),
            "ci_upper": round(float(ci_upper), 4),
            "is_significant": bool(p_value < 0.05),
            "effect_size_pct": round(float(lift * 100), 2),
        }
