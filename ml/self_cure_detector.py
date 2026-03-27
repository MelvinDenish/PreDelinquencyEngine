# pyre-ignore-all-errors
"""
Self-Cure Detection — P7
Identifies customers who recovered WITHOUT intervention (spontaneous recovery).

Value: Two direct benefits:
1. Training signal: Self-cured customers are strong negative examples — they
   look stressed but recover. Adding them as "0" labels prevents false positives.
2. Avoid over-intervention: If a customer has a high self-cure propensity score,
   a costly RM call may be unnecessary — a light touch is enough.
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
logger = logging.getLogger(__name__)


class SelfCureDetector:
    """
    Detects unprompted customer recovery after entering watch/critical tier.

    A self-cure event is when:
    - Customer enters watch/critical tier
    - No intervention was sent (or it was in holdout)
    - Customer's risk score returns to stable within 45 days

    Common patterns found in self-curers:
    - Delayed salary credit followed by large credit (payroll caught up)
    - One-time expense spike (medical/wedding) that normalised
    - Seasonal income dip (gig workers, agriculture) that recovered
    """

    RECOVERY_WINDOW_DAYS = 45
    STABLE_THRESHOLD = 0.40  # Score below this = stable

    def __init__(self):
        self.self_cure_model = None
        self.self_cure_rate_by_segment = {}
        self._patterns = []

    def identify_self_cures(self, risk_scores_df: pd.DataFrame,
                            interventions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify self-cure events from historical score data.

        Args:
            risk_scores_df:  DataFrame with columns: customer_id, risk_score,
                             risk_tier, scored_at
            interventions_df: DataFrame with columns: customer_id, sent_at, status

        Returns:
            DataFrame of self-cure events with customer_id, entry_date,
            recovery_date, duration_days, had_intervention
        """
        events = []

        # Find customers who entered watch/critical
        watch_entries = risk_scores_df[
            risk_scores_df["risk_tier"].isin(["watch", "critical"])
        ].sort_values("scored_at")

        # Group by customer and find first entry to watch tier
        first_watch = (
            watch_entries.groupby("customer_id")["scored_at"].min().reset_index()
        )
        first_watch.columns = ["customer_id", "watch_entry_date"]

        for _, row in first_watch.iterrows():
            cid = row["customer_id"]
            entry_date = pd.to_datetime(row["watch_entry_date"])
            recovery_window_end = entry_date + pd.Timedelta(days=self.RECOVERY_WINDOW_DAYS)

            # Check if customer had an intervention in this window
            if interventions_df is not None and not interventions_df.empty:
                cust_interventions = interventions_df[
                    (interventions_df["customer_id"] == cid) &
                    (pd.to_datetime(interventions_df["sent_at"]) >= entry_date) &
                    (pd.to_datetime(interventions_df["sent_at"]) <= recovery_window_end)
                ]
                had_intervention = len(cust_interventions) > 0
            else:
                had_intervention = False

            # Check if score returned to stable
            later_scores = risk_scores_df[
                (risk_scores_df["customer_id"] == cid) &
                (pd.to_datetime(risk_scores_df["scored_at"]) > entry_date) &
                (pd.to_datetime(risk_scores_df["scored_at"]) <= recovery_window_end)
            ]

            if later_scores.empty:
                continue

            stable_scores = later_scores[
                later_scores["risk_score"] < self.STABLE_THRESHOLD
            ]

            if stable_scores.empty:
                continue

            recovery_date = pd.to_datetime(stable_scores["scored_at"].min())
            duration_days = (recovery_date - entry_date).days

            events.append({
                "customer_id": cid,
                "watch_entry_date": entry_date,
                "recovery_date": recovery_date,
                "duration_days": duration_days,
                "had_intervention": had_intervention,
                "is_self_cure": not had_intervention,
            })

        result = pd.DataFrame(events)
        if not result.empty:
            self_cures = result[result["is_self_cure"]]
            logger.info(
                f"[SelfCure] Found {len(result)} recovery events, "
                f"{len(self_cures)} self-cures ({100*len(self_cures)/max(1,len(result)):.1f}%)"
            )
        return result

    def compute_self_cure_rate_by_segment(
        self, events_df: pd.DataFrame, customers_df: pd.DataFrame
    ) -> Dict[str, float]:
        """Compute self-cure rate per customer segment."""
        if events_df.empty or customers_df.empty:
            return {}

        merged = events_df.merge(
            customers_df[["customer_id", "employment_type"]],
            on="customer_id", how="left"
        )

        rates = {}
        for segment, group in merged.groupby("employment_type"):
            self_cures = group["is_self_cure"].sum()
            total = len(group)
            rates[segment] = round(self_cures / max(1, total), 3)

        self.self_cure_rate_by_segment = rates
        logger.info(f"[SelfCure] Rates by segment: {rates}")
        return rates

    def get_self_cure_propensity(self, customer: Dict) -> float:
        """
        Estimate self-cure propensity for a customer based on segment rates
        and historical patterns. Higher = more likely to self-cure.
        """
        segment = customer.get("employment_type", "salaried_private")
        base_rate = self.self_cure_rate_by_segment.get(segment, 0.20)

        # Boost if it looks like a temp spike
        adjustments = 0.0

        # Salary-delay but otherwise stable history → likely late payroll, self-cures
        if customer.get("salary_delay_days", 0) > 0 and customer.get("dti_ratio", 0.5) < 0.4:
            adjustments += 0.10

        # Single recent spike in discretionary spend (medical/event)
        if (customer.get("spend_volatility_3m", 0) > 0.5 and
                customer.get("salary_delay_days", 0) == 0):
            adjustments += 0.08

        # Long tenure customers statistically self-cure more often
        tenure = customer.get("tenure_months", 0)
        if tenure > 60:
            adjustments += 0.05
        elif tenure > 24:
            adjustments += 0.02

        return min(0.95, base_rate + adjustments)
