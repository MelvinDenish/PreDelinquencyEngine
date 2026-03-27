# pyre-ignore-all-errors
"""
Household / Joint Account Risk Aggregation — P11
Household-level risk scoring when multiple accounts are linked.

Value: A customer looks fine individually — until you see their spouse
       missed 2 EMIs last month. Joint-income households share financial stress;
       the engine should see the household, not just the account.
       This prevents a whole class of false negatives where individual scores are stable
       but household stress is high.
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
logger = logging.getLogger(__name__)


class HouseholdRiskAggregator:
    """
    Aggregates risk signals across linked accounts in a household.

    Household linkage sources (in priority order):
    1. Explicit `household_id` field in customers table (set by bank during KYC)
    2. Shared address + same phone prefix (heuristic linking)
    3. Joint account flagged in account_type field

    Household risk model:
    - household_max_risk: max individual risk in household
    - household_mean_risk: average risk
    - household_has_critical: any member in critical tier
    - household_combined_dti: sum of all individual DTI ratios
    - household_income_coverage: total household income / total household debt obligations
    """

    def get_household_members(
        self, customer_id: str, engine
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve all household members linked to a given customer.
        Returns DataFrame or None if no household linkage.
        """
        try:
            df = pd.read_sql("""
                SELECT c.*,
                       r.risk_score, r.risk_tier, r.scored_at
                FROM customers c
                LEFT JOIN LATERAL (
                    SELECT risk_score, risk_tier, scored_at
                    FROM risk_scores rs
                    WHERE rs.customer_id = c.customer_id
                    ORDER BY scored_at DESC LIMIT 1
                ) r ON true
                WHERE c.household_id = (
                    SELECT household_id FROM customers WHERE customer_id = %s
                )
                AND c.customer_id != %s
                AND c.household_id IS NOT NULL
            """, engine, params=(customer_id, customer_id))
            return df if not df.empty else None
        except Exception as e:
            logger.debug(f"[Household] could not load members for {customer_id}: {e}")
            return None

    def compute_household_features(
        self,
        customer: Dict,
        members: Optional[pd.DataFrame],
    ) -> Dict[str, float]:
        """
        Compute household-aggregated risk features.

        Args:
            customer: dict of the primary customer's features
            members:  DataFrame of linked household members with risk scores

        Returns:
            Dict of household-level features to append to feature vector
        """
        cust_risk = float(customer.get("risk_score", 0.0) or 0.0)
        cust_dti = float(customer.get("dti_ratio", 0.0) or 0.0)
        cust_salary = float(customer.get("monthly_salary", 0.0) or 0.0)

        if members is None or members.empty:
            # No household data — return neutral features
            return {
                "household_max_risk": cust_risk,
                "household_mean_risk": cust_risk,
                "household_has_critical": int(cust_risk >= 0.7),
                "household_combined_dti": cust_dti,
                "household_size": 1,
                "household_income_diversity": 1.0,  # single earner
                "household_weakest_member_risk": cust_risk,
            }

        # All member risk scores
        member_risks = members["risk_score"].fillna(0.0).astype(float).tolist()
        all_risks = [cust_risk] + member_risks

        member_dtis = members["dti_ratio"].fillna(0.0).astype(float).tolist() if "dti_ratio" in members.columns else []
        all_dtis = [cust_dti] + member_dtis

        member_salaries = members["monthly_salary"].fillna(0.0).astype(float).tolist() if "monthly_salary" in members.columns else []
        all_salaries = [cust_salary] + member_salaries

        total_income = sum(all_salaries)
        total_debt_service = sum(
            s * d for s, d in zip(all_salaries, all_dtis)
        )
        combined_dti = total_debt_service / max(1.0, total_income)

        # Income diversity: coefficient of variation of household incomes
        # Low diversity = single earner = higher household risk
        if len(all_salaries) > 1 and np.std(all_salaries) > 0:
            income_cv = np.std(all_salaries) / max(1.0, np.mean(all_salaries))
            income_diversity = float(1.0 / (1.0 + income_cv))
        else:
            income_diversity = 1.0

        features = {
            "household_max_risk": round(max(all_risks), 4),
            "household_mean_risk": round(float(np.mean(all_risks)), 4),
            "household_has_critical": int(any(r >= 0.7 for r in all_risks)),
            "household_combined_dti": round(float(combined_dti), 4),
            "household_size": len(all_risks),
            "household_income_diversity": round(income_diversity, 4),
            "household_weakest_member_risk": round(max(member_risks) if member_risks else 0.0, 4),
        }

        if features["household_has_critical"] and cust_risk < 0.5:
            logger.info(
                f"[Household] Cross-household risk alert: customer risk={cust_risk:.2f} "
                f"but household_max={features['household_max_risk']:.2f}"
            )

        return features

    def get_household_risk_adjustment(
        self, individual_score: float, household_features: Dict
    ) -> float:
        """
        Compute adjusted risk score accounting for household stress.
        Only ever increases the score (conservative — never hides risk).

        Adjustment logic:
        - +0.05 if any household member is in critical tier
        - +0.03 if household combined DTI > 0.5
        - +0.04 if household max risk > individual + 0.2 (unseen stress)
        """
        adjustment = 0.0

        if household_features.get("household_has_critical", 0):
            adjustment += 0.05

        if household_features.get("household_combined_dti", 0) > 0.5:
            adjustment += 0.03

        if household_features.get("household_max_risk", 0) > individual_score + 0.20:
            adjustment += 0.04

        adjusted = min(1.0, individual_score + adjustment)
        return round(adjusted, 4)
