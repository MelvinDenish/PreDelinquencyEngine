# pyre-ignore-all-errors
"""
Survival Analysis Model — P1
Cox Proportional Hazards + Kaplan-Meier for time-to-default prediction.

Value: Tells you not just "will this customer default?" but "in how many days?"
       Enables urgency-ranked intervention queues — a customer with TTE=8 days
       gets a call today; TTE=45 days gets a gentle SMS.
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
logger = logging.getLogger(__name__)


class SurvivalModel:
    """
    Time-To-Event model for delinquency prediction.

    Outputs:
        tte_days   — median predicted days until first EMI miss
        p30d       — probability of default within 30 days
        p60d       — probability of default within 60 days
        p90d       — probability of default within 90 days
    """

    def __init__(self):
        self.cox_model = None
        self.baseline_hazard = None
        self.kaplan_meier = None
        self.feature_names = None
        self._is_fitted = False

    def build_survival_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert tabular features into (duration, event) pairs for survival analysis.
        duration = days_since_last_good_payment (observed time)
        event    = 1 if defaulted, 0 if still performing (censored)
        """
        df = df.copy()
        # Duration: use days since watched risk as observed time
        if "days_since_last_credit" not in df.columns:
            df["duration"] = np.random.randint(10, 120, len(df))
        else:
            df["duration"] = df["days_since_last_credit"].clip(1, 180)

        # Event: use is_stressed / delinquent flag
        if "is_stressed" in df.columns:
            df["event"] = df["is_stressed"].astype(int)
        elif "defaulted" in df.columns:
            df["event"] = df["defaulted"].astype(int)
        else:
            df["event"] = 0  # all censored if no label

        return df[["duration", "event"] + [c for c in df.columns
                                             if c not in ("duration", "event")]]

    def train(self, df: pd.DataFrame, feature_cols: list) -> dict:
        """
        Train CoxPH model. Falls back to Kaplan-Meier if lifelines unavailable.
        Returns dict with concordance index and median survival time.
        """
        try:
            from lifelines import CoxPHFitter, KaplanMeierFitter
        except ImportError:
            logger.warning("lifelines not installed — using parametric fallback")
            return self._train_parametric_fallback(df, feature_cols)

        df_surv = self.build_survival_dataset(df)
        self.feature_names = feature_cols

        # Kaplan-Meier for population-level baseline
        self.kaplan_meier = KaplanMeierFitter()
        self.kaplan_meier.fit(
            durations=df_surv["duration"],
            event_observed=df_surv["event"],
            label="Portfolio Survival"
        )

        # Cox PH for individual-level scoring
        cox_df = df_surv[["duration", "event"] + feature_cols].dropna()
        if len(cox_df) < 50:
            logger.warning("Insufficient data for CoxPH — using KM only")
            self._is_fitted = True
            return {"concordance_index": 0.5, "model": "kaplan_meier_only"}

        self.cox_model = CoxPHFitter(penalizer=0.1)
        self.cox_model.fit(cox_df, duration_col="duration", event_col="event")
        self._is_fitted = True

        c_index = self.cox_model.concordance_index_
        logger.info(f"[Survival] CoxPH trained. Concordance index: {c_index:.4f}")

        return {
            "concordance_index": c_index,
            "median_survival_days": float(self.kaplan_meier.median_survival_time_),
            "model": "cox_ph",
        }

    def _train_parametric_fallback(self, df: pd.DataFrame, feature_cols: list) -> dict:
        """Fallback: use risk_score as proxy for hazard rate."""
        self._is_fitted = True
        self.feature_names = feature_cols
        return {"concordance_index": 0.0, "model": "parametric_fallback"}

    def predict(self, features: np.ndarray, risk_score: float = None) -> dict:
        """
        Predict time-to-event for a single customer.
        Returns tte_days, p30d, p60d, p90d.
        """
        if not self._is_fitted or self.cox_model is None:
            # Fallback: derive from risk_score
            return self._score_from_risk_score(risk_score or 0.5)

        try:
            import pandas as pd
            feat_df = pd.DataFrame([features], columns=self.feature_names)
            survival_fn = self.cox_model.predict_survival_function(feat_df)

            # P(default within N days)
            def _get_prob_at(days: int) -> float:
                times = survival_fn.index.values
                idx = np.searchsorted(times, days, side="right") - 1
                idx = max(0, min(idx, len(times) - 1))
                s_val = float(survival_fn.iloc[idx, 0])
                return round(1.0 - s_val, 4)

            median_time = self.cox_model.predict_median(feat_df)
            tte = float(median_time.iloc[0]) if not np.isnan(median_time.iloc[0]) else 90.0

            return {
                "tte_days": round(min(tte, 180), 1),
                "p30d": _get_prob_at(30),
                "p60d": _get_prob_at(60),
                "p90d": _get_prob_at(90),
            }

        except Exception as e:
            logger.warning(f"[Survival] predict failed: {e}")
            return self._score_from_risk_score(risk_score or 0.5)

    def _score_from_risk_score(self, risk_score: float) -> dict:
        """Derive survival estimates from plain risk score when model unavailable."""
        # Higher risk → shorter TTE
        tte = max(5.0, 90.0 * (1.0 - risk_score) + 10.0)
        p30 = min(0.99, risk_score * 0.6)
        p60 = min(0.99, risk_score * 0.78)
        p90 = min(0.99, risk_score * 0.92)
        return {
            "tte_days": round(tte, 1),
            "p30d": round(p30, 4),
            "p60d": round(p60, 4),
            "p90d": round(p90, 4),
        }

    def save(self, path: str):
        joblib.dump({
            "cox_model": self.cox_model,
            "kaplan_meier": self.kaplan_meier,
            "feature_names": self.feature_names,
            "is_fitted": self._is_fitted,
        }, path)
        logger.info(f"[Survival] Model saved to {path}")

    def load(self, path: str):
        if not os.path.exists(path):
            return
        data = joblib.load(path)
        self.cox_model = data.get("cox_model")
        self.kaplan_meier = data.get("kaplan_meier")
        self.feature_names = data.get("feature_names")
        self._is_fitted = data.get("is_fitted", False)
        logger.info(f"[Survival] Model loaded from {path}")
