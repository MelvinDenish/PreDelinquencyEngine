"""
SHAP Explainability Layer
Computes SHAP values for XGBoost predictions to identify
the most influential behavioral drivers of each risk score.
"""
import os
import sys
import numpy as np
import shap
import logging
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """SHAP-based model explainability for XGBoost predictions."""

    def __init__(self, model, feature_names: List[str]):
        """
        Initialize with a trained XGBoost model.
        Args:
            model: XGBoost model instance (or the booster)
            feature_names: List of feature names in order
        """
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)
        logger.info(f"[SHAP] Explainer initialized with {len(feature_names)} features")

    def explain_single(self, features: np.ndarray, top_k: int = 5) -> Dict:
        """
        Explain a single prediction.
        Returns dict with SHAP values, top drivers, and natural language explanation.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        shap_values = self.explainer.shap_values(features)

        if isinstance(shap_values, list):
            # Binary classification: use positive class
            sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            sv = shap_values[0]

        # Create feature-SHAP pairs sorted by absolute contribution
        feature_contributions = [
            {
                "feature": self.feature_names[i],
                "shap_value": float(sv[i]),
                "feature_value": float(features[0][i]),
                "direction": "increases_risk" if sv[i] > 0 else "decreases_risk",
            }
            for i in range(len(sv))
        ]

        # Sort by absolute SHAP value
        feature_contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        # Top drivers
        top_drivers = feature_contributions[:top_k]

        # Generate natural language explanation
        explanation = self._generate_explanation(top_drivers)

        return {
            "shap_values": {self.feature_names[i]: float(sv[i]) for i in range(len(sv))},
            "top_drivers": top_drivers,
            "explanation": explanation,
            "base_value": float(self.explainer.expected_value[1]
                               if isinstance(self.explainer.expected_value, (list, np.ndarray))
                               else self.explainer.expected_value),
        }

    def explain_batch(self, features: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Explain a batch of predictions."""
        return [self.explain_single(features[i:i+1], top_k) for i in range(len(features))]

    def _generate_explanation(self, top_drivers: List[Dict]) -> str:
        """Generate natural language explanation from top SHAP drivers."""
        FEATURE_DESCRIPTIONS = {
            "discretionary_spend_7d": "discretionary spending (dining, entertainment) in the last 7 days",
            "discretionary_spend_30d": "discretionary spending in the last 30 days",
            "atm_withdrawals_count_7d": "ATM withdrawal frequency in the last 7 days",
            "atm_withdrawals_count_30d": "ATM withdrawal frequency in the last 30 days",
            "lending_app_txn_count_7d": "lending app transactions in the last 7 days",
            "lending_app_txn_count_30d": "lending app transactions in the last 30 days",
            "weighted_lending_risk_7d": "high-risk lending activity in the last 7 days",
            "weighted_lending_risk_30d": "high-risk lending activity in the last 30 days",
            "savings_balance_pct_change_7d": "savings balance change in the last 7 days",
            "failed_autodebits_count_7d": "failed auto-debit payments in the last 7 days",
            "failed_autodebits_count_30d": "failed auto-debit payments in the last 30 days",
            "total_spend_7d": "total spending in the last 7 days",
            "total_spend_30d": "total spending in the last 30 days",
            "txn_count_7d": "transaction volume in the last 7 days",
            "txn_count_30d": "transaction volume in the last 30 days",
            "avg_txn_amount_7d": "average transaction size in the last 7 days",
            "max_txn_amount_7d": "maximum transaction size in the last 7 days",
            "salary_delay_days": "salary credit delay",
            "utility_payment_delay_avg": "utility payment delays",
            "discretionary_spend_trend": "change in discretionary spending pattern",
            "credit_score": "credit score",
            "age": "customer age",
            "tenure_months": "account tenure",
            "product_count": "number of banking products",
            "has_credit_card": "credit card ownership",
            "has_personal_loan": "personal loan status",
            "has_mortgage": "mortgage status",
            "avg_monthly_spend_3m": "average monthly spending over 3 months",
            "spend_volatility_3m": "spending pattern volatility over 3 months",
        }

        risk_factors = []
        protective_factors = []

        for driver in top_drivers[:3]:
            feature = driver["feature"]
            desc = FEATURE_DESCRIPTIONS.get(feature, feature.replace("_", " "))
            value = driver["feature_value"]

            if driver["direction"] == "increases_risk":
                risk_factors.append(f"{desc} ({value:.1f})")
            else:
                protective_factors.append(f"{desc} ({value:.1f})")

        parts = []
        if risk_factors:
            parts.append(f"Key risk drivers: {', '.join(risk_factors)}")
        if protective_factors:
            parts.append(f"Protective factors: {', '.join(protective_factors)}")

        return ". ".join(parts) + "." if parts else "No significant risk drivers identified."
