# pyre-ignore-all-errors
"""
LIME Explainability Module
Local Interpretable Model-agnostic Explanations for delinquency predictions.
Provides per-instance feature contribution analysis complementing SHAP.
"""
import os
import sys
import numpy as np
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

# Feature explanation templates for LIME (same domain as SHAP but alternative method)
LIME_FEATURE_DESCRIPTIONS = {
    "lending_app_txn_count_7d": "borrowing from lending apps in last 7 days",
    "lending_app_txn_count_30d": "borrowing from lending apps in last 30 days",
    "weighted_lending_risk_7d": "weighted lending risk exposure (7-day)",
    "weighted_lending_risk_30d": "weighted lending risk exposure (30-day)",
    "failed_autodebits_count_7d": "failed automatic debit attempts (7-day)",
    "failed_autodebits_count_30d": "failed automatic debit attempts (30-day)",
    "discretionary_spend_7d": "non-essential spending in last 7 days",
    "discretionary_spend_30d": "non-essential spending in last 30 days",
    "atm_withdrawals_count_7d": "ATM cash withdrawals (7-day)",
    "atm_withdrawals_count_30d": "ATM cash withdrawals (30-day)",
    "savings_balance_pct_change_7d": "savings account balance change",
    "salary_delay_days": "salary payment delay from expected date",
    "utility_payment_delay_avg": "average utility bill payment delay",
    "discretionary_spend_trend": "trend in discretionary spending",
    "credit_score": "credit bureau score",
    "avg_monthly_spend_3m": "average monthly spending (3 months)",
    "spend_volatility_3m": "spending pattern volatility (3 months)",
}


class LIMEExplainer:
    """LIME-based explainer for delinquency risk models."""

    def __init__(self, predict_fn, feature_names: list, training_data: np.ndarray = None):
        """
        Initialize LIME explainer.

        Args:
            predict_fn: Function that takes (N, features) array and returns probabilities
            feature_names: List of feature names
            training_data: Optional training data for better perturbation
        """
        try:
            from lime.lime_tabular import LimeTabularExplainer
        except ImportError:
            raise ImportError("lime package required: pip install lime")

        self.predict_fn = predict_fn
        self.feature_names = feature_names

        # Create LIME explainer
        self.explainer = LimeTabularExplainer(
            training_data=training_data if training_data is not None else np.zeros((10, len(feature_names))),
            feature_names=feature_names,
            class_names=["stable", "at_risk"],
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )
        logger.info(f"[LIME] Explainer initialized with {len(feature_names)} features")

    def _predict_wrapper(self, X):
        """Wrap predict function to return [P(stable), P(at_risk)]."""
        probs = self.predict_fn(X)
        if probs.ndim == 1:
            return np.column_stack([1 - probs, probs])
        return probs

    def explain_single(self, instance: np.ndarray, top_k: int = 5,
                       num_samples: int = 500) -> dict:
        """
        Generate LIME explanation for a single instance.

        Args:
            instance: Shape (1, features) or (features,)
            top_k: Number of top contributing features
            num_samples: Number of perturbation samples

        Returns:
            dict with top_drivers, weights, explanation text
        """
        if instance.ndim == 2:
            instance = instance[0]

        explanation = self.explainer.explain_instance(
            instance,
            self._predict_wrapper,
            num_features=top_k,
            num_samples=num_samples,
            top_labels=1,
        )

        # Extract feature contributions
        # Get explanation for the "at_risk" class (label 1)
        try:
            feature_weights = explanation.as_list(label=1)
        except Exception:
            feature_weights = explanation.as_list()

        drivers = []
        for feature_expr, weight in feature_weights[:top_k]:
            # Parse feature name from LIME's expression (e.g. "feature_name <= 3.00")
            feature_name = feature_expr.split(" ")[0].strip()
            # Find closest matching feature name
            matched_name = feature_name
            for fn in self.feature_names:
                if fn in feature_expr:
                    matched_name = fn
                    break

            description = LIME_FEATURE_DESCRIPTIONS.get(
                matched_name, f"{matched_name} contribution"
            )

            drivers.append({
                "feature": matched_name,
                "expression": feature_expr,
                "weight": float(weight),
                "direction": "increases_risk" if weight > 0 else "decreases_risk",
                "description": description,
            })

        # Natural language explanation
        risk_drivers = [d for d in drivers if d["weight"] > 0]
        protective = [d for d in drivers if d["weight"] < 0]

        explanation_parts = []
        if risk_drivers:
            top_risk = risk_drivers[0]
            explanation_parts.append(
                f"Primary risk driver: {top_risk['description']} "
                f"(weight: {top_risk['weight']:.3f})"
            )
        if protective:
            top_protect = protective[0]
            explanation_parts.append(
                f"Protective factor: {top_protect['description']} "
                f"(weight: {top_protect['weight']:.3f})"
            )

        explanation_text = ". ".join(explanation_parts) if explanation_parts else "No significant drivers identified."

        # Prediction probabilities
        prediction = self._predict_wrapper(instance.reshape(1, -1))

        return {
            "top_drivers": drivers,
            "explanation": explanation_text,
            "prediction_proba": {
                "stable": float(prediction[0][0]),
                "at_risk": float(prediction[0][1]),
            },
            "method": "LIME",
            "num_samples": num_samples,
        }

    def explain_batch(self, X: np.ndarray, top_k: int = 5,
                      num_samples: int = 200) -> list:
        """Explain multiple instances."""
        results = []
        for i in range(len(X)):
            try:
                result = self.explain_single(X[i:i+1], top_k=top_k,
                                             num_samples=num_samples)
                results.append(result)
            except Exception as e:
                logger.warning(f"[LIME] Failed for instance {i}: {e}")
                results.append({"error": str(e), "method": "LIME"})
        return results

    def get_feature_importance(self, X: np.ndarray, num_samples: int = 200) -> dict:
        """
        Compute global feature importance by averaging LIME weights
        across multiple instances.
        """
        importance_accumulator = {fn: [] for fn in self.feature_names}

        sample_size = min(len(X), 50)
        indices = np.random.choice(len(X), sample_size, replace=False)

        for idx in indices:
            try:
                result = self.explain_single(X[idx:idx+1], top_k=len(self.feature_names),
                                             num_samples=num_samples)
                for driver in result["top_drivers"]:
                    if driver["feature"] in importance_accumulator:
                        importance_accumulator[driver["feature"]].append(abs(driver["weight"]))
            except Exception:
                continue

        # Average importance
        global_importance = {}
        for feature, weights in importance_accumulator.items():
            if weights:
                global_importance[feature] = float(np.mean(weights))
            else:
                global_importance[feature] = 0.0

        return dict(sorted(global_importance.items(), key=lambda x: x[1], reverse=True))
