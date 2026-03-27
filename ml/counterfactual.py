# pyre-ignore-all-errors
"""
Counterfactual Explanations — P5
Gradient-based + greedy search for actionable feature changes.

Value: Turns "risk score = 0.74" into:
       "If salary delay dropped from 8 days to 0, risk would fall to 0.41"
       "If discretionary spend reduced by Rs.4,200/month, risk would drop to 0.49"
       This is the difference between a black box and a coaching tool for RMs.
"""
import os
import sys
import logging
import numpy as np
from typing import List, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
logger = logging.getLogger(__name__)


class CounterfactualGenerator:
    """
    Generates actionable counterfactual explanations:
    "What feature changes would bring this customer's risk below the watch threshold?"

    Method: Greedy feature perturbation — sorts by SHAP importance, perturbs
    one feature at a time (in actionable direction), stops when target score reached.
    Falls back to DiCE if available.
    """

    # Features that are actionable (can realistically change)
    ACTIONABLE_FEATURES = {
        "salary_delay_days":             ("decrease", 0.0, "Reduce salary delay to 0 days"),
        "discretionary_spend_7d":        ("decrease", 0.5, "Reduce discretionary spend by {delta:.0f}"),
        "discretionary_spend_30d":       ("decrease", 0.5, "Reduce monthly discretionary spend by {delta:.0f}"),
        "lending_app_txn_count_7d":      ("decrease", 0.0, "Avoid lending app usage"),
        "failed_autodebits_count_7d":    ("decrease", 0.0, "Ensure sufficient balance before debit dates"),
        "failed_autodebits_count_30d":   ("decrease", 0.0, "Resolve failed auto-debit pattern"),
        "savings_balance_pct_change_7d": ("increase", 0.0, "Increase savings balance by {delta:.1%}"),
        "fd_premature_closures_90d":     ("decrease", 0.0, "Avoid premature FD closures"),
        "sip_stoppages_90d":             ("decrease", 0.0, "Resume stopped SIPs"),
        "atm_withdrawals_count_7d":      ("decrease", 0.0, "Reduce ATM cash withdrawals"),
        "num_active_loans":              ("decrease", None, "Consolidate or close a loan"),
        "dti_ratio":                     ("decrease", 0.3, "Reduce debt-to-income ratio to {target:.0%}"),
    }

    # Features that CANNOT change (immutable)
    IMMUTABLE_FEATURES = {
        "age", "gender", "tenure_months", "customer_id",
        "credit_score",  # not directly actionable in short term
        "region", "city", "state",
    }

    def __init__(self, target_threshold: float = 0.50, max_steps: int = 5):
        """
        Args:
            target_threshold: Risk score to get BELOW (default = watch boundary)
            max_steps:        Maximum features to change (keep explanations concise)
        """
        self.target_threshold = target_threshold
        self.max_steps = max_steps

    def generate(
        self,
        features: np.ndarray,
        feature_names: List[str],
        predict_fn,
        shap_drivers: Optional[List[Dict]] = None,
        current_score: Optional[float] = None,
    ) -> Dict:
        """
        Generate counterfactual for a single customer.

        Args:
            features:      Feature vector (1D numpy array)
            feature_names: Names matching feature vector positions
            predict_fn:    Function: features_2d → probability score
            shap_drivers:  Pre-computed SHAP drivers (used to prioritize perturbation order)
            current_score: Current risk score (computed if not provided)

        Returns:
            dict with 'actions', 'projected_score', 'risk_reduction', 'achievable'
        """
        feat = features.copy().astype(float)
        feat_dict = dict(zip(feature_names, feat))

        if current_score is None:
            current_score = float(predict_fn(feat.reshape(1, -1))[0])

        if current_score <= self.target_threshold:
            return {
                "actions": [],
                "current_score": round(current_score, 4),
                "projected_score": round(current_score, 4),
                "risk_reduction": 0.0,
                "achievable": True,
                "note": "Already below target threshold",
            }

        # Order features to perturb by SHAP importance (most impactful first)
        if shap_drivers:
            ordered_features = [d["feature"] for d in shap_drivers
                                 if d["feature"] in self.ACTIONABLE_FEATURES
                                 and d["feature"] in feat_dict]
        else:
            ordered_features = [f for f in feature_names if f in self.ACTIONABLE_FEATURES]

        actions = []
        modified_feat = feat.copy()
        score = current_score

        for feat_name in ordered_features:
            if len(actions) >= self.max_steps:
                break
            if feat_name not in feat_dict or feat_name not in feature_names:
                continue

            feat_idx = feature_names.index(feat_name)
            direction, target_val, description_template = self.ACTIONABLE_FEATURES[feat_name]
            original_val = modified_feat[feat_idx]

            # Compute perturbation
            if direction == "decrease":
                if target_val is not None:
                    new_val = target_val
                else:
                    new_val = max(0.0, original_val * 0.5)
            else:  # increase
                if target_val is not None:
                    new_val = max(original_val * 1.3, original_val + 0.1)
                else:
                    new_val = original_val * 1.3

            if abs(new_val - original_val) < 1e-6:
                continue

            # Apply perturbation and re-score
            trial_feat = modified_feat.copy()
            trial_feat[feat_idx] = new_val
            try:
                new_score = float(predict_fn(trial_feat.reshape(1, -1))[0])
            except Exception:
                continue

            if new_score >= score:
                continue  # Skip if this change doesn't help

            delta = original_val - new_val if direction == "decrease" else new_val - original_val

            # Format description
            try:
                description = description_template.format(
                    delta=abs(delta), target=new_val
                )
            except (KeyError, ValueError):
                description = description_template

            actions.append({
                "feature": feat_name,
                "direction": direction,
                "original_value": round(float(original_val), 3),
                "counterfactual_value": round(float(new_val), 3),
                "score_before": round(score, 4),
                "score_after": round(new_score, 4),
                "score_delta": round(score - new_score, 4),
                "description": description,
            })

            modified_feat = trial_feat
            score = new_score

            if score <= self.target_threshold:
                break  # Already achieved target

        return {
            "actions": actions,
            "current_score": round(current_score, 4),
            "projected_score": round(score, 4),
            "risk_reduction": round(current_score - score, 4),
            "achievable": score <= self.target_threshold,
            "target_threshold": self.target_threshold,
            "steps_used": len(actions),
        }
