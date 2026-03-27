# pyre-ignore-all-errors
"""
Fairness & Bias Auditing
Evaluates model fairness using BOTH Fairlearn AND AIF360 (IBM).
Measures demographic parity, equalized odds, disparate impact, and more.
"""
import os
import sys
import numpy as np
import pandas as pd
import logging
from typing import Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Fairlearn Metrics
# ─────────────────────────────────────────────
def compute_fairlearn_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: pd.DataFrame,
) -> Dict:
    """Compute fairness metrics using Fairlearn's MetricFrame."""
    try:
        from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
        from sklearn.metrics import accuracy_score, precision_score, recall_score
    except ImportError:
        logger.warning("Fairlearn not installed. Skipping Fairlearn audit.")
        return {"error": "Fairlearn not installed"}

    results = {}

    for col in sensitive_features.columns:
        sf = sensitive_features[col].values

        metric_frame = MetricFrame(
            metrics={
                "accuracy": accuracy_score,
                "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
                "recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
                "selection_rate": selection_rate,
            },
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sf,
        )

        group_metrics = metric_frame.by_group.to_dict()
        overall_metrics = metric_frame.overall.to_dict()

        try:
            dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=sf)
        except Exception:
            dpd = None

        results[col] = {
            "overall": overall_metrics,
            "by_group": {k: {str(gk): float(gv) for gk, gv in v.items()} for k, v in group_metrics.items()},
            "demographic_parity_difference": float(dpd) if dpd is not None else None,
            "is_fair": abs(dpd) < 0.1 if dpd is not None else None,
        }

        logger.info(f"[Fairlearn] {col}: DPD = {dpd:.4f}" if dpd else f"[Fairlearn] {col}: computed")

    return results


# ─────────────────────────────────────────────
# AIF360 (IBM) Metrics
# ─────────────────────────────────────────────
def compute_aif360_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: pd.DataFrame,
) -> Dict:
    """
    Compute fairness metrics using IBM AIF360.
    Measures disparate impact, statistical parity difference,
    equal opportunity difference, and average odds difference.
    """
    try:
        from aif360.datasets import BinaryLabelDataset
        from aif360.metrics import ClassificationMetric
    except ImportError:
        logger.warning("AIF360 not installed. Skipping AIF360 audit.")
        return {"error": "AIF360 not installed"}

    results = {}

    for col in sensitive_features.columns:
        sf_values = sensitive_features[col].values

        # AIF360 needs numeric sensitive attributes
        unique_vals = sorted(pd.Series(sf_values).dropna().unique())
        if len(unique_vals) < 2:
            results[col] = {"warning": f"Less than 2 groups for {col}"}
            continue

        # Map to numeric if needed
        val_map = {v: i for i, v in enumerate(unique_vals)}
        sf_numeric = np.array([val_map.get(v, 0) for v in sf_values])

        # Build AIF360 datasets
        # Privileged group = group with highest index (often majority/higher income)
        privileged_groups = [{col: len(unique_vals) - 1}]
        unprivileged_groups = [{col: 0}]

        # Create DataFrame for AIF360
        df_true = pd.DataFrame({
            "label": y_true.astype(int),
            col: sf_numeric,
        })
        df_pred = pd.DataFrame({
            "label": y_pred.astype(int),
            col: sf_numeric,
        })

        try:
            dataset_true = BinaryLabelDataset(
                favorable_label=0,  # stable = favorable
                unfavorable_label=1,  # at_risk = unfavorable
                df=df_true,
                label_names=["label"],
                protected_attribute_names=[col],
            )
            dataset_pred = BinaryLabelDataset(
                favorable_label=0,
                unfavorable_label=1,
                df=df_pred,
                label_names=["label"],
                protected_attribute_names=[col],
            )

            metric = ClassificationMetric(
                dataset_true,
                dataset_pred,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups,
            )

            disparate_impact = metric.disparate_impact()
            stat_parity_diff = metric.statistical_parity_difference()
            equal_opp_diff = metric.equal_opportunity_difference()
            avg_odds_diff = metric.average_odds_difference()

            results[col] = {
                "disparate_impact": float(disparate_impact) if not np.isnan(disparate_impact) else None,
                "statistical_parity_difference": float(stat_parity_diff),
                "equal_opportunity_difference": float(equal_opp_diff),
                "average_odds_difference": float(avg_odds_diff),
                "is_fair_disparate_impact": (0.8 <= disparate_impact <= 1.25) if not np.isnan(disparate_impact) else None,
                "is_fair_stat_parity": abs(stat_parity_diff) < 0.1,
                "group_mapping": {str(v): int(i) for v, i in val_map.items()},
                "privileged_group": str(unique_vals[-1]),
                "unprivileged_group": str(unique_vals[0]),
            }

            logger.info(f"[AIF360] {col}: DI={disparate_impact:.3f}, SPD={stat_parity_diff:.4f}")

        except Exception as e:
            logger.warning(f"[AIF360] Error for {col}: {e}")
            results[col] = {"error": str(e)}

    return results


# ─────────────────────────────────────────────
# Combined Bias Audit
# ─────────────────────────────────────────────
def run_bias_audit(model, X: np.ndarray, y: np.ndarray,
                   demographics: pd.DataFrame) -> Dict:
    """
    Run complete bias audit using BOTH Fairlearn AND AIF360.
    demographics DataFrame should have columns like 'age', 'gender', 'region', 'income_bracket'.
    """
    y_pred = model.predict(X)

    sensitive_cols = [c for c in demographics.columns
                     if c in ("gender", "region", "income_bracket", "age_group")]

    if not sensitive_cols:
        return {"warning": "No sensitive features found for fairness audit"}

    sensitive_df = demographics[sensitive_cols].reset_index(drop=True)

    # Create age groups if 'age' exists
    if "age" in demographics.columns and "age_group" not in demographics.columns:
        sensitive_df["age_group"] = pd.cut(
            demographics["age"],
            bins=[0, 30, 45, 60, 100],
            labels=["18-30", "31-45", "46-60", "60+"]
        )

    # Run both frameworks
    fairlearn_results = compute_fairlearn_metrics(y, y_pred, sensitive_df)
    aif360_results = compute_aif360_metrics(y, y_pred, sensitive_df)

    # Overall verdict combining both
    fairlearn_fair = all(
        r.get("is_fair", True) for r in fairlearn_results.values()
        if isinstance(r, dict) and "is_fair" in r
    )
    aif360_fair = all(
        r.get("is_fair_stat_parity", True) for r in aif360_results.values()
        if isinstance(r, dict) and "is_fair_stat_parity" in r
    )

    combined = {
        "fairlearn": fairlearn_results,
        "aif360": aif360_results,
        "verdict": "PASS" if (fairlearn_fair and aif360_fair) else "FAIL - Review required",
        "frameworks_used": ["Fairlearn", "AIF360"],
    }

    logger.info(f"[Fairness] Combined verdict: {combined['verdict']}")
    return combined
