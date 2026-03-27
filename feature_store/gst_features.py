# pyre-ignore-all-errors
"""
GST Invoice Features — P10
Self-employed / MSME distress signals from GST filing patterns.

Value: For salaried customers you have salary credits as the heartbeat of financial health.
       For self-employed customers (35M+ in India), the equivalent heartbeat is GST filing.
       A gap in GST invoice issuance or a sharp drop in declared turnover is
       the earliest detectable stress signal for this segment — weeks before a missed EMI.
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
logger = logging.getLogger(__name__)


class GSTFeatureExtractor:
    """
    Extracts GST-linked distress features for self-employed and MSME customers.

    In production: consumes GST data from bank's business-account transaction feed
    (GST payments, vendor payments with GST, GSTR-3B filing events).
    In this implementation: derives proxy features from existing transaction patterns.
    """

    GST_PAYMENT_DESCRIPTION_KEYWORDS = [
        "gst", "gstn", "tax_challan", "gst_payment", "igst", "cgst", "sgst"
    ]

    def compute_gst_features(
        self, customer_id: str, transactions: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Compute GST-linked features from transaction history.

        Args:
            customer_id:  Customer identifier
            transactions: DataFrame with columns: amount, description,
                          merchant_category, timestamp, txn_type

        Returns:
            Dict of GST features for the feature vector
        """
        if transactions.empty:
            return self._empty_features()

        transactions = transactions.copy()
        transactions["timestamp"] = pd.to_datetime(transactions["timestamp"])
        transactions["month"] = transactions["timestamp"].dt.to_period("M")

        now = transactions["timestamp"].max()
        cutoff_90d = now - pd.Timedelta(days=90)
        cutoff_180d = now - pd.Timedelta(days=180)

        recent = transactions[transactions["timestamp"] >= cutoff_90d]
        historical = transactions[transactions["timestamp"] >= cutoff_180d]

        # Detect GST payment transactions
        desc_col = "description" if "description" in transactions.columns else "merchant_category"
        gst_mask = transactions[desc_col].str.lower().str.contains(
            "|".join(self.GST_PAYMENT_DESCRIPTION_KEYWORDS), na=False
        )
        gst_txns = transactions[gst_mask]
        gst_recent = gst_txns[gst_txns["timestamp"] >= cutoff_90d]
        gst_hist = gst_txns[gst_txns["timestamp"] >= cutoff_180d]

        # Feature 1: Monthly GST filing regularity
        if len(gst_hist) >= 1:
            months_with_gst = gst_hist.groupby("month").size()
            unique_months = historical["month"].nunique()
            gst_filing_regularity = min(1.0, len(months_with_gst) / max(1, unique_months))
        else:
            gst_filing_regularity = 0.0

        # Feature 2: GST gap — months since last GST payment
        if len(gst_txns) > 0:
            last_gst_date = gst_txns["timestamp"].max()
            gst_gap_months = (now - last_gst_date).days / 30.0
        else:
            gst_gap_months = 6.0  # treat as 6 months if never seen

        # Feature 3: Business inflow trend (proxy for revenue)
        # Large credits to business account = customer turnover
        large_credits = transactions[
            (transactions["txn_type"].str.contains("credit", case=False, na=False)) &
            (transactions["amount"] > 50_000)
        ]
        recent_credits = large_credits[large_credits["timestamp"] >= cutoff_90d]
        hist_credits = large_credits[large_credits["timestamp"] >= cutoff_180d]

        recent_inflow = float(recent_credits["amount"].sum())
        hist_monthly_avg = float(hist_credits["amount"].sum()) / max(1, 6)  # 6 months
        recent_monthly_avg = recent_inflow / 3.0  # 3 months

        if hist_monthly_avg > 0:
            inflow_trend_pct = (recent_monthly_avg - hist_monthly_avg) / hist_monthly_avg
        else:
            inflow_trend_pct = 0.0

        # Feature 4: GST payment amount trend (declining = shrinking business)
        if len(gst_txns) >= 4:
            gst_amounts = gst_txns.sort_values("timestamp")["amount"].values
            half = len(gst_amounts) // 2
            recent_avg = float(gst_amounts[half:].mean())
            older_avg = float(gst_amounts[:half].mean())
            gst_amount_trend = (recent_avg - older_avg) / max(1.0, older_avg)
        else:
            gst_amount_trend = 0.0

        # Feature 5: Vendor payment regularity (supply chain continuity)
        vendor_keywords = ["supplier", "vendor", "purchase", "raw_material", "b2b"]
        vendor_mask = transactions[desc_col].str.lower().str.contains(
            "|".join(vendor_keywords), na=False
        )
        vendor_txns_recent = transactions[vendor_mask & (transactions["timestamp"] >= cutoff_90d)]
        vendor_payment_count_90d = len(vendor_txns_recent)

        features = {
            "gst_filing_regularity_6m": round(gst_filing_regularity, 3),
            "gst_gap_months": round(gst_gap_months, 1),
            "business_inflow_trend_pct": round(inflow_trend_pct, 3),
            "gst_amount_trend_pct": round(gst_amount_trend, 3),
            "vendor_payment_count_90d": float(vendor_payment_count_90d),
            "gst_payments_90d": float(len(gst_recent)),
            "gst_total_amount_90d": float(gst_recent["amount"].sum()) if len(gst_recent) > 0 else 0.0,
        }

        logger.debug(f"[GST] Features for {customer_id}: {features}")
        return features

    def _empty_features(self) -> Dict[str, float]:
        return {
            "gst_filing_regularity_6m": 0.0,
            "gst_gap_months": 6.0,
            "business_inflow_trend_pct": 0.0,
            "gst_amount_trend_pct": 0.0,
            "vendor_payment_count_90d": 0.0,
            "gst_payments_90d": 0.0,
            "gst_total_amount_90d": 0.0,
        }

    @staticmethod
    def is_applicable(employment_type: str) -> bool:
        """GST features are only meaningful for self-employed / MSME."""
        return employment_type in (
            "self_employed", "business_owner", "professional", "freelancer"
        )
