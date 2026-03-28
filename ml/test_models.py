# pyre-ignore-all-errors
"""
Pre-Delinquency Engine — Model Testing & Evaluation Script
===========================================================
Loads the trained models (XGBoost, LightGBM, LSTM), rebuilds the
test dataset from PostgreSQL, evaluates each model on a held-out
20% test split, computes ensemble scores, and writes all results
to a structured JSON file for downstream documentation.

Usage:
    python -m ml.test_models
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    log_loss,
    brier_score_loss,
    roc_curve,
)
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import ModelConfig, PostgresConfig
from ml.dataset_builder import build_training_dataset, build_temporal_dataset
from ml.xgboost_model import XGBoostDelinquencyModel
from ml.lightgbm_model import LightGBMDelinquencyModel
from ml.lstm_model import LSTMDelinquencyModel
from ml.ensemble import EnsembleScorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PDI-ModelTest")

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'test_results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def _safe_float(val):
    """Convert numpy types to plain Python float for JSON serialisation."""
    if isinstance(val, (np.floating, np.float32, np.float64)):
        return float(val)
    if isinstance(val, (np.integer, np.int32, np.int64)):
        return int(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def evaluate_model(name: str, y_true: np.ndarray, y_proba: np.ndarray,
                   threshold: float = 0.5) -> dict:
    """Compute comprehensive metrics for a single model."""
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # ROC curve points (subsample for JSON size)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
    # keep ~50 points
    step = max(1, len(fpr) // 50)
    roc_points = list(zip(fpr[::step].tolist(), tpr[::step].tolist()))

    # Precision–Recall curve points
    prec_arr, rec_arr, pr_thresholds = precision_recall_curve(y_true, y_proba)
    step_pr = max(1, len(prec_arr) // 50)
    pr_points = list(zip(rec_arr[::step_pr].tolist(), prec_arr[::step_pr].tolist()))

    metrics = {
        "model_name": name,
        "auc_roc": _safe_float(roc_auc_score(y_true, y_proba)),
        "average_precision": _safe_float(average_precision_score(y_true, y_proba)),
        "accuracy": _safe_float(accuracy_score(y_true, y_pred)),
        "precision": _safe_float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": _safe_float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": _safe_float(f1_score(y_true, y_pred, zero_division=0)),
        "log_loss": _safe_float(log_loss(y_true, y_proba)),
        "brier_score": _safe_float(brier_score_loss(y_true, y_proba)),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "specificity": _safe_float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "classification_report": classification_report(y_true, y_pred, output_dict=True,
                                                       zero_division=0),
        "roc_curve_points": roc_points,
        "pr_curve_points": pr_points,
        "threshold_used": threshold,
        "total_samples": int(len(y_true)),
        "positive_samples": int(y_true.sum()),
        "negative_samples": int(len(y_true) - y_true.sum()),
    }

    # Clean nested dicts
    for k, v in list(metrics.items()):
        if isinstance(v, dict):
            metrics[k] = {kk: _safe_float(vv) if not isinstance(vv, dict) else {
                kkk: _safe_float(vvv) for kkk, vvv in vv.items()
            } for kk, vv in v.items()}

    return metrics


def test_pipeline():
    """Full model testing pipeline."""
    start_time = time.time()
    print("=" * 70)
    print("PRE-DELINQUENCY ENGINE — MODEL TESTING PIPELINE")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # ──────────────────────────────────────────────
    # 1. Rebuild dataset (same seed → same split)
    # ──────────────────────────────────────────────
    print("\n[1/6] Building test dataset from PostgreSQL...")
    X_tab, y_tab, feature_names, customer_ids = build_training_dataset()
    if X_tab is None:
        print("ERROR: Cannot build dataset. Ensure data exists in PostgreSQL.")
        return

    X_seq, y_seq, cids_seq = build_temporal_dataset()

    # Reproduce the exact same 80/20 split used during training
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_tab, y_tab, np.arange(len(y_tab)),
        test_size=0.2, stratify=y_tab, random_state=42,
    )
    print(f"  Dataset size:  {len(y_tab)} total")
    print(f"  Training set:  {len(y_train)} ({y_train.mean()*100:.1f}% positive)")
    print(f"  Test set:      {len(y_test)} ({y_test.mean()*100:.1f}% positive)")
    print(f"  Features:      {len(feature_names)}")

    all_results = {
        "run_timestamp": datetime.now().isoformat(),
        "dataset": {
            "total_samples": len(y_tab),
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "positive_rate_train": _safe_float(y_train.mean()),
            "positive_rate_test": _safe_float(y_test.mean()),
            "split_method": "stratified 80/20, random_state=42",
        },
        "models": {},
    }

    # ──────────────────────────────────────────────
    # 2. Test XGBoost
    # ──────────────────────────────────────────────
    print("\n[2/6] Testing XGBoost model...")
    xgb_path = os.path.join(MODEL_DIR, "xgboost_model.joblib")
    xgb_model = XGBoostDelinquencyModel()
    xgb_model.load(xgb_path)
    print(f"  Loaded from: {xgb_path}")

    xgb_test_probs = xgb_model.predict_proba(X_test)
    xgb_train_probs = xgb_model.predict_proba(X_train)
    xgb_test_metrics = evaluate_model("XGBoost", y_test, xgb_test_probs)
    xgb_train_metrics = evaluate_model("XGBoost (train)", y_train, xgb_train_probs)

    # Feature importance
    if xgb_model.feature_names:
        importances = xgb_model.model.feature_importances_
        fi = dict(zip(xgb_model.feature_names, importances.tolist()))
        fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
        xgb_test_metrics["feature_importance"] = fi_sorted

    xgb_test_metrics["train_auc"] = xgb_train_metrics["auc_roc"]
    xgb_test_metrics["overfit_gap"] = round(xgb_train_metrics["auc_roc"] - xgb_test_metrics["auc_roc"], 4)

    all_results["models"]["xgboost"] = xgb_test_metrics

    print(f"  ✓ Test AUC-ROC:       {xgb_test_metrics['auc_roc']:.4f}")
    print(f"  ✓ Train AUC-ROC:      {xgb_train_metrics['auc_roc']:.4f}")
    print(f"  ✓ Overfit gap:        {xgb_test_metrics['overfit_gap']:.4f}")
    print(f"  ✓ Precision:          {xgb_test_metrics['precision']:.4f}")
    print(f"  ✓ Recall:             {xgb_test_metrics['recall']:.4f}")
    print(f"  ✓ F1 Score:           {xgb_test_metrics['f1_score']:.4f}")
    print(f"  ✓ Average Precision:  {xgb_test_metrics['average_precision']:.4f}")
    cm = xgb_test_metrics["confusion_matrix"]
    print(f"  ✓ Confusion Matrix:   TP={cm['true_positives']}, FP={cm['false_positives']}, "
          f"TN={cm['true_negatives']}, FN={cm['false_negatives']}")

    # ──────────────────────────────────────────────
    # 3. Test LightGBM
    # ──────────────────────────────────────────────
    print("\n[3/6] Testing LightGBM model...")
    lgb_path = os.path.join(MODEL_DIR, "lightgbm_model.joblib")
    lgb_model = LightGBMDelinquencyModel()
    lgb_model.load(lgb_path)
    print(f"  Loaded from: {lgb_path}")

    lgb_test_probs = lgb_model.predict_proba(X_test)
    lgb_train_probs = lgb_model.predict_proba(X_train)
    lgb_test_metrics = evaluate_model("LightGBM", y_test, lgb_test_probs)
    lgb_train_metrics = evaluate_model("LightGBM (train)", y_train, lgb_train_probs)

    # Feature importance from saved model
    if lgb_model.feature_importances_:
        fi_sorted = dict(sorted(lgb_model.feature_importances_.items(),
                                key=lambda x: x[1], reverse=True))
        lgb_test_metrics["feature_importance"] = {k: _safe_float(v) for k, v in fi_sorted.items()}

    lgb_test_metrics["train_auc"] = lgb_train_metrics["auc_roc"]
    lgb_test_metrics["overfit_gap"] = round(lgb_train_metrics["auc_roc"] - lgb_test_metrics["auc_roc"], 4)
    lgb_test_metrics["num_trees"] = lgb_model.model.num_trees() if lgb_model.model else "N/A"

    all_results["models"]["lightgbm"] = lgb_test_metrics

    print(f"  ✓ Test AUC-ROC:       {lgb_test_metrics['auc_roc']:.4f}")
    print(f"  ✓ Train AUC-ROC:      {lgb_train_metrics['auc_roc']:.4f}")
    print(f"  ✓ Overfit gap:        {lgb_test_metrics['overfit_gap']:.4f}")
    print(f"  ✓ Precision:          {lgb_test_metrics['precision']:.4f}")
    print(f"  ✓ Recall:             {lgb_test_metrics['recall']:.4f}")
    print(f"  ✓ F1 Score:           {lgb_test_metrics['f1_score']:.4f}")
    print(f"  ✓ Average Precision:  {lgb_test_metrics['average_precision']:.4f}")
    cm = lgb_test_metrics["confusion_matrix"]
    print(f"  ✓ Confusion Matrix:   TP={cm['true_positives']}, FP={cm['false_positives']}, "
          f"TN={cm['true_negatives']}, FN={cm['false_negatives']}")

    # ──────────────────────────────────────────────
    # 4. Test LSTM
    # ──────────────────────────────────────────────
    lstm_test_metrics = None
    lstm_test_probs = None
    if X_seq is not None and len(X_seq) > 50:
        print("\n[4/6] Testing LSTM model...")
        lstm_path = os.path.join(MODEL_DIR, "lstm_model.pt")
        if os.path.exists(lstm_path):
            lstm_model = LSTMDelinquencyModel()
            lstm_model.load(lstm_path)
            print(f"  Loaded from: {lstm_path}")

            # Match test set customer IDs to temporal sequences
            test_cids = customer_ids[idx_test]
            lstm_probs_list = []
            lstm_labels_list = []
            matched = 0

            for i, cid in enumerate(test_cids):
                seq_idx = np.where(cids_seq == cid)[0]
                if len(seq_idx) > 0:
                    prob = lstm_model.predict_proba(X_seq[seq_idx[0]:seq_idx[0]+1])[0]
                    lstm_probs_list.append(float(prob))
                    lstm_labels_list.append(y_test[i])
                    matched += 1

            if matched > 20:
                lstm_test_probs_arr = np.array(lstm_probs_list)
                lstm_labels_arr = np.array(lstm_labels_list)
                lstm_test_metrics = evaluate_model("LSTM", lstm_labels_arr, lstm_test_probs_arr)
                lstm_test_metrics["matched_samples"] = matched
                lstm_test_metrics["total_temporal_sequences"] = len(X_seq)
                all_results["models"]["lstm"] = lstm_test_metrics

                # Build full-length prob array for ensemble (zeros where unmatched)
                lstm_test_probs = np.zeros(len(X_test))
                for i, cid in enumerate(test_cids):
                    seq_idx = np.where(cids_seq == cid)[0]
                    if len(seq_idx) > 0:
                        lstm_test_probs[i] = lstm_model.predict_proba(
                            X_seq[seq_idx[0]:seq_idx[0]+1]
                        )[0]

                print(f"  ✓ Test AUC-ROC:       {lstm_test_metrics['auc_roc']:.4f}")
                print(f"  ✓ Precision:          {lstm_test_metrics['precision']:.4f}")
                print(f"  ✓ Recall:             {lstm_test_metrics['recall']:.4f}")
                print(f"  ✓ F1 Score:           {lstm_test_metrics['f1_score']:.4f}")
                print(f"  ✓ Matched samples:    {matched}/{len(test_cids)}")
            else:
                print(f"  ⚠ Insufficient matched samples ({matched}), skipping LSTM evaluation")
        else:
            print(f"  ⚠ LSTM model not found at {lstm_path}")
    else:
        print("\n[4/6] LSTM — skipped (insufficient temporal data)")

    # ──────────────────────────────────────────────
    # 5. Test Ensemble (fixed-weight)
    # ──────────────────────────────────────────────
    print("\n[5/6] Testing Ensemble (weighted combination)...")
    ensemble = EnsembleScorer()

    ensemble_probs = ensemble.combine_batch(
        xgb_probs=xgb_test_probs,
        lgb_probs=lgb_test_probs,
        lstm_probs=lstm_test_probs,
        tft_probs=None,  # TFT model not saved separately
    )
    ensemble_metrics = evaluate_model("Ensemble (Fixed-Weight)", y_test, ensemble_probs)

    # Risk tier distribution
    tiers = [ensemble.score_to_risk_tier(float(p)) for p in ensemble_probs]
    tier_dist = {
        "critical": tiers.count("critical"),
        "watch": tiers.count("watch"),
        "stable": tiers.count("stable"),
    }
    tier_pct = {k: round(v / len(tiers) * 100, 1) for k, v in tier_dist.items()}
    ensemble_metrics["risk_tier_distribution"] = tier_dist
    ensemble_metrics["risk_tier_percentages"] = tier_pct
    ensemble_metrics["weights_used"] = {
        "xgboost": ensemble.xgb_weight,
        "lightgbm": ensemble.lgb_weight,
        "lstm": ensemble.lstm_weight,
        "tft": ensemble.tft_weight,
    }

    all_results["models"]["ensemble_fixed_weight"] = ensemble_metrics

    print(f"  ✓ Ensemble AUC-ROC:   {ensemble_metrics['auc_roc']:.4f}")
    print(f"  ✓ Precision:          {ensemble_metrics['precision']:.4f}")
    print(f"  ✓ Recall:             {ensemble_metrics['recall']:.4f}")
    print(f"  ✓ F1 Score:           {ensemble_metrics['f1_score']:.4f}")
    print(f"  ✓ Risk Tiers:         Critical={tier_dist['critical']} ({tier_pct['critical']}%), "
          f"Watch={tier_dist['watch']} ({tier_pct['watch']}%), "
          f"Stable={tier_dist['stable']} ({tier_pct['stable']}%)")

    # ──────────────────────────────────────────────
    # 6. Model Comparison Summary
    # ──────────────────────────────────────────────
    print("\n[6/6] Generating comparison summary...")

    comparison = []
    for model_key, metrics in all_results["models"].items():
        comparison.append({
            "model": metrics.get("model_name", model_key),
            "auc_roc": metrics.get("auc_roc"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1_score": metrics.get("f1_score"),
            "accuracy": metrics.get("accuracy"),
            "avg_precision": metrics.get("average_precision"),
            "brier_score": metrics.get("brier_score"),
        })

    all_results["comparison_summary"] = comparison

    # Determine best model
    best = max(comparison, key=lambda x: x.get("auc_roc", 0))
    all_results["best_model"] = best["model"]
    all_results["best_auc"] = best["auc_roc"]

    elapsed = round(time.time() - start_time, 2)
    all_results["elapsed_seconds"] = elapsed

    # ──────────────────────────────────────────────
    # Save results to JSON
    # ──────────────────────────────────────────────
    results_path = os.path.join(RESULTS_DIR, "model_test_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_safe_float)
    print(f"\n  Results saved to: {results_path}")

    # ──────────────────────────────────────────────
    # Print final summary table
    # ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MODEL TESTING RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<30} {'AUC-ROC':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 70)
    for row in comparison:
        print(f"{row['model']:<30} {row['auc_roc']:>10.4f} {row['precision']:>10.4f} "
              f"{row['recall']:>10.4f} {row['f1_score']:>10.4f}")
    print("-" * 70)
    print(f"  Best Model: {best['model']} (AUC-ROC: {best['auc_roc']:.4f})")
    print(f"  Elapsed:    {elapsed}s")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    test_pipeline()
