# pyre-ignore-all-errors
"""
Enhanced Model Testing — Banking-Standard Metrics
====================================================
Adds Gini Coefficient, KS Statistic, Lift/Gain analysis,
Decile Analysis, and optimal threshold search on top of
standard ML metrics. Produces a summary JSON used by the
documentation generator.

Usage:
    python -m ml.enhanced_test
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score, log_loss,
    brier_score_loss, roc_curve, matthews_corrcoef,
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
logger = logging.getLogger("PDI-EnhancedTest")

MODEL_DIR  = os.path.join(os.path.dirname(__file__), '..', 'models')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'test_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

_f = lambda v: float(v) if isinstance(v, (np.floating, np.integer)) else v


# ═════════════════════════════════════════════════════════
# Banking-Standard Metric Functions
# ═════════════════════════════════════════════════════════

def gini_coefficient(y_true, y_proba):
    """Gini = 2 × AUC − 1.  Industry-standard credit risk discriminative power."""
    auc = roc_auc_score(y_true, y_proba)
    return round(2 * auc - 1, 4)


def ks_statistic(y_true, y_proba):
    """
    Kolmogorov–Smirnov Statistic: maximum separation between
    the cumulative distributions of positive and negative classes.
    Banking benchmark: KS > 0.40 is considered good.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ks = np.max(tpr - fpr)
    return round(float(ks), 4)


def decile_analysis(y_true, y_proba):
    """
    Sort customers into 10 deciles by predicted risk.
    Returns per-decile capture rate and cumulative lift.
    This is the standard 'gains table' used in banking model validation.
    """
    df = pd.DataFrame({"y": y_true, "p": y_proba})
    df["decile"] = pd.qcut(df["p"], 10, labels=False, duplicates="drop")
    df["decile"] = df["decile"].max() - df["decile"]  # decile 0 = highest risk

    table = []
    total_pos = df["y"].sum()
    cum_pos = 0
    for d in sorted(df["decile"].unique()):
        grp = df[df["decile"] == d]
        n = len(grp)
        pos = int(grp["y"].sum())
        cum_pos += pos
        rate = pos / n if n else 0
        capture = cum_pos / total_pos if total_pos else 0
        baseline = (d + 1) / df["decile"].nunique()
        lift = capture / baseline if baseline else 0
        table.append({
            "decile": int(d + 1),
            "count": int(n),
            "positives": pos,
            "event_rate": round(rate, 4),
            "cumulative_capture": round(capture, 4),
            "cumulative_lift": round(lift, 4),
        })
    return table


def optimal_threshold(y_true, y_proba):
    """
    Find the threshold that maximises Youden's J = Sensitivity + Specificity − 1.
    Also finds threshold that maximises F1.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    best_j_idx = np.argmax(j_scores)
    best_j_thresh = float(thresholds[best_j_idx])

    # F1-optimal threshold
    prec_arr, rec_arr, pr_thresholds = precision_recall_curve(y_true, y_proba)
    f1_arr = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + 1e-8)
    best_f1_idx = np.argmax(f1_arr)
    best_f1_thresh = float(pr_thresholds[best_f1_idx])

    return {
        "youden_j_threshold": round(best_j_thresh, 4),
        "youden_j_value": round(float(j_scores[best_j_idx]), 4),
        "sensitivity_at_j": round(float(tpr[best_j_idx]), 4),
        "specificity_at_j": round(float(1 - fpr[best_j_idx]), 4),
        "f1_optimal_threshold": round(best_f1_thresh, 4),
        "f1_at_optimal": round(float(f1_arr[best_f1_idx]), 4),
    }


def evaluate_at_threshold(y_true, y_proba, threshold):
    """Compute key metrics at a given decision threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        "threshold": round(threshold, 4),
        "precision": round(_f(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(_f(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(_f(f1_score(y_true, y_pred, zero_division=0)), 4),
        "accuracy": round(_f(accuracy_score(y_true, y_pred)), 4),
        "specificity": round(tn / (tn + fp), 4) if (tn + fp) else 0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def full_evaluation(name, y_true, y_proba):
    """Run all standard + banking metrics for a model."""
    auc = roc_auc_score(y_true, y_proba)
    gini = gini_coefficient(y_true, y_proba)
    ks = ks_statistic(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)
    logloss = log_loss(y_true, y_proba)
    opt = optimal_threshold(y_true, y_proba)

    # Default threshold (0.5)
    at_05 = evaluate_at_threshold(y_true, y_proba, 0.5)

    # Optimal Youden threshold
    at_j = evaluate_at_threshold(y_true, y_proba, opt["youden_j_threshold"])

    # Optimal F1 threshold
    at_f1 = evaluate_at_threshold(y_true, y_proba, opt["f1_optimal_threshold"])

    # MCC
    y_pred_05 = (y_proba >= 0.5).astype(int)
    mcc = matthews_corrcoef(y_true, y_pred_05)

    deciles = decile_analysis(y_true, y_proba)

    return {
        "model_name": name,
        # ── Threshold-independent ──
        "auc_roc": round(auc, 4),
        "gini_coefficient": gini,
        "ks_statistic": ks,
        "average_precision_auc_pr": round(ap, 4),
        "brier_score": round(brier, 4),
        "log_loss": round(logloss, 4),
        "mcc_at_0.5": round(float(mcc), 4),
        # ── At default threshold 0.5 ──
        "at_threshold_0.5": at_05,
        # ── At optimal thresholds ──
        "optimal_thresholds": opt,
        "at_youden_threshold": at_j,
        "at_f1_optimal_threshold": at_f1,
        # ── Decile / gains table ──
        "decile_analysis": deciles,
    }


# ═════════════════════════════════════════════════════════
# Main pipeline
# ═════════════════════════════════════════════════════════

def run_enhanced_test():
    start = time.time()
    print("=" * 70)
    print("ENHANCED MODEL TESTING — BANKING-STANDARD METRICS")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # 1. Build dataset
    print("\n[1/5] Building dataset...")
    X_tab, y_tab, feature_names, customer_ids = build_training_dataset()
    if X_tab is None:
        print("ERROR: Cannot build dataset.")
        return
    X_seq, y_seq, cids_seq = build_temporal_dataset()

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_tab, y_tab, np.arange(len(y_tab)),
        test_size=0.2, stratify=y_tab, random_state=42,
    )
    print(f"  Test set: {len(y_test)} samples ({y_test.mean()*100:.1f}% positive)")

    results = {"timestamp": datetime.now().isoformat(), "models": {}}

    # 2. XGBoost
    print("\n[2/5] Evaluating XGBoost...")
    xgb = XGBoostDelinquencyModel()
    xgb.load(os.path.join(MODEL_DIR, "xgboost_model.joblib"))
    xgb_probs = xgb.predict_proba(X_test)
    results["models"]["xgboost"] = full_evaluation("XGBoost", y_test, xgb_probs)
    r = results["models"]["xgboost"]
    print(f"  AUC: {r['auc_roc']}  Gini: {r['gini_coefficient']}  KS: {r['ks_statistic']}")

    # 3. LightGBM
    print("\n[3/5] Evaluating LightGBM...")
    lgb = LightGBMDelinquencyModel()
    lgb.load(os.path.join(MODEL_DIR, "lightgbm_model.joblib"))
    lgb_probs = lgb.predict_proba(X_test)
    results["models"]["lightgbm"] = full_evaluation("LightGBM", y_test, lgb_probs)
    r = results["models"]["lightgbm"]
    print(f"  AUC: {r['auc_roc']}  Gini: {r['gini_coefficient']}  KS: {r['ks_statistic']}")

    # 4. LSTM
    lstm_probs = None
    if X_seq is not None and len(X_seq) > 50:
        print("\n[4/5] Evaluating LSTM...")
        lstm_path = os.path.join(MODEL_DIR, "lstm_model.pt")
        if os.path.exists(lstm_path):
            lstm = LSTMDelinquencyModel()
            lstm.load(lstm_path)
            test_cids = customer_ids[idx_test]
            lstm_probs = np.zeros(len(X_test))
            for i, cid in enumerate(test_cids):
                seq_idx = np.where(cids_seq == cid)[0]
                if len(seq_idx) > 0:
                    lstm_probs[i] = lstm.predict_proba(X_seq[seq_idx[0]:seq_idx[0]+1])[0]
            results["models"]["lstm"] = full_evaluation("LSTM", y_test, lstm_probs)
            r = results["models"]["lstm"]
            print(f"  AUC: {r['auc_roc']}  Gini: {r['gini_coefficient']}  KS: {r['ks_statistic']}")
    else:
        print("\n[4/5] LSTM — skipped")

    # 5. Ensemble
    print("\n[5/5] Evaluating Ensemble...")
    ens = EnsembleScorer()
    ens_probs = ens.combine_batch(xgb_probs=xgb_probs, lgb_probs=lgb_probs,
                                   lstm_probs=lstm_probs, tft_probs=None)
    results["models"]["ensemble"] = full_evaluation("Ensemble", y_test, ens_probs)
    r = results["models"]["ensemble"]
    print(f"  AUC: {r['auc_roc']}  Gini: {r['gini_coefficient']}  KS: {r['ks_statistic']}")

    # ── Comparison ──
    comparison = []
    for key, m in results["models"].items():
        comparison.append({
            "model": m["model_name"],
            "auc_roc": m["auc_roc"],
            "gini": m["gini_coefficient"],
            "ks_stat": m["ks_statistic"],
            "avg_precision": m["average_precision_auc_pr"],
            "brier": m["brier_score"],
            "precision_05": m["at_threshold_0.5"]["precision"],
            "recall_05": m["at_threshold_0.5"]["recall"],
            "f1_05": m["at_threshold_0.5"]["f1"],
            "best_f1": m["at_f1_optimal_threshold"]["f1"],
            "best_f1_thresh": m["optimal_thresholds"]["f1_optimal_threshold"],
            "recall_at_optimal": m["at_youden_threshold"]["recall"],
            "precision_at_optimal": m["at_youden_threshold"]["precision"],
        })
    results["comparison"] = comparison

    # Best model
    best = max(comparison, key=lambda x: x["gini"])
    results["best_model"] = best["model"]
    results["elapsed_seconds"] = round(time.time() - start, 1)

    # Save
    path = os.path.join(RESULTS_DIR, "enhanced_test_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=_f)
    print(f"\n  Results saved to: {path}")

    # Summary table
    print("\n" + "=" * 100)
    print("ENHANCED RESULTS SUMMARY")
    print("=" * 100)
    print(f"{'Model':<12} {'AUC':>7} {'Gini':>7} {'KS':>7} {'AP':>7} {'Brier':>7} "
          f"{'P@0.5':>7} {'R@0.5':>7} {'F1@0.5':>7} {'BestF1':>7} {'@Thr':>7}")
    print("-" * 100)
    for c in comparison:
        print(f"{c['model']:<12} {c['auc_roc']:>7.4f} {c['gini']:>7.4f} "
              f"{c['ks_stat']:>7.4f} {c['avg_precision']:>7.4f} {c['brier']:>7.4f} "
              f"{c['precision_05']:>7.4f} {c['recall_05']:>7.4f} {c['f1_05']:>7.4f} "
              f"{c['best_f1']:>7.4f} {c['best_f1_thresh']:>7.4f}")
    print("-" * 100)
    print(f"  Best model (by Gini): {best['model']}")
    print("=" * 100)

    return results


if __name__ == "__main__":
    run_enhanced_test()
