# pyre-ignore-all-errors
"""
BentoML Model Serving Service
Production-grade model serving with BentoML for the PDI delinquency ensemble.
Provides /predict and /explain endpoints with request validation.
"""
import os
import sys
import numpy as np
import logging

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# ─────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────
_xgb_model = None
_lgb_model = None

_ensemble = None
_shap_explainer = None
_lime_explainer = None


def _load_models():
    """Load all models on startup."""
    global _xgb_model, _lgb_model, _ensemble, _shap_explainer, _lime_explainer

    from ml.ensemble import EnsembleScorer
    _ensemble = EnsembleScorer()

    # XGBoost
    try:
        from ml.xgboost_model import XGBoostDelinquencyModel
        xgb_path = os.path.join(MODEL_DIR, "xgboost_model.joblib")
        if os.path.exists(xgb_path):
            _xgb_model = XGBoostDelinquencyModel()
            _xgb_model.load(xgb_path)
            logger.info("[BentoML] XGBoost loaded")
    except Exception as e:
        logger.warning(f"[BentoML] XGBoost load failed: {e}")

    # LightGBM
    try:
        from ml.lightgbm_model import LightGBMDelinquencyModel
        lgb_path = os.path.join(MODEL_DIR, "lightgbm_model.joblib")
        if os.path.exists(lgb_path):
            _lgb_model = LightGBMDelinquencyModel()
            _lgb_model.load(lgb_path)
            logger.info("[BentoML] LightGBM loaded")
    except Exception as e:
        logger.warning(f"[BentoML] LightGBM load failed: {e}")



    # SHAP
    if _xgb_model is not None:
        try:
            from ml.explainability import SHAPExplainer
            _shap_explainer = SHAPExplainer(_xgb_model.get_booster(), _xgb_model.feature_names)
            logger.info("[BentoML] SHAP loaded")
        except Exception as e:
            logger.warning(f"[BentoML] SHAP load failed: {e}")

    # LIME
    predict_fn = _xgb_model.predict_proba if _xgb_model else None
    if predict_fn:
        try:
            from ml.lime_explainer import LIMEExplainer
            _lime_explainer = LIMEExplainer(
                predict_fn=predict_fn,
                feature_names=_xgb_model.feature_names,
            )
            logger.info("[BentoML] LIME loaded")
        except Exception as e:
            logger.warning(f"[BentoML] LIME load failed: {e}")


# ─────────────────────────────────────────────
# BentoML Service
# ─────────────────────────────────────────────
try:
    import bentoml
    from bentoml.io import JSON, NumpyNdarray

    svc = bentoml.Service("pdi_delinquency_scorer", runners=[])

    @svc.on_startup
    def startup():
        _load_models()

    @svc.api(input=NumpyNdarray(), output=JSON())
    def predict(features: np.ndarray) -> dict:
        """
        Predict delinquency risk from feature array.
        Input: numpy array of shape (N, num_features)
        Output: risk scores, tiers, and model contributions
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        results = []
        for i in range(len(features)):
            row = features[i:i+1]

            xgb_prob = float(_xgb_model.predict_proba(row)[0]) if _xgb_model else None
            lgb_prob = float(_lgb_model.predict_proba(row)[0]) if _lgb_model else None

            ensemble_score = _ensemble.combine(xgb_prob, lgb_prob)
            risk_tier = _ensemble.score_to_risk_tier(ensemble_score)
            credit_score = _ensemble.score_to_credit_score(ensemble_score)

            results.append({
                "ensemble_score": ensemble_score,
                "risk_tier": risk_tier,
                "credit_score_mapped": credit_score,
                "xgboost_score": xgb_prob,
                "lightgbm_score": lgb_prob,
                "model_contributions": _ensemble.get_model_contributions(
                    xgb_prob, lgb_prob
                ),
            })

        return {"predictions": results}

    @svc.api(input=NumpyNdarray(), output=JSON())
    def explain(features: np.ndarray) -> dict:
        """
        Get SHAP and LIME explanations for predictions.
        Input: numpy array of shape (1, num_features) or (num_features,)
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        result = {}

        # SHAP explanation
        if _shap_explainer:
            try:
                shap_result = _shap_explainer.explain_single(features)
                result["shap"] = shap_result
            except Exception as e:
                result["shap"] = {"error": str(e)}

        # LIME explanation
        if _lime_explainer:
            try:
                lime_result = _lime_explainer.explain_single(features)
                result["lime"] = lime_result
            except Exception as e:
                result["lime"] = {"error": str(e)}

        # Prediction
        xgb_prob = float(_xgb_model.predict_proba(features)[0]) if _xgb_model else None
        lgb_prob = float(_lgb_model.predict_proba(features)[0]) if _lgb_model else None

        result["prediction"] = {
            "ensemble_score": _ensemble.combine(xgb_prob, lgb_prob),
            "risk_tier": _ensemble.score_to_risk_tier(_ensemble.combine(xgb_prob, lgb_prob)),
        }

        return result

    @svc.api(input=JSON(), output=JSON())
    def health(data: dict) -> dict:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "models": {
                "xgboost": _xgb_model is not None,
                "lightgbm": _lgb_model is not None,
                "lstm": False,  # LSTM removed — TFT replaces it
                "shap": _shap_explainer is not None,
                "lime": _lime_explainer is not None,
            },
            "service": "BentoML PDI Delinquency Scorer",
        }

    logger.info("[BentoML] Service defined successfully")

except ImportError:
    logger.warning("[BentoML] BentoML not installed. Service not available.")
    svc = None
