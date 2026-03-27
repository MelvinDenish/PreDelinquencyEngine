"""
FastAPI Scoring Service — v3.0
Provides REST endpoints for real-time delinquency risk scoring.
Integrates XGBoost + LightGBM + LSTM + TFT ensemble with meta-learner stacking,
cold-start handling, segment classification, product action proposals,
SHAP + LIME explainability, Cassandra storage, and Prometheus metrics.
"""
import os
import sys
import json
import logging
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import redis as redis_lib
import psycopg2
from psycopg2.extras import execute_values
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import (
    PostgresConfig, RedisConfig, ModelConfig, ScoringConfig, FeastConfig,
)
from ml.xgboost_model import XGBoostDelinquencyModel
from ml.lightgbm_model import LightGBMDelinquencyModel
from ml.lstm_model import LSTMDelinquencyModel
from ml.ensemble import EnsembleScorer, StackingEnsemble
from ml.explainability import SHAPExplainer
from ml.tft_model import TFTDelinquencyModel
from ml.cold_start import ColdStartScorer
from ml.segment_classifier import CustomerSegmentClassifier
from scoring_service.sequence_cache import SequenceCache
from intervention.product_actions import ProductActionEngine
from scoring_service.cassandra_client import write_risk_score as cassandra_write_risk_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Pre-Delinquency Intervention Engine - Scoring Service",
    description="Real-time risk scoring: XGBoost + LightGBM + LSTM + TFT ensemble "
                "with meta-learner stacking, cold-start handling, segment classification, "
                "SHAP + LIME explainability, Prometheus metrics, Cassandra storage",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Prometheus Metrics (prometheus-fastapi-instrumentator)
# ─────────────────────────────────────────────
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    _instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_instrument_requests_inprogress=True,
        inprogress_labels=True,
    ).instrument(app)
    _prometheus_enabled = True
    logger.info("Prometheus instrumentation enabled — metrics at /metrics")
except ImportError:
    _prometheus_enabled = False
    logger.warning("prometheus-fastapi-instrumentator not installed; /metrics disabled")

# ─────────────────────────────────────────────
# Global model instances (loaded on startup)
# ─────────────────────────────────────────────
xgb_model: Optional[XGBoostDelinquencyModel] = None
lgb_model: Optional[LightGBMDelinquencyModel] = None
lstm_model: Optional[LSTMDelinquencyModel] = None
tft_model: Optional[TFTDelinquencyModel] = None
ensemble: Optional[EnsembleScorer] = None
stacker: Optional[StackingEnsemble] = None
shap_explainer: Optional[SHAPExplainer] = None
lime_explainer = None
redis_client: Optional[redis_lib.Redis] = None
cold_start_scorer = ColdStartScorer()
segment_classifier = CustomerSegmentClassifier()
sequence_cache: Optional[SequenceCache] = None
product_engine = ProductActionEngine()


# ─────────────────────────────────────────────
# Request/Response Models
# ─────────────────────────────────────────────
class ScoreRequest(BaseModel):
    customer_id: str

class BatchScoreRequest(BaseModel):
    customer_ids: List[str]

class ScoreResponse(BaseModel):
    customer_id: str
    risk_score: float
    risk_tier: str
    credit_score_mapped: int
    segment_type: Optional[str] = None
    is_cold_start: bool = False
    xgboost_score: float
    lightgbm_score: Optional[float] = None
    lstm_score: Optional[float] = None
    tft_score: Optional[float] = None
    ensemble_score: float
    meta_learner_used: bool = False
    top_shap_features: Optional[list] = None
    top_lime_features: Optional[list] = None
    explanation: Optional[str] = None
    product_actions: Optional[list] = None
    scored_at: str

class HealthResponse(BaseModel):
    status: str
    models_loaded: dict
    timestamp: str


# ─────────────────────────────────────────────
# Feature Retrieval
# ─────────────────────────────────────────────
def get_features_from_redis(customer_id: str) -> dict:
    """Retrieve features from Redis (streaming features)."""
    key = f"features:streaming:{customer_id}"
    data = redis_client.hgetall(key)
    if data:
        return {k.decode() if isinstance(k, bytes) else k:
                float(v.decode() if isinstance(v, bytes) else v)
                for k, v in data.items()}
    return {}


def get_features_from_db(customer_id: str) -> dict:
    """Retrieve features from PostgreSQL (batch + streaming)."""
    conn = psycopg2.connect(
        host=PostgresConfig.HOST, port=PostgresConfig.PORT,
        user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
        dbname=PostgresConfig.DB,
    )
    cursor = conn.cursor()
    features = {}

    # Streaming features
    cursor.execute("SELECT * FROM streaming_features WHERE customer_id = %s", (customer_id,))
    row = cursor.fetchone()
    if row:
        cols = [desc[0] for desc in cursor.description]
        for col, val in zip(cols, row):
            if col not in ("customer_id", "updated_at") and val is not None:
                features[col] = float(val)

    # Batch features
    cursor.execute("SELECT * FROM batch_features WHERE customer_id = %s", (customer_id,))
    row = cursor.fetchone()
    if row:
        cols = [desc[0] for desc in cursor.description]
        for col, val in zip(cols, row):
            if col not in ("customer_id", "updated_at") and val is not None:
                if isinstance(val, bool):
                    features[col] = 1.0 if val else 0.0
                else:
                    try:
                        features[col] = float(val)
                    except (ValueError, TypeError):
                        pass

    cursor.close()
    conn.close()
    return features


def assemble_feature_vector(customer_id: str) -> np.ndarray:
    """Assemble ordered feature vector for model inference."""
    features = get_features_from_redis(customer_id)
    db_features = get_features_from_db(customer_id)
    merged = {**db_features, **features}

    vector = []
    for col in ModelConfig.FEATURE_COLUMNS:
        vector.append(merged.get(col, 0.0))

    return np.array(vector, dtype=np.float32)


# ─────────────────────────────────────────────
# Score Storage
# ─────────────────────────────────────────────
def store_risk_score(score_data: dict):
    """Store risk score in both Redis and PostgreSQL."""
    customer_id = score_data["customer_id"]

    # Redis (for fast access)
    redis_client.hset(f"risk_score:{customer_id}", mapping={
        "risk_score": str(score_data["risk_score"]),
        "risk_tier": score_data["risk_tier"],
        "credit_score": str(score_data["credit_score_mapped"]),
        "scored_at": score_data["scored_at"],
    })
    redis_client.expire(f"risk_score:{customer_id}", 86400)

    # PostgreSQL (for historical tracking)
    conn = psycopg2.connect(
        host=PostgresConfig.HOST, port=PostgresConfig.PORT,
        user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
        dbname=PostgresConfig.DB,
    )
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO risk_scores
        (customer_id, risk_score, risk_tier, credit_score_mapped,
         xgboost_score, lstm_score, ensemble_score, top_shap_features, model_version)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        (
            customer_id, score_data["risk_score"], score_data["risk_tier"],
            score_data["credit_score_mapped"], score_data.get("xgboost_score"),
            score_data.get("lstm_score"), score_data["ensemble_score"],
            json.dumps(score_data.get("top_shap_features", [])),
            "v2.0",
        )
    )
    conn.commit()
    cursor.close()
    conn.close()


# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────
@app.on_event("startup")
async def load_models():
    """Load all models on startup."""
    global xgb_model, lgb_model, lstm_model, tft_model, ensemble, stacker
    global shap_explainer, lime_explainer, redis_client, sequence_cache

    # Expose Prometheus /metrics endpoint
    if _prometheus_enabled:
        _instrumentator.expose(app, endpoint="/metrics")

    redis_client = redis_lib.Redis(
        host=RedisConfig.HOST, port=RedisConfig.PORT, db=RedisConfig.DB,
    )

    MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

    # Load XGBoost
    try:
        xgb_path = os.path.join(MODEL_DIR, "xgboost_model.joblib")
        if os.path.exists(xgb_path):
            xgb_model = XGBoostDelinquencyModel()
            xgb_model.load(xgb_path)
            logger.info("XGBoost model loaded")
    except Exception as e:
        logger.warning(f"XGBoost load failed: {e}")

    # Load LightGBM
    try:
        lgb_path = os.path.join(MODEL_DIR, "lightgbm_model.joblib")
        if os.path.exists(lgb_path):
            lgb_model = LightGBMDelinquencyModel()
            lgb_model.load(lgb_path)
            logger.info("LightGBM model loaded")
    except Exception as e:
        logger.warning(f"LightGBM load failed: {e}")

    # Load LSTM
    try:
        lstm_path = os.path.join(MODEL_DIR, "lstm_model.pt")
        if os.path.exists(lstm_path):
            lstm_model = LSTMDelinquencyModel()
            lstm_model.load(lstm_path)
            logger.info("LSTM model loaded")
    except Exception as e:
        logger.warning(f"LSTM load failed: {e}")

    # Load TFT
    try:
        tft_path = os.path.join(MODEL_DIR, "tft_model.pt")
        if os.path.exists(tft_path):
            tft_model = TFTDelinquencyModel()
            tft_model.load(tft_path)
            logger.info("TFT model loaded")
    except Exception as e:
        logger.warning(f"TFT load failed: {e}")

    # SHAP
    if xgb_model is not None:
        try:
            shap_explainer = SHAPExplainer(xgb_model.get_booster(), xgb_model.feature_names)
            logger.info("SHAP explainer initialized")
        except Exception as e:
            logger.warning(f"SHAP init failed: {e}")

    # LIME
    if xgb_model is not None:
        try:
            from ml.lime_explainer import LIMEExplainer
            lime_explainer = LIMEExplainer(
                predict_fn=xgb_model.predict_proba,
                feature_names=xgb_model.feature_names,
            )
            logger.info("LIME explainer initialized")
        except Exception as e:
            logger.warning(f"LIME init failed: {e}")

    # Initialize ensemble + StackingEnsemble
    ensemble = EnsembleScorer()
    meta_path = os.path.join(MODEL_DIR, "meta_learner.joblib")
    stacker = StackingEnsemble(meta_learner_path=meta_path)
    logger.info(f"Meta-learner loaded: {stacker.meta_learner is not None}")

    # Sequence cache
    try:
        sequence_cache = SequenceCache()
        cached_count = sequence_cache.get_cached_customer_count()
        logger.info(f"Sequence cache connected ({cached_count} cached customers)")
    except Exception as e:
        logger.warning(f"Sequence cache unavailable: {e}")

    logger.info(
        f"Scoring service v3.0 ready "
        f"(XGB:{xgb_model is not None} LGB:{lgb_model is not None} "
        f"LSTM:{lstm_model is not None} TFT:{tft_model is not None} "
        f"MetaLearner:{stacker.meta_learner is not None})"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded={
            "xgboost": xgb_model is not None,
            "lightgbm": lgb_model is not None,
            "lstm": lstm_model is not None,
            "tft": tft_model is not None,
            "meta_learner": stacker.meta_learner is not None if stacker else False,
            "shap": shap_explainer is not None,
            "lime": lime_explainer is not None,
        },
        timestamp=datetime.now().isoformat(),
    )


@app.post("/score", response_model=ScoreResponse)
async def score_customer(request: ScoreRequest):
    """Score a single customer using 4-model ensemble + meta-learner stacking."""
    if xgb_model is None and lgb_model is None:
        raise HTTPException(status_code=503, detail="No models loaded")

    customer_id = request.customer_id

    # Step 1: Assemble feature vector
    features = assemble_feature_vector(customer_id)
    features_2d = features.reshape(1, -1)

    # Step 1b: Classify customer segment
    segment_type = "salaried"  # default
    try:
        feature_dict = dict(zip(ModelConfig.FEATURE_COLUMNS, features))
        segment_info = segment_classifier.classify(feature_dict)
        segment_type = segment_info["segment_type"]
    except Exception as e:
        logger.warning(f"Segment classification failed: {e}")

    # Step 1c: Cold-start check
    is_cold_start = cold_start_scorer.is_cold_start(customer_id)

    # Step 2: XGBoost inference
    xgb_prob = float(xgb_model.predict_proba(features_2d)[0]) if xgb_model else None

    # Step 3: LightGBM inference
    lgb_prob = float(lgb_model.predict_proba(features_2d)[0]) if lgb_model else None

    # Step 4: TFT inference (from sequence cache)
    tft_prob = None
    if tft_model is not None and sequence_cache is not None:
        cached = sequence_cache.get_tft_sequence(customer_id)
        if cached is not None:
            try:
                seq = cached["sequence"].reshape(1, *cached["sequence"].shape)
                stat = cached["static"].reshape(1, -1)
                tft_prob = float(tft_model.predict_proba(seq, stat)[0])
                # Cache attention weights for dashboard
                attn = tft_model.get_attention_weights(seq, stat)
                if attn is not None:
                    sequence_cache.cache_attention_weights(customer_id, attn)
            except Exception as e:
                logger.warning(f"TFT inference failed: {e}")

    # Step 5: LSTM inference (skip for single-score without sequences)
    lstm_prob = None

    # Step 6: Ensemble scoring (meta-learner or fixed weights)
    meta_learner_used = False
    if stacker and stacker.meta_learner is not None and not is_cold_start:
        customer_meta = {
            "income_bracket": "middle",  # Retrieved from features if available
            "segment_type": segment_type,
            "tenure_months": int(features_2d[0][ModelConfig.FEATURE_COLUMNS.index("tenure_months")])
                if "tenure_months" in ModelConfig.FEATURE_COLUMNS else 36,
            "credit_score": int(features_2d[0][ModelConfig.FEATURE_COLUMNS.index("credit_score")])
                if "credit_score" in ModelConfig.FEATURE_COLUMNS else 700,
        }
        final_score = stacker.combine_stacked(
            xgb_prob=xgb_prob, lgb_prob=lgb_prob,
            tft_prob=tft_prob, lstm_prob=lstm_prob,
            customer_meta=customer_meta,
        )
        meta_learner_used = True
    else:
        final_score = ensemble.combine(xgb_prob, lgb_prob, lstm_prob, tft_prob)

    # Step 6b: Cold-start scoring cap
    if is_cold_start:
        cs_result = cold_start_scorer.score(customer_id, features_2d[0])
        final_score = min(final_score, cs_result["risk_score"])

    risk_tier = ensemble.score_to_risk_tier(final_score)
    credit_score = ensemble.score_to_credit_score(final_score)

    # Cold-start cap: never assign critical
    if is_cold_start and risk_tier == "critical":
        risk_tier = "watch"

    # Step 7: SHAP explanation
    top_shap = None
    explanation = None
    if shap_explainer:
        try:
            shap_result = shap_explainer.explain_single(features_2d)
            top_shap = shap_result["top_drivers"]
            explanation = shap_result["explanation"]
        except Exception as e:
            logger.warning(f"SHAP failed: {e}")

    # Step 8: LIME explanation
    top_lime = None
    if lime_explainer:
        try:
            lime_result = lime_explainer.explain_single(features_2d)
            top_lime = lime_result["top_drivers"]
            if not explanation:
                explanation = lime_result.get("explanation")
        except Exception as e:
            logger.warning(f"LIME failed: {e}")

    scored_at = datetime.now().isoformat()

    # Step 9: Product action proposals (for watch/critical)
    actions = []
    if risk_tier in ("watch", "critical"):
        try:
            feature_dict = dict(zip(ModelConfig.FEATURE_COLUMNS, features))
            actions = product_engine.generate_proposals(
                customer_id, feature_dict, final_score, risk_tier
            )
        except Exception as e:
            logger.warning(f"Product actions failed: {e}")

    # Step 10: Store results
    score_data = {
        "customer_id": customer_id,
        "risk_score": final_score,
        "risk_tier": risk_tier,
        "credit_score_mapped": credit_score,
        "xgboost_score": xgb_prob,
        "lightgbm_score": lgb_prob,
        "lstm_score": lstm_prob,
        "tft_score": tft_prob,
        "ensemble_score": final_score,
        "segment_type": segment_type,
        "is_cold_start": is_cold_start,
        "meta_learner_used": meta_learner_used,
        "top_shap_features": top_shap,
        "top_lime_features": top_lime,
        "scored_at": scored_at,
    }
    try:
        store_risk_score(score_data)
    except Exception as e:
        logger.warning(f"PostgreSQL/Redis score storage failed: {e}")

    # Cassandra — high-throughput time-series risk score storage
    try:
        cassandra_write_risk_score(
            customer_id=customer_id,
            risk_score=final_score,
            risk_tier=risk_tier,
            credit_score=credit_score,
            xgboost_score=xgb_prob,
            lightgbm_score=lgb_prob,
            lstm_score=lstm_prob,
            ensemble_score=final_score,
            top_features=json.dumps(top_shap) if top_shap else None,
        )
    except Exception as e:
        logger.warning(f"Cassandra score storage failed (non-blocking): {e}")

    return ScoreResponse(
        customer_id=customer_id,
        risk_score=final_score,
        risk_tier=risk_tier,
        credit_score_mapped=credit_score,
        segment_type=segment_type,
        is_cold_start=is_cold_start,
        xgboost_score=xgb_prob or 0.0,
        lightgbm_score=lgb_prob,
        lstm_score=lstm_prob,
        tft_score=tft_prob,
        ensemble_score=final_score,
        meta_learner_used=meta_learner_used,
        top_shap_features=top_shap,
        top_lime_features=top_lime,
        explanation=explanation,
        product_actions=[a["action_type"] for a in actions] if actions else None,
        scored_at=scored_at,
    )


@app.post("/score/batch")
async def score_batch(request: BatchScoreRequest):
    """Score multiple customers."""
    results = []
    for customer_id in request.customer_ids:
        try:
            req = ScoreRequest(customer_id=customer_id)
            result = await score_customer(req)
            results.append(result.dict())
        except Exception as e:
            results.append({"customer_id": customer_id, "error": str(e)})
    return {"results": results}


@app.get("/score/{customer_id}")
async def get_score(customer_id: str):
    """Get latest stored risk score for a customer."""
    conn = psycopg2.connect(
        host=PostgresConfig.HOST, port=PostgresConfig.PORT,
        user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
        dbname=PostgresConfig.DB,
    )
    cursor = conn.cursor()
    cursor.execute(
        """SELECT risk_score, risk_tier, credit_score_mapped, xgboost_score,
                  lstm_score, ensemble_score, top_shap_features, scored_at
           FROM risk_scores WHERE customer_id = %s
           ORDER BY scored_at DESC LIMIT 1""",
        (customer_id,)
    )
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="No score found for customer")

    return {
        "customer_id": customer_id,
        "risk_score": float(row[0]),
        "risk_tier": row[1],
        "credit_score_mapped": row[2],
        "xgboost_score": float(row[3]) if row[3] else None,
        "lstm_score": float(row[4]) if row[4] else None,
        "ensemble_score": float(row[5]) if row[5] else None,
        "top_shap_features": row[6],
        "scored_at": row[7].isoformat() if row[7] else None,
    }


@app.get("/explain/{customer_id}")
async def explain_customer(customer_id: str):
    """Get both SHAP and LIME explanations for a customer."""
    features = assemble_feature_vector(customer_id)
    features_2d = features.reshape(1, -1)

    result = {"customer_id": customer_id}

    if shap_explainer:
        try:
            result["shap"] = shap_explainer.explain_single(features_2d)
        except Exception as e:
            result["shap"] = {"error": str(e)}

    if lime_explainer:
        try:
            result["lime"] = lime_explainer.explain_single(features_2d)
        except Exception as e:
            result["lime"] = {"error": str(e)}

    return result


if __name__ == "__main__":
    uvicorn.run(app, host=ScoringConfig.HOST, port=ScoringConfig.PORT)
