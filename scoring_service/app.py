# pyre-ignore-all-errors
"""
FastAPI Scoring Service — v3.0
Provides REST endpoints for real-time delinquency risk scoring.
Integrates XGBoost + LightGBM + TFT ensemble with meta-learner stacking, calibrated PD,
cold-start handling, segment classification, product action proposals,
SHAP + LIME explainability, Cassandra storage, and Prometheus metrics.
"""
import os
import sys
import json
import logging
from datetime import datetime
from typing import List, Optional

import re
import numpy as np
import pandas as pd
import redis as redis_lib
import psycopg2
from psycopg2.extras import execute_values
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, validator
from typing import Optional as Opt
import uvicorn

# Rate limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    _limiter = Limiter(key_func=get_remote_address)
    _RATELIMIT_ENABLED = True
except ImportError:
    _RATELIMIT_ENABLED = False
    _limiter = None

# Auth & Audit
from scoring_service.auth import (
    TokenPayload, require_role, authenticate_user_db, create_access_token,
    get_current_user,
)
from scoring_service.audit import (
    write_audit_event, get_request_ip, AuditEvent, mask_email, mask_phone,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import threading
from config.settings import (
    PostgresConfig, RedisConfig, ModelConfig, ScoringConfig, FeastConfig,
)
from ml.xgboost_model import XGBoostDelinquencyModel
from ml.lightgbm_model import LightGBMDelinquencyModel
from ml.ensemble import EnsembleScorer, StackingEnsemble
from ml.calibration import ProbabilityCalibrator
from ml.explainability import SHAPExplainer
from ml.tft_model import TFTDelinquencyModel
from ml.cold_start import ColdStartScorer
from ml.segment_classifier import CustomerSegmentClassifier
from scoring_service.sequence_cache import SequenceCache
from intervention.product_actions import ProductActionEngine
from scoring_service.cassandra_client import write_risk_score as cassandra_write_risk_score

from ml.survival_model import SurvivalModel
from ml.conformal import ConformalPredictor
from ml.ab_holdout import ABHoldout
from ml.shadow_scorer import ShadowScorer
from ml.uplift_model import UpliftModel
from ml.counterfactual import CounterfactualGenerator
from scoring_service.grpc_server import start_grpc_server
from intervention.notification_dispatcher import process_and_notify
from scoring_service.whatif import router as whatif_router

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Barclays Pre-Delinquency Intervention Engine",
    description="Real-time credit risk scoring and proactive intervention platform. "
                "XGBoost + LightGBM + TFT ensemble with calibrated PD, meta-learner stacking, "
                "SHAP + LIME explainability, JWT/API-key auth, RBAC, full audit logging.",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── Rate limiting ───
if _RATELIMIT_ENABLED:
    app.state.limiter = _limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ─── CORS: locked to specific origins (no wildcard) ───
_ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:8050,http://localhost:8000",
    ).split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
)

# ─── Request size limit middleware ───
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

class _RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 1_048_576:  # 1 MB
            return JSONResponse({"error": "Request body too large (max 1 MB)"}, status_code=413)
        return await call_next(request)

app.add_middleware(_RequestSizeLimitMiddleware)

# ─── Register sub-routers ───
app.include_router(whatif_router)

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
tft_model: Optional[TFTDelinquencyModel] = None
ensemble: Optional[EnsembleScorer] = None
stacker: Optional[StackingEnsemble] = None
calibrator: Optional[ProbabilityCalibrator] = None
shap_explainer: Optional[SHAPExplainer] = None
lime_explainer = None
redis_client: Optional[redis_lib.Redis] = None
cold_start_scorer = ColdStartScorer()
segment_classifier = CustomerSegmentClassifier()
sequence_cache: Optional[SequenceCache] = None
product_engine = ProductActionEngine()

# Phase 2 global instances
survival_model: Optional[SurvivalModel] = None
conformal_predictor: Optional[ConformalPredictor] = None
ab_holdout = ABHoldout()
shadow_scorer = ShadowScorer()
uplift_model: Optional[UpliftModel] = None
counterfactual_gen = CounterfactualGenerator()


# ─────────────────────────────────────────────
# Request/Response Models
# ─────────────────────────────────────────────
class ScoreRequest(BaseModel):
    customer_id: str = Field(..., min_length=1, max_length=50, pattern=r'^[A-Za-z0-9_\-]+$')

class BatchScoreRequest(BaseModel):
    customer_ids: List[str] = Field(..., max_items=500)

class NotifyRequest(BaseModel):
    """Typed, validated payload for the /notify endpoint. Prevents injection attacks."""
    customer_id: str = Field(..., min_length=1, max_length=50, pattern=r'^[A-Za-z0-9_\-]+$')
    customer_name: str = Field(..., min_length=1, max_length=200)
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_tier: str = Field(..., pattern=r'^(stable|low_watch|medium_watch|high_watch|watch|critical|severe)$')
    alert_message: str = Field(..., min_length=1, max_length=2000)
    city: Opt[str] = Field(None, max_length=100)
    phone: Opt[str] = Field(None, max_length=20)
    email: Opt[str] = Field(None, max_length=200)
    salary: Opt[float] = Field(None, ge=0)

    @validator('alert_message', 'customer_name', pre=True)
    def strip_html(cls, v):
        """Remove HTML tags to prevent stored XSS."""
        if not v:
            return v
        # Remove HTML tags
        clean = re.sub(r'<[^>]+>', '', str(v))
        # Remove script-like patterns
        clean = re.sub(r'(?i)(javascript:|vbscript:|on\w+=)', '', clean)
        return clean.strip()

    @validator('email')
    def validate_email(cls, v):
        if v and '@' not in v:
            raise ValueError('Invalid email format')
        return v

class AuthTokenRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1, max_length=200)

class ScoreResponse(BaseModel):
    customer_id: str
    risk_score: float
    risk_tier: str
    credit_score_mapped: int
    segment_type: Optional[str] = None
    is_cold_start: bool = False
    xgboost_score: float
    lightgbm_score: Optional[float] = None
    calibrated_pd: Optional[float] = None
    tft_score: Optional[float] = None
    ensemble_score: float
    meta_learner_used: bool = False
    top_shap_features: Optional[list] = None
    top_lime_features: Optional[list] = None
    explanation: Optional[str] = None
    product_actions: Optional[list] = None
    # Phase 2 fields
    tte_days: Optional[float] = None
    p30d: Optional[float] = None
    p60d: Optional[float] = None
    p90d: Optional[float] = None
    risk_score_lower: Optional[float] = None
    risk_score_upper: Optional[float] = None
    confidence_flag: Optional[str] = None
    uplift_score: Optional[float] = None
    holdout_group: Optional[str] = None
    shadow_score: Optional[float] = None
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
         xgboost_score, ensemble_score, top_shap_features, model_version)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
        (
            customer_id, score_data["risk_score"], score_data["risk_tier"],
            score_data["credit_score_mapped"], score_data.get("xgboost_score"),
            score_data["ensemble_score"],
            json.dumps(score_data.get("top_shap_features", [])),
            "v3.0",
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
    global xgb_model, lgb_model, tft_model, ensemble, stacker, calibrator
    global shap_explainer, lime_explainer, redis_client, sequence_cache
    global survival_model, conformal_predictor, uplift_model

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

    # Load probability calibrator (IFRS 9 PD)
    try:
        cal_path = os.path.join(MODEL_DIR, "calibrator.joblib")
        calibrator = ProbabilityCalibrator()
        calibrator.load(cal_path)
        logger.info(f"Probability calibrator loaded: {calibrator._is_fitted}")
    except Exception as e:
        logger.warning(f"Calibrator load failed: {e}")

    # Sequence cache
    try:
        sequence_cache = SequenceCache()
        cached_count = sequence_cache.get_cached_customer_count()
        logger.info(f"Sequence cache connected ({cached_count} cached customers)")
    except Exception as e:
        logger.warning(f"Sequence cache unavailable: {e}")

    # Phase 2: Load survival model (P1)
    try:
        surv_path = os.path.join(MODEL_DIR, "survival_model.joblib")
        survival_model = SurvivalModel()
        survival_model.load(surv_path)
        logger.info(f"Survival model loaded: {survival_model._is_fitted}")
    except Exception as e:
        logger.warning(f"Survival model load failed: {e}")

    # Phase 2: Load conformal predictor (P6)
    try:
        conf_path = os.path.join(MODEL_DIR, "conformal_predictor.joblib")
        conformal_predictor = ConformalPredictor()
        conformal_predictor.load(conf_path)
        logger.info(f"Conformal predictor loaded: {conformal_predictor._is_calibrated}")
    except Exception as e:
        logger.warning(f"Conformal predictor load failed: {e}")

    # Phase 2: Load uplift model (P9)
    try:
        uplift_path = os.path.join(MODEL_DIR, "uplift_model.joblib")
        uplift_model = UpliftModel()
        uplift_model.load(uplift_path)
        logger.info(f"Uplift model loaded: {uplift_model._is_fitted}")
    except Exception as e:
        logger.warning(f"Uplift model load failed: {e}")

    # Phase 2: Load shadow candidate model (P8)
    try:
        shadow_path = os.path.join(MODEL_DIR, "shadow_candidate.joblib")
        shadow_scorer.load_candidate(shadow_path)
    except Exception as e:
        logger.info(f"No shadow candidate model: {e}")

    # Phase 2: Start gRPC server alongside REST (P12)
    def _grpc_score_fn(cid):
        import asyncio
        loop = asyncio.new_event_loop()
        return loop.run_until_complete(score_customer(ScoreRequest(customer_id=cid)))

    def _grpc_health_fn():
        return {"status": "healthy", "version": "4.0",
                "models_loaded": xgb_model is not None}

    try:
        start_grpc_server(_grpc_score_fn, _grpc_health_fn)
    except Exception as e:
        logger.info(f"gRPC server not started: {e}")

    logger.info(
        f"Scoring service v4.0 ready "
        f"(XGB:{xgb_model is not None} LGB:{lgb_model is not None} "
        f"TFT:{tft_model is not None} Calibrator:{calibrator is not None} "
        f"MetaLearner:{stacker.meta_learner is not None} "
        f"Survival:{survival_model is not None} "
        f"Conformal:{conformal_predictor is not None} "
        f"Uplift:{uplift_model is not None})"
    )



# ─────────────────────────────────────────────
# Auth endpoint — public
# ─────────────────────────────────────────────
@app.post("/auth/token", tags=["Auth"])
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Obtain a JWT access token.
    Accepts application/x-www-form-urlencoded with username + password fields.
    """
    user = authenticate_user_db(form_data.username, form_data.password)
    if not user:
        write_audit_event(
            event_type=AuditEvent.LOGIN_FAILURE,
            actor_id=form_data.username,
            actor_role="unknown",
            action="login_attempt",
            outcome="FAILURE",
            request_ip=get_request_ip(request),
        )
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(subject=user.sub, role=user.role)
    write_audit_event(
        event_type=AuditEvent.LOGIN_SUCCESS,
        actor_id=user.sub,
        actor_role=user.role,
        action="login",
        outcome="SUCCESS",
        request_ip=get_request_ip(request),
    )
    return {"access_token": token, "token_type": "bearer", "role": user.role}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check — public, returns minimal system status."""
    return HealthResponse(
        status="healthy",
        models_loaded={
            "xgboost": xgb_model is not None,
            "lightgbm": lgb_model is not None,
            "tft": tft_model is not None,
            "calibrator": calibrator is not None if calibrator else False,
            "meta_learner": stacker.meta_learner is not None if stacker else False,
            "shap": shap_explainer is not None,
            "lime": lime_explainer is not None,
        },
        timestamp=datetime.now().isoformat(),
    )


# ─────────────────────────────────────────────
# Notification endpoint — requires risk_officer+
# ─────────────────────────────────────────────
@app.post("/notify", tags=["Interventions"])
async def notify_customer(
    payload: NotifyRequest,
    request: Request,
    current_user: TokenPayload = Depends(require_role("risk_officer", "admin", "service_account")),
):
    """
    Send intervention notification for a customer.
    Called by n8n workflow after scoring.

    Expects JSON:
    {
        "customer_id": "...",
        "customer_name": "...",
        "risk_score": 0.72,
        "risk_tier": "watch",
        "alert_message": "...",
        "city": "Delhi",
        "phone": "+91...",      # optional
        "email": "..."          # optional
    }
    """
    customer_id = payload.customer_id
    risk_score = payload.risk_score
    alert_message = payload.alert_message
    customer_name = payload.customer_name
    risk_tier = payload.risk_tier

    # Build customer dict — phone/email fallback to env test values only if not provided
    customer = {
        "customer_id": customer_id,
        "first_name": customer_name.split()[0] if customer_name else "",
        "last_name": " ".join(customer_name.split()[1:]) if customer_name else "",
        "phone": payload.phone or os.getenv("TEST_PHONE_TO", ""),
        "email": payload.email or os.getenv("SMTP_USER", ""),
        "city": payload.city or "",
        "monthly_salary": payload.salary or 50000,
    }

    # Build intervention dict
    intervention = {
        "risk_score": risk_score,
        "risk_tier": risk_tier,
        "intervention_type": "proactive_outreach" if risk_score >= 0.7 else "wellness_checkin",
        "shap_drivers": [],
    }

    # Build messages dict
    messages = {
        "sms": f"[Barclays PDI] {alert_message}"[:1600],
        "email_subject": f"Barclays - Risk Alert for {customer_name}",
        "email_html": f"""
            <div style='font-family:Arial;padding:20px;'>
                <h2 style='color:#0a2463;'>⚠️ Pre-Delinquency Alert</h2>
                <p><strong>Customer:</strong> {customer_name}</p>
                <p><strong>Risk Score:</strong> {risk_score:.1%}</p>
                <p><strong>Risk Tier:</strong> {risk_tier.upper()}</p>
                <hr/>
                <pre style='background:#f5f5f5;padding:10px;'>{alert_message}</pre>
                <hr/>
                <p style='color:#666;font-size:12px;'>
                    Sent by Barclays Pre-Delinquency Intervention Engine
                </p>
            </div>
        """,
        "email_text": alert_message,
        "whatsapp": f"🏦 Barclays PDI Alert\n{alert_message}",
        "push_title": f"Risk Alert: {customer_name}",
        "push_body": alert_message,
        "rm_call_script": f"Contact {customer_name} ({customer_id}). Risk: {risk_score:.1%}. {alert_message}",
        "collector_brief": f"Escalation for {customer_name}. Risk: {risk_score:.1%}.",
    }

    try:
        from intervention.notification_dispatcher import dispatch_notification
        results = dispatch_notification(customer, intervention, messages)
        channels = [r.get("channel", "unknown") for r in results]
        logger.info(f"[Notify] Dispatched {len(results)} notifications for {customer_id}")
        write_audit_event(
            event_type=AuditEvent.NOTIFY_DISPATCH,
            actor_id=current_user.sub,
            actor_role=current_user.role,
            action="notify_dispatch",
            outcome="SUCCESS",
            customer_id=customer_id,
            request_ip=get_request_ip(request),
            details={"risk_tier": risk_tier, "channels": channels, "num_sent": len(results)},
        )
        return {
            "status": "dispatched",
            "customer_id": customer_id,
            "risk_score": risk_score,
            "channels_attempted": len(results),
            "results": results,
        }
    except Exception as e:
        logger.error(f"[Notify] Failed for {customer_id}: {e}")
        write_audit_event(
            event_type=AuditEvent.NOTIFY_DISPATCH,
            actor_id=current_user.sub,
            actor_role=current_user.role,
            action="notify_dispatch",
            outcome="FAILURE",
            customer_id=customer_id,
            request_ip=get_request_ip(request),
            details={"error": str(e)[:200]},
        )
        return {"status": "error", "customer_id": customer_id, "error": str(e)}


@app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
async def score_customer(
    request: ScoreRequest,
    http_request: Request,
    current_user: TokenPayload = Depends(require_role("analyst", "risk_officer", "admin", "service_account")),
):
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

    # Step 5: Ensemble scoring (meta-learner or fixed weights)
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
            tft_prob=tft_prob,
            customer_meta=customer_meta,
        )
        meta_learner_used = True
    else:
        final_score = ensemble.combine(xgb_prob, lgb_prob, tft_prob=tft_prob)

    # Step 5b: Probability calibration (IFRS 9 PD)
    if calibrator and calibrator._is_fitted:
        final_score = calibrator.calibrate(final_score)

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

    # Step 10 (P1): Survival analysis — TTE, p30d, p60d, p90d
    tte_days = None
    p30d = None
    p60d = None
    p90d = None
    if survival_model is not None and survival_model._is_fitted:
        try:
            surv = survival_model.predict(features, risk_score=final_score)
            tte_days = surv["tte_days"]
            p30d = surv["p30d"]
            p60d = surv["p60d"]
            p90d = surv["p90d"]
        except Exception as e:
            logger.warning(f"Survival prediction failed: {e}")
    else:
        # Fallback: derive from risk score
        surv_fallback = SurvivalModel()._score_from_risk_score(final_score)
        tte_days = surv_fallback["tte_days"]
        p30d = surv_fallback["p30d"]
        p60d = surv_fallback["p60d"]
        p90d = surv_fallback["p90d"]

    # Step 11 (P6): Conformal prediction interval
    risk_score_lower = None
    risk_score_upper = None
    confidence_flag = None
    if conformal_predictor is not None:
        lower, upper = conformal_predictor.predict_interval(final_score)
        risk_score_lower = lower
        risk_score_upper = upper
        confidence_flag = conformal_predictor.uncertainty_flag(lower, upper)

    # Step 12 (P4): A/B holdout check
    holdout_assignment = ab_holdout.get_assignment(customer_id)
    holdout_group = holdout_assignment["group"]
    try:
        ab_holdout.save_assignment(customer_id, risk_tier)
    except Exception:
        pass

    # Step 13 (P8): Shadow scoring
    shadow_score_val = None
    try:
        shadow_score_val = shadow_scorer.shadow_score(customer_id, features)
        if shadow_score_val is not None:
            shadow_scorer.persist_shadow_score(
                customer_id, final_score, shadow_score_val
            )
    except Exception:
        pass

    # Step 14 (P9): Uplift score
    uplift_score_val = None
    if uplift_model is not None:
        try:
            uplift_score_val = uplift_model.predict_uplift_single(features)
        except Exception:
            pass

    # Step 15: Store results (expanded with Phase 2 columns)
    score_data = {
        "customer_id": customer_id,
        "risk_score": final_score,
        "risk_tier": risk_tier,
        "credit_score_mapped": credit_score,
        "xgboost_score": xgb_prob,
        "lightgbm_score": lgb_prob,
        "calibrated_pd": final_score,
        "tft_score": tft_prob,
        "ensemble_score": final_score,
        "segment_type": segment_type,
        "is_cold_start": is_cold_start,
        "meta_learner_used": meta_learner_used,
        "top_shap_features": top_shap,
        "top_lime_features": top_lime,
        "tte_days": tte_days,
        "p30d": p30d,
        "p60d": p60d,
        "p90d": p90d,
        "risk_score_lower": risk_score_lower,
        "risk_score_upper": risk_score_upper,
        "confidence_flag": confidence_flag,
        "uplift_score": uplift_score_val,
        "shadow_score": shadow_score_val,
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
            calibrated_pd=final_score,
            ensemble_score=final_score,
            top_features=json.dumps(top_shap) if top_shap else None,
        )
    except Exception as e:
        logger.warning(f"Cassandra score storage failed (non-blocking): {e}")

    # --- NEW: Trigger Notification Dispatcher for n8n demo ---
    if final_score >= 0.5:
        demo_customer = {
            "customer_id": customer_id,
            "first_name": "Demo",
            "last_name": "Customer",
            "email": os.getenv("SMTP_USER", ""),  # Sends email to yourself
            "phone": os.getenv("TEST_PHONE_TO", ""), # Gets your test phone number
            "monthly_salary": 50000,
            "dti_ratio": 0.45
        }
        demo_intervention = {
            "risk_score": final_score,
            "risk_tier": risk_tier,
            "intervention_type": "escalation_call" if final_score >= 0.75 else "wellness_checkin",
            "shap_drivers": top_shap if top_shap else []
        }
        logger.info(f"Triggering background notification for {customer_id}")
        threading.Thread(target=process_and_notify, args=(demo_customer, demo_intervention)).start()

    # Audit the scoring event
    write_audit_event(
        event_type=AuditEvent.SCORE_REQUEST,
        actor_id=current_user.sub,
        actor_role=current_user.role,
        action="score_customer",
        outcome="SUCCESS",
        customer_id=customer_id,
        request_ip=get_request_ip(http_request),
        details={
            "risk_tier": risk_tier,
            "is_cold_start": is_cold_start,
            "segment_type": segment_type,
        },
    )

    return ScoreResponse(
        customer_id=customer_id,
        risk_score=final_score,
        risk_tier=risk_tier,
        credit_score_mapped=credit_score,
        segment_type=segment_type,
        is_cold_start=is_cold_start,
        xgboost_score=xgb_prob or 0.0,
        lightgbm_score=lgb_prob,
        calibrated_pd=final_score,
        tft_score=tft_prob,
        ensemble_score=final_score,
        meta_learner_used=meta_learner_used,
        top_shap_features=top_shap,
        top_lime_features=top_lime,
        explanation=explanation,
        product_actions=[a["action_type"] for a in actions] if actions else None,
        tte_days=tte_days,
        p30d=p30d,
        p60d=p60d,
        p90d=p90d,
        risk_score_lower=risk_score_lower,
        risk_score_upper=risk_score_upper,
        confidence_flag=confidence_flag,
        uplift_score=uplift_score_val,
        holdout_group=holdout_group,
        shadow_score=shadow_score_val,
        scored_at=scored_at,
    )


@app.post("/score/batch", tags=["Scoring"])
async def score_batch(
    request: BatchScoreRequest,
    http_request: Request,
    current_user: TokenPayload = Depends(require_role("risk_officer", "admin", "service_account")),
):
    """Score multiple customers. Requires risk_officer role or higher."""
    results = []
    for customer_id in request.customer_ids:
        try:
            req = ScoreRequest(customer_id=customer_id)
            result = await score_customer(req, http_request, current_user)
            results.append(result.dict())
        except Exception as e:
            results.append({"customer_id": customer_id, "error": str(e)})
    write_audit_event(
        event_type=AuditEvent.SCORE_BATCH_REQUEST,
        actor_id=current_user.sub,
        actor_role=current_user.role,
        action="score_batch",
        outcome="SUCCESS",
        request_ip=get_request_ip(http_request),
        details={"batch_size": len(request.customer_ids), "success_count": sum(1 for r in results if "error" not in r)},
    )
    return {"results": results}


@app.get("/score/{customer_id}", tags=["Scoring"])
async def get_score(
    customer_id: str,
    http_request: Request,
    current_user: TokenPayload = Depends(require_role("analyst", "risk_officer", "admin", "read_only", "service_account")),
):
    """Get latest stored risk score for a customer."""
    conn = psycopg2.connect(
        host=PostgresConfig.HOST, port=PostgresConfig.PORT,
        user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
        dbname=PostgresConfig.DB,
    )
    cursor = conn.cursor()
    cursor.execute(
        """SELECT risk_score, risk_tier, credit_score_mapped, xgboost_score,
                  ensemble_score, top_shap_features, scored_at
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
        "ensemble_score": float(row[4]) if row[4] else None,
        "top_shap_features": row[5],
        "scored_at": row[6].isoformat() if row[6] else None,
    }


@app.get("/explain/{customer_id}", tags=["Explainability"])
async def explain_customer(
    customer_id: str,
    http_request: Request,
    current_user: TokenPayload = Depends(require_role("analyst", "risk_officer", "admin")),
):
    """Get SHAP, LIME, and counterfactual explanations for a customer."""
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

    # P5: Counterfactual explanations
    try:
        def _predict_fn(x):
            return xgb_model.predict_proba(x) if xgb_model else np.full(len(x), 0.5)

        shap_drivers = result.get("shap", {}).get("top_drivers", [])
        cf = counterfactual_gen.generate(
            features=features,
            feature_names=ModelConfig.FEATURE_COLUMNS,
            predict_fn=_predict_fn,
            shap_drivers=shap_drivers,
        )
        result["counterfactuals"] = cf
    except Exception as e:
        result["counterfactuals"] = {"error": str(e)}

    return result


if __name__ == "__main__":
    uvicorn.run(app, host=ScoringConfig.HOST, port=ScoringConfig.PORT)
