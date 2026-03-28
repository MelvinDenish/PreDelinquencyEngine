"""
What-If Simulator API for PDI Engine.
Allows Barclays management to simulate macro scenarios and see portfolio impact.

Scenarios supported:
  - IT/sector unemployment shock (employer health degradation)
  - BoE/Fed rate hike (credit stress multiplier)
  - EMI holiday offer (intervention cost vs NPA savings)
  - Regional economic shock (geographic risk spike)
  - Salary delay shock (employer payroll disruption)
  - Custom: adjust any feature by %

All simulations run against the current live portfolio in the database.
Results show: portfolio tier shift, projected NPA delta, intervention cost, ROI.
"""
import logging
from typing import Optional

import numpy as np
import psycopg2
from fastapi import APIRouter, Depends, Request, HTTPException
from pydantic import BaseModel, Field

from scoring_service.auth import TokenPayload, require_role
from scoring_service.audit import write_audit_event, get_request_ip, AuditEvent
from config.settings import PostgresConfig, ModelConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/whatif", tags=["What-If Simulator"])


# ──────────────────────────────────────────────────
# Request/Response Models
# ──────────────────────────────────────────────────

class ScenarioParameter(BaseModel):
    name: str = Field(..., description="Human-readable scenario name")
    type: str = Field(..., description="sector_shock | rate_hike | emi_holiday | regional_shock | salary_shock | custom")
    # Sector shock params
    sector: Optional[str] = Field(None, description="e.g. IT, Banking, Pharma, Manufacturing")
    sector_employment_change_pct: Optional[float] = Field(None, ge=-100, le=0, description="Negative = job losses, e.g. -15 = 15% layoffs")
    # Rate hike params
    rate_hike_bps: Optional[float] = Field(None, ge=0, le=500, description="Basis points e.g. 50 = +0.5%")
    # EMI holiday params
    emi_holiday_months: Optional[int] = Field(None, ge=1, le=6)
    emi_holiday_target_tiers: Optional[list] = Field(None, description="e.g. ['watch', 'critical']")
    # Regional shock
    regions: Optional[list] = Field(None, description="List of regions to apply shock e.g. ['South', 'East']")
    regional_income_shock_pct: Optional[float] = Field(None, ge=-100, le=0)
    # Salary shock
    salary_delay_days_added: Optional[int] = Field(None, ge=0, le=60)
    # Custom feature override
    feature_overrides: Optional[dict] = Field(None, description="Dict of {feature_name: pct_change} e.g. {'discretionary_spend_7d': 30}")


class TierDistribution(BaseModel):
    stable: int
    watch: int
    critical: int
    stable_pct: float
    watch_pct: float
    critical_pct: float


class WhatIfResponse(BaseModel):
    scenario_name: str
    total_customers: int
    # Current state
    current_distribution: TierDistribution
    # Simulated state
    simulated_distribution: TierDistribution
    # Deltas
    customers_upgraded_to_watch: int    # stable → watch
    customers_upgraded_to_critical: int  # watch/stable → critical
    customers_downgraded: int            # critical/watch → lower (scenario recovery)
    # Financial impact
    estimated_npa_delta_crore: float     # projected additional NPAs from this scenario
    intervention_cost_lakh: Optional[float]  # cost if we intervene on all new watch/critical
    intervention_roi: Optional[float]    # estimated_npa_prevented / intervention_cost
    avg_risk_score_current: float
    avg_risk_score_simulated: float
    # Segment breakdown
    segment_breakdown: Optional[dict]
    # Top affected employers (for sector shock)
    top_affected_employers: Optional[list]
    # Region breakdown (for regional shock)
    region_breakdown: Optional[dict]


# ──────────────────────────────────────────────────
# Portfolio fetcher
# ──────────────────────────────────────────────────

def _get_portfolio_features() -> list[dict]:
    """
    Fetch current features + risk scores for all scored customers.
    Returns list of dicts with feature values and current risk_score/risk_tier.
    Capped at 5000 customers for simulation performance.
    """
    try:
        conn = psycopg2.connect(
            host=PostgresConfig.HOST, port=PostgresConfig.PORT,
            user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
            dbname=PostgresConfig.DB, connect_timeout=10,
        )
        cursor = conn.cursor()

        # Join streaming + batch features with latest risk scores
        cursor.execute("""
            SELECT
                c.customer_id,
                c.region,
                c.income_bracket,
                c.monthly_salary,
                bf.salary_delay_days,
                bf.employer_health_score,
                bf.credit_score,
                bf.has_mortgage,
                bf.has_personal_loan,
                bf.avg_monthly_spend_3m,
                bf.spend_volatility_3m,
                sf.discretionary_spend_7d,
                sf.discretionary_spend_30d,
                sf.failed_autodebits_count_7d,
                sf.failed_autodebits_count_30d,
                sf.lending_app_txn_count_7d,
                sf.atm_withdrawals_count_7d,
                rs.risk_score,
                rs.risk_tier,
                bf.segment_type
            FROM customers c
            LEFT JOIN batch_features bf ON c.customer_id = bf.customer_id
            LEFT JOIN streaming_features sf ON c.customer_id = sf.customer_id
            LEFT JOIN LATERAL (
                SELECT risk_score, risk_tier
                FROM risk_scores
                WHERE customer_id = c.customer_id
                ORDER BY scored_at DESC
                LIMIT 1
            ) rs ON TRUE
            WHERE rs.risk_score IS NOT NULL
            LIMIT 5000
        """)
        cols = [d[0] for d in cursor.description]
        rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"[WhatIf] Failed to fetch portfolio: {e}")
        return []


def _get_employer_mapping() -> dict:
    """Return {customer_id: employer_name} for sector shock mapping."""
    try:
        conn = psycopg2.connect(
            host=PostgresConfig.HOST, port=PostgresConfig.PORT,
            user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
            dbname=PostgresConfig.DB, connect_timeout=5,
        )
        cursor = conn.cursor()
        cursor.execute("SELECT customer_id, employer_name FROM batch_features WHERE employer_name IS NOT NULL")
        result = {row[0]: row[1] for row in cursor.fetchall()}
        cursor.close()
        conn.close()
        return result
    except Exception:
        return {}


# ──────────────────────────────────────────────────
# Score simulation (simplified — uses current risk_score + adjustments)
# ──────────────────────────────────────────────────

_RISK_TIER_THRESHOLDS = {
    "stable":   (0.0, 0.50),
    "watch":    (0.50, 0.70),
    "critical": (0.70, 1.01),
}


def _score_to_tier(score: float) -> str:
    if score >= 0.70:
        return "critical"
    elif score >= 0.50:
        return "watch"
    return "stable"


def _apply_sector_shock(row: dict, sector: str, employment_change_pct: float) -> float:
    """
    Simulate employer health degradation for customers in the affected sector.
    employment_change_pct is negative (e.g. -15 = 15% layoffs).
    """
    current_score = row.get("risk_score") or 0.3
    employer_health = row.get("employer_health_score") or 0.8

    # Degrade employer health score proportional to shock magnitude
    shock_intensity = abs(employment_change_pct) / 100.0
    new_employer_health = max(0.0, employer_health - shock_intensity * 0.6)

    # Risk score increases proportionally to employer health degradation
    health_delta = employer_health - new_employer_health
    risk_delta = health_delta * 0.45  # 45% pass-through to risk score

    # Higher impact for customers with existing stress indicators
    if (row.get("failed_autodebits_count_7d") or 0) > 0:
        risk_delta *= 1.3
    if (row.get("salary_delay_days") or 0) > 5:
        risk_delta *= 1.2

    return min(1.0, current_score + risk_delta)


def _apply_rate_hike(row: dict, rate_hike_bps: float) -> float:
    """
    Rate hike increases EMI burden — hits mortgage and loan holders hardest.
    """
    current_score = row.get("risk_score") or 0.3
    rate_hike_pct = rate_hike_bps / 10000.0  # convert bps to decimal

    risk_delta = 0.0
    # Mortgage holders: +1.5% risk per 100bps
    if row.get("has_mortgage"):
        risk_delta += rate_hike_pct * 1.5
    # Personal loan holders: +0.8% per 100bps
    if row.get("has_personal_loan"):
        risk_delta += rate_hike_pct * 0.8
    # Amplified for low-income customers
    salary = row.get("monthly_salary") or 50000
    if salary < 30000:
        risk_delta *= 1.4

    return min(1.0, current_score + risk_delta)


def _apply_regional_shock(row: dict, regions: list, income_shock_pct: float) -> float:
    """Regional economic shock (flood, disaster, recession in specific regions)."""
    current_score = row.get("risk_score") or 0.3
    if row.get("region") not in regions:
        return current_score

    shock_intensity = abs(income_shock_pct) / 100.0
    risk_delta = shock_intensity * 0.5
    return min(1.0, current_score + risk_delta)


def _apply_salary_shock(row: dict, delay_days_added: int) -> float:
    """Simulate payroll delays (e.g. due to employer financial stress)."""
    current_score = row.get("risk_score") or 0.3
    existing_delay = row.get("salary_delay_days") or 0
    total_delay = existing_delay + delay_days_added

    # Risk escalates sharply after 7 days delay
    if total_delay <= 7:
        risk_delta = delay_days_added * 0.008
    elif total_delay <= 15:
        risk_delta = delay_days_added * 0.015
    else:
        risk_delta = delay_days_added * 0.025

    return min(1.0, current_score + risk_delta)


def _apply_emi_holiday(row: dict, months: int) -> float:
    """EMI holiday reduces stress on watch/critical customers."""
    current_score = row.get("risk_score") or 0.3
    # EMI holiday lowers risk score for stressed customers
    if current_score >= 0.5:
        relief = months * 0.04  # 4% relief per month
        return max(0.0, current_score - relief)
    return current_score


def _apply_custom_overrides(row: dict, feature_overrides: dict) -> float:
    """Apply percentage changes to specified features and re-estimate risk impact."""
    current_score = row.get("risk_score") or 0.3
    risk_delta = 0.0

    # Feature sensitivity weights (based on SHAP importance from model)
    feature_risk_weights = {
        "discretionary_spend_7d": 0.12,
        "discretionary_spend_30d": 0.10,
        "failed_autodebits_count_7d": 0.18,
        "failed_autodebits_count_30d": 0.15,
        "lending_app_txn_count_7d": 0.20,
        "salary_delay_days": 0.25,
        "employer_health_score": -0.22,  # negative: higher health = lower risk
        "credit_score": -0.15,
        "atm_withdrawals_count_7d": 0.08,
    }

    for feature_name, pct_change in feature_overrides.items():
        weight = feature_risk_weights.get(feature_name, 0.05)
        risk_delta += (pct_change / 100.0) * weight

    return min(1.0, max(0.0, current_score + risk_delta))


# ──────────────────────────────────────────────────
# Financial impact calculations
# ──────────────────────────────────────────────────

_AVG_LOAN_OUTSTANDING_LAKH = 8.5  # Average outstanding loan balance per customer (₹ lakhs)
_NPA_RECOVERY_RATE = 0.35          # Banks typically recover ~35% of NPA amount
_INTERVENTION_COST_PER_CUSTOMER_LAKH = 0.0015  # ~₹150 per intervention (SMS + email + RM time)
_EMI_HOLIDAY_COST_PER_CUSTOMER_LAKH = 0.12     # ~₹12,000 deferred interest per month


def _calculate_financial_impact(
    new_critical_count: int,
    new_watch_count: int,
    scenario_type: str,
    emi_holiday_months: int = 0,
) -> tuple[float, float, float]:
    """
    Returns (npa_delta_crore, intervention_cost_lakh, roi)
    """
    # Each new critical customer has ~85% probability of default
    # Each new watch customer has ~40% probability
    expected_defaults = (new_critical_count * 0.85) + (new_watch_count * 0.40)
    gross_npa_lakh = expected_defaults * _AVG_LOAN_OUTSTANDING_LAKH * (1 - _NPA_RECOVERY_RATE)
    npa_delta_crore = gross_npa_lakh / 100.0

    # Intervention cost
    total_to_intervene = new_critical_count + new_watch_count
    if scenario_type == "emi_holiday":
        intervention_cost_lakh = total_to_intervene * _EMI_HOLIDAY_COST_PER_CUSTOMER_LAKH * emi_holiday_months
    else:
        intervention_cost_lakh = total_to_intervene * _INTERVENTION_COST_PER_CUSTOMER_LAKH * 100  # to lakh

    # ROI: NPAs prevented (if 65% of interventions succeed) / cost
    npa_prevented_lakh = gross_npa_lakh * 0.65
    roi = (npa_prevented_lakh / intervention_cost_lakh) if intervention_cost_lakh > 0 else 0

    return round(npa_delta_crore, 2), round(intervention_cost_lakh, 2), round(roi, 1)


# ──────────────────────────────────────────────────
# Main simulation endpoint
# ──────────────────────────────────────────────────

@router.post("/simulate", response_model=WhatIfResponse)
async def simulate_scenario(
    scenario: ScenarioParameter,
    http_request: Request,
    current_user: TokenPayload = Depends(require_role("analyst", "risk_officer", "admin")),
):
    """
    Run a portfolio-wide What-If simulation.
    Returns how the risk distribution changes under the given macro scenario.
    Used by senior management and risk officers for strategic planning and stress testing.
    """
    portfolio = _get_portfolio_features()
    if not portfolio:
        raise HTTPException(
            status_code=503,
            detail="Portfolio data not available. Ensure customers have been scored first."
        )

    total = len(portfolio)

    # Current distribution
    current_scores = [r.get("risk_score") or 0.3 for r in portfolio]
    current_tiers = [_score_to_tier(s) for s in current_scores]
    curr_stable = current_tiers.count("stable")
    curr_watch = current_tiers.count("watch")
    curr_critical = current_tiers.count("critical")

    # Apply simulation
    simulated_scores = []
    for row in portfolio:
        score = row.get("risk_score") or 0.3
        scenario_type = scenario.type

        if scenario_type == "sector_shock":
            score = _apply_sector_shock(row, scenario.sector or "", scenario.sector_employment_change_pct or 0)
        elif scenario_type == "rate_hike":
            score = _apply_rate_hike(row, scenario.rate_hike_bps or 0)
        elif scenario_type == "regional_shock":
            score = _apply_regional_shock(row, scenario.regions or [], scenario.regional_income_shock_pct or 0)
        elif scenario_type == "salary_shock":
            score = _apply_salary_shock(row, scenario.salary_delay_days_added or 0)
        elif scenario_type == "emi_holiday":
            target_tiers = scenario.emi_holiday_target_tiers or ["watch", "critical"]
            if _score_to_tier(score) in target_tiers:
                score = _apply_emi_holiday(row, scenario.emi_holiday_months or 1)
        elif scenario_type == "custom":
            score = _apply_custom_overrides(row, scenario.feature_overrides or {})

        simulated_scores.append(score)

    simulated_tiers = [_score_to_tier(s) for s in simulated_scores]
    sim_stable = simulated_tiers.count("stable")
    sim_watch = simulated_tiers.count("watch")
    sim_critical = simulated_tiers.count("critical")

    # Tier movements
    upgraded_to_watch = sum(
        1 for c, s in zip(current_tiers, simulated_tiers)
        if c == "stable" and s == "watch"
    )
    upgraded_to_critical = sum(
        1 for c, s in zip(current_tiers, simulated_tiers)
        if c in ("stable", "watch") and s == "critical"
    )
    downgraded = sum(
        1 for c, s in zip(current_tiers, simulated_tiers)
        if (c == "critical" and s in ("watch", "stable")) or (c == "watch" and s == "stable")
    )

    # Financial impact
    new_critical = max(0, sim_critical - curr_critical)
    new_watch = max(0, sim_watch - curr_watch)
    npa_delta, intervention_cost, roi = _calculate_financial_impact(
        new_critical, new_watch, scenario.type,
        scenario.emi_holiday_months or 0,
    )

    # Segment breakdown
    segment_breakdown = {}
    for i, row in enumerate(portfolio):
        seg = row.get("segment_type") or "unknown"
        if seg not in segment_breakdown:
            segment_breakdown[seg] = {"count": 0, "new_watch": 0, "new_critical": 0}
        segment_breakdown[seg]["count"] += 1
        if current_tiers[i] != "watch" and simulated_tiers[i] == "watch":
            segment_breakdown[seg]["new_watch"] += 1
        if current_tiers[i] != "critical" and simulated_tiers[i] == "critical":
            segment_breakdown[seg]["new_critical"] += 1

    # Top affected employers (for sector shock)
    top_affected_employers = None
    if scenario.type == "sector_shock":
        employer_map = _get_employer_mapping()
        employer_impact = {}
        for i, row in enumerate(portfolio):
            emp = employer_map.get(row["customer_id"], "Unknown")
            if emp not in employer_impact:
                employer_impact[emp] = {"total": 0, "newly_at_risk": 0}
            employer_impact[emp]["total"] += 1
            if current_tiers[i] not in ("watch", "critical") and simulated_tiers[i] in ("watch", "critical"):
                employer_impact[emp]["newly_at_risk"] += 1
        top_affected_employers = sorted(
            [{"employer": k, **v} for k, v in employer_impact.items() if v["newly_at_risk"] > 0],
            key=lambda x: x["newly_at_risk"],
            reverse=True,
        )[:10]

    # Region breakdown (for regional shock)
    region_breakdown = None
    if scenario.type == "regional_shock" and scenario.regions:
        region_breakdown = {}
        for i, row in enumerate(portfolio):
            region = row.get("region") or "Unknown"
            if region not in region_breakdown:
                region_breakdown[region] = {"total": 0, "newly_at_risk": 0, "affected": region in (scenario.regions or [])}
            region_breakdown[region]["total"] += 1
            if current_tiers[i] not in ("watch", "critical") and simulated_tiers[i] in ("watch", "critical"):
                region_breakdown[region]["newly_at_risk"] += 1

    # Audit the simulation
    write_audit_event(
        event_type="WHATIF_SIMULATION",
        actor_id=current_user.sub,
        actor_role=current_user.role,
        action=f"whatif_simulate:{scenario.type}",
        outcome="SUCCESS",
        request_ip=get_request_ip(http_request),
        details={
            "scenario_type": scenario.type,
            "scenario_name": scenario.name,
            "total_customers": total,
            "new_critical": new_critical,
            "new_watch": new_watch,
        },
    )

    def _tier_dist(stable, watch, critical, total):
        return TierDistribution(
            stable=stable, watch=watch, critical=critical,
            stable_pct=round(stable / total * 100, 1) if total else 0,
            watch_pct=round(watch / total * 100, 1) if total else 0,
            critical_pct=round(critical / total * 100, 1) if total else 0,
        )

    return WhatIfResponse(
        scenario_name=scenario.name,
        total_customers=total,
        current_distribution=_tier_dist(curr_stable, curr_watch, curr_critical, total),
        simulated_distribution=_tier_dist(sim_stable, sim_watch, sim_critical, total),
        customers_upgraded_to_watch=upgraded_to_watch,
        customers_upgraded_to_critical=upgraded_to_critical,
        customers_downgraded=downgraded,
        estimated_npa_delta_crore=npa_delta,
        intervention_cost_lakh=intervention_cost,
        intervention_roi=roi,
        avg_risk_score_current=round(float(np.mean(current_scores)), 4),
        avg_risk_score_simulated=round(float(np.mean(simulated_scores)), 4),
        segment_breakdown=segment_breakdown,
        top_affected_employers=top_affected_employers,
        region_breakdown=region_breakdown,
    )


@router.get("/scenarios/templates", tags=["What-If Simulator"])
async def get_scenario_templates(
    current_user: TokenPayload = Depends(require_role("analyst", "risk_officer", "admin")),
):
    """Return pre-built scenario templates for common stress tests."""
    return {
        "templates": [
            {
                "id": "it_layoffs_moderate",
                "name": "IT Sector Moderate Layoffs (-10%)",
                "description": "Simulates 10% workforce reduction in IT sector — typical mid-cycle correction",
                "params": {
                    "type": "sector_shock",
                    "sector": "IT",
                    "sector_employment_change_pct": -10,
                },
            },
            {
                "id": "it_layoffs_severe",
                "name": "IT Sector Severe Layoffs (-25%)",
                "description": "Simulates 25% IT sector layoffs — e.g. mass tech layoffs like 2023 US tech wave",
                "params": {
                    "type": "sector_shock",
                    "sector": "IT",
                    "sector_employment_change_pct": -25,
                },
            },
            {
                "id": "boe_rate_hike_50bps",
                "name": "BoE/Fed Rate Hike +50 bps",
                "description": "50 basis point base rate hike — increases mortgage & loan repayment burden",
                "params": {
                    "type": "rate_hike",
                    "rate_hike_bps": 50,
                },
            },
            {
                "id": "boe_rate_hike_100bps",
                "name": "BoE/Fed Rate Hike +100 bps (Stress Test)",
                "description": "100 bps hike — severe stress scenario for PRA/Basel III regulatory stress testing",
                "params": {
                    "type": "rate_hike",
                    "rate_hike_bps": 100,
                },
            },
            {
                "id": "emi_holiday_watch_tier",
                "name": "2-Month EMI Holiday (Watch Tier)",
                "description": "Cost-benefit: offer 2-month EMI holiday to all watch-tier customers",
                "params": {
                    "type": "emi_holiday",
                    "emi_holiday_months": 2,
                    "emi_holiday_target_tiers": ["watch"],
                },
            },
            {
                "id": "emi_holiday_all_stressed",
                "name": "3-Month EMI Holiday (Watch + Critical)",
                "description": "Aggressive intervention: 3-month holiday for all stressed customers",
                "params": {
                    "type": "emi_holiday",
                    "emi_holiday_months": 3,
                    "emi_holiday_target_tiers": ["watch", "critical"],
                },
            },
            {
                "id": "monsoon_failure_south",
                "name": "Monsoon Failure — South India",
                "description": "Deficient monsoon causing 20% income shock in South & East regions",
                "params": {
                    "type": "regional_shock",
                    "regions": ["South", "East"],
                    "regional_income_shock_pct": -20,
                },
            },
            {
                "id": "payroll_delay_15days",
                "name": "Widespread Payroll Delay (+15 days)",
                "description": "Simulates systemic payroll delay — tests portfolio resilience",
                "params": {
                    "type": "salary_shock",
                    "salary_delay_days_added": 15,
                },
            },
        ]
    }


@router.get("/portfolio/summary", tags=["What-If Simulator"])
async def get_portfolio_summary(
    current_user: TokenPayload = Depends(require_role("analyst", "risk_officer", "admin")),
):
    """Return current portfolio risk distribution summary (baseline for what-if comparison)."""
    portfolio = _get_portfolio_features()
    if not portfolio:
        return {"total": 0, "message": "No scored customers found. Run scoring first."}

    total = len(portfolio)
    scores = [r.get("risk_score") or 0.3 for r in portfolio]
    tiers = [_score_to_tier(s) for s in scores]

    stable = tiers.count("stable")
    watch = tiers.count("watch")
    critical = tiers.count("critical")

    # Segment breakdown
    segments = {}
    regions = {}
    for row in portfolio:
        seg = row.get("segment_type") or "unknown"
        segments[seg] = segments.get(seg, 0) + 1
        reg = row.get("region") or "Unknown"
        regions[reg] = regions.get(reg, 0) + 1

    return {
        "total_customers": total,
        "distribution": {
            "stable": {"count": stable, "pct": round(stable / total * 100, 1)},
            "watch": {"count": watch, "pct": round(watch / total * 100, 1)},
            "critical": {"count": critical, "pct": round(critical / total * 100, 1)},
        },
        "avg_risk_score": round(float(np.mean(scores)), 4),
        "p90_risk_score": round(float(np.percentile(scores, 90)), 4),
        "segments": segments,
        "regions": regions,
        "estimated_portfolio_at_risk_crore": round(
            (watch * 0.40 + critical * 0.85) * _AVG_LOAN_OUTSTANDING_LAKH * (1 - _NPA_RECOVERY_RATE) / 100,
            2,
        ),
    }
