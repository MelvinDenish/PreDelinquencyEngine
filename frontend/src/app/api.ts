// API service to call the real scoring backend
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface ScoreResult {
  customer_id: string;
  risk_score: number;
  risk_tier: string;
  credit_score_mapped: number;
  xgboost_score: number | null;
  lightgbm_score: number | null;
  lstm_score: number | null;
  tft_score: number | null;
  ensemble_score: number;
  top_shap_features: { feature: string; value: number }[];
  top_lime_features: { feature: string; value: number }[];
  explanation: string;
  product_actions: string[];
  tte_days: number | null;
  is_cold_start: boolean;
  meta_learner_used: boolean;
  scored_at: string;
  // conformal
  risk_score_lower?: number;
  risk_score_upper?: number;
  confidence_flag?: string;
  // uplift
  uplift_score?: number;
  holdout_group?: string;
  shadow_score?: number;
}

export interface NotifyResult {
  status: string;
  customer_id: string;
  risk_score?: number;
  channels_attempted?: number;
  results?: Array<{ channel: string; status: string; detail?: string }>;
  error?: string;
}

export interface HealthResult {
  status: string;
  models_loaded: Record<string, boolean>;
  timestamp: string;
}

// Score a customer via the real ML ensemble
export async function scoreCustomer(customerId: string): Promise<ScoreResult> {
  const res = await fetch(`${API_BASE}/score`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ customer_id: customerId }),
  });
  if (!res.ok) {
    throw new Error(`Score API returned ${res.status}: ${await res.text()}`);
  }
  return res.json();
}

// Send notification via real dispatcher (SMS/Email/WhatsApp)
export async function notifyCustomer(payload: {
  customer_id: string;
  customer_name: string;
  risk_score: number;
  risk_tier: string;
  alert_message: string;
  city?: string;
  salary?: number;
}): Promise<NotifyResult> {
  const res = await fetch(`${API_BASE}/notify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(`Notify API returned ${res.status}: ${await res.text()}`);
  }
  return res.json();
}

// Check service health
export async function checkHealth(): Promise<HealthResult> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

// Explain a customer (SHAP + LIME + Counterfactuals)
export async function explainCustomer(customerId: string) {
  const res = await fetch(`${API_BASE}/explain/${customerId}`);
  if (!res.ok) throw new Error(`Explain API failed: ${res.status}`);
  return res.json();
}

// ─── What-If Simulator ───────────────────────────────────────────────────────

export interface TierDistribution {
  stable: number;
  watch: number;
  critical: number;
  stable_pct: number;
  watch_pct: number;
  critical_pct: number;
}

export interface WhatIfResult {
  scenario_name: string;
  total_customers: number;
  current_distribution: TierDistribution;
  simulated_distribution: TierDistribution;
  customers_upgraded_to_watch: number;
  customers_upgraded_to_critical: number;
  customers_downgraded: number;
  estimated_npa_delta_crore: number;
  intervention_cost_lakh: number | null;
  intervention_roi: number | null;
  avg_risk_score_current: number;
  avg_risk_score_simulated: number;
  segment_breakdown: Record<string, { count: number; new_watch: number; new_critical: number }> | null;
  top_affected_employers: Array<{ employer: string; total: number; newly_at_risk: number }> | null;
  region_breakdown: Record<string, { total: number; newly_at_risk: number; affected: boolean }> | null;
}

export interface PortfolioSummary {
  total_customers: number;
  distribution: {
    stable: { count: number; pct: number };
    watch: { count: number; pct: number };
    critical: { count: number; pct: number };
  };
  avg_risk_score: number;
  p90_risk_score: number;
  segments: Record<string, number>;
  regions: Record<string, number>;
  estimated_portfolio_at_risk_crore: number;
}

export interface ScenarioTemplate {
  id: string;
  name: string;
  description: string;
  params: Record<string, unknown>;
}

export async function runWhatIfSimulation(params: Record<string, unknown>): Promise<WhatIfResult> {
  const res = await fetch(`${API_BASE}/whatif/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`What-If simulation failed (${res.status}): ${text}`);
  }
  return res.json();
}

export async function getScenarioTemplates(): Promise<{ templates: ScenarioTemplate[] }> {
  const res = await fetch(`${API_BASE}/whatif/scenarios/templates`);
  if (!res.ok) throw new Error(`Templates fetch failed: ${res.status}`);
  return res.json();
}

export async function getPortfolioSummary(): Promise<PortfolioSummary> {
  const res = await fetch(`${API_BASE}/whatif/portfolio/summary`);
  if (!res.ok) throw new Error(`Portfolio summary failed: ${res.status}`);
  return res.json();
}
