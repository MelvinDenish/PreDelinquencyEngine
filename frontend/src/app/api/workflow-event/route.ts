/**
 * POST /api/workflow-event
 * Receives pipeline events pushed by the n8n workflow after each run.
 * Stores the last 100 events in a server-side in-memory ring buffer
 * that the frontend polls via GET /api/workflow-events.
 */
import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export interface WorkflowEvent {
  event_id: string;
  event_time: string;
  pipeline_run_ms: number;
  customer_id: string;
  customer_name: string;
  city: string;
  region: string;
  segment: string;
  credit_score: number;
  monthly_salary: number;
  txn_id: string;
  txn_type: string;
  merchant_category: string;
  amount: number;
  is_stress_signal: boolean;
  risk_score: number;
  risk_tier: "critical" | "watch" | "stable";
  risk_tier_icon: string;
  xgboost_score: number | null;
  lightgbm_score: number | null;
  tft_score: number | null;
  ensemble_score: number;
  calibrated_pd: number | null;
  tte_days: number | null;
  uplift_score: number | null;
  confidence_flag: string;
  meta_learner_used: boolean;
  top_shap_features: { feature: string; value: number }[];
  product_actions: string[];
  ml_source: string;
  intervention_type: string;
  notify_status: string;
  channels_sent: string;
  channels_count: number;
  notify_results: { channel: string; status: string }[];
  stages: { stage: string; ts: string; status: string; label: string }[];
}

const MAX_EVENTS = 100;
const WEBHOOK_SECRET = "pdi-n8n-webhook-secret";

export async function POST(req: NextRequest) {
  // Verify the shared secret from n8n
  const secret = req.headers.get("x-pdi-secret");
  if (secret !== WEBHOOK_SECRET) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  let body: WorkflowEvent;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  // Validate minimal required fields
  if (!body.event_id || !body.customer_id || !body.risk_tier) {
    return NextResponse.json({ error: "Missing required fields" }, { status: 422 });
  }

  // File based persistence using process.cwd() instead of global memory
  try {
    const dataPath = path.resolve(process.cwd(), "pdi_events.json");
    let events: WorkflowEvent[] = [];
    if (fs.existsSync(dataPath)) {
      events = JSON.parse(fs.readFileSync(dataPath, "utf-8"));
    }
    
    events.unshift(body);
    if (events.length > MAX_EVENTS) {
      events = events.slice(0, MAX_EVENTS);
    }
    
    fs.writeFileSync(dataPath, JSON.stringify(events, null, 2), "utf-8");
  } catch (e) {
    console.error("Failed to write to pdi_events.json", e);
  }

  return NextResponse.json({ status: "ok", event_id: body.event_id });
}
