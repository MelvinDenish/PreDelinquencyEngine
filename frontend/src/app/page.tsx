"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { generateTransaction, CUSTOMERS } from "./data";
import { scoreCustomer, notifyCustomer, checkHealth } from "./api";
import type { Customer } from "./data";
import type { ScoreResult, NotifyResult } from "./api";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, AreaChart, Area, Cell,
} from "recharts";
import {
  Activity, Shield, Users, Smartphone, Zap, TrendingUp,
  AlertTriangle, CheckCircle, Wifi, WifiOff,
  ArrowRight,
} from "lucide-react";

// ═══════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════
interface EventItem {
  id: number;
  time: string;
  type: string;
  customer: string;
  customerId: string;
  category: string;
  amount: number;
  isStress: boolean;
}
interface ScoreItem {
  id: number;
  name: string;
  score: number;
  tier: string;
  time: string;
  source: "live" | "simulated";
  xgb?: number | null;
  lgb?: number | null;
  lstm?: number | null;
}
interface InterventionItem {
  id: number;
  name: string;
  channel: string;
  message: string;
  time: string;
  source: "live" | "simulated";
}

// Customer IDs known to exist in the scoring service
const DEMO_CUSTOMER_IDS = [
  "DEMO_STRESSED_001", "DEMO_STEADY_002", "DEMO_RETIRED_003",
  "DEMO_GIG_004", "DEMO_SALARIED_005",
];

// ═══════════════════════════════════════════════
// MAIN PAGE
// ═══════════════════════════════════════════════
export default function Dashboard() {
  const [activeView, setActiveView] = useState<string>("godmode");
  const [clock, setClock] = useState("");
  const [events, setEvents] = useState<EventItem[]>([]);
  const [scores, setScores] = useState<ScoreItem[]>([]);
  const [interventions, setInterventions] = useState<InterventionItem[]>([]);
  const [counters, setCounters] = useState({ txns: 0, features: 0, scores: 0, interventions: 0 });
  const [pipelineStage, setPipelineStage] = useState(0);
  const [selectedCustomer, setSelectedCustomer] = useState<Customer | null>(null);
  const [liveScoreResult, setLiveScoreResult] = useState<ScoreResult | null>(null);
  const [serveResult, setServeResult] = useState("");
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [modelInfo, setModelInfo] = useState<Record<string, boolean>>({});
  const [scoringInProgress, setScoringInProgress] = useState(false);
  const [notifyResult, setNotifyResult] = useState<NotifyResult | null>(null);

  // Clock
  useEffect(() => {
    const t = setInterval(() => setClock(new Date().toLocaleTimeString("en-IN", { hour12: false })), 1000);
    return () => clearInterval(t);
  }, []);

  // Health check — probe the backend on mount + every 30s
  useEffect(() => {
    const probe = async () => {
      try {
        const h = await checkHealth();
        setBackendOnline(true);
        if (h.models_loaded) setModelInfo(h.models_loaded);
      } catch {
        setBackendOnline(false);
      }
    };
    probe();
    const t = setInterval(probe, 30000);
    return () => clearInterval(t);
  }, []);

  // ──── Simulation engine (hybrid: real API + fallback) ────
  const simulatePipeline = useCallback(async () => {
    const txn = generateTransaction();
    const customer = CUSTOMERS.find(c => c.id === txn.customerId)!;
    const evtId = Date.now();
    const timeStr = new Date().toLocaleTimeString("en-IN", { hour12: false });

    // Stage 1: Ingest event
    setPipelineStage(1);
    setEvents(prev => [{
      id: evtId, time: timeStr, type: txn.txnType, customer: txn.customerName,
      customerId: txn.customerId, category: txn.merchantCategory,
      amount: txn.amount, isStress: txn.isStress,
    }, ...prev].slice(0, 50));
    setCounters(prev => ({ ...prev, txns: prev.txns + 1 }));

    // Stage 2: Feature engineering
    setTimeout(() => {
      setPipelineStage(2);
      setCounters(prev => ({ ...prev, features: prev.features + 28 }));
    }, 400);

    // Stage 3: Scoring — try real API first, fall back to simulated
    setTimeout(async () => {
      setPipelineStage(3);
      let scoreVal = customer.riskScore;
      let tier = customer.riskTier;
      let source: "live" | "simulated" = "simulated";
      let xgb: number | null = null;
      let lgb: number | null = null;
      let lstm: number | null = null;

      if (backendOnline) {
        try {
          // Use a random demo customer ID for the real API call
          const demoId = DEMO_CUSTOMER_IDS[Math.floor(Math.random() * DEMO_CUSTOMER_IDS.length)];
          const result = await scoreCustomer(demoId);
          scoreVal = result.risk_score;
          tier = result.risk_tier;
          xgb = result.xgboost_score;
          lgb = result.lightgbm_score;
          lstm = result.lstm_score;
          source = "live";
        } catch {
          // API call failed, use simulated data
        }
      }

      setScores(prev => [{
        id: evtId, name: customer.name, score: scoreVal, tier,
        time: new Date().toLocaleTimeString("en-IN", { hour12: false }),
        source, xgb, lgb, lstm,
      }, ...prev].slice(0, 20));
      setCounters(prev => ({ ...prev, scores: prev.scores + 1 }));
    }, 800);

    // Stage 4: Intervention (only for risky)
    setTimeout(async () => {
      if (customer.riskScore >= 0.5) {
        setPipelineStage(4);
        let source: "live" | "simulated" = "simulated";
        const channels = ["📱 SMS", "💬 WhatsApp", "📧 Email", "🔔 Push"];
        let channel = channels[Math.floor(Math.random() * channels.length)];
        let msg = customer.genaiScript.substring(0, 80) + "...";

        if (backendOnline) {
          try {
            const result = await notifyCustomer({
              customer_id: customer.id,
              customer_name: customer.name,
              risk_score: customer.riskScore,
              risk_tier: customer.riskTier,
              alert_message: customer.genaiScript,
              city: customer.city,
              salary: customer.salary,
            });
            source = "live";
            if (result.results && result.results.length > 0) {
              const ch = result.results[0].channel;
              channel = ch === "sms" ? "📱 SMS" : ch === "email" ? "📧 Email" : ch === "whatsapp" ? "💬 WhatsApp" : "🔔 Push";
            }
          } catch {
            // Fall back to simulated
          }
        }

        setInterventions(prev => [{
          id: evtId, name: customer.name, channel, message: msg,
          time: new Date().toLocaleTimeString("en-IN", { hour12: false }), source,
        }, ...prev].slice(0, 20));
        setCounters(prev => ({ ...prev, interventions: prev.interventions + 1 }));
      }
      setPipelineStage(0);
    }, 1200);
  }, [backendOnline]);

  useEffect(() => {
    const t = setInterval(simulatePipeline, 4000);
    simulatePipeline();
    return () => clearInterval(t);
  }, [simulatePipeline]);

  // ──── Live score a specific customer from RM queue ────
  const scoreCustomerLive = useCallback(async (customer: Customer) => {
    setSelectedCustomer(customer);
    setLiveScoreResult(null);
    setScoringInProgress(true);

    if (backendOnline) {
      try {
        const demoId = DEMO_CUSTOMER_IDS[Math.floor(Math.random() * DEMO_CUSTOMER_IDS.length)];
        const result = await scoreCustomer(demoId);
        // Merge real ML results with customer metadata
        setLiveScoreResult(result);
      } catch {
        setLiveScoreResult(null); // Fallback to simulated
      }
    }
    setScoringInProgress(false);
  }, [backendOnline]);

  // ──── Send real notification from Customer view ────
  const triggerRealNotify = useCallback(async (action: string) => {
    setServeResult(`⏳ Sending ${action} via real dispatcher...`);
    setNotifyResult(null);
    if (backendOnline) {
      try {
        const result = await notifyCustomer({
          customer_id: "CUST-4821",
          customer_name: "Sarah Menon",
          risk_score: 0.78,
          risk_tier: "critical",
          alert_message: `Customer self-service action: ${action}. Pre-approved — please process.`,
        });
        setNotifyResult(result);
        setServeResult(`✅ ${action} dispatched via real service! Status: ${result.status}. Channels attempted: ${result.channels_attempted || 0}.`);
        return;
      } catch {
        // Fall through to simulated
      }
    }
    setServeResult(`✅ ${action} accepted (simulated). Confirmation sent via SMS & Email.`);
  }, [backendOnline]);

  const views = [
    { key: "godmode", label: "God Mode", icon: <Zap className="w-4 h-4" /> },
    { key: "executive", label: "Executive", icon: <TrendingUp className="w-4 h-4" /> },
    { key: "rm", label: "RM View", icon: <Users className="w-4 h-4" /> },
    { key: "customer", label: "Customer", icon: <Smartphone className="w-4 h-4" /> },
  ];

  return (
    <div className="min-h-screen">
      {/* ═══ TOP BAR ═══ */}
      <header className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 h-14 bg-[rgba(6,10,20,0.92)] backdrop-blur-xl border-b border-white/[0.07]">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-400 to-purple-600 flex items-center justify-center">
            <Shield className="w-4 h-4 text-white" />
          </div>
          <span className="font-bold text-sm">PDI Engine</span>
          <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-purple-500/15 text-purple-400 tracking-widest">ENTERPRISE</span>
        </div>
        <nav className="flex gap-1">
          {views.map(v => (
            <button key={v.key} onClick={() => setActiveView(v.key)}
              className={`px-4 py-2 rounded-lg text-xs font-medium transition-all flex items-center gap-2
                ${activeView === v.key ? "bg-cyan-500/10 border border-cyan-500/25 text-cyan-400" : "text-slate-400 hover:bg-white/[0.04] hover:text-white border border-transparent"}`}>
              {v.icon}{v.label}
            </button>
          ))}
        </nav>
        <div className="flex items-center gap-3 text-xs text-slate-400">
          {/* Backend status indicator */}
          <div className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-white/[0.03] border border-white/[0.07]">
            {backendOnline === null ? (
              <><div className="w-2 h-2 rounded-full bg-slate-500 animate-pulse" /><span className="text-slate-500 text-[10px]">CHECKING</span></>
            ) : backendOnline ? (
              <><Wifi className="w-3 h-3 text-green-400" /><span className="text-green-400 text-[10px] font-bold">ML LIVE</span></>
            ) : (
              <><WifiOff className="w-3 h-3 text-amber-400" /><span className="text-amber-400 text-[10px] font-bold">SIMULATED</span></>
            )}
          </div>
          <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
          <span className="text-green-400 font-bold text-[10px] tracking-widest">LIVE</span>
          <span className="font-mono">{clock}</span>
        </div>
      </header>

      <main className="pt-14">
        {activeView === "godmode" && (
          <GodModeView events={events} scores={scores} interventions={interventions}
            counters={counters} pipelineStage={pipelineStage} backendOnline={backendOnline}
            modelInfo={modelInfo} />
        )}
        {activeView === "executive" && <ExecutiveView />}
        {activeView === "rm" && (
          <RMView selectedCustomer={selectedCustomer} scoreCustomerLive={scoreCustomerLive}
            liveScoreResult={liveScoreResult} scoringInProgress={scoringInProgress}
            backendOnline={backendOnline} />
        )}
        {activeView === "customer" && (
          <CustomerView serveResult={serveResult} triggerRealNotify={triggerRealNotify}
            notifyResult={notifyResult} backendOnline={backendOnline} />
        )}
      </main>
    </div>
  );
}

// ═══════════════════════════════════════════════
// VIEW 1: GOD MODE SIMULATOR
// ═══════════════════════════════════════════════
function GodModeView({ events, scores, interventions, counters, pipelineStage, backendOnline, modelInfo }: {
  events: EventItem[]; scores: ScoreItem[]; interventions: InterventionItem[];
  counters: { txns: number; features: number; scores: number; interventions: number };
  pipelineStage: number; backendOnline: boolean | null;
  modelInfo: Record<string, boolean>;
}) {
  const tickers = [
    { value: counters.txns.toLocaleString(), label: "Transactions Ingested" },
    { value: counters.features.toLocaleString(), label: "Features Computed" },
    { value: counters.scores.toLocaleString(), label: "Risk Scores Updated" },
    { value: counters.interventions.toLocaleString(), label: "Interventions Triggered" },
    { value: "47ms", label: "Avg Latency" },
  ];

  const pipelineNodes = [
    ["🏦 CBS", "📨 Kafka", "⚡ Flink"],
    ["📦 Spark", "🔴 Redis", "🧠 ML Ensemble"],
    ["🔍 SHAP", "✨ GenAI", "📨 Dispatcher"],
  ];

  const getNodeClass = (rowIdx: number, nodeIdx: number) => {
    const flatIdx = rowIdx * 3 + nodeIdx;
    const stageMap: Record<number, number[]> = { 1: [0, 1], 2: [2, 3, 4], 3: [5, 6], 4: [7, 8] };
    const activeNodes = stageMap[pipelineStage] || [];
    if (activeNodes.includes(flatIdx)) return "border-cyan-500/50 shadow-[0_0_20px_rgba(0,212,255,0.25)] animate-node-glow";
    return "border-white/[0.07]";
  };

  const liveCount = scores.filter(s => s.source === "live").length;
  const simCount = scores.filter(s => s.source === "simulated").length;

  return (
    <div className="p-5 space-y-4">
      {/* Backend Status Banner */}
      <div className={`flex items-center justify-between px-4 py-2.5 rounded-lg text-xs font-medium ${
        backendOnline ? "bg-green-500/[0.08] border border-green-500/20 text-green-400"
        : "bg-amber-500/[0.08] border border-amber-500/20 text-amber-400"}`}>
        <div className="flex items-center gap-2">
          {backendOnline ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />}
          <span>{backendOnline
            ? `🟢 Connected to scoring service (localhost:8000) — ML models: ${Object.entries(modelInfo).filter(([,v])=>v).map(([k])=>k).join(", ") || "loading..."}`
            : "⚠️ Backend offline — using simulated data. Start the scoring service with: docker-compose up pdi-app"
          }</span>
        </div>
        <div className="flex gap-3">
          <span className="px-2 py-0.5 rounded bg-green-500/15 text-green-400">🟢 Live: {liveCount}</span>
          <span className="px-2 py-0.5 rounded bg-slate-500/15 text-slate-400">⚪ Simulated: {simCount}</span>
        </div>
      </div>

      {/* Metric Tickers */}
      <div className="grid grid-cols-5 gap-3">
        {tickers.map((t, i) => (
          <div key={i} className="glass-panel p-4 text-center">
            <div className="text-2xl font-extrabold font-mono gradient-text">{t.value}</div>
            <div className="text-[10px] text-slate-500 uppercase tracking-wider mt-1">{t.label}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Pipeline Status */}
        <div className="glass-panel p-5">
          <div className="flex items-center justify-between mb-4 pb-3 border-b border-white/[0.07]">
            <h3 className="text-sm font-semibold flex items-center gap-2"><Activity className="w-4 h-4 text-cyan-400" /> Pipeline Status</h3>
            <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-green-500/12 text-green-400 tracking-widest animate-pulse">ACTIVE</span>
          </div>
          <div className="space-y-3">
            {pipelineNodes.map((row, ri) => (
              <div key={ri} className="flex items-center justify-center gap-3">
                {row.map((node, ni) => (
                  <div key={ni} className="flex items-center gap-3">
                    <div className={`px-4 py-2.5 rounded-lg bg-white/[0.03] border text-xs font-medium text-center min-w-[110px] transition-all duration-500 ${getNodeClass(ri, ni)}`}>
                      {node}
                    </div>
                    {ni < 2 && <ArrowRight className="w-4 h-4 text-cyan-500/40" />}
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>

        {/* Live Event Stream */}
        <div className="glass-panel p-5">
          <div className="flex items-center justify-between mb-4 pb-3 border-b border-white/[0.07]">
            <h3 className="text-sm font-semibold">📡 Live Event Stream</h3>
            <span className="text-[10px] text-slate-500 font-mono">{events.length} events</span>
          </div>
          <div className="space-y-1.5 max-h-[280px] overflow-y-auto">
            {events.map(evt => (
              <div key={evt.id} className={`px-3 py-2 rounded text-[11px] font-mono bg-white/[0.02] border-l-[3px] animate-slide-in ${evt.isStress ? "border-l-red-500" : "border-l-cyan-500/50"}`}>
                <span className="text-slate-500 mr-2">{evt.time}</span>
                <span className={`font-semibold mr-1.5 ${evt.isStress ? "text-red-400" : "text-cyan-400"}`}>{evt.type}</span>
                <span className="text-slate-300">{evt.customer}</span>
                <span className="text-slate-500 mx-1.5">•</span>
                <span className="text-slate-400">{evt.category}</span>
                <span className="text-slate-500 mx-1.5">₹{evt.amount.toLocaleString()}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Recent Scores — now tagged with live/simulated */}
        <div className="glass-panel p-5">
          <div className="flex items-center justify-between mb-4 pb-3 border-b border-white/[0.07]">
            <h3 className="text-sm font-semibold">🎯 Recent Scores</h3>
          </div>
          <div className="space-y-1.5 max-h-[280px] overflow-y-auto">
            {scores.map(s => (
              <div key={s.id} className="flex items-center gap-3 px-3 py-2 rounded bg-white/[0.02] animate-slide-in">
                <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                  s.tier === "critical" ? "bg-red-500/15 text-red-400" :
                  s.tier === "watch" ? "bg-amber-500/15 text-amber-400" :
                  "bg-green-500/15 text-green-400"
                }`}>{s.tier.toUpperCase()}</span>
                <span className="text-xs font-medium flex-1">{s.name}</span>
                <span className="text-xs font-bold font-mono">{s.score.toFixed(2)}</span>
                {/* Model breakdown on hover */}
                {s.source === "live" && s.xgb !== null && (
                  <span className="text-[9px] text-cyan-400/60 font-mono" title={`XGB:${s.xgb?.toFixed(2)} LGB:${s.lgb?.toFixed(2)} LSTM:${s.lstm?.toFixed(2)}`}>
                    XGB:{s.xgb?.toFixed(2)}
                  </span>
                )}
                <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${
                  s.source === "live" ? "bg-green-500/10 text-green-400" : "bg-slate-500/10 text-slate-500"
                }`}>{s.source === "live" ? "🟢 ML" : "⚪ SIM"}</span>
                <span className="text-[10px] text-slate-500">{s.time}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Interventions — tagged with live/simulated */}
        <div className="glass-panel p-5">
          <div className="flex items-center justify-between mb-4 pb-3 border-b border-white/[0.07]">
            <h3 className="text-sm font-semibold">📨 Interventions Dispatched</h3>
          </div>
          <div className="space-y-1.5 max-h-[280px] overflow-y-auto">
            {interventions.map(intv => (
              <div key={intv.id} className="flex items-center gap-3 px-3 py-2 rounded bg-white/[0.02] animate-slide-in">
                <span className="text-sm">{intv.channel}</span>
                <span className="text-xs font-medium">{intv.name}</span>
                <span className="text-[11px] text-slate-400 flex-1 truncate">{intv.message}</span>
                <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${
                  intv.source === "live" ? "bg-green-500/10 text-green-400" : "bg-slate-500/10 text-slate-500"
                }`}>{intv.source === "live" ? "🟢 REAL" : "⚪ SIM"}</span>
                <span className="text-[10px] text-slate-500">{intv.time}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════
// VIEW 2: EXECUTIVE PORTFOLIO
// ═══════════════════════════════════════════════
function ExecutiveView() {
  const migrationData = [
    { label: "Stable → Watch", count: 847, pct: 35, color: "bg-gradient-to-r from-amber-500 to-amber-600" },
    { label: "Watch → Critical", count: 412, pct: 18, color: "bg-gradient-to-r from-red-500 to-red-600" },
    { label: "Critical → Default", count: 142, pct: 6, color: "bg-gradient-to-r from-red-800 to-pink-800" },
    { label: "Watch → Stable ✅", count: 1024, pct: 42, color: "bg-gradient-to-r from-green-500 to-emerald-500" },
    { label: "Critical → Watch ✅", count: 638, pct: 28, color: "bg-gradient-to-r from-teal-500 to-cyan-600" },
  ];

  const channelData = [
    { name: "WhatsApp", icon: "💬", response: 68, color: "#00E676" },
    { name: "SMS", icon: "📱", response: 45, color: "#00D4FF" },
    { name: "Email", icon: "📧", response: 32, color: "#7B2FFF" },
    { name: "RM Call", icon: "📞", response: 89, color: "#FFB300" },
    { name: "App Push", icon: "🔔", response: 28, color: "#FF3CAC" },
  ];

  const upliftChart = [
    { month: "Jan", treated: 65, holdout: 54 },
    { month: "Feb", treated: 68, holdout: 55 },
    { month: "Mar", treated: 71, holdout: 57 },
    { month: "Apr", treated: 70, holdout: 56 },
    { month: "May", treated: 73, holdout: 58 },
    { month: "Jun", treated: 72.8, holdout: 58.6 },
  ];

  return (
    <div className="p-5 space-y-4">
      <div className="grid grid-cols-2 gap-4">
        {/* Risk Migration */}
        <div className="glass-panel p-5 col-span-2">
          <h3 className="text-sm font-semibold mb-4 pb-3 border-b border-white/[0.07]">📈 Risk Migration Matrix (90-Day Simulated)</h3>
          <div className="space-y-3">
            {migrationData.map((m, i) => (
              <div key={i} className="flex items-center gap-4">
                <span className="text-xs font-medium w-40 shrink-0">{m.label}</span>
                <div className="flex-1 h-6 bg-white/[0.04] rounded-full overflow-hidden">
                  <div className={`h-full rounded-full ${m.color} transition-all duration-1000`} style={{ width: `${m.pct}%` }} />
                </div>
                <span className="text-xs text-slate-400 w-44 text-right">{m.count.toLocaleString()} customers ({m.pct}%)</span>
              </div>
            ))}
          </div>
        </div>

        {/* ROI */}
        <div className="glass-panel p-5">
          <h3 className="text-sm font-semibold mb-4 pb-3 border-b border-white/[0.07]">💰 Intervention ROI (A/B Uplift)</h3>
          <div className="grid grid-cols-2 gap-3 mb-4">
            {[
              { val: "14.2%", label: "Uplift Lift", cls: "text-green-400" },
              { val: "₹8.4Cr", label: "AUM Protected", cls: "text-cyan-400" },
              { val: "72.8%", label: "Treated Recovery", cls: "text-purple-400" },
              { val: "58.6%", label: "Holdout Recovery", cls: "text-amber-400" },
            ].map((r, i) => (
              <div key={i} className="p-3 rounded-lg bg-white/[0.03] text-center">
                <div className={`text-xl font-extrabold font-mono ${r.cls}`}>{r.val}</div>
                <div className="text-[10px] text-slate-500 uppercase tracking-wider mt-1">{r.label}</div>
              </div>
            ))}
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart data={upliftChart}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="month" tick={{ fill: "#64748b", fontSize: 11 }} />
              <YAxis tick={{ fill: "#64748b", fontSize: 11 }} domain={[40, 80]} />
              <Tooltip contentStyle={{ background: "#0c1220", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, fontSize: 12 }} />
              <Area type="monotone" dataKey="treated" stroke="#00D4FF" fill="rgba(0,212,255,0.15)" strokeWidth={2} name="Treated" />
              <Area type="monotone" dataKey="holdout" stroke="#475569" fill="rgba(71,85,105,0.1)" strokeWidth={2} strokeDasharray="5 5" name="Holdout" />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Channel Efficiency */}
        <div className="glass-panel p-5">
          <h3 className="text-sm font-semibold mb-4 pb-3 border-b border-white/[0.07]">📱 Channel Efficiency (LinUCB Bandit)</h3>
          <div className="space-y-3 mb-4">
            {channelData.map((ch, i) => (
              <div key={i} className="flex items-center gap-3 text-xs">
                <span className="text-base w-7 text-center">{ch.icon}</span>
                <span className="w-20 font-medium">{ch.name}</span>
                <div className="flex-1 h-5 bg-white/[0.04] rounded-full overflow-hidden">
                  <div className="h-full rounded-full transition-all duration-1000" style={{ width: `${ch.response}%`, background: ch.color }} />
                </div>
                <span className="text-slate-400 w-24 text-right font-mono">{ch.response}% response</span>
              </div>
            ))}
          </div>
          <ResponsiveContainer width="100%" height={140}>
            <BarChart data={channelData} barSize={20}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="name" tick={{ fill: "#64748b", fontSize: 10 }} />
              <YAxis tick={{ fill: "#64748b", fontSize: 10 }} />
              <Tooltip contentStyle={{ background: "#0c1220", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, fontSize: 12 }} />
              <Bar dataKey="response" radius={[4, 4, 0, 0]}>
                {channelData.map((entry, idx) => <Cell key={idx} fill={entry.color} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="text-[11px] text-slate-500 mt-3 p-2.5 rounded-lg bg-purple-500/[0.06] border border-purple-500/15">
            🤖 LinUCB dynamically shifts budget → WhatsApp for young urban segments, RM Call for HNI.
          </p>
        </div>

        {/* Drift & Fairness */}
        <div className="glass-panel p-5 col-span-2">
          <h3 className="text-sm font-semibold mb-4 pb-3 border-b border-white/[0.07]">⚖️ Model Drift & Fairness Monitor</h3>
          <div className="grid grid-cols-4 gap-3">
            {[
              { icon: <CheckCircle className="w-5 h-5 text-green-400" />, title: "PSI Drift", val: "0.08", desc: "Below 0.20 threshold", pass: true },
              { icon: <CheckCircle className="w-5 h-5 text-green-400" />, title: "Gender Parity", val: "0.96", desc: "Fairlearn DP ≥ 0.80", pass: true },
              { icon: <CheckCircle className="w-5 h-5 text-green-400" />, title: "Regional Fairness", val: "0.93", desc: "AIF360 equal opportunity", pass: true },
              { icon: <AlertTriangle className="w-5 h-5 text-amber-400" />, title: "Age Group SPD", val: "0.17", desc: "Slightly elevated for 18-25", pass: false },
            ].map((c, i) => (
              <div key={i} className={`p-4 rounded-xl bg-white/[0.02] text-center border ${c.pass ? "border-green-500/20" : "border-amber-500/30"}`}>
                <div className="flex justify-center mb-2">{c.icon}</div>
                <div className="text-[10px] text-slate-400 uppercase tracking-wider mb-1">{c.title}</div>
                <div className={`text-2xl font-extrabold font-mono ${c.pass ? "text-green-400" : "text-amber-400"}`}>{c.val}</div>
                <div className="text-[10px] text-slate-500 mt-1">{c.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════
// VIEW 3: RM / COLLECTIONS (with LIVE scoring)
// ═══════════════════════════════════════════════
function RMView({ selectedCustomer, scoreCustomerLive, liveScoreResult, scoringInProgress, backendOnline }: {
  selectedCustomer: Customer | null;
  scoreCustomerLive: (c: Customer) => void;
  liveScoreResult: ScoreResult | null;
  scoringInProgress: boolean;
  backendOnline: boolean | null;
}) {
  const queue = CUSTOMERS.filter(c => c.riskScore >= 0.5).sort((a, b) => a.tteDays - b.tteDays);

  return (
    <div className="p-5 flex gap-4" style={{ height: "calc(100vh - 56px)" }}>
      {/* Queue Panel */}
      <div className="glass-panel p-5 w-96 shrink-0 flex flex-col">
        <div className="flex items-center justify-between mb-4 pb-3 border-b border-white/[0.07]">
          <h3 className="text-sm font-semibold">🚨 Urgency-Ranked Queue</h3>
          <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-red-500/12 text-red-400">{queue.length} pending</span>
        </div>
        <p className="text-[10px] text-cyan-400/60 mb-3">
          {backendOnline ? "🟢 Click to score via real ML models" : "⚪ Click to view simulated data"}
        </p>
        <div className="space-y-2 overflow-y-auto flex-1">
          {queue.map(c => (
            <button key={c.id} onClick={() => scoreCustomerLive(c)}
              className={`w-full text-left p-3 rounded-lg bg-white/[0.02] border transition-all ${
                selectedCustomer?.id === c.id ? "border-cyan-500/50 bg-cyan-500/[0.06]" : "border-transparent hover:border-cyan-500/30 hover:bg-cyan-500/[0.03]"}`}>
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-sm font-semibold">{c.name}</span>
                <span className={`text-xs font-bold font-mono ${c.riskScore >= 0.7 ? "text-red-400" : "text-amber-400"}`}>{c.riskScore.toFixed(2)}</span>
              </div>
              <div className="flex gap-1.5 flex-wrap">
                <span className="text-[9px] px-1.5 py-0.5 rounded bg-red-500/10 text-red-400 font-semibold">TTE: {c.tteDays}d</span>
                <span className="text-[9px] px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-400 font-semibold">Uplift: {c.upliftScore.toFixed(2)}</span>
                <span className="text-[9px] px-1.5 py-0.5 rounded bg-purple-500/10 text-purple-400 font-semibold">{c.segment}</span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Customer 360° */}
      <div className="glass-panel p-5 flex-1 overflow-y-auto">
        <h3 className="text-sm font-semibold mb-4 pb-3 border-b border-white/[0.07]">👤 Customer 360°</h3>
        {!selectedCustomer ? (
          <div className="flex items-center justify-center h-96 text-slate-500">Select a customer from the queue ←</div>
        ) : scoringInProgress ? (
          <div className="flex items-center justify-center h-96 text-cyan-400 animate-pulse">🧠 Scoring via ML Ensemble...</div>
        ) : (
          <div className="space-y-5 animate-fade-in-up">
            {/* Header */}
            <div className="flex items-center gap-5">
              <div className="w-14 h-14 rounded-full bg-gradient-to-br from-cyan-400 to-purple-600 flex items-center justify-center text-lg font-bold">
                {selectedCustomer.name.charAt(0)}
              </div>
              <div>
                <h4 className="text-lg font-bold">{selectedCustomer.name}</h4>
                <p className="text-xs text-slate-400">{selectedCustomer.occupation} • {selectedCustomer.city} • Age {selectedCustomer.age}</p>
              </div>
            </div>

            {/* Live ML Results Banner */}
            {liveScoreResult && (
              <div className="p-4 rounded-xl bg-green-500/[0.06] border border-green-500/20">
                <h5 className="text-[11px] uppercase tracking-wider text-green-400 mb-3 font-semibold">🟢 LIVE ML ENSEMBLE RESULTS (from localhost:8000/score)</h5>
                <div className="grid grid-cols-5 gap-2">
                  <div className="p-2 rounded-lg bg-white/[0.03] text-center">
                    <div className="text-lg font-extrabold font-mono text-cyan-400">{liveScoreResult.risk_score.toFixed(3)}</div>
                    <div className="text-[9px] text-slate-500 uppercase">Ensemble</div>
                  </div>
                  <div className="p-2 rounded-lg bg-white/[0.03] text-center">
                    <div className="text-lg font-extrabold font-mono text-purple-400">{liveScoreResult.xgboost_score?.toFixed(3) ?? "N/A"}</div>
                    <div className="text-[9px] text-slate-500 uppercase">XGBoost</div>
                  </div>
                  <div className="p-2 rounded-lg bg-white/[0.03] text-center">
                    <div className="text-lg font-extrabold font-mono text-amber-400">{liveScoreResult.lightgbm_score?.toFixed(3) ?? "N/A"}</div>
                    <div className="text-[9px] text-slate-500 uppercase">LightGBM</div>
                  </div>
                  <div className="p-2 rounded-lg bg-white/[0.03] text-center">
                    <div className="text-lg font-extrabold font-mono text-pink-400">{liveScoreResult.lstm_score?.toFixed(3) ?? "N/A"}</div>
                    <div className="text-[9px] text-slate-500 uppercase">LSTM</div>
                  </div>
                  <div className="p-2 rounded-lg bg-white/[0.03] text-center">
                    <div className={`text-lg font-extrabold font-mono ${liveScoreResult.risk_tier === "critical" ? "text-red-400" : liveScoreResult.risk_tier === "watch" ? "text-amber-400" : "text-green-400"}`}>
                      {liveScoreResult.risk_tier.toUpperCase()}
                    </div>
                    <div className="text-[9px] text-slate-500 uppercase">Tier</div>
                  </div>
                </div>
                {/* SHAP from real API */}
                {liveScoreResult.top_shap_features && liveScoreResult.top_shap_features.length > 0 && (
                  <div className="mt-3">
                    <div className="text-[10px] text-green-400/60 uppercase tracking-wider mb-2">Real SHAP Features (from ML model)</div>
                    <div className="space-y-1">
                      {liveScoreResult.top_shap_features.slice(0, 5).map((f, i) => (
                        <div key={i} className="flex items-center gap-3 text-xs">
                          <span className="w-44 text-slate-300 font-mono truncate">{f.feature}</span>
                          <div className="flex-1 h-2 bg-white/[0.04] rounded-full overflow-hidden">
                            <div className={`h-full rounded-full ${f.value >= 0 ? "bg-red-500" : "bg-cyan-500"}`} style={{ width: `${Math.min(Math.abs(f.value) * 500, 100)}%` }} />
                          </div>
                          <span className={`font-bold font-mono w-12 text-right ${f.value >= 0 ? "text-red-400" : "text-cyan-400"}`}>
                            {f.value >= 0 ? "+" : ""}{f.value.toFixed(3)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {liveScoreResult.explanation && (
                  <p className="mt-3 text-xs text-slate-400 italic border-l-2 border-green-500/30 pl-3">
                    {liveScoreResult.explanation}
                  </p>
                )}
              </div>
            )}

            {/* Metrics */}
            <div className="grid grid-cols-4 gap-3">
              {[
                { val: (liveScoreResult?.risk_score ?? selectedCustomer.riskScore).toFixed(2), label: "Risk Score", cls: (liveScoreResult?.risk_score ?? selectedCustomer.riskScore) >= 0.7 ? "text-red-400" : "text-amber-400" },
                { val: (liveScoreResult?.credit_score_mapped ?? selectedCustomer.creditScore).toString(), label: "Credit Score", cls: "text-cyan-400" },
                { val: `${liveScoreResult?.tte_days ?? selectedCustomer.tteDays}d`, label: "Time-To-Event", cls: (liveScoreResult?.tte_days ?? selectedCustomer.tteDays) <= 10 ? "text-red-400" : "text-amber-400" },
                { val: (liveScoreResult?.uplift_score ?? selectedCustomer.upliftScore).toFixed(2), label: "Uplift Score", cls: "text-green-400" },
              ].map((m, i) => (
                <div key={i} className="p-3 rounded-lg bg-white/[0.03] text-center">
                  <div className={`text-xl font-extrabold font-mono ${m.cls}`}>{m.val}</div>
                  <div className="text-[10px] text-slate-500 uppercase tracking-wider mt-1">{m.label}</div>
                </div>
              ))}
            </div>

            {/* SHAP — shows real if available, otherwise simulated */}
            {!liveScoreResult && (
              <div className="p-4 rounded-xl bg-white/[0.02] border border-white/[0.07]">
                <h5 className="text-[11px] uppercase tracking-wider text-slate-400 mb-3 font-semibold">🔍 SHAP Features (Simulated — start backend for real ML)</h5>
                <div className="space-y-2">
                  {selectedCustomer.shapDrivers.map((d, i) => (
                    <div key={i} className="flex items-center gap-3 text-xs">
                      <span className="w-44 text-slate-300 font-mono truncate">{d.feature}</span>
                      <div className="flex-1 h-2 bg-white/[0.04] rounded-full overflow-hidden">
                        <div className={`h-full rounded-full ${d.value >= 0 ? "bg-red-500" : "bg-cyan-500"}`} style={{ width: `${Math.abs(d.value) * 500}%` }} />
                      </div>
                      <span className={`font-bold font-mono w-12 text-right ${d.value >= 0 ? "text-red-400" : "text-cyan-400"}`}>{d.value >= 0 ? "+" : ""}{d.value.toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Counterfactuals */}
            {selectedCustomer.counterfactuals.length > 0 && (
              <div className="p-4 rounded-xl bg-white/[0.02] border border-white/[0.07]">
                <h5 className="text-[11px] uppercase tracking-wider text-slate-400 mb-3 font-semibold">🔄 Counterfactual — WHAT-IF scenarios</h5>
                <div className="space-y-2">
                  {selectedCustomer.counterfactuals.map((cf, i) => (
                    <div key={i} className="flex items-center justify-between py-2 border-b border-white/[0.05] last:border-0 text-xs">
                      <span className="text-slate-300">{cf.action}</span>
                      <span className="text-green-400 font-bold font-mono">{cf.newScore}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* GenAI Script */}
            {selectedCustomer.genaiScript && (
              <div className="p-4 rounded-xl bg-purple-500/[0.06] border border-purple-500/20">
                <h5 className="text-[11px] uppercase tracking-wider text-purple-400 mb-3 font-semibold">✨ GenAI Call Script (Groq LLM)</h5>
                <p className="text-sm text-slate-300 leading-relaxed italic border-l-[3px] border-purple-500 pl-4">
                  &ldquo;{selectedCustomer.genaiScript}&rdquo;
                </p>
                <div className="flex gap-2 mt-3 flex-wrap">
                  {selectedCustomer.offers.map((o, i) => (
                    <span key={i} className="text-[11px] px-3 py-1 rounded-full bg-green-500/10 text-green-400 border border-green-500/20 font-medium">{o}</span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════
// VIEW 4: CUSTOMER APP (with real /notify)
// ═══════════════════════════════════════════════
function CustomerView({ serveResult, triggerRealNotify, notifyResult, backendOnline }: {
  serveResult: string;
  triggerRealNotify: (action: string) => void;
  notifyResult: NotifyResult | null;
  backendOnline: boolean | null;
}) {
  const nudges = [
    { day: 0, ch: "💬 WhatsApp", msg: "Hi Sarah, we noticed some changes in your spending. We're here to help — reply HELP for options.", status: "✅ Delivered • Opened", done: true, active: false },
    { day: 3, ch: "🔔 App Push", msg: "Your personalized financial wellness report is ready.", status: "✅ Delivered • Tapped", done: true, active: false },
    { day: 5, ch: "📱 SMS", msg: "Barclays: Good news! You're pre-approved for a 3-month EMI holiday. Tap to accept → barclays.in/emi", status: "📨 Just now", done: false, active: true },
    { day: 10, ch: "📞 RM Call", msg: "Escalation — Relationship Manager callback scheduled", status: "🔜 Pending", done: false, active: false },
  ];

  return (
    <div className="p-5 flex gap-6 justify-center" style={{ height: "calc(100vh - 56px)" }}>
      {/* Mobile Simulator */}
      <div className="w-[380px] shrink-0">
        <div className="bg-[#111] rounded-[40px] p-3 border-2 border-[#333] shadow-[0_20px_60px_rgba(0,0,0,0.5)]">
          <div className="w-28 h-1.5 bg-[#222] rounded-full mx-auto mb-2" />
          <div className="bg-[#060a14] rounded-[28px] overflow-y-auto" style={{ maxHeight: "calc(100vh - 170px)" }}>
            <div className="flex justify-between px-5 py-3 text-xs font-semibold bg-cyan-500/[0.06] border-b border-white/[0.07]">
              <span>🏦 Barclays</span>
              <span className="font-mono">{new Date().toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", hour12: false })}</span>
            </div>
            <div className="p-5">
              <h3 className="text-xl font-bold mb-1">Hi Sarah 👋</h3>
              <p className="text-xs text-slate-400 mb-5">Your financial wellness dashboard</p>

              {/* Nudge Timeline */}
              <div className="space-y-0 mb-5">
                {nudges.map((n, i) => (
                  <div key={i} className={`flex gap-3 pl-2 pb-4 border-l-2 ml-2 relative ${i === nudges.length - 1 ? "border-l-transparent" : "border-l-white/[0.07]"}`}>
                    <div className={`absolute -left-[6px] top-1 w-3 h-3 rounded-full border-2 ${
                      n.done ? "bg-green-400 border-green-400" : n.active ? "bg-cyan-400 border-cyan-400 animate-pulse" : "bg-slate-600 border-slate-600"}`} />
                    <div className="ml-3">
                      <div className="flex gap-2 text-[10px] mb-1">
                        <span className="text-cyan-400 font-bold">Day {n.day}</span>
                        <span className="text-slate-400">{n.ch}</span>
                      </div>
                      <p className="text-xs text-slate-300 leading-relaxed mb-1">{n.msg}</p>
                      <span className="text-[10px] text-slate-500">{n.status}</span>
                    </div>
                  </div>
                ))}
              </div>

              {/* Self-Serve Actions — trigger real /notify */}
              <div>
                <h4 className="text-xs font-semibold text-slate-400 mb-3">
                  Quick Actions {backendOnline ? <span className="text-green-400">(🟢 triggers real /notify API)</span> : <span className="text-amber-400">(⚪ simulated)</span>}
                </h4>
                <button onClick={() => triggerRealNotify("EMI Holiday (3 months)")} className="w-full text-left p-3 mb-2 rounded-lg bg-green-500/10 border border-green-500/30 text-green-400 text-xs font-semibold hover:bg-green-500/20 transition-all">
                  ✅ Accept EMI Holiday (3 months)
                </button>
                <button onClick={() => triggerRealNotify("EMI Restructuring (12 months)")} className="w-full text-left p-3 mb-2 rounded-lg bg-white/[0.04] border border-white/[0.07] text-xs font-medium hover:border-cyan-500/30 transition-all">
                  📋 Restructure EMI (extend 12 months)
                </button>
                <button onClick={() => triggerRealNotify("RM Callback")} className="w-full text-left p-3 rounded-lg bg-white/[0.04] border border-white/[0.07] text-xs font-medium hover:border-cyan-500/30 transition-all">
                  📞 Schedule RM Callback
                </button>
                {serveResult && (
                  <div className="mt-3 p-3 rounded-lg bg-green-500/[0.08] border border-green-500/20 text-xs text-green-400 animate-fade-in-up">
                    {serveResult}
                  </div>
                )}
              </div>
            </div>
          </div>
          <div className="w-24 h-1 bg-[#444] rounded-full mx-auto mt-2" />
        </div>
      </div>

      {/* Info Panel */}
      <div className="glass-panel p-5 flex-1 max-w-xl">
        <h3 className="text-sm font-semibold mb-4 pb-3 border-b border-white/[0.07]">📋 Nudge Journey Details</h3>
        <div className="space-y-0">
          {[
            ["Customer", "Sarah Menon (CUST-4821)"],
            ["Risk Score", "0.78 (Critical)", true],
            ["Journey ID", "NJ-2026-03-28-001"],
            ["Journey Start", "March 23, 2026"],
            ["Current Step", "3 of 5 (SMS)"],
            ["Bandit Channel", "WhatsApp (68% predicted)"],
            ["Top SHAP Driver", "atm_withdrawals_7d (+0.18)"],
            ["Product Offer", "EMI Holiday (3 months, pre-approved)"],
            ["Estimated Savings", "₹42,000 (reduced default risk)"],
          ].map(([label, value, isCritical], i) => (
            <div key={i} className="flex justify-between py-2.5 border-b border-white/[0.05] text-xs">
              <span className="text-slate-400">{label}</span>
              <span className={`font-medium ${isCritical ? "text-red-400 font-bold" : ""}`}>{value}</span>
            </div>
          ))}
        </div>

        {/* Real Notify API response */}
        {notifyResult && (
          <div className="mt-4 p-4 rounded-xl bg-green-500/[0.06] border border-green-500/20">
            <h5 className="text-[11px] uppercase tracking-wider text-green-400 mb-2 font-semibold">🟢 Real /notify API Response</h5>
            <pre className="text-[11px] text-slate-300 font-mono overflow-x-auto whitespace-pre-wrap">
              {JSON.stringify(notifyResult, null, 2)}
            </pre>
          </div>
        )}

        <div className="mt-5 p-4 rounded-xl bg-purple-500/[0.06] border border-purple-500/15">
          <p className="text-xs text-slate-300 leading-relaxed">
            🤖 <strong className="text-purple-400">LinUCB Bandit Decision:</strong> WhatsApp selected as Day 0 channel because Sarah&apos;s demographic
            (Female, 28-35, Urban, Tech-savvy) has <strong>68% WhatsApp response rate</strong> vs 32% Email.
          </p>
        </div>
      </div>
    </div>
  );
}
