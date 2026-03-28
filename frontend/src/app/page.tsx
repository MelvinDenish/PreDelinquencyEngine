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
import Link from "next/link";
import gsap from "gsap";
import {
  Activity, Shield, Users, Smartphone, Zap, TrendingUp,
  AlertTriangle, CheckCircle, Wifi, WifiOff,
  ArrowRight, BarChart3, Lock, ChevronDown, ChevronUp,
  Phone, Clock, Target, ShieldAlert, Stethoscope, Briefcase,
  TrendingDown, AlertCircle, Check,
} from "lucide-react";

// ═══════════════════════════════════════════════
// ANIMATED NUMBER — GSAP count-up on mount
// ═══════════════════════════════════════════════
function AnimatedNumber({ value, prefix = "", suffix = "", duration = 1.2, decimals = 0 }: {
  value: number; prefix?: string; suffix?: string; duration?: number; decimals?: number;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const tweenRef = useRef<{ val: number }>({ val: 0 });

  useEffect(() => {
    if (!ref.current) return;
    tweenRef.current.val = 0;
    gsap.to(tweenRef.current, {
      val: value,
      duration,
      ease: "power2.out",
      onUpdate: () => {
        if (ref.current) {
          const v = tweenRef.current.val;
          ref.current.textContent = prefix + (decimals > 0 ? v.toFixed(decimals) : Math.round(v).toLocaleString()) + suffix;
        }
      },
    });
  }, [value, prefix, suffix, duration, decimals]);

  return <span ref={ref}>{prefix}0{suffix}</span>;
}

// ═══════════════════════════════════════════════
// TOAST SYSTEM
// ═══════════════════════════════════════════════
interface ToastItem {
  id: number;
  message: string;
  type: "success" | "info";
}

function ToastContainer({ toasts, onRemove }: { toasts: ToastItem[]; onRemove: (id: number) => void }) {
  return (
    <div className="toast-container">
      {toasts.map(t => (
        <ToastNotification key={t.id} toast={t} onRemove={onRemove} />
      ))}
    </div>
  );
}

function ToastNotification({ toast, onRemove }: { toast: ToastItem; onRemove: (id: number) => void }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    gsap.fromTo(el,
      { x: 80, opacity: 0 },
      { x: 0, opacity: 1, duration: 0.4, ease: "power3.out" },
    );
    const timeout = setTimeout(() => {
      gsap.to(el, {
        x: 80, opacity: 0, duration: 0.3, ease: "power2.in",
        onComplete: () => onRemove(toast.id),
      });
    }, 3000);
    return () => clearTimeout(timeout);
  }, [toast.id, onRemove]);

  return (
    <div ref={ref} className={`toast ${toast.type === "success" ? "toast-success" : "toast-info"}`}>
      <Check className="w-4 h-4 shrink-0" />
      <span>{toast.message}</span>
    </div>
  );
}

// ═══════════════════════════════════════════════
// SKELETON LOADER — shimmer placeholder for tab transitions
// ═══════════════════════════════════════════════
function TabSkeleton() {
  return (
    <div className="p-5 space-y-4 animate-fade-in-up">
      <div className="grid grid-cols-6 gap-3">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="skeleton-card p-3.5 text-center" style={{ height: 80 }}>
            <div className="skeleton" style={{ height: 24, width: "60%", margin: "0 auto 8px" }} />
            <div className="skeleton" style={{ height: 10, width: "80%", margin: "0 auto" }} />
          </div>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div className="skeleton-card" style={{ height: 260 }} />
        <div className="skeleton-card" style={{ height: 260 }} />
      </div>
      <div className="skeleton-card" style={{ height: 200 }} />
    </div>
  );
}

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
  const [activeView, setActiveView] = useState<string>("executive");
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
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const [tabLoading, setTabLoading] = useState(false);

  const addToast = useCallback((message: string, type: "success" | "info" = "success") => {
    const id = Date.now() + Math.random();
    setToasts(prev => [...prev, { id, message, type }]);
  }, []);

  const removeToast = useCallback((id: number) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  // Tab switching with skeleton shimmer
  const switchTab = useCallback((key: string) => {
    if (key === activeView) return;
    setTabLoading(true);
    setActiveView(key);
    // Brief shimmer then reveal
    setTimeout(() => setTabLoading(false), 400);
  }, [activeView]);

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
      let tier: string = customer.riskTier;
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
        const channels = ["SMS", "WhatsApp", "Email", "Push"];
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
              channel = ch === "sms" ? "SMS" : ch === "email" ? "Email" : ch === "whatsapp" ? "WhatsApp" : "Push";
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
        setLiveScoreResult(result);
        addToast(`${customer.name} scored: ${result.risk_score.toFixed(2)} (${result.risk_tier})`, "info");
      } catch {
        setLiveScoreResult(null);
      }
    }
    setScoringInProgress(false);
  }, [backendOnline, addToast]);

  // ──── Send real notification from Customer view ────
  const triggerRealNotify = useCallback(async (action: string) => {
    setServeResult(`Sending ${action} via dispatcher...`);
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
        setServeResult(`${action} dispatched via real service. Status: ${result.status}. Channels attempted: ${result.channels_attempted || 0}.`);
        addToast(`${action} dispatched successfully`, "success");
        return;
      } catch {
        // Fall through to simulated
      }
    }
    setServeResult(`${action} accepted (simulated). Confirmation sent via SMS & Email.`);
    addToast(`${action} accepted — confirmation sent`, "success");
  }, [backendOnline, addToast]);

  const views = [
    { key: "executive", label: "Portfolio Overview", icon: <TrendingUp className="w-4 h-4" /> },
    { key: "rm", label: "Relationship Manager", icon: <Users className="w-4 h-4" /> },
    { key: "customer", label: "Customer Service", icon: <Smartphone className="w-4 h-4" /> },
    { key: "godmode", label: "Operations Centre", icon: <Activity className="w-4 h-4" /> },
  ];

  return (
    <div className="min-h-screen">
      {/* ═══ TOP BAR ═══ */}
      <header className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 h-14"
        style={{ background: "#0a1525", borderBottom: "1px solid rgba(255,255,255,0.08)" }}>

        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2.5">
            <div style={{
              width: 32, height: 32, borderRadius: 6,
              background: "rgba(0,174,239,0.1)",
              display: "flex", alignItems: "center", justifyContent: "center",
            }}>
              <Shield className="w-4 h-4 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span style={{ fontSize: 15, fontWeight: 700, color: "#FFFFFF", letterSpacing: "0.04em" }}>BARCLAYS</span>
                <span style={{ width: 1, height: 16, background: "rgba(255,255,255,0.2)", display: "inline-block" }} />
                <span style={{ fontSize: 11, fontWeight: 600, color: "rgba(255,255,255,0.7)", letterSpacing: "0.02em" }}>
                  Pre-Delinquency Intelligence
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex gap-0.5">
          {views.map(v => (
            <button key={v.key} onClick={() => switchTab(v.key)}
              className="px-3.5 py-1.5 text-xs font-medium transition-all flex items-center gap-1.5"
              style={{
                background: activeView === v.key ? "rgba(255,255,255,0.15)" : "transparent",
                borderBottom: activeView === v.key ? "2px solid #FFFFFF" : "2px solid transparent",
                color: activeView === v.key ? "#FFFFFF" : "rgba(255,255,255,0.55)",
                borderRadius: "4px 4px 0 0",
              }}>
              {v.icon}{v.label}
            </button>
          ))}
          <Link href="/whatif"
            className="px-3.5 py-1.5 text-xs font-medium transition-all flex items-center gap-1.5 ml-2"
            style={{
              background: "rgba(255,255,255,0.08)",
              border: "1px solid rgba(255,255,255,0.15)",
              color: "rgba(255,255,255,0.7)",
              textDecoration: "none",
              borderRadius: 4,
            }}>
            <BarChart3 className="w-3.5 h-3.5" />
            Stress Testing
          </Link>
        </nav>

        {/* Status row */}
        <div className="flex items-center gap-3 text-xs">
          <div className="flex items-center gap-1.5 px-2 py-1 rounded"
            style={{ background: "rgba(255,255,255,0.08)" }}>
            <Lock className="w-2.5 h-2.5 text-white/60" />
            <span style={{ color: "rgba(255,255,255,0.6)", fontSize: 9, fontWeight: 600, letterSpacing: "0.05em" }}>SECURED</span>
          </div>
          <div className="flex items-center gap-1.5 px-2 py-1 rounded"
            style={{ background: "rgba(255,255,255,0.08)" }}>
            {backendOnline === null ? (
              <><div className="w-2 h-2 rounded-full bg-white/40" /><span style={{ color: "rgba(255,255,255,0.5)", fontSize: 9 }}>CONNECTING</span></>
            ) : backendOnline ? (
              <><Wifi className="w-3 h-3 text-green-300" /><span style={{ color: "#86EFAC", fontSize: 9, fontWeight: 600 }}>ML ONLINE</span></>
            ) : (
              <><WifiOff className="w-3 h-3 text-amber-300" /><span style={{ color: "#FCD34D", fontSize: 9, fontWeight: 600 }}>SIMULATED</span></>
            )}
          </div>
          <span className="font-mono" style={{ color: "rgba(255,255,255,0.5)", fontSize: 11 }}>{clock}</span>
        </div>
      </header>

      <ToastContainer toasts={toasts} onRemove={removeToast} />

      <main className="pt-14">
        {tabLoading ? <TabSkeleton /> : (
          <>
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
          </>
        )}
      </main>
    </div>
  );
}

// ═══════════════════════════════════════════════
// VIEW: OPERATIONS CENTRE (Admin)
// ═══════════════════════════════════════════════
function GodModeView({ events, scores, interventions, counters, pipelineStage, backendOnline, modelInfo }: {
  events: EventItem[]; scores: ScoreItem[]; interventions: InterventionItem[];
  counters: { txns: number; features: number; scores: number; interventions: number };
  pipelineStage: number; backendOnline: boolean | null;
  modelInfo: Record<string, boolean>;
}) {
  const eventStreamRef = useRef<HTMLDivElement>(null);

  // Auto-scroll event stream to top when new events arrive
  useEffect(() => {
    if (eventStreamRef.current && events.length > 0) {
      gsap.to(eventStreamRef.current, { scrollTop: 0, duration: 0.3, ease: "power2.out" });
    }
  }, [events.length]);

  const pipelineNodes = [
    ["CBS", "Kafka", "Flink"],
    ["Spark", "Redis", "ML Ensemble"],
    ["SHAP", "GenAI", "Dispatcher"],
  ];

  const getNodeClass = (rowIdx: number, nodeIdx: number) => {
    const flatIdx = rowIdx * 3 + nodeIdx;
    const stageMap: Record<number, number[]> = { 1: [0, 1], 2: [2, 3, 4], 3: [5, 6], 4: [7, 8] };
    const activeNodes = stageMap[pipelineStage] || [];
    if (activeNodes.includes(flatIdx)) return "border-blue-500 bg-cyan-500/8 animate-node-glow";
    return "border-white/[0.08]";
  };

  const liveCount = scores.filter(s => s.source === "live").length;
  const simCount = scores.filter(s => s.source === "simulated").length;

  return (
    <div className="p-5 space-y-4">
      {/* Service Status Banner */}
      <div className={`flex items-center justify-between px-4 py-2.5 rounded-lg text-xs font-medium ${
        backendOnline ? "bg-green-500/10 border border-green-500/25 text-emerald-400"
        : "bg-amber-500/10 border border-amber-500/25 text-amber-400"}`}>
        <div className="flex items-center gap-2">
          {backendOnline ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />}
          <span>{backendOnline
            ? `Connected to scoring service — ML models: ${Object.entries(modelInfo).filter(([,v])=>v).map(([k])=>k).join(", ") || "loading..."}`
            : "Backend offline — using simulated data. Start the service with: docker-compose up pdi-app"
          }</span>
        </div>
        <div className="flex gap-3">
          <span className="px-2 py-0.5 rounded bg-green-100 text-emerald-400">Live: {liveCount}</span>
          <span className="px-2 py-0.5 rounded bg-white/[0.05] text-slate-300">Simulated: {simCount}</span>
        </div>
      </div>

      {/* Metric Tickers — live animated counters */}
      <div className="grid grid-cols-5 gap-3">
        {[
          { value: counters.txns, label: "Transactions Ingested", suffix: "" },
          { value: counters.features, label: "Features Computed", suffix: "" },
          { value: counters.scores, label: "Risk Scores Updated", suffix: "" },
          { value: counters.interventions, label: "Interventions Triggered", suffix: "" },
        ].map((t, i) => (
          <div key={i} className="glass-panel p-4 text-center">
            <div className="text-2xl font-bold font-mono text-slate-100">
              <AnimatedNumber value={t.value} duration={0.6} />
            </div>
            <div className="text-[10px] text-slate-500 uppercase tracking-wider mt-1">{t.label}</div>
          </div>
        ))}
        <div className="glass-panel p-4 text-center">
          <div className="text-2xl font-bold font-mono text-slate-100">47ms</div>
          <div className="text-[10px] text-slate-500 uppercase tracking-wider mt-1">Avg Latency</div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Pipeline Status */}
        <div className="glass-panel p-5">
          <div className="flex items-center justify-between mb-4 pb-3 border-b border-white/[0.06]">
            <h3 className="text-sm font-semibold flex items-center gap-2 text-slate-100"><Activity className="w-4 h-4 text-cyan-400" /> Pipeline Status</h3>
            <span className="text-[10px] font-semibold px-2 py-0.5 rounded bg-green-500/10 text-emerald-400 border border-green-500/25">ACTIVE</span>
          </div>
          <div className="space-y-3">
            {pipelineNodes.map((row, ri) => (
              <div key={ri} className="flex items-center justify-center gap-3">
                {row.map((node, ni) => (
                  <div key={ni} className="flex items-center gap-3">
                    <div className={`px-4 py-2.5 rounded-md bg-white/[0.03] border text-xs font-medium text-center min-w-[110px] transition-all duration-500 text-slate-200 ${getNodeClass(ri, ni)}`}>
                      {node}
                    </div>
                    {ni < 2 && <ArrowRight className="w-4 h-4 text-slate-300" />}
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>

        {/* Live Event Stream */}
        <div className="glass-panel p-5">
          <div className="flex items-center justify-between mb-4 pb-3 border-b border-white/[0.06]">
            <h3 className="text-sm font-semibold text-slate-100">Live Event Stream</h3>
            <span className="text-[10px] text-slate-400 font-mono">{events.length} events</span>
          </div>
          <div ref={eventStreamRef} className="space-y-1.5 max-h-[280px] overflow-y-auto">
            {events.map(evt => (
              <div key={evt.id} className={`px-3 py-2 rounded text-[11px] font-mono bg-white/[0.03] border-l-[3px] animate-slide-in ${evt.isStress ? "border-l-red-500" : "border-l-blue-400"}`}>
                <span className="text-slate-400 mr-2">{evt.time}</span>
                <span className={`font-semibold mr-1.5 ${evt.isStress ? "text-red-400" : "text-cyan-400"}`}>{evt.type}</span>
                <span className="text-slate-200">{evt.customer}</span>
                <span className="text-slate-300 mx-1.5">&middot;</span>
                <span className="text-slate-500">{evt.category}</span>
                <span className="text-slate-500 mx-1.5">₹{evt.amount.toLocaleString()}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Recent Scores */}
        <div className="glass-panel p-5">
          <div className="flex items-center justify-between mb-4 pb-3 border-b border-white/[0.06]">
            <h3 className="text-sm font-semibold text-slate-100">Recent Risk Scores</h3>
          </div>
          <div className="space-y-1.5 max-h-[280px] overflow-y-auto">
            {scores.map(s => (
              <div key={s.id} className="flex items-center gap-3 px-3 py-2 rounded bg-white/[0.03] animate-slide-in">
                <span className={`text-[10px] font-semibold px-2 py-0.5 rounded ${
                  s.tier === "critical" ? "bg-red-500/10 text-red-400 border border-red-500/25" :
                  s.tier === "watch" ? "bg-amber-500/10 text-amber-400 border border-amber-500/25" :
                  "bg-green-500/10 text-emerald-400 border border-green-500/25"
                }`}>{s.tier.toUpperCase()}</span>
                <span className="text-xs font-medium flex-1 text-slate-200">{s.name}</span>
                <span className="text-xs font-bold font-mono text-slate-100">{s.score.toFixed(2)}</span>
                {s.source === "live" && s.xgb !== null && (
                  <span className="text-[9px] text-blue-500 font-mono" title={`XGB:${s.xgb?.toFixed(2)} LGB:${s.lgb?.toFixed(2)} LSTM:${s.lstm?.toFixed(2)}`}>
                    XGB:{s.xgb?.toFixed(2)}
                  </span>
                )}
                <span className={`text-[9px] font-semibold px-1.5 py-0.5 rounded ${
                  s.source === "live" ? "bg-green-500/10 text-emerald-400 border border-green-500/25" : "bg-white/[0.05] text-slate-500"
                }`}>{s.source === "live" ? "LIVE" : "SIM"}</span>
                <span className="text-[10px] text-slate-400">{s.time}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Interventions */}
        <div className="glass-panel p-5">
          <div className="flex items-center justify-between mb-4 pb-3 border-b border-white/[0.06]">
            <h3 className="text-sm font-semibold text-slate-100">Interventions Dispatched</h3>
          </div>
          <div className="space-y-1.5 max-h-[280px] overflow-y-auto">
            {interventions.map(intv => (
              <div key={intv.id} className="flex items-center gap-3 px-3 py-2 rounded bg-white/[0.03] animate-slide-in">
                <span className="text-sm">{intv.channel}</span>
                <span className="text-xs font-medium text-slate-200">{intv.name}</span>
                <span className="text-[11px] text-slate-500 flex-1 truncate">{intv.message}</span>
                <span className={`text-[9px] font-semibold px-1.5 py-0.5 rounded ${
                  intv.source === "live" ? "bg-green-500/10 text-emerald-400 border border-green-500/25" : "bg-white/[0.05] text-slate-500"
                }`}>{intv.source === "live" ? "LIVE" : "SIM"}</span>
                <span className="text-[10px] text-slate-400">{intv.time}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════
// VIEW: PORTFOLIO OVERVIEW (Risk Officers / Executives)
// ═══════════════════════════════════════════════
function ExecutiveView() {
  const migrationData = [
    { label: "Stable to Watch", count: 847, pct: 35, color: "bg-amber-500/100" },
    { label: "Watch to Critical", count: 412, pct: 18, color: "bg-red-500/100" },
    { label: "Critical to Default", count: 142, pct: 6, color: "bg-red-800" },
    { label: "Watch to Stable (recovered)", count: 1024, pct: 42, color: "bg-green-500/100" },
    { label: "Critical to Watch (recovered)", count: 638, pct: 28, color: "bg-teal-500" },
  ];

  const channelData = [
    { name: "WhatsApp", response: 68, color: "#34D399" },
    { name: "SMS", response: 45, color: "#22D3EE" },
    { name: "Email", response: 32, color: "#A78BFA" },
    { name: "RM Call", response: 89, color: "#FBBF24" },
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
      {/* KPI Summary Bar — animated count-up on load */}
      <div className="grid grid-cols-6 gap-3">
        {[
          { num: 4217, prefix: "₹", suffix: "Cr", label: "Total AUM", sub: "Retail portfolio", accent: "#00395D" },
          { num: 24891, prefix: "", suffix: "", label: "Active Customers", sub: "Scored this week", accent: "#22D3EE" },
          { num: 1.83, prefix: "", suffix: "%", label: "Gross NPA", sub: "Down 0.12% from last month", accent: "#34D399", decimals: 2 },
          { num: 8.4, prefix: "₹", suffix: "Cr", label: "AUM Protected", sub: "Via interventions (90d)", accent: "#A78BFA", decimals: 1 },
          { num: 2847, prefix: "", suffix: "", label: "Interventions", sub: "Dispatched this month", accent: "#FBBF24" },
          { num: 64, prefix: "", suffix: "%", label: "Response Rate", sub: "Across all channels", accent: "#34D399" },
        ].map((kpi, i) => (
          <div key={i} className="glass-panel p-3.5 text-center border-t-2" style={{ borderTopColor: kpi.accent }}>
            <div className="text-xl font-bold font-mono text-slate-100">
              <AnimatedNumber value={kpi.num} prefix={kpi.prefix} suffix={kpi.suffix} decimals={kpi.decimals ?? 0} duration={1.4} />
            </div>
            <div className="text-[10px] text-slate-500 uppercase tracking-wider mt-1 font-semibold">{kpi.label}</div>
            <div className="text-[9px] text-slate-400 mt-0.5">{kpi.sub}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Risk Migration */}
        <div className="glass-panel p-6 col-span-2">
          <h3 className="text-sm font-semibold mb-6 pb-3 border-b border-white/[0.06] text-slate-100">Risk Migration Matrix (90-Day)</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
            {migrationData.map((m, i) => (
              <div key={i} className="flex items-center gap-4">
                <span className="text-xs font-medium w-52 shrink-0 text-slate-300">{m.label}</span>
                <div className="flex-1 h-7 bg-white/[0.05] rounded overflow-hidden">
                  <div className={`h-full rounded ${m.color} transition-all duration-1000`} style={{ width: `${m.pct}%` }} />
                </div>
                <span className="text-xs text-slate-500 w-44 text-right">{m.count.toLocaleString()} customers ({m.pct}%)</span>
              </div>
            ))}
          </div>
        </div>

        {/* ROI */}
        <div className="glass-panel p-5">
          <h3 className="text-sm font-semibold mb-4 pb-3 border-b border-white/[0.06] text-slate-100">Intervention ROI — A/B Uplift Analysis</h3>
          <div className="grid grid-cols-2 gap-4 mb-5">
            {[
              { val: "14.2%", label: "Uplift Lift", color: "#34D399" },
              { val: "₹8.4Cr", label: "AUM Protected", color: "#22D3EE" },
              { val: "72.8%", label: "Treated Recovery", color: "#A78BFA" },
              { val: "58.6%", label: "Holdout Recovery", color: "#FBBF24" },
            ].map((r, i) => (
              <div key={i} className="p-4 rounded-lg bg-white/[0.03] text-center">
                <div className="text-2xl font-bold font-mono" style={{ color: r.color }}>{r.val}</div>
                <div className="text-[10px] text-slate-500 uppercase tracking-wider mt-1.5">{r.label}</div>
              </div>
            ))}
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart data={upliftChart}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="month" tick={{ fill: "#94A3B8", fontSize: 11 }} />
              <YAxis tick={{ fill: "#94A3B8", fontSize: 11 }} domain={[40, 80]} />
              <Tooltip contentStyle={{ background: "#132337", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 6, fontSize: 12, color: "#E2E8F0" }} />
              <Area type="monotone" dataKey="treated" stroke="#22D3EE" fill="rgba(0,119,182,0.1)" strokeWidth={2} name="Treated" />
              <Area type="monotone" dataKey="holdout" stroke="#475569" fill="rgba(148,163,184,0.05)" strokeWidth={2} strokeDasharray="5 5" name="Holdout" />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Channel Efficiency */}
        <div className="glass-panel p-5">
          <h3 className="text-sm font-semibold mb-4 pb-3 border-b border-white/[0.06] text-slate-100">Channel Efficiency — LinUCB Bandit</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }} className="mb-5">
            {channelData.map((ch, i) => (
              <div key={i} className="flex items-center gap-3 text-xs">
                <span className="w-20 font-medium text-slate-200">{ch.name}</span>
                <div className="flex-1 h-6 bg-white/[0.05] rounded overflow-hidden">
                  <div className="h-full rounded transition-all duration-1000" style={{ width: `${ch.response}%`, background: ch.color }} />
                </div>
                <span className="text-slate-500 w-24 text-right font-mono">{ch.response}%</span>
              </div>
            ))}
          </div>
          <ResponsiveContainer width="100%" height={170}>
            <BarChart data={channelData} barSize={28} barGap={8} barCategoryGap="25%">
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="name" tick={{ fill: "#94A3B8", fontSize: 10 }} />
              <YAxis tick={{ fill: "#94A3B8", fontSize: 10 }} />
              <Tooltip contentStyle={{ background: "#132337", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 6, fontSize: 12, color: "#E2E8F0" }} />
              <Bar dataKey="response" radius={[4, 4, 0, 0]}>
                {channelData.map((entry, idx) => <Cell key={idx} fill={entry.color} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="text-[11px] text-slate-500 mt-3 p-2.5 rounded-lg bg-cyan-500/8 border border-cyan-500/20">
            LinUCB dynamically allocates budget: WhatsApp for young urban segments, RM Call for HNI.
          </p>
        </div>

        {/* Employer Contagion Radar */}
        <div className="glass-panel p-5 col-span-2">
          <div className="flex items-center justify-between mb-4 pb-3 border-b border-white/[0.06]">
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-semibold text-slate-100">Employer Contagion Radar</h3>
              <span className="text-[9px] text-slate-400">MCA filings, News NLP, Transaction patterns</span>
            </div>
            <span className="badge-live">REAL-TIME</span>
          </div>
          <div className="space-y-2">
            {[
              { name: "Byju's / Think & Learn", customers: 847, healthScore: 0.23, atRisk: 312, trend: -0.31, signal: "Mass layoffs confirmed — 4,000+ employees affected. Salary delays 18+ days.", tier: "critical" as const },
              { name: "Paytm / One97 Comm.", customers: 312, healthScore: 0.31, atRisk: 87, trend: -0.22, signal: "Regulatory compliance action → restructuring. Hiring freeze, 15% workforce reduction.", tier: "critical" as const },
              { name: "Zomato Ltd.", customers: 423, healthScore: 0.38, atRisk: 156, trend: -0.15, signal: "Gig worker payment cycle delayed from weekly → bi-weekly. Driver attrition ↑40%.", tier: "watch" as const },
              { name: "Wipro Technologies", customers: 1234, healthScore: 0.45, atRisk: 89, trend: -0.08, signal: "Q3 revenue miss, variable pay reduced to 60%. Voluntary separation scheme active.", tier: "watch" as const },
              { name: "Tata Consultancy (TCS)", customers: 2891, healthScore: 0.82, atRisk: 12, trend: +0.02, signal: "Stable. Record hiring, salary increments on track. No stress signals.", tier: "stable" as const },
              { name: "Infosys Ltd.", customers: 1567, healthScore: 0.78, atRisk: 23, trend: -0.01, signal: "Minor bench increase, but financials strong. Monitoring only.", tier: "stable" as const },
            ].map((emp, i) => (
              <div key={i} className="flex items-center gap-4 px-3 py-2.5 rounded-lg transition-all hover:bg-white/[0.03]"
                style={{ background: emp.tier === "critical" ? "rgba(248,113,113,0.08)" : emp.tier === "watch" ? "rgba(251,191,36,0.08)" : "transparent",
                  borderLeft: `3px solid ${emp.tier === "critical" ? "#F87171" : emp.tier === "watch" ? "#FBBF24" : "#34D399"}` }}>
                <div className="w-44 shrink-0">
                  <div className="text-xs font-semibold text-slate-100">{emp.name}</div>
                  <div className="text-[10px] text-slate-400">{emp.customers.toLocaleString()} customers</div>
                </div>
                {/* Health Score gauge */}
                <div className="w-28 shrink-0">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[9px] text-slate-400 uppercase">Health</span>
                    <span className={`text-xs font-bold font-mono ${emp.healthScore < 0.35 ? "text-red-400" : emp.healthScore < 0.5 ? "text-amber-400" : "text-emerald-400"}`}>
                      {emp.healthScore.toFixed(2)}
                    </span>
                  </div>
                  <div className="h-1.5 rounded-full overflow-hidden bg-white/[0.05]">
                    <div className="h-full rounded-full transition-all" style={{
                      width: `${emp.healthScore * 100}%`,
                      background: emp.healthScore < 0.35 ? "#F87171" : emp.healthScore < 0.5 ? "#FBBF24" : "#34D399",
                    }} />
                  </div>
                </div>
                {/* At-risk count */}
                <div className="w-24 text-center shrink-0">
                  <div className={`text-sm font-bold font-mono ${emp.atRisk > 100 ? "text-red-400" : emp.atRisk > 30 ? "text-amber-400" : "text-slate-400"}`}>
                    {emp.atRisk}
                  </div>
                  <div className="text-[9px] text-slate-400">newly at-risk</div>
                </div>
                {/* Trend */}
                <div className="w-16 text-center shrink-0">
                  <span className={`text-xs font-bold font-mono ${emp.trend < -0.1 ? "text-red-400" : emp.trend < 0 ? "text-amber-400" : "text-emerald-400"}`}>
                    {emp.trend > 0 ? "+" : ""}{emp.trend.toFixed(2)}
                  </span>
                  <div className="text-[9px] text-slate-400">30d delta</div>
                </div>
                {/* Signal */}
                <div className="flex-1 text-[11px] text-slate-500 leading-snug">{emp.signal}</div>
                {/* Tier badge */}
                <span className={`shrink-0 ${emp.tier === "critical" ? "badge-critical" : emp.tier === "watch" ? "badge-watch" : "badge-stable"}`}>
                  {emp.tier}
                </span>
              </div>
            ))}
          </div>
          <p className="text-[11px] text-slate-400 mt-3 p-2.5 rounded bg-white/[0.03] border border-white/[0.06]">
            Sources: MCA quarterly filings, BSE/NSE announcements, News NLP (sentiment), employee transaction pattern anomaly detection. Refreshes every 6 hours.
          </p>
        </div>

        {/* Early Warning Signals */}
        <div className="glass-panel p-5">
          <div className="flex items-center justify-between mb-4 pb-3 border-b border-white/[0.06]">
            <h3 className="text-sm font-semibold text-slate-100">Early Warning Signals</h3>
            <span className="text-[9px] text-slate-400">Auto-detected from transaction patterns</span>
          </div>
          <div className="space-y-2.5">
            {[
              { signal: "UPI decline rate", current: "3.8%", prev: "2.3%", change: "+65%", severity: "amber" as const, detail: "Mumbai metro region — potential cash-flow stress" },
              { signal: "Lending app registrations", current: "1,847", prev: "1,302", change: "+42%", severity: "red" as const, detail: "↑ Payday lender access in 18-30 age group" },
              { signal: "Salary credit delays", current: "4.2d avg", prev: "2.1d", change: "+100%", severity: "red" as const, detail: "IT sector — correlates with Wipro/Byju's employer stress" },
              { signal: "Medical txn surge", current: "₹2.8Cr", prev: "₹2.1Cr", change: "+35%", severity: "red" as const, detail: "Bangalore cluster — possible health event contagion" },
              { signal: "ATM cash spikes", current: "+28%", prev: "baseline", change: "+28%", severity: "amber" as const, detail: "Gig worker segment — cash-hoarding behavior" },
              { signal: "Auto-debit bounces", current: "6.4%", prev: "4.1%", change: "+56%", severity: "red" as const, detail: "Home loan EMIs — post-rate-hike stress in NCR region" },
            ].map((s, i) => (
              <div key={i} className="flex items-center gap-3 px-3 py-2 rounded-lg"
                style={{ background: s.severity === "red" ? "rgba(248,113,113,0.08)" : "rgba(251,191,36,0.08)",
                  borderLeft: `3px solid ${s.severity === "red" ? "#F87171" : "#FBBF24"}` }}>
                <div className="w-44 shrink-0">
                  <div className="text-xs font-semibold text-slate-100">{s.signal}</div>
                  <div className="text-[10px] text-slate-400">{s.detail}</div>
                </div>
                <div className="flex items-center gap-2 w-32 shrink-0">
                  <span className="text-[10px] text-slate-400 line-through">{s.prev}</span>
                  <span className="text-[10px] text-slate-400">&rarr;</span>
                  <span className="text-xs font-bold font-mono" style={{ color: s.severity === "red" ? "#F87171" : "#FBBF24" }}>{s.current}</span>
                </div>
                <span className="text-[10px] font-bold px-2 py-0.5 rounded font-mono"
                  style={{ background: s.severity === "red" ? "rgba(248,113,113,0.15)" : "rgba(251,191,36,0.15)",
                    color: s.severity === "red" ? "#F87171" : "#FBBF24" }}>
                  {s.change} MoM
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Portfolio Health Scorecard */}
        <div className="glass-panel p-5">
          <div className="flex items-center justify-between mb-4 pb-3 border-b border-white/[0.06]">
            <h3 className="text-sm font-semibold text-slate-100">Portfolio Health Scorecard</h3>
            <span className="text-[9px] text-slate-400">Composite risk index</span>
          </div>
          {/* Central gauge */}
          <div className="flex flex-col items-center mb-4">
            <div className="relative w-32 h-32">
              <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
                <circle cx="50" cy="50" r="42" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="8" />
                <circle cx="50" cy="50" r="42" fill="none" stroke="url(#healthGrad)" strokeWidth="8"
                  strokeDasharray={`${72.4 * 2.64} ${100 * 2.64}`} strokeLinecap="round" />
                <defs><linearGradient id="healthGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#34D399" /><stop offset="50%" stopColor="#22D3EE" /><stop offset="100%" stopColor="#FBBF24" />
                </linearGradient></defs>
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className="text-3xl font-bold font-mono text-emerald-400">72.4</span>
                <span className="text-[9px] text-slate-400 uppercase tracking-wider">of 100</span>
              </div>
            </div>
            <span className="mt-1 badge-stable">HEALTHY</span>
          </div>
          <div className="space-y-2">
            {[
              { metric: "NPA Ratio (Gross)", value: "1.83%", target: "< 3.0%", pass: true },
              { metric: "Provision Coverage", value: "78.4%", target: "> 70%", pass: true },
              { metric: "Watch → Critical Rate", value: "18%", target: "< 25%", pass: true },
              { metric: "Intervention Response", value: "64%", target: "> 50%", pass: true },
              { metric: "Model PSI Drift", value: "0.08", target: "< 0.20", pass: true },
              { metric: "Consent Compliance", value: "97.2%", target: "> 95%", pass: true },
              { metric: "Avg Time-to-Intervene", value: "2.4h", target: "< 4h", pass: true },
              { metric: "Age Fairness SPD", value: "0.17", target: "< 0.10", pass: false },
            ].map((m, i) => (
              <div key={i} className="flex items-center justify-between px-2 py-1.5 rounded text-xs"
                style={{ background: m.pass ? "transparent" : "rgba(251,191,36,0.08)" }}>
                <span className="text-slate-300">{m.metric}</span>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-slate-400">{m.target}</span>
                  <span className={`font-bold font-mono ${m.pass ? "text-emerald-400" : "text-amber-400"}`}>{m.value}</span>
                  {m.pass ? <CheckCircle className="w-3 h-3 text-emerald-400" /> : <AlertTriangle className="w-3 h-3 text-amber-400" />}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Drift & Fairness */}
        <div className="glass-panel p-5 col-span-2">
          <h3 className="text-sm font-semibold mb-4 pb-3 border-b border-white/[0.06] text-slate-100">Model Drift & Fairness Monitor</h3>
          <div className="grid grid-cols-4 gap-3">
            {[
              { icon: <CheckCircle className="w-5 h-5 text-emerald-400" />, title: "PSI Drift", val: "0.08", desc: "Below 0.20 threshold", pass: true },
              { icon: <CheckCircle className="w-5 h-5 text-emerald-400" />, title: "Gender Parity", val: "0.96", desc: "Fairlearn DP >= 0.80", pass: true },
              { icon: <CheckCircle className="w-5 h-5 text-emerald-400" />, title: "Regional Fairness", val: "0.93", desc: "AIF360 equal opportunity", pass: true },
              { icon: <AlertTriangle className="w-5 h-5 text-amber-400" />, title: "Age Group SPD", val: "0.17", desc: "Slightly elevated for 18-25", pass: false },
            ].map((c, i) => (
              <div key={i} className={`p-4 rounded-lg text-center border ${c.pass ? "bg-green-500/10 border-green-500/25" : "bg-amber-500/10 border-amber-500/25"}`}>
                <div className="flex justify-center mb-2">{c.icon}</div>
                <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">{c.title}</div>
                <div className={`text-2xl font-bold font-mono ${c.pass ? "text-emerald-400" : "text-amber-400"}`}>{c.val}</div>
                <div className="text-[10px] text-slate-400 mt-1">{c.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════
// RM PRE-CALL AI BRIEF COMPONENT
// ═══════════════════════════════════════════════
function RMPreCallBrief({ customer }: { customer: Customer }) {
  const [objOpen, setObjOpen] = useState(false);

  const categoryMeta: Record<string, { label: string; color: string; bg: string; border: string; Icon: typeof Shield }> = {
    medical:          { label: "Medical Emergency",    color: "#EF4444", bg: "rgba(239,68,68,0.07)",   border: "rgba(239,68,68,0.25)",   Icon: Stethoscope },
    business:         { label: "Business Stress",      color: "#F59E0B", bg: "rgba(245,158,11,0.07)",  border: "rgba(245,158,11,0.25)",  Icon: Briefcase },
    lifestyle:        { label: "Lifestyle Overspend",  color: "#A855F7", bg: "rgba(168,85,247,0.07)",  border: "rgba(168,85,247,0.25)",  Icon: TrendingDown },
    income_volatility:{ label: "Income Volatility",    color: "#F97316", bg: "rgba(249,115,22,0.07)",  border: "rgba(249,115,22,0.25)",  Icon: AlertCircle },
  };
  const cat = categoryMeta[customer.stressCategory];
  const CatIcon = cat.Icon;

  const conversionDiff = customer.callConversionToday - customer.callConversionDelay;
  const urgency = customer.tteDays <= 7 ? "CALL NOW" : customer.tteDays <= 14 ? "CALL TODAY" : "THIS WEEK";
  const urgencyColor = customer.tteDays <= 7 ? "#EF4444" : customer.tteDays <= 14 ? "#F59E0B" : "#00AEEF";

  if (customer.riskTier === "stable" || !customer.aiOpener) return null;

  return (
    <div className="rounded-lg overflow-hidden border border-white/[0.08] bg-[#132337]">
      {/* Header bar */}
      <div className="flex items-center justify-between px-4 py-2.5"
        style={{ background: "rgba(0,174,239,0.08)", borderBottom: "1px solid rgba(0,174,239,0.15)" }}>
        <div className="flex items-center gap-2">
          <span style={{ fontSize: 10, fontWeight: 700, color: "#FFFFFF", letterSpacing: "0.08em" }}>PRE-CALL INTELLIGENCE BRIEF</span>
          <span style={{ fontSize: 9, color: "rgba(255,255,255,0.5)" }}>SHAP + LLM Analysis</span>
        </div>
        <span className="text-[10px] font-bold px-2 py-0.5 rounded"
          style={{ background: customer.tteDays <= 7 ? "rgba(248,113,113,0.15)" : customer.tteDays <= 14 ? "rgba(251,191,36,0.15)" : "rgba(34,211,238,0.15)",
            color: urgencyColor }}>
          {urgency}
        </span>
      </div>

      <div className="p-4 space-y-3">
        {/* Row 1: Stress Trigger + Call Window */}
        <div className="flex gap-3">
          <div className="flex-1 p-3 rounded-lg border" style={{ background: cat.bg, borderColor: cat.border }}>
            <div className="flex items-center gap-2 mb-1.5">
              <CatIcon className="w-3.5 h-3.5" style={{ color: cat.color }} />
              <span style={{ fontSize: 9, fontWeight: 700, color: cat.color, letterSpacing: "0.06em" }}>STRESS TRIGGER: {cat.label.toUpperCase()}</span>
            </div>
            <p className="text-xs leading-relaxed text-slate-200">{customer.stressTrigger}</p>
            {customer.lifeEvent && (
              <div className="mt-2 inline-flex items-center gap-1.5 px-2 py-0.5 rounded"
                style={{ background: "rgba(248,113,113,0.08)", border: "1px solid #FECACA" }}>
                <AlertTriangle className="w-2.5 h-2.5 text-red-400" />
                <span style={{ fontSize: 9, color: "#F87171", fontWeight: 700 }}>{customer.lifeEvent}</span>
              </div>
            )}
          </div>
          <div className="w-44 p-3 rounded-lg bg-white/[0.03] border border-white/[0.08]">
            <div className="flex items-center gap-1.5 mb-2">
              <Clock className="w-3 h-3 text-cyan-400" />
              <span style={{ fontSize: 9, fontWeight: 700, color: "#22D3EE", letterSpacing: "0.06em" }}>BEST CALL WINDOW</span>
            </div>
            <div className="text-sm font-bold mb-1 text-slate-100">{customer.callBestTime}</div>
            <div className="flex items-center gap-1.5">
              <div className="flex-1 h-1.5 rounded-full overflow-hidden bg-white/[0.06]">
                <div className="h-full rounded-full bg-green-500/100" style={{ width: `${customer.callAnswerRate}%` }} />
              </div>
              <span style={{ fontSize: 10, color: "#34D399", fontWeight: 700 }}>{customer.callAnswerRate}%</span>
            </div>
            <div style={{ fontSize: 9, color: "#64748B", marginTop: 3 }}>historical answer rate</div>
          </div>
        </div>

        {/* Row 2: Conversion Forecast */}
        <div className="p-3 rounded-lg bg-white/[0.03] border border-white/[0.08]">
          <div className="flex items-center gap-2 mb-2.5">
            <Target className="w-3.5 h-3.5 text-violet-400" />
            <span style={{ fontSize: 9, fontWeight: 700, color: "#A78BFA", letterSpacing: "0.06em" }}>POSITIVE OUTCOME FORECAST</span>
            <span className="ml-auto text-[9px] font-bold" style={{ color: conversionDiff > 0.1 ? "#F87171" : "#34D399" }}>
              {conversionDiff > 0.1 ? `${Math.round(conversionDiff * 100)}% worse if delayed` : "Timing not urgent"}
            </span>
          </div>
          <div className="grid grid-cols-2 gap-3">
            {[
              { label: "Call Today", value: customer.callConversionToday, color: "#22C55E" },
              { label: "Delay 7 Days", value: customer.callConversionDelay, color: conversionDiff > 0.1 ? "#EF4444" : "#22C55E" },
            ].map((item, i) => (
              <div key={i}>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-[10px] text-slate-500">{item.label}</span>
                  <span style={{ fontSize: 14, fontWeight: 700, color: item.color, fontFamily: "monospace" }}>
                    {Math.round(item.value * 100)}%
                  </span>
                </div>
                <div className="h-2 rounded-full overflow-hidden bg-white/[0.06]">
                  <div className="h-full rounded-full transition-all duration-700"
                    style={{ width: `${item.value * 100}%`, background: item.color }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Row 3: Empathy-First Opener */}
        <div className="p-3 rounded-lg bg-cyan-500/8 border border-cyan-500/20">
          <div className="flex items-center gap-2 mb-2">
            <Phone className="w-3.5 h-3.5 text-cyan-400" />
            <span style={{ fontSize: 9, fontWeight: 700, color: "#22D3EE", letterSpacing: "0.06em" }}>RECOMMENDED OPENER</span>
            <span className="text-[9px] text-slate-400">Do not lead with the missed payment</span>
          </div>
          <p className="text-xs leading-relaxed italic text-slate-200"
            style={{ borderLeft: "2px solid #0077B6", paddingLeft: 10 }}>
            &ldquo;{customer.aiOpener}&rdquo;
          </p>
        </div>

        {/* Row 4: Objection Playbook */}
        {customer.objections.length > 0 && (
          <div className="rounded-lg overflow-hidden border border-white/[0.08]">
            <button onClick={() => setObjOpen(o => !o)}
              className="w-full flex items-center justify-between px-3 py-2.5 text-left transition-all hover:bg-white/[0.03] bg-[#132337]">
              <div className="flex items-center gap-2">
                <ShieldAlert className="w-3.5 h-3.5 text-amber-400" />
                <span style={{ fontSize: 9, fontWeight: 700, color: "#FBBF24", letterSpacing: "0.06em" }}>
                  OBJECTION PLAYBOOK ({customer.objections.length} predicted)
                </span>
              </div>
              {objOpen ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
            </button>
            {objOpen && (
              <div className="border-t border-white/[0.06]">
                {customer.objections.map((obj, i) => (
                  <div key={i} className="px-3 py-3 grid grid-cols-2 gap-3"
                    style={{ borderBottom: i < customer.objections.length - 1 ? "1px solid rgba(255,255,255,0.05)" : "none" }}>
                    <div>
                      <div style={{ fontSize: 9, color: "#F87171", fontWeight: 700, marginBottom: 4 }}>CUSTOMER SAYS</div>
                      <p className="text-xs text-slate-300 italic">&ldquo;{obj.q}&rdquo;</p>
                    </div>
                    <div>
                      <div style={{ fontSize: 9, color: "#34D399", fontWeight: 700, marginBottom: 4 }}>YOUR RESPONSE</div>
                      <p className="text-xs text-slate-300">{obj.a}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Row 5: Compliance Guardrails */}
        {customer.guardrails.length > 0 && (
          <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/25">
            <div className="flex items-center gap-2 mb-2">
              <AlertTriangle className="w-3.5 h-3.5 text-red-400" />
              <span style={{ fontSize: 9, fontWeight: 700, color: "#F87171", letterSpacing: "0.06em" }}>REGULATORY GUARDRAILS — DO NOT MENTION</span>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {customer.guardrails.map((g, i) => (
                <span key={i} className="text-[10px] px-2 py-0.5 rounded font-medium bg-red-100 text-red-400 border border-red-500/25">
                  ✕ {g}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════
// VIEW: RELATIONSHIP MANAGER (Analyst)
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
      <div className="glass-panel p-5 w-80 shrink-0 flex flex-col">
        <div className="flex items-center justify-between mb-4 pb-3 border-b border-white/[0.06]">
          <h3 className="text-sm font-semibold text-slate-100">Priority Queue</h3>
          <span className="text-[10px] font-semibold px-2 py-0.5 rounded bg-red-500/10 text-red-400 border border-red-500/25">{queue.length} pending</span>
        </div>
        <p className="text-[10px] text-slate-400 mb-3">
          {backendOnline ? "Click to score via ML engine" : "Click for simulated data"}
        </p>
        <div className="space-y-2 overflow-y-auto flex-1">
          {queue.map(c => (
            <button key={c.id} onClick={() => scoreCustomerLive(c)}
              className={`w-full text-left p-3 rounded-lg border transition-all ${
                selectedCustomer?.id === c.id ? "border-blue-400 bg-cyan-500/8" : "border-white/[0.06] hover:border-blue-300 hover:bg-white/[0.03]"}`}>
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-sm font-semibold text-slate-100">{c.name}</span>
                <span className={`text-xs font-bold font-mono ${c.riskScore >= 0.7 ? "text-red-400" : "text-amber-400"}`}>{c.riskScore.toFixed(2)}</span>
              </div>
              <div className="flex gap-1.5 flex-wrap">
                <span className="text-[9px] px-1.5 py-0.5 rounded bg-red-500/10 text-red-400 font-semibold border border-red-500/25">TTE: {c.tteDays}d</span>
                <span className="text-[9px] px-1.5 py-0.5 rounded bg-cyan-500/8 text-cyan-400 font-semibold border border-cyan-500/25">Uplift: {c.upliftScore.toFixed(2)}</span>
                <span className="text-[9px] px-1.5 py-0.5 rounded bg-violet-500/10 text-violet-400 font-semibold border border-violet-500/25">{c.segment}</span>
              </div>
              {/* Brief conversion hint */}
              {c.riskTier !== "stable" && (
                <div className="mt-1.5 flex items-center gap-1.5">
                  <span style={{ fontSize: 9, color: c.callConversionToday >= 0.6 ? "#34D399" : "#FBBF24" }}>
                    {Math.round(c.callConversionToday * 100)}% conversion today
                  </span>
                  <span className="text-[9px] text-slate-400">{c.callBestTime.split("(")[0].trim()}</span>
                </div>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Customer 360° + AI Brief */}
      <div className="glass-panel p-5 flex-1 overflow-y-auto">
        <div className="flex items-center justify-between mb-4 pb-3 border-b border-white/[0.06]">
          <h3 className="text-sm font-semibold text-slate-100">Customer 360 &mdash; Intelligence Brief</h3>
          {selectedCustomer && selectedCustomer.riskTier !== "stable" && (
            <span className="text-[9px] text-slate-400 font-medium">
              Auto-generated from SHAP + transaction patterns
            </span>
          )}
        </div>

        {!selectedCustomer ? (
          <div className="flex flex-col items-center justify-center h-96 gap-3 text-slate-400">
            <Users className="w-10 h-10 text-slate-300" />
            <p className="text-sm">Select a customer from the queue to view their intelligence brief</p>
          </div>
        ) : scoringInProgress ? (
          <div className="flex items-center justify-center h-96 text-cyan-400">Scoring via ML Ensemble...</div>
        ) : (
          <div className="space-y-4 animate-fade-in-up">
            {/* Customer Header */}
            <div className="flex items-center gap-4">
              <div className="w-14 h-14 rounded-full bg-slate-700 flex items-center justify-center text-lg font-bold shrink-0 text-white">
                {selectedCustomer.name.charAt(0)}
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-3 flex-wrap">
                  <h4 className="text-lg font-bold text-slate-100">{selectedCustomer.name}</h4>
                  <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                    selectedCustomer.riskTier === "critical" ? "badge-critical" :
                    selectedCustomer.riskTier === "watch" ? "badge-watch" : "badge-stable"
                  }`}>{selectedCustomer.riskTier.toUpperCase()}</span>
                  {selectedCustomer.lifeEvent && (
                    <span style={{ fontSize: 9, color: "#F87171", fontWeight: 700, padding: "2px 6px",
                      background: "rgba(248,113,113,0.08)", border: "1px solid #FECACA", borderRadius: 4 }}>
                      {selectedCustomer.lifeEvent}
                    </span>
                  )}
                </div>
                <p className="text-xs text-slate-500 mt-0.5">{selectedCustomer.occupation} &middot; {selectedCustomer.city} &middot; Age {selectedCustomer.age} &middot; ₹{selectedCustomer.salary.toLocaleString()}/mo</p>
              </div>
              <div className="text-right shrink-0">
                <div className={`text-3xl font-bold font-mono ${selectedCustomer.riskScore >= 0.7 ? "text-red-400" : "text-amber-400"}`}>
                  {(liveScoreResult?.risk_score ?? selectedCustomer.riskScore).toFixed(2)}
                </div>
                <div className="text-[10px] text-slate-400 uppercase tracking-wider">Risk Score</div>
              </div>
            </div>

            {/* ── AI PRE-CALL BRIEF ── */}
            <RMPreCallBrief customer={selectedCustomer} />

            {/* Key Metrics */}
            <div className="grid grid-cols-4 gap-3">
              {[
                { val: (liveScoreResult?.credit_score_mapped ?? selectedCustomer.creditScore).toString(), label: "Credit Score", color: "#22D3EE" },
                { val: `${liveScoreResult?.tte_days ?? selectedCustomer.tteDays}d`, label: "Time-To-Event", color: (liveScoreResult?.tte_days ?? selectedCustomer.tteDays) <= 10 ? "#F87171" : "#FBBF24" },
                { val: (liveScoreResult?.uplift_score ?? selectedCustomer.upliftScore).toFixed(2), label: "Uplift Score", color: "#34D399" },
                { val: `${Math.round(selectedCustomer.callConversionToday * 100)}%`, label: "Call Conversion", color: "#A78BFA" },
              ].map((m, i) => (
                <div key={i} className="p-3 rounded-lg bg-white/[0.03] text-center border border-white/[0.06]">
                  <div className="text-xl font-bold font-mono" style={{ color: m.color }}>{m.val}</div>
                  <div className="text-[10px] text-slate-400 uppercase tracking-wider mt-1">{m.label}</div>
                </div>
              ))}
            </div>

            {/* Live ML Results */}
            {liveScoreResult && (
              <div className="p-4 rounded-lg bg-green-500/10 border border-green-500/25">
                <h5 className="text-[11px] uppercase tracking-wider text-emerald-400 mb-3 font-semibold">LIVE ML ENSEMBLE RESULTS</h5>
                <div className="grid grid-cols-4 gap-2">
                  {[
                    { val: liveScoreResult.risk_score.toFixed(3), label: "Ensemble", color: "#22D3EE" },
                    { val: liveScoreResult.xgboost_score?.toFixed(3) ?? "N/A", label: "XGBoost", color: "#A78BFA" },
                    { val: liveScoreResult.lightgbm_score?.toFixed(3) ?? "N/A", label: "LightGBM", color: "#FBBF24" },
                    { val: liveScoreResult.lstm_score?.toFixed(3) ?? "N/A", label: "LSTM", color: "#F472B6" },
                  ].map((item, i) => (
                    <div key={i} className="p-2 rounded-lg bg-white/[0.03] text-center border border-green-500/20">
                      <div className="text-base font-bold font-mono" style={{ color: item.color }}>{item.val}</div>
                      <div className="text-[9px] text-slate-500 uppercase">{item.label}</div>
                    </div>
                  ))}
                </div>
                {liveScoreResult.top_shap_features && liveScoreResult.top_shap_features.length > 0 && (
                  <div className="mt-3 space-y-1">
                    <div className="text-[10px] text-emerald-400 uppercase tracking-wider mb-1.5">Live SHAP Features</div>
                    {liveScoreResult.top_shap_features.slice(0, 4).map((f, i) => (
                      <div key={i} className="flex items-center gap-3 text-xs">
                        <span className="w-40 text-slate-300 font-mono truncate">{f.feature}</span>
                        <div className="flex-1 h-1.5 bg-white/[0.05] rounded-full overflow-hidden">
                          <div className={`h-full rounded-full ${f.value >= 0 ? "bg-red-500/100" : "bg-cyan-500/80"}`} style={{ width: `${Math.min(Math.abs(f.value) * 500, 100)}%` }} />
                        </div>
                        <span className={`font-bold font-mono w-12 text-right text-[11px] ${f.value >= 0 ? "text-red-400" : "text-cyan-400"}`}>
                          {f.value >= 0 ? "+" : ""}{f.value.toFixed(3)}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* SHAP (simulated) */}
            {!liveScoreResult && (
              <div className="p-4 rounded-lg bg-white/[0.03] border border-white/[0.08]">
                <h5 className="text-[11px] uppercase tracking-wider text-slate-500 mb-3 font-semibold">SHAP Risk Drivers (Simulated)</h5>
                <div className="space-y-2">
                  {selectedCustomer.shapDrivers.map((d, i) => (
                    <div key={i} className="flex items-center gap-3 text-xs">
                      <span className="w-44 text-slate-300 font-mono truncate">{d.feature}</span>
                      <div className="flex-1 h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
                        <div className={`h-full rounded-full ${d.value >= 0 ? "bg-red-500/100" : "bg-cyan-500/80"}`} style={{ width: `${Math.abs(d.value) * 500}%` }} />
                      </div>
                      <span className={`font-bold font-mono w-12 text-right ${d.value >= 0 ? "text-red-400" : "text-cyan-400"}`}>{d.value >= 0 ? "+" : ""}{d.value.toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Counterfactuals */}
            {selectedCustomer.counterfactuals.length > 0 && (
              <div className="p-4 rounded-lg bg-white/[0.03] border border-white/[0.08]">
                <h5 className="text-[11px] uppercase tracking-wider text-slate-500 mb-3 font-semibold">Counterfactual Analysis &mdash; Risk Reduction Paths</h5>
                <div className="space-y-2">
                  {selectedCustomer.counterfactuals.map((cf, i) => (
                    <div key={i} className="flex items-center justify-between py-2 border-b border-white/[0.06] last:border-0 text-xs">
                      <span className="text-slate-300">{cf.action}</span>
                      <span className="text-emerald-400 font-bold font-mono">{cf.newScore}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Full GenAI Script */}
            {selectedCustomer.genaiScript && (
              <div className="p-4 rounded-lg bg-violet-500/10 border border-violet-500/25">
                <h5 className="text-[11px] uppercase tracking-wider text-violet-400 mb-3 font-semibold">Full Call Script (LLM Generated)</h5>
                <p className="text-sm text-slate-300 leading-relaxed italic border-l-[3px] border-violet-400 pl-4">
                  &ldquo;{selectedCustomer.genaiScript}&rdquo;
                </p>
                <div className="flex gap-2 mt-3 flex-wrap">
                  {selectedCustomer.offers.map((o, i) => (
                    <span key={i} className="text-[11px] px-3 py-1 rounded bg-green-500/10 text-emerald-400 border border-green-500/25 font-medium">{o}</span>
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
// VIEW: CUSTOMER SERVICE (Self-Service Portal)
// ═══════════════════════════════════════════════
function CustomerView({ serveResult, triggerRealNotify, notifyResult, backendOnline }: {
  serveResult: string;
  triggerRealNotify: (action: string) => void;
  notifyResult: NotifyResult | null;
  backendOnline: boolean | null;
}) {
  const nudges = [
    { day: 0, ch: "WhatsApp", msg: "Hi Sarah, we noticed some changes in your spending. We're here to help — reply HELP for options.", status: "Delivered, Opened", done: true, active: false },
    { day: 3, ch: "App Push", msg: "Your personalized financial wellness report is ready.", status: "Delivered, Tapped", done: true, active: false },
    { day: 5, ch: "SMS", msg: "Barclays: Good news! You're pre-approved for a 3-month EMI holiday. Tap to accept.", status: "Just now", done: false, active: true },
    { day: 10, ch: "RM Call", msg: "Escalation — Relationship Manager callback scheduled", status: "Pending", done: false, active: false },
  ];

  return (
    <div className="p-5 flex gap-6 justify-center" style={{ height: "calc(100vh - 56px)" }}>
      {/* Mobile Simulator */}
      <div className="w-[380px] shrink-0">
        <div className="bg-[#0a0f1a] rounded-[40px] p-3 border-2 border-white/[0.1] shadow-xl">
          <div className="w-28 h-1.5 bg-white/[0.1] rounded-full mx-auto mb-2" />
          <div className="bg-[#0e1726] rounded-[28px] overflow-y-auto" style={{ maxHeight: "calc(100vh - 170px)" }}>
            <div className="flex justify-between px-5 py-3 text-xs font-semibold border-b border-white/[0.08]"
              style={{ background: "#0a1525", color: "white", borderRadius: "28px 28px 0 0" }}>
              <span>Barclays</span>
              <span className="font-mono">{new Date().toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", hour12: false })}</span>
            </div>
            <div className="p-5">
              <h3 className="text-xl font-bold mb-1 text-white">Hello Sarah</h3>
              <p className="text-xs text-slate-400 mb-5">Your financial wellness dashboard</p>

              {/* Nudge Timeline */}
              <div className="space-y-0 mb-5">
                {nudges.map((n, i) => (
                  <div key={i} className={`flex gap-3 pl-2 pb-4 border-l-2 ml-2 relative ${i === nudges.length - 1 ? "border-l-transparent" : "border-l-slate-200"}`}>
                    <div className={`absolute -left-[6px] top-1 w-3 h-3 rounded-full border-2 ${
                      n.done ? "bg-green-500/100 border-green-500" : n.active ? "bg-cyan-500/80 border-blue-500" : "bg-slate-300 border-slate-300"}`} />
                    <div className="ml-3">
                      <div className="flex gap-2 text-[10px] mb-1">
                        <span className="text-cyan-400 font-bold">Day {n.day}</span>
                        <span className="text-slate-400">{n.ch}</span>
                      </div>
                      <p className="text-xs text-slate-300 leading-relaxed mb-1">{n.msg}</p>
                      <span className="text-[10px] text-slate-400">{n.status}</span>
                    </div>
                  </div>
                ))}
              </div>

              {/* Self-Serve Actions — trigger real /notify */}
              <div>
                <h4 className="text-xs font-semibold text-slate-500 mb-3">
                  Quick Actions {backendOnline ? <span className="text-emerald-400">(Live API)</span> : <span className="text-amber-400">(Simulated)</span>}
                </h4>
                <button onClick={() => triggerRealNotify("EMI Holiday (3 months)")} className="w-full text-left p-3 mb-2 rounded-lg bg-green-500/10 border border-green-500/25 text-emerald-400 text-xs font-semibold hover:bg-green-100 transition-all">
                  Accept EMI Holiday (3 months)
                </button>
                <button onClick={() => triggerRealNotify("EMI Restructuring (12 months)")} className="w-full text-left p-3 mb-2 rounded-lg bg-white/[0.03] border border-white/[0.08] text-slate-200 text-xs font-medium hover:border-blue-300 transition-all">
                  Restructure EMI (extend 12 months)
                </button>
                <button onClick={() => triggerRealNotify("RM Callback")} className="w-full text-left p-3 rounded-lg bg-white/[0.03] border border-white/[0.08] text-slate-200 text-xs font-medium hover:border-blue-300 transition-all">
                  Schedule RM Callback
                </button>
                {serveResult && (
                  <div className="mt-3 p-3 rounded-lg bg-green-500/10 border border-green-500/25 text-xs text-emerald-400 animate-fade-in-up">
                    {serveResult}
                  </div>
                )}
              </div>
            </div>
          </div>
          <div className="w-24 h-1 bg-white/[0.15] rounded-full mx-auto mt-2" />
        </div>
      </div>

      {/* Info Panel */}
      <div className="glass-panel p-5 flex-1 max-w-xl">
        <h3 className="text-sm font-semibold mb-4 pb-3 border-b border-white/[0.06] text-slate-100">Nudge Journey Details</h3>
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
            <div key={i} className="flex justify-between py-2.5 border-b border-white/[0.06] text-xs">
              <span className="text-slate-500">{label}</span>
              <span className={`font-medium ${isCritical ? "text-red-400 font-bold" : "text-slate-200"}`}>{value}</span>
            </div>
          ))}
        </div>

        {/* Real Notify API response */}
        {notifyResult && (
          <div className="mt-4 p-4 rounded-lg bg-green-500/10 border border-green-500/25">
            <h5 className="text-[11px] uppercase tracking-wider text-emerald-400 mb-2 font-semibold">Live API Response</h5>
            <pre className="text-[11px] text-slate-300 font-mono overflow-x-auto whitespace-pre-wrap">
              {JSON.stringify(notifyResult, null, 2)}
            </pre>
          </div>
        )}

        <div className="mt-5 p-4 rounded-lg bg-cyan-500/8 border border-cyan-500/20">
          <p className="text-xs text-slate-300 leading-relaxed">
            <strong className="text-cyan-400">LinUCB Bandit Decision:</strong> WhatsApp selected as Day 0 channel because Sarah&apos;s demographic
            (Female, 28-35, Urban, Tech-savvy) has <strong>68% WhatsApp response rate</strong> vs 32% Email.
          </p>
        </div>
      </div>
    </div>
  );
}
