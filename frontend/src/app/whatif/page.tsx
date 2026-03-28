"use client";

import { useState, useEffect, useCallback } from "react";
import {
  runWhatIfSimulation, getScenarioTemplates, getPortfolioSummary,
  type WhatIfResult, type PortfolioSummary, type ScenarioTemplate,
} from "../api";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, PieChart, Pie, Legend,
} from "recharts";
import {
  AlertTriangle, TrendingDown, TrendingUp, Zap, Play,
  ArrowLeft, Building2, CloudRain, DollarSign, Percent,
  Users, BarChart3, ShieldAlert, ChevronDown, ChevronRight,
  RefreshCw, Info,
} from "lucide-react";
import Link from "next/link";

// ─── Barclays brand colours ───────────────────────────────────────────────────
const B = {
  navy:       "#0B1929",
  blue:       "#00AEEF",
  lightBlue:  "#22D3EE",
  darkNavy:   "#0a1525",
  white:      "#F8FAFC",
  surface:    "#0B1929",
  card:       "#132337",
  cardBorder: "#243B53",
  stable:     "#34D399",
  watch:      "#FBBF24",
  critical:   "#F87171",
  muted:      "#94A3B8",
};

// ─── Scenario types ───────────────────────────────────────────────────────────
type ScenarioType = "sector_shock" | "rate_hike" | "emi_holiday" | "regional_shock" | "salary_shock" | "custom";

interface ScenarioConfig {
  type: ScenarioType;
  name: string;
  // sector shock
  sector?: string;
  sector_employment_change_pct?: number;
  // rate hike
  rate_hike_bps?: number;
  // emi holiday
  emi_holiday_months?: number;
  emi_holiday_target_tiers?: string[];
  // regional
  regions?: string[];
  regional_income_shock_pct?: number;
  // salary
  salary_delay_days_added?: number;
  // custom
  feature_overrides?: Record<string, number>;
}

const SECTORS = ["IT", "Banking", "Pharma", "Manufacturing", "Retail", "Real Estate", "Telecom", "Auto"];
const REGIONS = ["North", "South", "East", "West", "Central", "Northeast"];

function fmt(n: number, decimals = 1) {
  return n.toFixed(decimals);
}
function fmtCrore(n: number) {
  if (n >= 100) return `₹${fmt(n / 100, 1)}K Cr`;
  return `₹${fmt(n, 1)} Cr`;
}

// ─── Animated counter ─────────────────────────────────────────────────────────
function AnimatedNumber({ value, suffix = "", prefix = "" }: { value: number; suffix?: string; prefix?: string }) {
  const [display, setDisplay] = useState(0);
  useEffect(() => {
    const duration = 800;
    const start = Date.now();
    const from = display;
    const raf = () => {
      const elapsed = Date.now() - start;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay(from + (value - from) * eased);
      if (progress < 1) requestAnimationFrame(raf);
    };
    requestAnimationFrame(raf);
  }, [value]);
  return <span>{prefix}{fmt(display, value % 1 === 0 ? 0 : 1)}{suffix}</span>;
}

// ─── Tier badge ───────────────────────────────────────────────────────────────
function TierBadge({ tier }: { tier: string }) {
  const color = tier === "critical" ? B.critical : tier === "watch" ? B.watch : B.stable;
  const bg = tier === "critical" ? "rgba(239,68,68,0.15)" : tier === "watch" ? "rgba(245,158,11,0.15)" : "rgba(34,197,94,0.15)";
  return (
    <span style={{ background: bg, color, border: `1px solid ${color}30`, borderRadius: 4, padding: "2px 8px", fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em" }}>
      {tier}
    </span>
  );
}

// ─── KPI Card ─────────────────────────────────────────────────────────────────
function KpiCard({
  label, value, sub, delta, icon: Icon, color = B.blue, warning = false,
}: {
  label: string; value: React.ReactNode; sub?: string; delta?: number; icon: React.ElementType; color?: string; warning?: boolean;
}) {
  return (
    <div style={{
      background: B.card, border: `1px solid ${warning ? B.critical + "40" : B.cardBorder}`,
      borderRadius: 12, padding: "20px 24px", display: "flex", flexDirection: "column", gap: 8,
      boxShadow: warning ? `0 0 24px ${B.critical}15` : "none",
    }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span style={{ fontSize: 12, color: B.muted, fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.05em" }}>{label}</span>
        <div style={{ width: 32, height: 32, borderRadius: 8, background: `${color}20`, display: "flex", alignItems: "center", justifyContent: "center" }}>
          <Icon size={16} color={color} />
        </div>
      </div>
      <div style={{ fontSize: 28, fontWeight: 800, color: B.white, lineHeight: 1 }}>{value}</div>
      {sub && <div style={{ fontSize: 12, color: B.muted }}>{sub}</div>}
      {delta !== undefined && (
        <div style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 12 }}>
          {delta > 0
            ? <TrendingUp size={12} color={B.critical} />
            : <TrendingDown size={12} color={B.stable} />}
          <span style={{ color: delta > 0 ? B.critical : B.stable, fontWeight: 600 }}>
            {delta > 0 ? "+" : ""}{fmt(delta, 1)}
          </span>
          <span style={{ color: B.muted }}>vs current</span>
        </div>
      )}
    </div>
  );
}

// ─── Distribution bar ─────────────────────────────────────────────────────────
function DistributionBar({
  label, stable, watch, critical, total, highlight = false,
}: {
  label: string; stable: number; watch: number; critical: number; total: number; highlight?: boolean;
}) {
  const stPct = total > 0 ? (stable / total) * 100 : 0;
  const waPct = total > 0 ? (watch / total) * 100 : 0;
  const crPct = total > 0 ? (critical / total) * 100 : 0;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontSize: 13, fontWeight: 600, color: highlight ? B.blue : B.muted }}>{label}</span>
        <div style={{ display: "flex", gap: 12, fontSize: 12 }}>
          <span style={{ color: B.stable }}>{fmt(stPct)}% stable</span>
          <span style={{ color: B.watch }}>{fmt(waPct)}% watch</span>
          <span style={{ color: B.critical }}>{fmt(crPct)}% critical</span>
        </div>
      </div>
      <div style={{ height: 10, borderRadius: 5, overflow: "hidden", display: "flex", background: "#1e293b" }}>
        <div style={{ width: `${stPct}%`, background: B.stable, transition: "width 0.8s ease" }} />
        <div style={{ width: `${waPct}%`, background: B.watch, transition: "width 0.8s ease" }} />
        <div style={{ width: `${crPct}%`, background: B.critical, transition: "width 0.8s ease" }} />
      </div>
    </div>
  );
}

// ─── Slider ───────────────────────────────────────────────────────────────────
function Slider({
  label, value, min, max, step = 1, onChange, suffix = "", color = B.blue,
  description,
}: {
  label: string; value: number; min: number; max: number; step?: number;
  onChange: (v: number) => void; suffix?: string; color?: string; description?: string;
}) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <span style={{ fontSize: 13, fontWeight: 600, color: B.white }}>{label}</span>
          {description && <p style={{ fontSize: 11, color: B.muted, margin: "2px 0 0" }}>{description}</p>}
        </div>
        <span style={{
          fontSize: 15, fontWeight: 800, color, background: `${color}15`,
          border: `1px solid ${color}30`, borderRadius: 6, padding: "2px 10px",
          minWidth: 64, textAlign: "center",
        }}>
          {value > 0 ? "+" : ""}{value}{suffix}
        </span>
      </div>
      <div style={{ position: "relative", height: 6, background: "#1e293b", borderRadius: 3 }}>
        <div style={{
          position: "absolute", left: 0, top: 0, height: "100%", background: color,
          width: `${pct}%`, borderRadius: 3, transition: "width 0.1s",
        }} />
        <input
          type="range" min={min} max={max} step={step} value={value}
          onChange={e => onChange(Number(e.target.value))}
          style={{
            position: "absolute", inset: 0, width: "100%", height: "100%",
            opacity: 0, cursor: "pointer", zIndex: 1,
          }}
        />
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: B.muted }}>
        <span>{min}{suffix}</span>
        <span>{max}{suffix}</span>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN PAGE
// ═══════════════════════════════════════════════════════════════════════════════
export default function WhatIfSimulator() {
  const [portfolio, setPortfolio] = useState<PortfolioSummary | null>(null);
  const [templates, setTemplates] = useState<ScenarioTemplate[]>([]);
  const [result, setResult] = useState<WhatIfResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeScenario, setActiveScenario] = useState<ScenarioType>("sector_shock");
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Scenario params state
  const [sector, setSector] = useState("IT");
  const [employmentChange, setEmploymentChange] = useState(-15);
  const [rateBps, setRateBps] = useState(50);
  const [emiMonths, setEmiMonths] = useState(2);
  const [emiTiers, setEmiTiers] = useState<string[]>(["watch", "critical"]);
  const [selectedRegions, setSelectedRegions] = useState<string[]>(["South"]);
  const [incomeShock, setIncomeShock] = useState(-20);
  const [salaryDelay, setSalaryDelay] = useState(15);
  const [customName, setCustomName] = useState("Custom Scenario");

  // Load portfolio baseline and templates
  useEffect(() => {
    getPortfolioSummary().then(setPortfolio).catch(() => null);
    getScenarioTemplates().then(d => setTemplates(d.templates)).catch(() => null);
  }, []);

  const buildParams = useCallback((): Record<string, unknown> => {
    const base: Record<string, unknown> = { type: activeScenario };
    switch (activeScenario) {
      case "sector_shock":
        return { ...base, name: `${sector} Layoff Shock (${employmentChange}%)`, sector, sector_employment_change_pct: employmentChange };
      case "rate_hike":
        return { ...base, name: `BoE/Fed Rate Hike +${rateBps}bps`, rate_hike_bps: rateBps };
      case "emi_holiday":
        return { ...base, name: `${emiMonths}-Month EMI Holiday (${emiTiers.join("/")})`, emi_holiday_months: emiMonths, emi_holiday_target_tiers: emiTiers };
      case "regional_shock":
        return { ...base, name: `Regional Shock: ${selectedRegions.join(", ")} (${incomeShock}%)`, regions: selectedRegions, regional_income_shock_pct: incomeShock };
      case "salary_shock":
        return { ...base, name: `Payroll Delay +${salaryDelay} days`, salary_delay_days_added: salaryDelay };
      default:
        return { ...base, name: customName };
    }
  }, [activeScenario, sector, employmentChange, rateBps, emiMonths, emiTiers, selectedRegions, incomeShock, salaryDelay, customName]);

  const runSimulation = useCallback(async (params?: Record<string, unknown>) => {
    setLoading(true);
    setError(null);
    try {
      const p = params || buildParams();
      const res = await runWhatIfSimulation(p);
      setResult(res);
    } catch (e: unknown) {
      setError((e as Error).message || "Simulation failed");
    } finally {
      setLoading(false);
    }
  }, [buildParams]);

  const loadTemplate = (template: ScenarioTemplate) => {
    const p = template.params as Record<string, unknown>;
    setActiveScenario(p.type as ScenarioType);
    if (p.sector) setSector(p.sector as string);
    if (p.sector_employment_change_pct) setEmploymentChange(p.sector_employment_change_pct as number);
    if (p.rate_hike_bps) setRateBps(p.rate_hike_bps as number);
    if (p.emi_holiday_months) setEmiMonths(p.emi_holiday_months as number);
    if (p.emi_holiday_target_tiers) setEmiTiers(p.emi_holiday_target_tiers as string[]);
    if (p.regions) setSelectedRegions(p.regions as string[]);
    if (p.regional_income_shock_pct) setIncomeShock(p.regional_income_shock_pct as number);
    if (p.salary_delay_days_added) setSalaryDelay(p.salary_delay_days_added as number);
    runSimulation({ ...p, name: template.name });
  };

  // Tier shift chart data
  const tierShiftData = result ? [
    { name: "Stable", current: result.current_distribution.stable, simulated: result.simulated_distribution.stable },
    { name: "Watch", current: result.current_distribution.watch, simulated: result.simulated_distribution.watch },
    { name: "Critical", current: result.current_distribution.critical, simulated: result.simulated_distribution.critical },
  ] : [];

  const pieCurrent = result ? [
    { name: "Stable", value: result.current_distribution.stable, color: B.stable },
    { name: "Watch", value: result.current_distribution.watch, color: B.watch },
    { name: "Critical", value: result.current_distribution.critical, color: B.critical },
  ] : [];

  const pieSimulated = result ? [
    { name: "Stable", value: result.simulated_distribution.stable, color: B.stable },
    { name: "Watch", value: result.simulated_distribution.watch, color: B.watch },
    { name: "Critical", value: result.simulated_distribution.critical, color: B.critical },
  ] : [];

  const scenarioTypes: { id: ScenarioType; label: string; icon: React.ElementType; color: string }[] = [
    { id: "sector_shock", label: "Sector / Employer Shock", icon: Building2, color: B.critical },
    { id: "rate_hike", label: "BoE/Fed Rate Hike", icon: Percent, color: "#8B5CF6" },
    { id: "emi_holiday", label: "EMI Holiday Relief", icon: ShieldAlert, color: B.stable },
    { id: "regional_shock", label: "Regional Economic Shock", icon: CloudRain, color: B.watch },
    { id: "salary_shock", label: "Payroll Delay Shock", icon: DollarSign, color: "#F97316" },
    { id: "custom", label: "Custom Scenario", icon: Zap, color: B.blue },
  ];

  return (
    <div style={{ background: B.surface, minHeight: "100vh", fontFamily: "'Inter', sans-serif", color: B.white }}>

      {/* ── Top Nav ── */}
      <div style={{
        background: B.darkNavy, borderBottom: `1px solid ${B.cardBorder}`,
        padding: "0 32px", display: "flex", alignItems: "center", gap: 20, height: 60,
        position: "sticky", top: 0, zIndex: 100,
      }}>
        <Link href="/" style={{ display: "flex", alignItems: "center", gap: 8, textDecoration: "none", color: B.muted }}>
          <ArrowLeft size={16} />
          <span style={{ fontSize: 13 }}>Dashboard</span>
        </Link>
        <div style={{ width: 1, height: 20, background: B.cardBorder }} />
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            width: 28, height: 28, borderRadius: 6, background: `${B.blue}20`,
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <BarChart3 size={14} color={B.blue} />
          </div>
          <div>
            <span style={{ fontSize: 14, fontWeight: 700, color: B.white }}>Portfolio What-If Simulator</span>
            <span style={{ fontSize: 11, color: B.muted, marginLeft: 10 }}>Macro stress testing & scenario analysis</span>
          </div>
        </div>
        {portfolio && (
          <div style={{ marginLeft: "auto", display: "flex", gap: 24 }}>
            <div style={{ textAlign: "right" }}>
              <div style={{ fontSize: 11, color: B.muted }}>Portfolio Size</div>
              <div style={{ fontSize: 14, fontWeight: 700 }}>{portfolio.total_customers.toLocaleString()}</div>
            </div>
            <div style={{ textAlign: "right" }}>
              <div style={{ fontSize: 11, color: B.muted }}>At Risk (₹ Cr)</div>
              <div style={{ fontSize: 14, fontWeight: 700, color: B.critical }}>
                {fmtCrore(portfolio.estimated_portfolio_at_risk_crore)}
              </div>
            </div>
          </div>
        )}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "340px 1fr", gap: 0, minHeight: "calc(100vh - 60px)" }}>

        {/* ── Left: Controls ── */}
        <div style={{
          background: B.card, borderRight: `1px solid ${B.cardBorder}`,
          padding: 24, display: "flex", flexDirection: "column", gap: 20, overflowY: "auto",
        }}>

          {/* Scenario type selector */}
          <div>
            <div style={{ fontSize: 11, fontWeight: 700, color: B.muted, textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 12 }}>
              Scenario Type
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {scenarioTypes.map(({ id, label, icon: Icon, color }) => (
                <button
                  key={id}
                  onClick={() => setActiveScenario(id)}
                  style={{
                    display: "flex", alignItems: "center", gap: 10, padding: "10px 14px",
                    borderRadius: 8, border: `1px solid ${activeScenario === id ? color + "60" : B.cardBorder}`,
                    background: activeScenario === id ? `${color}10` : "transparent",
                    color: activeScenario === id ? color : B.muted,
                    cursor: "pointer", fontSize: 13, fontWeight: activeScenario === id ? 600 : 400,
                    transition: "all 0.15s", textAlign: "left", width: "100%",
                  }}
                >
                  <Icon size={14} />
                  {label}
                  {activeScenario === id && <ChevronRight size={12} style={{ marginLeft: "auto" }} />}
                </button>
              ))}
            </div>
          </div>

          {/* Scenario params */}
          <div style={{ borderTop: `1px solid ${B.cardBorder}`, paddingTop: 20 }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: B.muted, textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 16 }}>
              Parameters
            </div>

            {activeScenario === "sector_shock" && (
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <div>
                  <label style={{ fontSize: 12, color: B.muted, display: "block", marginBottom: 6 }}>Sector</label>
                  <select
                    value={sector}
                    onChange={e => setSector(e.target.value)}
                    style={{
                      width: "100%", padding: "8px 12px", borderRadius: 8, fontSize: 13,
                      background: "#0a1628", border: `1px solid ${B.cardBorder}`, color: B.white,
                    }}
                  >
                    {SECTORS.map(s => <option key={s} value={s}>{s}</option>)}
                  </select>
                </div>
                <Slider
                  label="Employment Change"
                  description="Negative value = job losses"
                  value={employmentChange} min={-50} max={0} step={5}
                  onChange={setEmploymentChange} suffix="%" color={B.critical}
                />
              </div>
            )}

            {activeScenario === "rate_hike" && (
              <Slider
                label="Rate Hike"
                description="Repo rate increase in basis points (100bps = 1%)"
                value={rateBps} min={0} max={300} step={25}
                onChange={setRateBps} suffix=" bps" color="#8B5CF6"
              />
            )}

            {activeScenario === "emi_holiday" && (
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <Slider
                  label="Holiday Duration"
                  description="Number of months EMI is deferred"
                  value={emiMonths} min={1} max={6} step={1}
                  onChange={setEmiMonths} suffix=" mo" color={B.stable}
                />
                <div>
                  <label style={{ fontSize: 12, color: B.muted, display: "block", marginBottom: 8 }}>Target Tiers</label>
                  <div style={{ display: "flex", gap: 8 }}>
                    {["watch", "critical"].map(tier => (
                      <button
                        key={tier}
                        onClick={() => setEmiTiers(prev => prev.includes(tier) ? prev.filter(t => t !== tier) : [...prev, tier])}
                        style={{
                          padding: "6px 12px", borderRadius: 6, fontSize: 12, fontWeight: 600,
                          border: `1px solid ${emiTiers.includes(tier) ? (tier === "critical" ? B.critical : B.watch) + "60" : B.cardBorder}`,
                          background: emiTiers.includes(tier) ? (tier === "critical" ? `${B.critical}15` : `${B.watch}15`) : "transparent",
                          color: emiTiers.includes(tier) ? (tier === "critical" ? B.critical : B.watch) : B.muted,
                          cursor: "pointer", textTransform: "capitalize",
                        }}
                      >
                        {tier}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {activeScenario === "regional_shock" && (
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <div>
                  <label style={{ fontSize: 12, color: B.muted, display: "block", marginBottom: 8 }}>Affected Regions</label>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                    {REGIONS.map(region => (
                      <button
                        key={region}
                        onClick={() => setSelectedRegions(prev => prev.includes(region) ? prev.filter(r => r !== region) : [...prev, region])}
                        style={{
                          padding: "4px 10px", borderRadius: 6, fontSize: 12, fontWeight: 500,
                          border: `1px solid ${selectedRegions.includes(region) ? B.watch + "60" : B.cardBorder}`,
                          background: selectedRegions.includes(region) ? `${B.watch}15` : "transparent",
                          color: selectedRegions.includes(region) ? B.watch : B.muted,
                          cursor: "pointer",
                        }}
                      >
                        {region}
                      </button>
                    ))}
                  </div>
                </div>
                <Slider
                  label="Income Shock"
                  value={incomeShock} min={-50} max={0} step={5}
                  onChange={setIncomeShock} suffix="%" color={B.watch}
                />
              </div>
            )}

            {activeScenario === "salary_shock" && (
              <Slider
                label="Additional Salary Delay"
                description="Days added on top of existing delays"
                value={salaryDelay} min={0} max={45} step={5}
                onChange={setSalaryDelay} suffix=" days" color="#F97316"
              />
            )}

            {activeScenario === "custom" && (
              <div>
                <input
                  value={customName}
                  onChange={e => setCustomName(e.target.value)}
                  placeholder="Scenario name..."
                  style={{
                    width: "100%", padding: "8px 12px", borderRadius: 8, fontSize: 13,
                    background: "#0a1628", border: `1px solid ${B.cardBorder}`, color: B.white, marginBottom: 12,
                  }}
                />
                <p style={{ fontSize: 12, color: B.muted }}>Use the API directly with custom feature_overrides for advanced scenarios.</p>
              </div>
            )}
          </div>

          {/* Run button */}
          <button
            onClick={() => runSimulation()}
            disabled={loading}
            style={{
              width: "100%", padding: "14px", borderRadius: 10, fontSize: 14, fontWeight: 700,
              background: loading ? "#1e293b" : `linear-gradient(135deg, ${B.blue}, ${B.navy})`,
              color: loading ? B.muted : B.white, border: "none", cursor: loading ? "not-allowed" : "pointer",
              display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
              transition: "all 0.2s",
            }}
          >
            {loading ? <RefreshCw size={16} style={{ animation: "spin 1s linear infinite" }} /> : <Play size={16} />}
            {loading ? "Simulating..." : "Run Simulation"}
          </button>

          {error && (
            <div style={{
              padding: 12, borderRadius: 8, background: `${B.critical}15`,
              border: `1px solid ${B.critical}30`, fontSize: 12, color: B.critical,
            }}>
              {error}. Make sure customers are scored first.
            </div>
          )}

          {/* Quick templates */}
          {templates.length > 0 && (
            <div style={{ borderTop: `1px solid ${B.cardBorder}`, paddingTop: 20 }}>
              <button
                onClick={() => setShowAdvanced(v => !v)}
                style={{
                  display: "flex", alignItems: "center", gap: 6, fontSize: 11, fontWeight: 700,
                  color: B.muted, background: "none", border: "none", cursor: "pointer", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 12,
                }}
              >
                {showAdvanced ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                Quick Scenarios
              </button>
              {showAdvanced && (
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  {templates.slice(0, 6).map(t => (
                    <button
                      key={t.id}
                      onClick={() => loadTemplate(t)}
                      style={{
                        padding: "8px 12px", borderRadius: 8, fontSize: 12,
                        background: "transparent", border: `1px solid ${B.cardBorder}`,
                        color: B.muted, cursor: "pointer", textAlign: "left",
                        transition: "all 0.15s",
                      }}
                      onMouseEnter={e => { (e.target as HTMLButtonElement).style.borderColor = B.blue + "40"; (e.target as HTMLButtonElement).style.color = B.white; }}
                      onMouseLeave={e => { (e.target as HTMLButtonElement).style.borderColor = B.cardBorder; (e.target as HTMLButtonElement).style.color = B.muted; }}
                    >
                      {t.name}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── Right: Results ── */}
        <div style={{ padding: 32, overflowY: "auto" }}>

          {/* No result state */}
          {!result && !loading && (
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "60%", gap: 20, opacity: 0.6 }}>
              {portfolio && (
                <div style={{ width: "100%", maxWidth: 600, marginBottom: 20 }}>
                  <div style={{ fontSize: 14, fontWeight: 600, color: B.white, marginBottom: 16 }}>Current Portfolio Baseline</div>
                  <DistributionBar
                    label="All Customers"
                    stable={portfolio.distribution.stable.count}
                    watch={portfolio.distribution.watch.count}
                    critical={portfolio.distribution.critical.count}
                    total={portfolio.total_customers}
                    highlight
                  />
                </div>
              )}
              <BarChart3 size={48} color={B.blue} />
              <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: 16, fontWeight: 600, color: B.white }}>Configure a scenario and run the simulation</div>
                <div style={{ fontSize: 13, color: B.muted, marginTop: 6 }}>
                  See how macro events impact your entire loan portfolio in seconds
                </div>
              </div>
            </div>
          )}

          {/* Loading state */}
          {loading && (
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "60%", gap: 16 }}>
              <div style={{ width: 48, height: 48, border: `3px solid ${B.blue}30`, borderTop: `3px solid ${B.blue}`, borderRadius: "50%", animation: "spin 1s linear infinite" }} />
              <div style={{ fontSize: 14, color: B.muted }}>Simulating across {portfolio?.total_customers?.toLocaleString() || "all"} customers...</div>
            </div>
          )}

          {/* Results */}
          {result && !loading && (
            <div style={{ display: "flex", flexDirection: "column", gap: 24, animation: "fadeInUp 0.4s ease" }}>

              {/* Scenario header */}
              <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <div style={{
                  padding: "6px 14px", borderRadius: 8, background: `${B.blue}15`,
                  border: `1px solid ${B.blue}30`, fontSize: 13, fontWeight: 700, color: B.blue,
                }}>
                  {result.scenario_name}
                </div>
                <span style={{ fontSize: 12, color: B.muted }}>{result.total_customers.toLocaleString()} customers simulated</span>
              </div>

              {/* KPI grid */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16 }}>
                <KpiCard
                  label="New Watch Tier"
                  value={<AnimatedNumber value={result.customers_upgraded_to_watch} />}
                  sub="customers moved from stable"
                  delta={result.customers_upgraded_to_watch}
                  icon={AlertTriangle}
                  color={B.watch}
                  warning={result.customers_upgraded_to_watch > 100}
                />
                <KpiCard
                  label="New Critical Tier"
                  value={<AnimatedNumber value={result.customers_upgraded_to_critical} />}
                  sub="customers at high default risk"
                  delta={result.customers_upgraded_to_critical}
                  icon={ShieldAlert}
                  color={B.critical}
                  warning={result.customers_upgraded_to_critical > 50}
                />
                <KpiCard
                  label="Projected NPA Delta"
                  value={<>{fmtCrore(result.estimated_npa_delta_crore)}</>}
                  sub="additional NPAs from this scenario"
                  icon={TrendingDown}
                  color={B.critical}
                  warning={result.estimated_npa_delta_crore > 10}
                />
                <KpiCard
                  label="Intervention ROI"
                  value={result.intervention_roi ? <><AnimatedNumber value={result.intervention_roi} />x</> : "—"}
                  sub={result.intervention_cost_lakh ? `₹${result.intervention_cost_lakh.toFixed(1)}L intervention cost` : ""}
                  icon={TrendingUp}
                  color={B.stable}
                />
              </div>

              {/* Avg risk score delta */}
              <div style={{
                padding: "16px 20px", borderRadius: 10, background: B.card,
                border: `1px solid ${B.cardBorder}`, display: "flex", alignItems: "center", gap: 20,
              }}>
                <Info size={16} color={B.blue} />
                <span style={{ fontSize: 13, color: B.muted }}>Average portfolio risk score:</span>
                <span style={{ fontSize: 15, fontWeight: 700 }}>{fmt(result.avg_risk_score_current * 100, 1)}%</span>
                <span style={{ color: B.muted }}>→</span>
                <span style={{ fontSize: 15, fontWeight: 700, color: result.avg_risk_score_simulated > result.avg_risk_score_current ? B.critical : B.stable }}>
                  {fmt(result.avg_risk_score_simulated * 100, 1)}%
                </span>
                <span style={{
                  fontSize: 12, fontWeight: 600,
                  color: result.avg_risk_score_simulated > result.avg_risk_score_current ? B.critical : B.stable,
                  padding: "2px 8px", borderRadius: 4,
                  background: result.avg_risk_score_simulated > result.avg_risk_score_current ? `${B.critical}15` : `${B.stable}15`,
                }}>
                  {result.avg_risk_score_simulated > result.avg_risk_score_current ? "+" : ""}
                  {fmt((result.avg_risk_score_simulated - result.avg_risk_score_current) * 100, 2)} pp
                </span>
              </div>

              {/* Distribution comparison */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                <div style={{ background: B.card, border: `1px solid ${B.cardBorder}`, borderRadius: 12, padding: "20px 24px" }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: B.muted, marginBottom: 16 }}>Risk Distribution — Before vs After</div>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={tierShiftData} barGap={4}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="name" tick={{ fill: B.muted, fontSize: 12 }} />
                      <YAxis tick={{ fill: B.muted, fontSize: 11 }} />
                      <Tooltip
                        contentStyle={{ background: B.card, border: `1px solid ${B.cardBorder}`, borderRadius: 8, color: B.white }}
                      />
                      <Bar dataKey="current" name="Current" fill={B.navy} radius={[4, 4, 0, 0]} />
                      <Bar dataKey="simulated" name="Simulated" fill={B.blue} radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div style={{ background: B.card, border: `1px solid ${B.cardBorder}`, borderRadius: 12, padding: "20px 24px" }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: B.muted, marginBottom: 12 }}>Portfolio Composition Shift</div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 16, marginTop: 8 }}>
                    <DistributionBar
                      label="Before Scenario"
                      stable={result.current_distribution.stable}
                      watch={result.current_distribution.watch}
                      critical={result.current_distribution.critical}
                      total={result.total_customers}
                    />
                    <DistributionBar
                      label="After Scenario"
                      stable={result.simulated_distribution.stable}
                      watch={result.simulated_distribution.watch}
                      critical={result.simulated_distribution.critical}
                      total={result.total_customers}
                      highlight
                    />
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8, marginTop: 20 }}>
                    {[
                      { label: "Stable", delta: result.simulated_distribution.stable - result.current_distribution.stable, color: B.stable },
                      { label: "Watch", delta: result.simulated_distribution.watch - result.current_distribution.watch, color: B.watch },
                      { label: "Critical", delta: result.simulated_distribution.critical - result.current_distribution.critical, color: B.critical },
                    ].map(({ label, delta, color }) => (
                      <div key={label} style={{ textAlign: "center", padding: "8px", background: `${color}10`, borderRadius: 8, border: `1px solid ${color}20` }}>
                        <div style={{ fontSize: 10, color: B.muted, textTransform: "uppercase" }}>{label}</div>
                        <div style={{ fontSize: 16, fontWeight: 800, color: delta === 0 ? B.muted : (delta > 0 ? (label === "Stable" ? B.stable : color) : B.stable) }}>
                          {delta > 0 ? "+" : ""}{delta}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Segment breakdown */}
              {result.segment_breakdown && Object.keys(result.segment_breakdown).length > 0 && (
                <div style={{ background: B.card, border: `1px solid ${B.cardBorder}`, borderRadius: 12, padding: "20px 24px" }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: B.muted, marginBottom: 16 }}>Impact by Customer Segment</div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))", gap: 10 }}>
                    {Object.entries(result.segment_breakdown)
                      .sort((a, b) => (b[1].new_critical + b[1].new_watch) - (a[1].new_critical + a[1].new_watch))
                      .map(([seg, data]) => {
                        const newAtRisk = data.new_critical + data.new_watch;
                        const impactPct = data.count > 0 ? (newAtRisk / data.count) * 100 : 0;
                        return (
                          <div key={seg} style={{
                            padding: "12px 14px", borderRadius: 8,
                            background: newAtRisk > 0 ? `${B.critical}08` : "transparent",
                            border: `1px solid ${newAtRisk > 0 ? B.critical + "20" : B.cardBorder}`,
                          }}>
                            <div style={{ fontSize: 12, fontWeight: 600, color: B.white, textTransform: "capitalize", marginBottom: 6 }}>
                              {seg.replace(/_/g, " ")}
                            </div>
                            <div style={{ fontSize: 11, color: B.muted }}>{data.count} customers</div>
                            <div style={{ fontSize: 13, fontWeight: 700, color: newAtRisk > 0 ? B.critical : B.stable, marginTop: 4 }}>
                              {newAtRisk > 0 ? `+${newAtRisk} at risk` : "No impact"}
                            </div>
                            {newAtRisk > 0 && (
                              <div style={{ fontSize: 10, color: B.muted }}>({fmt(impactPct)}% of segment)</div>
                            )}
                          </div>
                        );
                      })}
                  </div>
                </div>
              )}

              {/* Top affected employers */}
              {result.top_affected_employers && result.top_affected_employers.length > 0 && (
                <div style={{ background: B.card, border: `1px solid ${B.cardBorder}`, borderRadius: 12, padding: "20px 24px" }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: B.muted, marginBottom: 16 }}>Top Affected Employers</div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    {result.top_affected_employers.slice(0, 8).map((emp, i) => {
                      const pct = emp.total > 0 ? (emp.newly_at_risk / emp.total) * 100 : 0;
                      return (
                        <div key={emp.employer} style={{ display: "flex", alignItems: "center", gap: 12 }}>
                          <span style={{ fontSize: 11, color: B.muted, minWidth: 16, textAlign: "right" }}>#{i + 1}</span>
                          <span style={{ fontSize: 13, color: B.white, minWidth: 140, flexShrink: 0 }}>{emp.employer}</span>
                          <div style={{ flex: 1, height: 6, background: "#1e293b", borderRadius: 3, overflow: "hidden" }}>
                            <div style={{ width: `${pct}%`, height: "100%", background: B.critical, borderRadius: 3 }} />
                          </div>
                          <span style={{ fontSize: 12, color: B.critical, fontWeight: 700, minWidth: 80, textAlign: "right" }}>
                            {emp.newly_at_risk} newly at risk
                          </span>
                          <span style={{ fontSize: 11, color: B.muted }}>/ {emp.total}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Recommended actions */}
              <div style={{
                padding: "20px 24px", borderRadius: 12,
                background: `linear-gradient(135deg, ${B.darkNavy}, ${B.navy}30)`,
                border: `1px solid ${B.blue}30`,
              }}>
                <div style={{ fontSize: 13, fontWeight: 700, color: B.blue, marginBottom: 12 }}>Recommended Actions</div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                  {result.customers_upgraded_to_critical > 0 && (
                    <div style={{ fontSize: 12, color: B.white, display: "flex", gap: 8, alignItems: "flex-start" }}>
                      <span style={{ color: B.critical, marginTop: 1 }}>•</span>
                      <span>Pre-flag {result.customers_upgraded_to_critical} customers for RM escalation calls before scenario materialises</span>
                    </div>
                  )}
                  {result.customers_upgraded_to_watch > 0 && (
                    <div style={{ fontSize: 12, color: B.white, display: "flex", gap: 8, alignItems: "flex-start" }}>
                      <span style={{ color: B.watch, marginTop: 1 }}>•</span>
                      <span>Queue wellness check-ins for {result.customers_upgraded_to_watch} newly watch-tier customers</span>
                    </div>
                  )}
                  {result.intervention_roi && result.intervention_roi > 5 && (
                    <div style={{ fontSize: 12, color: B.white, display: "flex", gap: 8, alignItems: "flex-start" }}>
                      <span style={{ color: B.stable, marginTop: 1 }}>•</span>
                      <span>Intervention ROI of {result.intervention_roi}x justifies proactive portfolio-wide outreach</span>
                    </div>
                  )}
                  {result.estimated_npa_delta_crore > 5 && (
                    <div style={{ fontSize: 12, color: B.white, display: "flex", gap: 8, alignItems: "flex-start" }}>
                      <span style={{ color: B.critical, marginTop: 1 }}>•</span>
                      <span>Consider provisioning {fmtCrore(result.estimated_npa_delta_crore * 0.1)} as contingency buffer (10% provision rate)</span>
                    </div>
                  )}
                </div>
              </div>

            </div>
          )}
        </div>
      </div>

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        @keyframes fadeInUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
        select option { background: #0D1F35; }
      `}</style>
    </div>
  );
}
