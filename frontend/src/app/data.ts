// Simulated banking data for Barclays UK/US market context
export interface Customer {
  id: string;
  name: string;
  age: number;
  city: string;
  region: string;
  salary: number;
  occupation: string;
  riskScore: number;
  riskTier: "stable" | "watch" | "critical";
  creditScore: number;
  tteDays: number;
  upliftScore: number;
  segment: string;
  shapDrivers: { feature: string; value: number }[];
  counterfactuals: { action: string; newScore: string }[];
  genaiScript: string;
  offers: string[];
  // ── RM Pre-Call AI Brief ────────────────────────────────────────────────────
  callBestTime: string;
  callAnswerRate: number;
  stressTrigger: string;
  stressCategory: "medical" | "business" | "lifestyle" | "income_volatility";
  callConversionToday: number;   // 0–1 probability
  callConversionDelay: number;   // probability if delayed 7 days
  aiOpener: string;
  objections: { q: string; a: string }[];
  guardrails: string[];
  lifeEvent?: string;            // e.g. "Medical Emergency Detected"
}

export const CUSTOMERS: Customer[] = [
  {
    id: "CUST-4821", name: "Sarah Mitchell", age: 32, city: "London", region: "South East",
    salary: 4200, occupation: "Marketing Executive", riskScore: 0.82, riskTier: "critical",
    creditScore: 621, tteDays: 12, upliftScore: 0.34, segment: "Young Professional",
    shapDrivers: [
      { feature: "atm_withdrawals_7d", value: 0.18 },
      { feature: "lending_app_txns_7d", value: 0.14 },
      { feature: "discretionary_spend_7d", value: 0.09 },
      { feature: "salary_delay_days", value: 0.07 },
      { feature: "dti_ratio", value: -0.04 },
    ],
    counterfactuals: [
      { action: "Reduce discretionary spend by £250/week", newScore: "0.45 (Stable)" },
      { action: "Consolidate payday loans into instalment plan", newScore: "0.38 (Stable)" },
      { action: "Setup Direct Debit for loan payments", newScore: "0.52 (Watch)" },
    ],
    genaiScript: "Hi Sarah, this is David from Barclays. I noticed some changes in your account recently — particularly some elevated ATM activity and a couple of short-term lending transactions. I want you to know we're here to help, not judge. We have a flexible 3-month payment holiday programme that could ease the pressure. Would you like me to walk you through it?",
    offers: ["3-month Payment Holiday", "12-month Tenure Extension", "Debt Consolidation Loan @ 5.9%"],
    callBestTime: "7–9 PM GMT (post-work)",
    callAnswerRate: 73,
    stressTrigger: "Payday loan spiral + ATM cash dependency spike over 7 days",
    stressCategory: "lifestyle",
    callConversionToday: 0.68,
    callConversionDelay: 0.41,
    aiOpener: "Hi Sarah, I noticed some financial pressure on your account this week. I'm calling because we have a 3-month breathing room option that costs nothing to activate — I wanted to reach out personally before your next payment date.",
    objections: [
      { q: "I'm managing, I'll sort it myself", a: "Of course — just 90 seconds to hear about a pre-approved option that won't affect your credit score?" },
      { q: "I'm switching to another bank", a: "That's completely your call. The payment holiday is already approved on your account regardless — let me explain it before you decide." },
    ],
    guardrails: ["missed payment", "default", "credit score drop", "collections team"],
  },
  {
    id: "CUST-7392", name: "James Robertson", age: 45, city: "Edinburgh", region: "Scotland",
    salary: 5800, occupation: "IT Manager", riskScore: 0.67, riskTier: "watch",
    creditScore: 694, tteDays: 28, upliftScore: 0.22, segment: "Mid-Career Professional",
    shapDrivers: [
      { feature: "utility_bill_misses_30d", value: 0.12 },
      { feature: "credit_utilization_ratio", value: 0.10 },
      { feature: "balance_velocity_7d", value: 0.08 },
      { feature: "employer_health_score", value: -0.05 },
      { feature: "tenure_months", value: -0.06 },
    ],
    counterfactuals: [
      { action: "Clear overdue utility bills (£840)", newScore: "0.41 (Stable)" },
      { action: "Reduce credit card utilisation to < 40%", newScore: "0.48 (Stable)" },
    ],
    genaiScript: "Good morning Mr. Robertson, this is your financial wellness advisor from Barclays. We noticed a couple of missed utility payments and higher-than-usual credit card usage this month. These are very common during the holiday season. Would you like me to set up an automatic bill payment schedule? We also have a 0% balance transfer offer that could help.",
    offers: ["0% Balance Transfer (6 months)", "Bill Pay Direct Debit Setup"],
    callBestTime: "9–11 AM GMT (pre-commute)",
    callAnswerRate: 61,
    stressTrigger: "Utility bill misses + elevated credit utilisation — likely holiday season overspend",
    stressCategory: "lifestyle",
    callConversionToday: 0.55,
    callConversionDelay: 0.48,
    aiOpener: "Good morning Mr. Robertson, this is your Barclays advisor. We noticed elevated credit usage this month — quite common around the holiday season — and I have a 0% balance restructuring option that costs you nothing to set up.",
    objections: [
      { q: "I'll clear it all next month", a: "That works too — but a 0% transfer now means no interest accrues meanwhile. Takes 2 minutes." },
      { q: "Too busy right now", a: "Understood, sir. Can I schedule a 10-minute call for tomorrow morning instead?" },
    ],
    guardrails: ["overdue bills", "poor financial management", "credit score drop"],
  },
  {
    id: "CUST-1156", name: "Emily Thompson", age: 28, city: "Manchester", region: "North West",
    salary: 2800, occupation: "School Teacher", riskScore: 0.91, riskTier: "critical",
    creditScore: 583, tteDays: 7, upliftScore: 0.41, segment: "Early Career",
    shapDrivers: [
      { feature: "medical_txns_30d", value: 0.22 },
      { feature: "lending_app_txns_7d", value: 0.16 },
      { feature: "cash_advance_count_30d", value: 0.13 },
      { feature: "salary_delay_days", value: 0.11 },
      { feature: "min_balance_breaches", value: 0.08 },
    ],
    counterfactuals: [
      { action: "Avail medical emergency loan @ 4.5%", newScore: "0.35 (Stable)" },
      { action: "3-month payment moratorium", newScore: "0.42 (Stable)" },
    ],
    genaiScript: "Hi Emily, this is your Barclays support team. We can see there have been some significant medical expenses recently, and we understand how stressful that can be. We want you to know — you're pre-approved for our Medical Emergency Support programme with reduced rates. Can I help you get started right now?",
    offers: ["Medical Emergency Loan @ 4.5%", "3-month Payment Moratorium", "Insurance Claim Support"],
    callBestTime: "6–8 PM GMT (after school hours)",
    callAnswerRate: 79,
    stressTrigger: "Acute medical emergency spend (£1,800 in 30d) triggering payday loan spiral",
    stressCategory: "medical",
    lifeEvent: "Medical Emergency Detected",
    callConversionToday: 0.74,
    callConversionDelay: 0.29,
    aiOpener: "Hi Emily, I can see you've been through a very difficult time recently with medical expenses — I'm calling specifically to help before your next payment date, not to add pressure.",
    objections: [
      { q: "I can't afford any payments right now", a: "That's exactly why I'm calling — the moratorium means zero payments for 3 months. Nothing to pay at all during that period." },
      { q: "I'll figure it out myself", a: "Of course, Emily. But the medical loan at 4.5% is significantly cheaper than any short-term lender. It's available right now for you." },
    ],
    guardrails: ["missed payment", "lending app", "collections", "default risk", "legal action"],
  },
  {
    id: "CUST-5590", name: "Richard Clarke", age: 52, city: "Bristol", region: "South West",
    salary: 9500, occupation: "Business Owner", riskScore: 0.55, riskTier: "watch",
    creditScore: 710, tteDays: 35, upliftScore: 0.15, segment: "HNI",
    shapDrivers: [
      { feature: "vat_filing_delay", value: 0.11 },
      { feature: "business_txn_volatility", value: 0.08 },
      { feature: "credit_utilization_ratio", value: 0.06 },
      { feature: "fixed_deposit_break", value: -0.03 },
      { feature: "tenure_months", value: -0.08 },
    ],
    counterfactuals: [
      { action: "File pending VAT returns", newScore: "0.38 (Stable)" },
      { action: "Reduce business credit line usage to < 50%", newScore: "0.42 (Stable)" },
    ],
    genaiScript: "Good afternoon Mr. Clarke, this is your dedicated relationship manager from Barclays. I noticed a slight delay in your recent VAT filings and some increased volatility in your business account. Given your long relationship with us, I wanted to proactively offer our Business Flex credit line that adjusts to seasonal cash flow patterns.",
    offers: ["Business Flex Credit Line", "VAT Payment Advisory", "Working Capital Optimisation"],
    callBestTime: "2–4 PM GMT (post-lunch, business hours)",
    callAnswerRate: 58,
    stressTrigger: "VAT filing delays + business cash flow volatility (seasonal pattern detected)",
    stressCategory: "business",
    callConversionToday: 0.44,
    callConversionDelay: 0.52,
    aiOpener: "Good afternoon Mr. Clarke, this is your dedicated RM. I'm calling proactively — we've noticed some cash flow patterns in your business account and I wanted to discuss our Business Flex facility before you need it.",
    objections: [
      { q: "My business is perfectly fine", a: "Absolutely, sir — this is entirely proactive. A Flex facility is cheaper to arrange now than during a cash crunch." },
      { q: "I'll call you back when convenient", a: "Of course, Mr. Clarke. I'll send a brief via your registered email — is the Bristol address still the best one?" },
    ],
    guardrails: ["tax compliance issue", "business failure", "revenue concerns", "HMRC penalty"],
  },
  {
    id: "CUST-8834", name: "Olivia Chen", age: 35, city: "Cambridge", region: "East",
    salary: 5200, occupation: "Software Engineer", riskScore: 0.23, riskTier: "stable",
    creditScore: 785, tteDays: 90, upliftScore: 0.05, segment: "Tech Professional",
    shapDrivers: [
      { feature: "salary_regularity", value: -0.12 },
      { feature: "savings_ratio", value: -0.09 },
      { feature: "credit_utilization_ratio", value: -0.06 },
      { feature: "discretionary_spend_7d", value: 0.03 },
      { feature: "online_shopping_7d", value: 0.02 },
    ],
    counterfactuals: [],
    genaiScript: "",
    offers: [],
    callBestTime: "N/A (stable)",
    callAnswerRate: 0,
    stressTrigger: "No stress triggers detected — monitoring only",
    stressCategory: "lifestyle",
    callConversionToday: 0,
    callConversionDelay: 0,
    aiOpener: "",
    objections: [],
    guardrails: [],
  },
  {
    id: "CUST-3247", name: "Marcus Johnson", age: 40, city: "Birmingham", region: "West Midlands",
    salary: 2100, occupation: "Deliveroo Driver", riskScore: 0.88, riskTier: "critical",
    creditScore: 545, tteDays: 5, upliftScore: 0.38, segment: "Gig Worker",
    shapDrivers: [
      { feature: "income_volatility_30d", value: 0.20 },
      { feature: "cash_advance_count_30d", value: 0.15 },
      { feature: "lending_app_txns_7d", value: 0.13 },
      { feature: "gambling_txns_7d", value: 0.10 },
      { feature: "min_balance_breaches", value: 0.08 },
    ],
    counterfactuals: [
      { action: "Stop gambling transactions", newScore: "0.52 (Watch)" },
      { action: "Consolidate payday loans", newScore: "0.41 (Stable)" },
      { action: "Enrol in income smoothing programme", newScore: "0.38 (Stable)" },
    ],
    genaiScript: "Hi Marcus, this is your Barclays support team. We've noticed a few short-term lending transactions on your account recently. We understand that gig work income can be unpredictable, and we have an income smoothing programme specifically designed for flexible workers like yourself. Would you like to hear more about it?",
    offers: ["Income Smoothing Programme", "Micro-Loan Consolidation", "Financial Literacy Workshop"],
    callBestTime: "12–2 PM GMT (midday break)",
    callAnswerRate: 67,
    stressTrigger: "Erratic gig income + payday loan cycle + gambling transactions creating debt spiral",
    stressCategory: "income_volatility",
    callConversionToday: 0.52,
    callConversionDelay: 0.21,
    aiOpener: "Hi Marcus, I'm calling from Barclays — just to help, nothing else. We have an income smoothing programme for gig workers that I'd like to explain to you.",
    objections: [
      { q: "I don't have time right now", a: "Just 2 minutes, Marcus. This programme could save you up to £200/month in payday loan fees." },
      { q: "Banks always cause problems", a: "I understand. I'm not calling about penalties — just a support option that's already available on your account." },
    ],
    guardrails: ["gambling transactions", "loan default", "account closure", "legal action"],
  },
];

export const TRANSACTION_TYPES = [
  "card_payment", "atm_withdrawal", "pos_swipe", "bacs", "faster_payment", "bill_payment", "cash_advance", "lending_app", "online_shopping"
];

export const MERCHANT_CATEGORIES = [
  "grocery", "dining", "fuel", "medical", "education", "entertainment",
  "lending_app", "gambling", "lottery", "payday_lender", "utility", "insurance"
];

export const STRESS_CATEGORIES = ["lending_app", "gambling", "lottery", "payday_lender", "cash_advance"];

export function generateTransaction() {
  const customer = CUSTOMERS[Math.floor(Math.random() * CUSTOMERS.length)];
  const category = MERCHANT_CATEGORIES[Math.floor(Math.random() * MERCHANT_CATEGORIES.length)];
  const amount = Math.floor(Math.random() * 1500) + 10;
  const isStress = STRESS_CATEGORIES.includes(category);
  return {
    customerId: customer.id,
    customerName: customer.name,
    city: customer.city,
    txnType: TRANSACTION_TYPES[Math.floor(Math.random() * TRANSACTION_TYPES.length)],
    merchantCategory: category,
    amount,
    isStress,
    status: Math.random() > 0.08 ? "success" : "failed",
    timestamp: new Date().toISOString(),
  };
}
