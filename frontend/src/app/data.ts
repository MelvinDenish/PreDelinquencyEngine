// Simulated banking data for Indian market context
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
}

export const CUSTOMERS: Customer[] = [
  {
    id: "CUST-4821", name: "Sarah Menon", age: 32, city: "Mumbai", region: "West",
    salary: 85000, occupation: "Marketing Executive", riskScore: 0.82, riskTier: "critical",
    creditScore: 621, tteDays: 12, upliftScore: 0.34, segment: "Young Professional",
    shapDrivers: [
      { feature: "atm_withdrawals_7d", value: 0.18 },
      { feature: "lending_app_txns_7d", value: 0.14 },
      { feature: "discretionary_spend_7d", value: 0.09 },
      { feature: "salary_delay_days", value: 0.07 },
      { feature: "dti_ratio", value: -0.04 },
    ],
    counterfactuals: [
      { action: "Reduce discretionary spend by ₹3,500/week", newScore: "0.45 (Stable)" },
      { action: "Consolidate payday loans into EMI", newScore: "0.38 (Stable)" },
      { action: "Setup auto-debit for EMI payments", newScore: "0.52 (Watch)" },
    ],
    genaiScript: "Hi Sarah, this is David from Barclays. I noticed some changes in your account recently — particularly some elevated ATM activity and a couple of lending app transactions. I want you to know we're here to help, not judge. We have a flexible 3-month EMI holiday program that could ease the pressure. Would you like me to walk you through it?",
    offers: ["3-month EMI Holiday", "12-month Tenure Extension", "Debt Consolidation Loan @ 10.5%"],
  },
  {
    id: "CUST-7392", name: "Rajesh Kumar", age: 45, city: "Delhi", region: "North",
    salary: 120000, occupation: "IT Manager", riskScore: 0.67, riskTier: "watch",
    creditScore: 694, tteDays: 28, upliftScore: 0.22, segment: "Mid-Career Professional",
    shapDrivers: [
      { feature: "utility_bill_misses_30d", value: 0.12 },
      { feature: "credit_utilization_ratio", value: 0.10 },
      { feature: "balance_velocity_7d", value: 0.08 },
      { feature: "employer_health_score", value: -0.05 },
      { feature: "tenure_months", value: -0.06 },
    ],
    counterfactuals: [
      { action: "Clear overdue utility bills (₹12,400)", newScore: "0.41 (Stable)" },
      { action: "Reduce credit card utilization to < 40%", newScore: "0.48 (Stable)" },
    ],
    genaiScript: "Good morning Mr. Kumar, this is your financial wellness advisor from Barclays. We noticed a couple of missed utility payments and higher-than-usual credit card usage this month. These are very common during festival season. Would you like me to set up an automatic bill payment schedule? We also have a 0% balance transfer offer that could help.",
    offers: ["0% Balance Transfer (6 months)", "Bill Pay Auto-Debit Setup"],
  },
  {
    id: "CUST-1156", name: "Priya Sharma", age: 28, city: "Bangalore", region: "South",
    salary: 55000, occupation: "School Teacher", riskScore: 0.91, riskTier: "critical",
    creditScore: 583, tteDays: 7, upliftScore: 0.41, segment: "Early Career",
    shapDrivers: [
      { feature: "medical_txns_30d", value: 0.22 },
      { feature: "lending_app_txns_7d", value: 0.16 },
      { feature: "cash_advance_count_30d", value: 0.13 },
      { feature: "salary_delay_days", value: 0.11 },
      { feature: "min_balance_breaches", value: 0.08 },
    ],
    counterfactuals: [
      { action: "Avail medical emergency loan @ 8.5%", newScore: "0.35 (Stable)" },
      { action: "3-month EMI moratorium", newScore: "0.42 (Stable)" },
    ],
    genaiScript: "Hi Priya, this is your Barclays support team. We can see there have been some significant medical expenses recently, and we understand how stressful that can be. We want you to know — you're pre-approved for our Medical Emergency Support program with reduced rates. Can I help you get started right now?",
    offers: ["Medical Emergency Loan @ 8.5%", "3-month EMI Moratorium", "Insurance Claim Support"],
  },
  {
    id: "CUST-5590", name: "Amit Patel", age: 52, city: "Ahmedabad", region: "West",
    salary: 200000, occupation: "Business Owner", riskScore: 0.55, riskTier: "watch",
    creditScore: 710, tteDays: 35, upliftScore: 0.15, segment: "HNI",
    shapDrivers: [
      { feature: "gst_filing_delay", value: 0.11 },
      { feature: "business_txn_volatility", value: 0.08 },
      { feature: "credit_utilization_ratio", value: 0.06 },
      { feature: "fixed_deposit_break", value: -0.03 },
      { feature: "tenure_months", value: -0.08 },
    ],
    counterfactuals: [
      { action: "File pending GST returns", newScore: "0.38 (Stable)" },
      { action: "Reduce business credit line usage to < 50%", newScore: "0.42 (Stable)" },
    ],
    genaiScript: "Good afternoon Mr. Patel, this is your dedicated relationship manager from Barclays. I noticed a slight delay in your recent GST filings and some increased volatility in your business account. Given your long relationship with us, I wanted to proactively offer our Business Flex credit line that adjusts to seasonal cash flow patterns.",
    offers: ["Business Flex Credit Line", "GST Payment Advisory", "Working Capital Optimization"],
  },
  {
    id: "CUST-8834", name: "Deepika Reddy", age: 35, city: "Hyderabad", region: "South",
    salary: 95000, occupation: "Software Engineer", riskScore: 0.23, riskTier: "stable",
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
  },
  {
    id: "CUST-3247", name: "Vikram Singh", age: 40, city: "Jaipur", region: "North",
    salary: 42000, occupation: "Taxi Driver", riskScore: 0.88, riskTier: "critical",
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
      { action: "Enroll in income smoothing program", newScore: "0.38 (Stable)" },
    ],
    genaiScript: "Namaste Vikram ji, Barclays se bol raha hoon. Hum dekh rahe hain ki pichle kuch dino mein aapke account mein kuch payday loan transactions aaye hain. Hum aapki madad karna chahte hain — hamare paas ek income smoothing program hai jo gig workers ke liye specially designed hai. Kya aap iske baare mein jaanna chahenge?",
    offers: ["Income Smoothing Program", "Micro-Loan Consolidation", "Financial Literacy Workshop"],
  },
];

export const TRANSACTION_TYPES = [
  "upi", "atm_withdrawal", "pos_swipe", "neft", "imps", "bill_payment", "cash_advance", "lending_app", "online_shopping"
];

export const MERCHANT_CATEGORIES = [
  "grocery", "dining", "fuel", "medical", "education", "entertainment",
  "lending_app", "gambling", "lottery", "payday_lender", "utility", "insurance"
];

export const STRESS_CATEGORIES = ["lending_app", "gambling", "lottery", "payday_lender", "cash_advance"];

export function generateTransaction() {
  const customer = CUSTOMERS[Math.floor(Math.random() * CUSTOMERS.length)];
  const category = MERCHANT_CATEGORIES[Math.floor(Math.random() * MERCHANT_CATEGORIES.length)];
  const amount = Math.floor(Math.random() * 25000) + 200;
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
