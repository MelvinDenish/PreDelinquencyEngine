/* ═══════════════════════════════════════════════════
   Pre-Delinquency Engine — Demo Frontend Script
   Interactive scoring demo, persona tabs, particles
   ═══════════════════════════════════════════════════ */

// ═══ CUSTOMER DATA FOR LIVE DEMO ═══
const CUSTOMERS = {
    sarah: {
        name: "Sarah Menon", age: 32, credit_score: 720, tenure: 48,
        occupation: "Marketing Executive", products: ["Credit Card", "Personal Loan"],
        features: {
            discretionary_spend_7d: 42000, atm_withdrawals_7d: 8,
            lending_app_txns_7d: 3, salary_delay: 0, failed_autodebits: 0,
            savings_pct_change: -18, credit_score: 720
        }
    },
    rajesh: {
        name: "Rajesh Kumar", age: 45, credit_score: 680, tenure: 96,
        occupation: "IT Manager", products: ["Credit Card", "Mortgage", "Personal Loan"],
        features: {
            discretionary_spend_7d: 28000, atm_withdrawals_7d: 3,
            lending_app_txns_7d: 0, salary_delay: 5, failed_autodebits: 2,
            savings_pct_change: -32, credit_score: 680
        }
    },
    priya: {
        name: "Priya Sharma", age: 28, credit_score: 780, tenure: 24,
        occupation: "Teacher", products: ["Credit Card"],
        features: {
            discretionary_spend_7d: 8000, atm_withdrawals_7d: 1,
            lending_app_txns_7d: 0, salary_delay: 0, failed_autodebits: 0,
            savings_pct_change: 5, credit_score: 780
        }
    },
    amit: {
        name: "Amit Patel", age: 52, credit_score: 620, tenure: 120,
        occupation: "Business Owner", products: ["Credit Card", "Business Loan", "Mortgage"],
        features: {
            discretionary_spend_7d: 65000, atm_withdrawals_7d: 12,
            lending_app_txns_7d: 5, salary_delay: 15, failed_autodebits: 4,
            savings_pct_change: -45, credit_score: 620
        }
    }
};

const LIFE_EVENTS = {
    medical: {
        label: "Medical Emergency",
        multipliers: { atm_withdrawals_7d: 2.5, discretionary_spend_7d: 1.8, lending_app_txns_7d: 2, savings_pct_change: 2 },
        shap_top: "medical_expenses"
    },
    jobloss: {
        label: "Job Loss",
        multipliers: { salary_delay: 30, failed_autodebits: 3, atm_withdrawals_7d: 1.5, savings_pct_change: 3 },
        shap_top: "salary_delay"
    },
    wedding: {
        label: "Wedding Expense",
        multipliers: { discretionary_spend_7d: 3, atm_withdrawals_7d: 2, lending_app_txns_7d: 1.5 },
        shap_top: "discretionary_spend"
    },
    none: {
        label: "No Stress Event",
        multipliers: {},
        shap_top: null
    }
};

const INTERVENTIONS = {
    critical: [
        "Hi {name}, we noticed some unusual activity on your account recently. At Barclays, your financial wellbeing matters to us. Would you like to explore our flexible payment options? We have a special 3-month EMI restructuring plan. Reply HELP or call 1800-XXX-XXXX anytime.",
        "Dear {name}, we understand that unexpected expenses can be stressful. Our team is here to help — no judgement, just solutions. We can offer up to 90-day payment flexibility on your existing obligations. Your dedicated RM, David, will call you today.",
    ],
    watch: [
        "Hi {name}, just checking in! We have some great financial planning tools that might interest you. Would you like a free consultation with our financial advisor? Visit barclays.in/wellness for personalized tips.",
        "Dear {name}, we've prepared some personalized savings recommendations based on your spending patterns. These could help you save up to ₹5,000/month. Check them out at barclays.in/smartsave",
    ],
    stable: [
        "Hi {name}, thank you for being a valued Barclays customer for {tenure} months! As a token of appreciation, you're pre-approved for our rewards program upgrade. Details at barclays.in/rewards"
    ]
};

// ═══ SCORING ENGINE (CLIENT-SIDE SIMULATION) ═══
function calculateScore(features, event) {
    const modifiedFeatures = { ...features };
    const multipliers = LIFE_EVENTS[event].multipliers;

    for (const [key, mult] of Object.entries(multipliers)) {
        if (key === "salary_delay" && typeof mult === "number" && mult > 10) {
            modifiedFeatures[key] = mult;
        } else if (modifiedFeatures[key] !== undefined) {
            modifiedFeatures[key] = Math.round(modifiedFeatures[key] * mult);
        }
    }

    // Simulate model scores
    const normalize = (val, min, max) => Math.min(1, Math.max(0, (val - min) / (max - min)));

    const atmRisk = normalize(modifiedFeatures.atm_withdrawals_7d, 0, 15);
    const spendRisk = normalize(modifiedFeatures.discretionary_spend_7d, 0, 80000);
    const lendingRisk = normalize(modifiedFeatures.lending_app_txns_7d, 0, 6);
    const salaryRisk = normalize(modifiedFeatures.salary_delay, 0, 30);
    const autoDebitRisk = normalize(modifiedFeatures.failed_autodebits, 0, 5);
    const savingsRisk = normalize(Math.abs(modifiedFeatures.savings_pct_change), 0, 50);
    const creditRisk = normalize(850 - modifiedFeatures.credit_score, 0, 350);

    const baseRisk = (atmRisk * 0.18 + spendRisk * 0.12 + lendingRisk * 0.15 +
                      salaryRisk * 0.2 + autoDebitRisk * 0.15 + savingsRisk * 0.1 + creditRisk * 0.1);

    const xgbScore = Math.min(0.99, Math.max(0.05, baseRisk + (Math.random() * 0.06 - 0.03)));
    const lgbScore = Math.min(0.99, Math.max(0.05, baseRisk + (Math.random() * 0.08 - 0.04)));
    const lstmScore = Math.min(0.99, Math.max(0.05, baseRisk * 0.9 + (Math.random() * 0.1 - 0.05)));

    const ensemble = 0.40 * xgbScore + 0.30 * lgbScore + 0.30 * lstmScore;

    // SHAP values
    const shapValues = [
        { name: "atm_withdrawals_7d", value: atmRisk * 0.18, dir: "pos" },
        { name: "salary_delay", value: salaryRisk * 0.16, dir: "pos" },
        { name: "lending_app_txns_7d", value: lendingRisk * 0.14, dir: "pos" },
        { name: "discretionary_spend_7d", value: spendRisk * 0.1, dir: "pos" },
        { name: "failed_autodebits", value: autoDebitRisk * 0.12, dir: "pos" },
        { name: "credit_score", value: creditRisk * -0.08, dir: "neg" },
        { name: "tenure_months", value: -0.03, dir: "neg" },
    ].sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 5);

    return {
        ensemble: Math.min(0.99, Math.max(0.05, ensemble)),
        xgb: xgbScore,
        lgb: lgbScore,
        lstm: lstmScore,
        shap: shapValues,
        modifiedFeatures
    };
}

function getTier(score) {
    if (score >= 0.7) return { label: "CRITICAL", color: "#FF5252", bg: "rgba(255,82,82,0.15)", border: "rgba(255,82,82,0.5)" };
    if (score >= 0.5) return { label: "WATCH", color: "#FFB300", bg: "rgba(255,179,0,0.15)", border: "rgba(255,179,0,0.5)" };
    return { label: "STABLE", color: "#00E676", bg: "rgba(0,230,118,0.15)", border: "rgba(0,230,118,0.5)" };
}

// ═══ LIVE DEMO ═══
function runDemoScoring() {
    const customerId = document.getElementById("demo-customer").value;
    const eventId = document.getElementById("demo-event").value;
    const customer = CUSTOMERS[customerId];
    const event = LIFE_EVENTS[eventId];
    const result = calculateScore(customer.features, eventId);
    const tier = getTier(result.ensemble);

    const btn = document.getElementById("btn-score");
    btn.innerHTML = '<span class="btn-icon">⏳</span> Scoring...';
    btn.disabled = true;

    // Simulate API latency
    setTimeout(() => {
        btn.innerHTML = '<span class="btn-icon">🧠</span> Score Customer';
        btn.disabled = false;

        const tierKey = result.ensemble >= 0.7 ? "critical" : (result.ensemble >= 0.5 ? "watch" : "stable");
        const msgs = INTERVENTIONS[tierKey];
        const msg = msgs[Math.floor(Math.random() * msgs.length)]
            .replace("{name}", customer.name.split(" ")[0])
            .replace("{tenure}", customer.tenure);

        const output = document.getElementById("demo-output");
        output.innerHTML = `
            <div class="result-header">
                <div class="result-score-circle" style="border-color:${tier.color}; background:${tier.bg};">
                    <span style="color:${tier.color}">${result.ensemble.toFixed(2)}</span>
                </div>
                <div class="result-tier" style="color:${tier.color}">● ${tier.label} RISK</div>
                <div style="font-size:0.85rem; color:var(--text-muted); margin-top:4px">${customer.name} • ${customer.occupation}</div>
            </div>
            <div class="result-models">
                <h4>Model Contributions</h4>
                ${renderBar("XGBoost (40%)", result.xgb, "linear-gradient(90deg,#00D4FF,#0088FF)")}
                ${renderBar("LightGBM (30%)", result.lgb, "linear-gradient(90deg,#7B2FFF,#B47CFF)")}
                ${renderBar("LSTM (30%)", result.lstm, "linear-gradient(90deg,#FF3CAC,#FF6B9D)")}
            </div>
            <div class="result-shap">
                <h4>SHAP Explanation — Why This Score?</h4>
                ${result.shap.map(s => `
                    <div class="shap-row">
                        <span style="width:180px;font-family:var(--font-mono);font-size:0.78rem;color:var(--text-muted)">${s.name}</span>
                        <div style="flex:1;display:flex;align-items:center;gap:4px;">
                            <div class="${s.dir === 'pos' ? 'shap-bar-pos' : 'shap-bar-neg'}" style="width:${Math.abs(s.value) / 0.2 * 100}%;min-width:4px;"></div>
                            <span style="font-size:0.78rem;font-weight:600;color:${s.dir === 'pos' ? '#FF5252' : '#00D4FF'}">${s.value >= 0 ? '+' : ''}${s.value.toFixed(3)}</span>
                        </div>
                    </div>
                `).join("")}
            </div>
            <div class="result-intervention">
                <h4>✨ GenAI Intervention Message (${tierKey === 'critical' ? '📱 SMS + RM Call' : tierKey === 'watch' ? '📧 Email' : '🎁 Rewards'})</h4>
                <p>"${msg}"</p>
            </div>
        `;
    }, 800);
}

function renderBar(label, value, gradient) {
    return `<div class="result-bar">
        <div class="result-bar-label"><span>${label}</span><span style="font-weight:700">${value.toFixed(2)}</span></div>
        <div class="result-bar-track"><div class="result-bar-fill" style="width:${value * 100}%;background:${gradient}"></div></div>
    </div>`;
}

// ═══ PERSONA TABS ═══
function showPersona(id) {
    document.querySelectorAll('.persona-content').forEach(el => el.style.display = 'none');
    document.querySelectorAll('.persona-tab').forEach(el => el.classList.remove('active'));
    document.getElementById('persona-' + id).style.display = 'block';
    event.target.classList.add('active');
}

// ═══ NAVBAR SCROLL ═══
const navbar = document.getElementById("navbar");
const sections = document.querySelectorAll("section");
const navLinks = document.querySelectorAll(".nav-links a");

window.addEventListener("scroll", () => {
    // Compact nav on scroll
    if (window.scrollY > 50) {
        navbar.classList.add("scrolled");
    } else {
        navbar.classList.remove("scrolled");
    }

    // Active section highlighting
    let current = "";
    sections.forEach(section => {
        const top = section.offsetTop - 100;
        if (window.scrollY >= top) current = section.getAttribute("id");
    });
    navLinks.forEach(a => {
        a.classList.remove("active");
        if (a.getAttribute("href") === "#" + current) a.classList.add("active");
    });
});

// ═══ SCROLL ANIMATIONS ═══
const observerOptions = { rootMargin: "-100px 0px", threshold: 0.1 };
const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = "1";
            entry.target.style.transform = "translateY(0)";
        }
    });
}, observerOptions);

document.querySelectorAll(".pipe-step, .arch-layer, .model-card, .tech-cat").forEach(el => {
    el.style.opacity = "0";
    el.style.transform = "translateY(30px)";
    el.style.transition = "all 0.6s ease";
    observer.observe(el);
});

// ═══ HERO PARTICLES ═══
function createParticles() {
    const container = document.getElementById("particles");
    if (!container) return;
    for (let i = 0; i < 40; i++) {
        const p = document.createElement("div");
        p.style.cssText = `
            position:absolute;
            width:${Math.random() * 4 + 1}px;
            height:${Math.random() * 4 + 1}px;
            background:rgba(0,212,255,${Math.random() * 0.3 + 0.1});
            border-radius:50%;
            top:${Math.random() * 100}%;
            left:${Math.random() * 100}%;
            animation: float ${Math.random() * 8 + 4}s ease-in-out infinite;
            animation-delay: ${Math.random() * 4}s;
        `;
        container.appendChild(p);
    }
}

// Add float animation
const style = document.createElement("style");
style.textContent = `
    @keyframes float {
        0%, 100% { transform: translate(0, 0) scale(1); opacity: 0.3; }
        25% { transform: translate(${Math.random()*40-20}px, -${Math.random()*30+10}px) scale(1.2); opacity: 0.6; }
        50% { transform: translate(${Math.random()*60-30}px, -${Math.random()*50+20}px) scale(0.8); opacity: 0.4; }
        75% { transform: translate(${Math.random()*40-20}px, -${Math.random()*20+5}px) scale(1.1); opacity: 0.5; }
    }
`;
document.head.appendChild(style);
createParticles();

// ═══ STAT COUNTER ANIMATION ═══
function animateStats() {
    const statAuc = document.getElementById("stat-auc");
    if (!statAuc) return;
    let val = 0;
    const target = 0.93;
    const interval = setInterval(() => {
        val += 0.02;
        if (val >= target) { val = target; clearInterval(interval); }
        statAuc.textContent = val.toFixed(2);
    }, 30);
}

const heroObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) animateStats();
    });
}, { threshold: 0.5 });
heroObserver.observe(document.getElementById("hero"));

// ═══ SMOOTH SCROLL FOR LINKS ═══
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener("click", function(e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute("href"));
        if (target) target.scrollIntoView({ behavior: "smooth" });
    });
});

console.log("🏦 Pre-Delinquency Intervention Engine — Demo Frontend Loaded");
