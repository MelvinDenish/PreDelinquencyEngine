# PDI Engine Frontend Architecture & Live Demonstration Plan

To successfully pitch the PDI Engine to bank executives, you cannot simply show terminal outputs or API JSON responses. They need to **see the business value**, **understand the operational workflow**, and **trust the AI**. 

Here is the blueprint for a robust frontend application and a high-impact demonstration script.

---

## 1. Frontend Architecture Plan

The frontend should be built as a modern, modular web application (e.g., Next.js with Tailwind CSS and Recharts/Chart.js) acting as the single pane of glass for the entire PDI ecosystem.

### A. The "God Mode" Simulator Dashboard
This is the core of the demo. It visualizes the hidden backend orchestration.
*   **Live Event Stream Component:** A scrolling terminal-like UI (but styled beautifully) showing Kafka events being published by the simulator and consumed by Flink.
*   **Pipeline Status Graph:** A live architectural diagram where components (Kafka, Flink, Spark, Redis, ML Models, Dispatcher) light up as data flows through them.
*   **Metric Tickers:** Real-time counters showing *Transactions Ingurgitated*, *Features Computed*, *Risk Scores Updated*, and *Interventions Triggered*.

### B. The Executive Portfolio View (The "Why")
This view proves to decision-makers that the system works at scale.
*   **Risk Migration Matrix:** A Sankey diagram showing customers moving from *Stable* → *Watch* → *Critical* over a 90-day simulated timeline.
*   **Intervention ROI Tracker (Uplift):** A chart comparing **Treated Recovery Rate** vs. **A/B Holdout Recovery Rate**, calculating the total capital saved (AUM protected) in real-time.
*   **Channel Efficiency (Bandit Results):** A breakdown of which channels (WhatsApp vs SMS vs RM Call) are performing best across different segments based on LinUCB bandit learning.
*   **Model Drift & Fairness Monitor:** A compliance dashboard showing that the PSI drift is below 0.20 and all models pass Fairlearn/AIF360 demographic parity checks.

### C. The Relationship Manager (RM) / Collections View (The "How")
This view shows how a human operator interacts with the AI.
*   **Urgency-Ranked Queue:** A list of customers sorted not just by risk score, but by **Time-To-Event (Survival Model)** and **Uplift Score** (prioritizing those who actually need a call).
*   **Customer 360 Drilldown:** Clicking a customer reveals:
    *   **The "Why" (SHAP/LIME):** "Risk score is 82% because: Salary delayed by 5 days (+15%), Missing GST filing (+10%)."
    *   **The "What-If" (Counterfactuals):** "If customer reduces discretionary spend by ₹3,500, risk drops to 45% (Stable)."
    *   **The "What Next" (GenAI Script):** The AI-generated, empathetic call script ready for the RM to read, along with authorized restructuring offers (e.g., "Pre-approved for 12-month tenure extension").

### D. The Customer App View (The "Experience")
A mobile-simulated view showing what the end customer actually sees.
*   **Nudge Journey Inbox:** Demonstrating the timeline (Day 0: WhatsApp nudge → Day 5: App push notification for a Wellness Check).
*   **Self-Serve Action (Product Actions):** Mocking the user clicking an SMS link to instantly accept an EMI Payment Holiday or EMI Restructuring offer within the bank's app.

---

## 2. The Live Demonstration Strategy (The "Pitch")

Do not show the entire system at once. Build the narrative through these 6 phases:

### Phase 1: "The Old Way vs. The New Way" (The Setup)
*   **Narrative:** "Banks currently react to missed payments (delinquency). By then, the customer is already stressed, and recovery costs are high. The PDI Engine predicts the stress *before* the payment is missed."
*   **Action:** Start the `data_generator`. Open the **God Mode Simulator Dashboard**.
*   **Visual:** Show raw transactions flowing in (Kafka) and the real-time scoring happening (Flink + Redis + Scoring Service). Point out that the system is tracking signals like "salary delays" and "GST filing gaps," which traditional models miss.

### Phase 2: "The Anatomy of a Prediction" (The Trust Building)
*   **Narrative:** "Black box AI is a regulatory risk and useless to your Relationship Managers. Our system explains every decision."
*   **Action:** Open the **RM View** and select a "High Risk" customer.
*   **Visual:** Show the SHAP explainability chart. 
    *   *Script:* "Look at Customer A. The score is 0.75. Why? Not because they are inherently risky, but because their DTI ratio spiked this month and their employer has delayed payroll across the company (Employer Health Score). Our Conformal Predictor gives us a 90% confidence bound on this score."

### Phase 3: "Making It Actionable" (The RM Enablement)
*   **Narrative:** "Knowing the risk isn't enough. How do we fix it without sounding like aggressive debt collectors?"
*   **Action:** Stay on the **Customer 360** view in the RM Dashboard.
*   **Visual:** Show the GenAI Call Script and Counterfactuals.
    *   *Script:* "The system generates this empathetic script for the RM. It also calculates that if this customer consolidates their secondary loan, their risk drops back to safe levels. The RM now acts as a financial wellness coach, not a collector."

### Phase 4: "Orchestrating the Nudge" (The Workflow)
*   **Narrative:** "Not everyone needs a costly human RM call. Most customers just need a gentle automated nudge."
*   **Action:** Open the **Customer App View** (mobile simulator) alongside the **Nudge Journey Dashboard**.
*   **Visual:** Trigger an automated WhatsApp message. Show the message popping up on the "mobile" screen. 
    *   *Script:* "Our 21-day Nudge Journey starts softly. A WhatsApp message on Day 0. If they don't respond, the system automatically escalates. Our LinUCB Bandit dynamically learns that *this specific demographic* responds better to WhatsApp than Email, boosting conversion rates without manual rule-setting."

### Phase 5: "Proving the ROI" (The Executive Hook)
*   **Narrative:** "How do we know this isn't just noise? How do we know these nudges actually prevent defaults?"
*   **Action:** Switch to the **Executive Portfolio View**.
*   **Visual:** Show the A/B Holdout Lift Chart and Uplift Model segments.
    *   *Script:* "We silently hold out 10% of risky customers as a control group. Here you can see that the group receiving our interventions has a 14% higher payment rate than the control group. Because we use Uplift Modeling (CATE), we never waste money intervening on customers who were going to 'self-cure' anyway."

### Phase 6: "Built for Enterprise Compliance" (The Closer)
*   **Narrative:** "This isn't a prototype. It's built for Day 1 production at a Tier-1 bank."
*   **Action:** Show the **Model Drift & Fairness Monitor**.
*   **Visual:** Highlight the AIF360/Fairlearn audit verdicts and the Evidently AI drift dashboard.
    *   *Script:* "The system continuously audits itself for demographic bias (gender, geographic region) to ensure fair lending compliance. If customer behaviors change—say, due to macroeconomic shifts—the PSI Drift Detector automatically triggers a retraining pipeline in MLflow before model accuracy degrades."

---

### Technical Requirements for the Frontend Stack (Recommendation)
*   **Framework:** Next.js (React) for fast, modular component development.
*   **State Management/API:** React Query for polling the FastAPI endpoints, or WebSockets (Socket.io) for the live Kafka/Flink visualizer.
*   **Styling:** Tailwind CSS combined with a component library like `shadcn/ui` to quickly build a polished, modern, enterprise-grade dark/light mode interface.
*   **Data Visualization:** Recharts (React) for simple line/bar charts; Apache ECharts for the complex Sankey diagrams (Risk Migration) and SHAP waterfall plots.
*   **Deployment:** Containerized as another Docker service within your existing [docker-compose.yml](file:///c:/Users/JKP/Barc/PreDelinquencyEngine/docker-compose.yml), listening on port `3000`.
