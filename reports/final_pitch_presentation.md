# 🎤 StockSense: Final Pitch & Executive Presentation
### IntelliData 2026 — Data Science Hackathon (Day 2)
**Institution:** Sri Eshwar College of Engineering, Coimbatore  
**Department:** Computer Science and Engineering  
**Industry Partner / Use Case:** NovaMart Retail Pvt. Ltd.  
**Team Name:** StockSense AI Consulting Unit  
**Challenge Line:** *“Every empty shelf is lost revenue. Can Data Science predict it before it happens?”*

---

## 👥 1. Team Ownership & Execution Matrix
| Role | Student Member | Primary Responsibilities |
|------|----------------|--------------------------|
| **Data Analyst** | *Student 1* | Data Auditing & Cleansing (6 Traps), Master Table Aggregation, 3 Hypothesis Tests, Pareto & EDA Insights |
| **ML Engineer** | *Student 2* | Leakage-free Feature Engineering (48 features), Time-Aware Validation, Model 1 & 2 Benchmark & Tuning |
| **Decision Intelligence** | *Student 3* | Safety Stock Logic, Reorder Optimization, Risk Tiering, Explainability Drivers, Streamlit Prototype |

---

## 🎯 2. The Core Business Question & Discovery

### The Challenge Faced by NovaMart:
NovaMart operates supermarkets across 6 Tamil Nadu cities (Chennai, Coimbatore, Madurai, Salem, Trichy, Erode). Prior to StockSense, replenishment was governed by manual store manager intuition and static rules. This caused:
- **Frequent Stock-Outs:** 14.8% of product-days suffered stockouts, causing **₹12.4 Lakhs in unfulfilled lost revenue**.
- **Overstocking of Slow-Movers:** Excess working capital locked in long shelf-life staples while fast-moving dairy spoiled.
- **Uncoordinated Promotions:** Marketing discounts increased demand by **+41.2%**, but store inventory stocked out within 48 hours because suppliers had a 2-4 day lead time.

### Key Discoveries from Round 1 Statistical Testing:
1. **Promotions Drive Demand ($p < 0.001$, Welch's t-test):** Significant sales uplift of +41.2%, proving promo flags must be a primary forecasting feature.
2. **Demand Varies by Store Type ($p < 0.001$, One-Way ANOVA):** Hypermarkets (Chennai) move 2.4× the volume of Express outlets (Madurai, Trichy), invalidating static chain-wide reorder minimums.
3. **Promotions Directly Cause Stock-Outs ($p < 0.001, \chi^2 = 84.1$):** Stockout rates double during promotional campaigns, requiring proactive pre-replenishment 48 hours before campaigns go live.

---

## 🔮 3. What We Can Predict & Model Performance (Round 2)

### Model 1 — 7-Day Forward Demand Forecast (Regression)
- **Target:** Total units sold over the upcoming 7-day replenishment cycle for every Store × Product pair.
- **Validation:** Strict time-aware holdout (last 14 days of data). Zero data leakage.
- **Champion Model:** **XGBoost Regressor**
  - **MAE:** `32.69 units` (beats Baseline 7-Day Mean by **6.1%**)
  - **RMSE:** `54.61 units`
  - **$R^2$ Score:** `0.9632`
- **Why MAE Matters:** Store managers operate with discrete units; an MAE of ~32 units across a weekly store turnover represents high planning accuracy.

### Model 2 — 7-Day Stock-Out Risk Classifier
- **Target:** Probability of stock depletion ($closing\_stock = 0$) within the 7-day horizon.
- **Champion Model:** **XGBoost Classifier (Class-Weighted)**
  - **ROC-AUC:** `0.739`
  - **Recall on Stock-outs:** `82.5%`
  - **Accuracy:** `67.1%`
- **Why Recall Matters:** In retail, **failing to catch a stock-out (False Negative) loses revenue and damages customer loyalty**. High recall ensures over 82% of at-risk products are caught and flagged for intervention.

---

## 🛡️ 4. Why the Results Can Be Trusted (Explainability & Validation)
Management will never trust a "black box" alert. StockSense provides transparent, human-readable decision drivers:
- **No Data Leakage:** All lag and rolling metrics are strictly computed using historical data ($t-1$ or earlier).
- **Feature Importance Hierarchy:**
  1. `days_of_inventory` (28.4%) — Current stock relative to recent burn rate.
  2. `rolling_mean_7` (21.2%) — Recent baseline sales velocity.
  3. `demand_lag_1` (14.5%) — Immediate yesterday momentum.
  4. `reorder_gap` (9.8%) — Proximity to minimum threshold.
  5. `promotion_flag` (8.2%) — Uplift demand elasticity.

### Example Store Manager Recommendation Card:
```
STORE S01 (COIMBATORE) — AAVIN MILK 1L (P101)
---------------------------------------------------------------------------------
Predicted 7-Day Demand : 485 units
Current Stock          : 250 units (Runway: 2.1 days)
Incoming Stock         : 80 units
Safety Stock Buffer    : 45 units (95% Cycle Service Level)
Recommended Reorder    : 180 units
Stock-out Probability  : 89% | Risk Tier: HIGH
---------------------------------------------------------------------------------
WHY?
• Active promotional discount driving demand surge
• Weekend footfall spike approaching
• Current runway is critically below supplier lead time (2 days)
---------------------------------------------------------------------------------
MANAGER ACTION:
🚨 URGENT: Raise purchase order of 180 units today with Supplier SUP01.
```

---

## 💼 5. How StockSense Reduces Business Loss & ROI
| Metric | Pre-StockSense Baseline | With StockSense AI | Projected Business Impact |
|--------|-------------------------|--------------------|---------------------------|
| **Stock-Out Incidents** | 14.8% of SKU-Days | < 4.5% of SKU-Days | **-69% Reduction in Out-of-Stock Events** |
| **Lost Sales Revenue** | ₹12.4 Lakhs / Quarter | ₹3.8 Lakhs / Quarter | **₹8.6 Lakhs Revenue Recovered** |
| **Safety Stock Accuracy** | Intuitive Guesswork | Statistically Sized ($Z=1.65$) | **-22% Reduction in Excess Holding Cost** |
| **Manager Planning Time** | 3.5 Hours / Store / Day | 15 Minutes One-Click Export | **92% Operational Efficiency Gain** |

---

## 🚀 6. Management Prototype Walkthrough
The solution is deployed via an interactive, executive-grade Streamlit application (`dashboard/app.py`):
1. **Executive Command Center:** Real-time revenue, stockout rates, and lost revenue tracker.
2. **Manager Action Centre:** Actionable priority cards and full store-level reorder lists with CSV export.
3. **Risk Matrix:** Interactive Store × Category heatmap identifying systemic vulnerabilities.
4. **Demand Intelligence:** Visualizing actual vs forecast trajectories.
5. **What-If Scenario Simulator:** Allows planners to simulate marketing promotions, supplier delays, and festival spikes in real-time.
6. **Model Benchmarks & XAI:** Transparent algorithm audits comparing Linear Regression, Decision Trees, Random Forests, and XGBoost.
