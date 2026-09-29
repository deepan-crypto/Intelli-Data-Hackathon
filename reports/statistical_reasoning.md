# 📊 NovaMart Retail — Statistical Reasoning & Hypothesis Testing (Round 1)
**Event:** IntelliData 2026 Data Science Hackathon  
**Team:** StockSense Consulting Unit | Sri Eshwar College of Engineering  

---

## Mandatory Statistical Analyses (3 Hypotheses)

### 📌 Analysis 1: Promotion Impact on Demand
- **Business Question:** Do promotional campaigns significantly increase units sold?
- **Null Hypothesis ($H_0$):** Mean sales volume during promotional periods equals mean sales volume during non-promotional periods (μ_promo = μ_non_promo).
- **Alternative Hypothesis ($H_1$):** Mean sales volume during promotional periods is significantly greater than during non-promotional periods (μ_promo > μ_non_promo).
- **Statistical Test:** Welch's Two-Sample Independent t-Test (unequal variances)
- **Test Results:**
  - **t-statistic:** `14.009`
  - **p-value:** `1.2238e-43` (Statistically Significant at $\alpha = 0.01$)
  - **Non-Promo Mean Daily Sales:** `37.74` units
  - **Promo Mean Daily Sales:** `50.43` units
  - **Sales Lift:** `+33.63%`
- **Business Interpretation & Action:**
  > With p-value < 0.001 (t = 14.01), we decisively REJECT H0. Promotions drive a statistically significant sales lift of +33.6% (average 50.4 units vs 37.7 units). Recommendation: Replenishment algorithms must include promotional flags to trigger pre-emptive stock uplift.

---

### 📌 Analysis 2: Store Format Variance in Customer Demand
- **Business Question:** Does mean customer demand differ significantly across store formats (Hypermarket, Supermarket, Express)?
- **Null Hypothesis ($H_0$):** Mean daily demand is identical across all store formats (μ_hyper = μ_super = μ_express).
- **Alternative Hypothesis ($H_1$):** At least one store format exhibits significantly different mean daily product demand.
- **Statistical Test:** One-Way Analysis of Variance (ANOVA)
- **Test Results:**
  - **F-statistic:** `7157.7828`
  - **p-value:** `0.0000e+00` (Statistically Significant at $\alpha = 0.01$)
- **Business Interpretation & Action:**
  > With F-statistic = 7157.78 and p-value < 0.001, we REJECT H0. Demand variance across store formats is highly significant. Hypermarkets exhibit substantially higher average turnover per SKU than Express outlets. Reorder points and safety stocks must be parameterized by store format rather than a one-size-fits-all threshold.

---

### 📌 Analysis 3: Promotional Status vs. Stock-out Occurrence
- **Business Question:** Is stock-out occurrence statistically dependent on promotional status?
- **Null Hypothesis ($H_0$):** Stock-out events and promotion campaigns are statistically independent.
- **Alternative Hypothesis ($H_1$):** Stock-out events are significantly associated with promotional campaigns.
- **Statistical Test:** Pearson’s Chi-Square Test of Independence (2x2 Contingency Table)
- **Test Results:**
  - **$\chi^2$ Statistic:** `71.0957`
  - **p-value:** `3.4032e-17` (Statistically Significant at $\alpha = 0.01$)
  - **Stockout Rate (Promotional Days):** `13.69%`
  - **Stockout Rate (Standard Days):** `9.12%`
- **Business Interpretation & Action:**
  > With Chi-square = 71.10 (p < 0.001), we REJECT H0. There is strong dependency between promotions and stock-outs. Stock-out rate during promotions (13.7%) is substantially higher than during regular periods (9.1%). NovaMart marketing teams are currently running promotions without coordinating supply chain replenishment lead times, creating avoidable lost sales.

---

## 📈 Key Exploratory Findings
1. **Pareto Concentration:** Top 3 categories (**Beverages**, **Dairy**, and **Snacks**) generate over 65% of company gross revenue.
2. **Weekend Surge:** Average footfall and unit demand rise by 32% on Saturdays and Sundays; replenishment schedules must dispatch on Thursday nights to protect weekend shelf availability.
3. **High Volatility Risk SKUs:** Dairy and Fresh Bakery present high demand variance and short shelf life (3-5 days), requiring tight daily safety stock buffers.
