# 🧠 StockSense: Demand Forecasting & Stock-Out Prevention Engine
> **“Every empty shelf is lost revenue. Can Data Science predict it before it happens?”**

[![IntelliData 2026](https://img.shields.io/badge/Hackathon-IntelliData%202026-blue.svg)](https://sece.ac.in)
[![Institution](https://img.shields.io/badge/Institution-Sri%20Eshwar%20College%20of%20Engineering-orange.svg)](https://sece.ac.in)
[![Department](https://img.shields.io/badge/Department-CSE%20III%20Year-green.svg)](https://sece.ac.in)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io)

---

## 🏢 Industry Use Case: NovaMart Retail Pvt. Ltd.
NovaMart Retail Pvt. Ltd. operates supermarket and hypermarket outlets across multiple cities in Tamil Nadu (Chennai, Coimbatore, Madurai, Salem, Trichy, Erode). Replenishment decisions historically relied on store manager intuition and simple heuristic rules.

This created two costly enterprise challenges:
1. **Frequent Stock-outs:** High-velocity products ran out of stock, leading to unfulfilled customer demand, lost gross revenue, and customer attrition.
2. **Overstocking & Working Capital Lock-up:** Slower-moving products were over-ordered, creating excess holding costs and risk of expiry in perishable categories (Dairy, Bakery).

**StockSense** delivers a management-ready decision-support system that **forecasts demand**, **quantifies stock-out probabilities**, **explains the primary drivers**, and **prescribes specific replenishment quantities** for store managers before shelves run empty.

---

## 🏗️ End-to-End Solution Architecture
```
┌─────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│   5 Raw CSVs    │ ───► │  Data Audit & Clean     │ ───► │   Master Table Grain    │
│  transactions   │      │  • 6 Data Traps Treated │      │   1 ROW = 1 DATE ×      │
│  products       │      │  • Deduplication        │      │   1 STORE × 1 PRODUCT   │
│  stores         │      │  • Arithmetic Reconcile │      │  (24,510 Observations)  │
│  inventory      │      └─────────────────────────┘      └────────────┬────────────┘
│  external       │                                                    │
└─────────────────┘                                                    ▼
┌─────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│ Decision Action Layer   │ ◄─── │ Permitted ML Algorithms │ ◄─── │ Leakage-Free Features   │
│ • Safety Stock (Z=1.65) │      │ • Model 1: Demand (XGB) │      │ • 48 Features           │
│ • Prescribed Reorder    │      │ • Model 2: Risk (XGB)   │      │ • Strict Lags (t-1..14) │
│ • Manager Action Cards  │      │ • Time-Aware Validation │      │ • Rolling Means/Std     │
│ • High/Med/Low Tiers    │      │ • MAE, RMSE, Recall, F1 │      │ • Climate & Promo Flags │
└───────────┬─────────────┘      └─────────────────────────┘      └─────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                      🖥️ Streamlit Interactive Management Prototype                        │
│   Executive Dashboard  │  Action Centre  │  Risk Matrix  │  What-If  │  Model XAI         │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository & Folder Structure
```
DS-hackathon-DAY-2/
├── data/
│   ├── raw/                              # Original source CSVs (with intentional traps)
│   │   ├── transactions.csv
│   │   ├── products.csv
│   │   ├── stores.csv
│   │   ├── inventory.csv
│   │   └── external_factors.csv
│   ├── processed/                        # Cleaned analytical tables
│   │   ├── master_table.csv              # Reconciled grain (24,510 rows)
│   │   ├── feature_matrix.csv            # 48 engineered ML features
│   │   └── manager_action_recommendations.csv # Final prescribed orders
│   └── generate_data.py                  # Synthetic data generator
├── notebooks/
│   ├── StockSense_End_to_End.ipynb       # Comprehensive Jupyter Notebook (Rounds 1, 2, 3)
│   └── generate_notebook.py              # Notebook automated builder
├── src/
│   ├── data_preparation.py               # Round 1: Auditing, cleaning, master merge
│   ├── eda_statistics.py                 # Round 1: 3 hypothesis tests & charts
│   ├── feature_engineering.py            # Round 2: Temporal, lag, rolling features
│   └── models.py                         # Round 2 & 3: ML training, validation, action layer
├── models/
│   ├── demand_forecast_model.pkl         # Trained Champion XGBoost Regressor
│   └── stockout_risk_model.pkl           # Trained Champion XGBoost Classifier
├── dashboard/
│   └── app.py                            # Round 3: Streamlit interactive decision prototype
├── reports/
│   ├── data_quality_report.md            # Round 1 Data Quality Audit
│   ├── statistical_reasoning.md          # Round 1 Statistical Hypothesis Tests
│   ├── model_comparison_report.md        # Round 2 ML Evaluation & Justification
│   ├── final_pitch_presentation.md       # Final pitch deck & presentation script
│   └── figures/                          # Exported EDA and analytical visualizations
│       ├── category_pareto.png
│       ├── promo_lift_box.png
│       ├── stockout_heatmap.png
│       ├── store_trends.png
│       ├── volatile_products.png
│       └── weekday_weekend.png
├── requirements.txt
└── README.md
```

---

## ⚡ Quickstart & Execution Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/StockSense.git
cd StockSense

# Install required dependencies
pip install -r requirements.txt
```

### 2. Run Complete Pipeline (End-to-End)
```bash
# Step 1: Generate Enterprise Raw Data with Intentional Traps
python data/generate_data.py

# Step 2: Round 1 Data Preparation & Master Table Assembly
python src/data_preparation.py

# Step 3: Round 1 EDA & Mandatory Statistical Hypothesis Testing
python src/eda_statistics.py

# Step 4: Round 2 Feature Engineering (Leakage-Free Construction)
python src/feature_engineering.py

# Step 5: Round 2 & 3 Machine Learning Models & Manager Action Prescriptions
python src/models.py
```

### 3. Launch Interactive Management Prototype
```bash
streamlit run dashboard/app.py
```
*The Streamlit web application will launch at `http://localhost:8501`.*

---

## 📋 Hackathon Rounds Breakdown

### Round 1: Data Understanding, Preparation & EDA
- **6 Data Quality Traps Remediation:**
  1. *Category Inconsistencies:* Standardized casing (`beverage` $\rightarrow$ `Beverages`).
  2. *Duplicate Transactions:* Deduplicated by `transaction_id`.
  3. *Negative/Impossible Quantities:* Filtered invalid non-positive lines.
  4. *Missing Values:* Imputed temperature via city medians; selling price via $MRP \times (1 - discount\%)$.
  5. *Inventory Arithmetic Mismatch:* Reconciled physical flow: $Closing = Opening + Received - Sold$.
  6. *Sparse History:* Applied category-level benchmark priors for newly launched SKUs.
- **Master Grain:** `ONE ROW = ONE DATE × ONE STORE × ONE PRODUCT` ($24,510$ observations).
- **3 Mandatory Hypothesis Tests:**
  - **Test 1 (Welch's t-test on Promotions):** $t = 38.41, p < 0.001$. Promotions drive a **+41.2% sales uplift**.
  - **Test 2 (One-Way ANOVA on Store Types):** $F = 412.5, p < 0.001$. Hypermarkets significantly outperform Express outlets.
  - **Test 3 (Chi-Square on Stock-Outs vs Promotions):** $\chi^2 = 84.1, p < 0.001$. Promotions significantly increase stock-out probability.

---

### Round 2: Feature Engineering & Machine Learning
- **48 Engineered Signals (Zero Leakage):**
  - Temporal (Day of week, weekend, month, cyclical sin/cos encodings, festival proximity).
  - Lags: Strictly historical ($t-1, t-2, t-7, t-14$).
  - Rolling Statistics: 7-day and 14-day rolling mean, std dev, min, and max using `shift(1)`.
  - Inventory Metrics: Days of inventory runway, reorder gap ($closing - reorder\_lvl$).
- **Model 1 — 7-Day Demand Forecasting (Regression):**
  | Algorithm | MAE (Units) | RMSE (Units) | MAPE (%) | $R^2$ Score | Status |
  |-----------|-------------|--------------|----------|-------------|--------|
  | Baseline (7-Day Mean) | 34.81 | 59.36 | 12.2% | 0.9566 | Evaluated |
  | Linear Regression | 35.89 | 60.95 | 12.7% | 0.9542 | Evaluated |
  | Decision Tree | 35.53 | 59.22 | 11.9% | 0.9568 | Evaluated |
  | Random Forest | 33.52 | 56.51 | 11.2% | 0.9606 | Evaluated |
  | **XGBoost Regressor** | **32.69** | **54.61** | **11.2%** | **0.9632** | 🏆 **WINNER** |
- **Model 2 — Stock-out Risk Classification:**
  | Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
  |-----------|----------|-----------|--------|----------|---------|--------|
  | Baseline (Majority) | 0.630 | 0.000 | 0.000 | 0.000 | 0.500 | Evaluated |
  | Decision Tree | 0.691 | 0.557 | 0.808 | 0.659 | 0.750 | Evaluated |
  | Random Forest | 0.686 | 0.553 | 0.784 | 0.649 | 0.724 | Evaluated |
  | **XGBoost Classifier** | **0.671** | **0.536** | **0.825** | **0.650** | **0.739** | 🏆 **WINNER** |

*Note on Metric Choice:* **Recall (82.5%)** is prioritized over pure accuracy because a False Negative (missed stockout) leads to lost revenue, whereas a False Positive merely flags an inventory review.

---

### Round 3: Intelligence Layer, Action Logic & Prototype

#### Business Replenishment Formulation:
$$\text{Recommended Stock} = \text{Predicted 7-Day Demand} + \text{Safety Stock}$$
$$\text{Safety Stock} = Z \times \sigma_{\text{7d}} \times \sqrt{\text{Lead Time Days}} \quad (Z = 1.65 \text{ for } 95\% \text{ Cycle Service Level})$$
$$\text{Reorder Quantity} = \max\left(0, \text{Recommended Stock} - \text{Current Stock} - \text{Incoming Stock}\right)$$

#### Operational Risk Tiers:
- 🔴 **HIGH RISK ($\ge 70\%$ Prob):** Urgent dispatch today before delivery cycle closure.
- 🟡 **MEDIUM RISK ($40\% - 70\%$ Prob):** Monitor buffer; reorder in next scheduled routine delivery.
- 🟢 **LOW RISK ($< 40\%$ Prob):** Inventory runway is healthy; no order required.

#### Example Manager Recommendation Card (from Prototype):
> **STORE S01 (COIMBATORE) — AAVIN MILK 1L (P101)**  
> **Predicted 7-day demand:** 485 units  
> **Current stock:** 250 units | **Incoming stock:** 80 units  
> **Stock-out probability:** 89% | **Risk Tier:** HIGH  
> **Recommended replenishment order:** **180 units**  
> **Why?** Active promotion driving surge, weekend footfall spike approaching, runway critically below supplier lead time.  
> **MANAGER ACTION:** 🚨 *Raise replenishment order today.*

---

## 👥 Team Roles & Hackathon Evaluation Alignment
- **Student 1 (Data Analyst):** Raw Data Cleaning, 6 Data Traps, Master Table Integration, Statistical Tests, Pareto EDA.
- **Student 2 (ML Engineer):** Feature Engineering Pipeline, Validation Splitting, Regression & Classification Benchmarks.
- **Student 3 (Decision Intelligence):** Safety Stock Math, Manager Action Centre, Explainability, Streamlit Dashboard.

---

## 🎓 Sri Eshwar College of Engineering | IntelliData 2026
*Department of Computer Science and Engineering*
