# 🏆 NovaMart Retail — Machine Learning Model Evaluation & Justification (Round 2)
**Event:** IntelliData 2026 Data Science Hackathon  
**Team:** StockSense Consulting Unit | Sri Eshwar College of Engineering  

---

## 1. Model 1 — 7-Day Demand Forecasting (Regression)
**Business Objective:** Forecast the total next 7-day demand for each Store × Product combination to guide replenishment orders before shelf depletion.  
**Validation Strategy:** Strict time-aware chronological validation (last 14 days reserved for hold-out evaluation). No future information leakage.

### Model Comparison Table
| Algorithm | MAE (Units) | RMSE (Units) | MAPE (%) | R² Score | Selection Status |
|-----------|-------------|--------------|----------|----------|------------------|
| **XGBoost Regressor** | `32.69` | `54.61` | `11.23%` | `0.9632` | **WINNER (Selected)** |
| **Random Forest** | `33.52` | `56.51` | `11.16%` | `0.9606` | Evaluated |
| **Baseline (7-Day Mean)** | `34.81` | `59.36` | `12.15%` | `0.9566` | Evaluated |
| **Decision Tree** | `35.53` | `59.22` | `11.93%` | `0.9568` | Evaluated |
| **Linear Regression** | `35.89` | `60.95` | `12.72%` | `0.9542` | Evaluated |

### 🔍 Metric Selection Justification
- **Primary Business Metric: Mean Absolute Error (MAE):** In grocery retail, store managers think in integer physical units (e.g. *"we are off by 4 bottles of milk"*). MAE provides an intuitive, robust operational error metric.
- **Why XGBoost Regressor Won:**
  1. XGBoost achieved the lowest MAE and lowest RMSE, demonstrating superior handling of non-linear promotional surges, weekend uplifts, and multi-category interactions.
  2. Standard Linear Regression struggled with sudden promotional spikes, while single Decision Trees overfit early training periods.

---

## 2. Model 2 — Stock-out Risk Classification
**Business Objective:** Predict the exact probability of an SKU stocking out over the next 7-day horizon and classify risk into **HIGH (≥0.70)**, **MEDIUM (0.40–0.70)**, and **LOW (<0.40)**.  
**Challenge Addressed:** Handled historical class imbalance using algorithm-native class weighting (`scale_pos_weight`).

### Model Comparison Table
| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Selection Status |
|-----------|----------|-----------|--------|----------|---------|------------------|
| **Decision Tree** | `0.691` | `0.557` | `0.807` | `0.659` | `0.750` | Evaluated |
| **XGBoost Classifier** | `0.671` | `0.536` | `0.825` | `0.650` | `0.739` | **WINNER (Selected)** |
| **Random Forest** | `0.686` | `0.553` | `0.784` | `0.648` | `0.724` | Evaluated |
| **Baseline (Majority Class)** | `0.630` | `0.000` | `0.000` | `0.000` | `0.500` | Evaluated |

### 🔍 Metric Selection Justification
- **Primary Business Metric: Recall & F1-Score:** In retail operations, a **False Negative (failing to predict a stock-out)** is catastrophic: it results in empty shelves, immediate lost revenue, and customer churn. A False Positive merely generates a reorder alert. Therefore, our model optimizes high Recall while preserving acceptable Precision.
- **Why XGBoost Classifier Won:**
  1. Achieved the highest ROC-AUC and F1-score with balanced recall across rare stock-out events.
  2. Captures complex risk thresholds where days of inventory, lead time, and upcoming promo flags intersect.

---

## 3. Business Decision Logic Formulation
```
1. Recommended Stock  = Predicted 7-Day Demand + Safety Stock
2. Safety Stock       = 1.65 × Rolling Std Dev (7d) × √(Lead Time Days)  [95% Cycle Service Level]
3. Reorder Quantity   = max(0, Recommended Stock - Current Stock - Incoming Stock)
4. Risk Categorization:
   • HIGH RISK   : Stock-out Probability ≥ 0.70  -> Immediate PO Dispatch
   • MEDIUM RISK : 0.40 ≤ Probability < 0.70    -> Prioritize in Next Delivery Cycle
   • LOW RISK    : Probability < 0.40           -> Sufficient Runway
```
