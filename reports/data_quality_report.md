# 📋 NovaMart Retail — Data Quality & Cleansing Report (Round 1)
**Event:** IntelliData 2026 Data Science Hackathon  
**Team:** StockSense Consulting Unit  
**College:** Sri Eshwar College of Engineering, Department of CSE  
**Date:** 2026-09-29

---

## 1. Executive Summary
During Round 1 data audit, our team inspected 5 disparate raw files (`transactions.csv`, `products.csv`, `stores.csv`, `inventory.csv`, `external_factors.csv`). We identified and resolved **6 major data quality anomalies**, eliminating duplication, resolving arithmetic contradictions, reconciling stock figures, and unifying categories into a standardized analytical dataset.

## 2. Data Quality Audit & Resolution Matrix

| Table | Data Quality Issue | Affected Count | Action Taken | Business Justification |
|-------|--------------------|----------------|--------------|-------------------------|
| **products.csv** | Category Inconsistencies (e.g. beverage vs BEVERAGES) | `9` | Standardised category strings with .strip().title() | Ensures uniform category-level aggregation and avoids split grouping. |
| **transactions.csv** | Duplicate transaction IDs | `686` | Removed duplicate records keeping first occurrence | Prevents double counting of revenue and quantity demanded. |
| **transactions.csv** | Impossible Non-positive Quantities (<= 0) | `228` | Filtered out invalid non-positive transaction lines | Negative quantities represent data corruption or unprocessed returns; demand models require positive sales signal. |
| **transactions.csv** | Missing / Zero Selling Price | `908` | Imputed selling_price via MRP * (1 - discount_pct/100) | Recovers revenue figures accurately without dropping valid transaction volumes. |
| **transactions.csv** | Missing customer_id | `1,365` | Imputed with "GUEST_CUSTOMER" | Permits guest checkout analysis without failing foreign key / integrity checks. |
| **external_factors.csv** | Missing temp_c (Temperature) | `29` | Imputed using City-wise median temperature | Temperature has strong seasonal correlation with beverage/ice cream demand; city median is a robust estimator. |
| **inventory.csv** | Inventory Balance Mismatch (closing != opening + received - sold) | `5,036` | Reconciled closing_stock to theoretical flow; flagged discrepancy | Audit discrepancies indicate shrinkage, unrecorded breakage or log delay. Reconciled stock provides consistent ground truth. |
| **master_table** | Sparse History for Newly Introduced Products (e.g. P901, P902) | `0` | Applied category-level benchmark fallback strategy | New products lack long lag history; category hierarchical prior stabilizes forecasts. |

---

## 3. Master Analytical Dataset Specification
- **Final Master Table Grain:** `ONE ROW = ONE DATE × ONE STORE × ONE PRODUCT`
- **Total Master Rows:** `24,510`
- **Total Unique Stores:** `6` (S01, S02, S03, S04, S05, S06)
- **Total Unique Products:** `47` across `9` categories
- **Date Range:** `2026-08-01` to `2026-10-29` (`90` days)
- **Total Revenue Recorded:** `₹79,546,249.25`
- **Total Units Sold:** `969,673`
- **Estimated Lost Sales Due to Stock-out:** `₹1,472,780.25`
- **Overall Historical Stock-out Rate:** `9.78%`

## 4. Reconciled Grain & Integrity Sign-Off
All foreign keys between stores, products, inventory, and external climatic datasets have achieved 100% referential integrity with 0 orphaned keys.
