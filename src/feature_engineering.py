"""
=============================================================================
IntelliData 2026 — Round 2: Feature Engineering Pipeline
=============================================================================
Builds leakage-free ML features from cleaned master table:
  1. Temporal Features:
     - day_of_week, is_weekend, month, week_of_year, festival_flag, holiday_flag
     - Cyclical encodings (sin/cos for day of week & month)
  2. Lag Features (Historical demand only, shifted strictly t-1 or earlier):
     - demand_lag_1, demand_lag_2, demand_lag_7, demand_lag_14
  3. Rolling Window Statistics (using shift(1) to prevent target leakage):
     - rolling_mean_7, rolling_mean_14, rolling_std_7, rolling_max_7, rolling_min_7
  4. Inventory Health Features:
     - days_of_inventory (closing_stock / (rolling_mean_7 + 1e-3))
     - inventory_to_demand_ratio
     - reorder_gap (closing_stock - reorder_level)
     - stock_below_reorder_flag
  5. Price & Promotion Features:
     - discount_pct, promotion_flag, mrp, cost_price, profit_margin
     - price_ratio (selling_price / mrp)
  6. Store & Product Categorical Attributes:
     - floor_area_sqft, avg_daily_customers, shelf_life_days, lead_time_days
     - Frequency & target encodings for high cardinality categorical features
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')


def engineer_features(master_df):
    """
    Transforms master_df into ML-ready feature matrix without data leakage.
    Target variables (next_7_day_demand, future_stockout_7d) are preserved.
    """
    print("\n--- Constructing Leakage-Free Feature Matrix ---")
    df = master_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['store_id', 'product_id', 'date']).reset_index(drop=True)

    group_keys = ['store_id', 'product_id']

    # 1. Temporal & Calendar Features
    print("   [1/6] Engineering Temporal & Calendar Signals...")
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['day_of_month'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
    df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)

    # Cyclical representations
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)

    # External weather/festival flags
    df['festival_flag'] = df['festival'].fillna(0).astype(int)
    df['holiday_flag'] = df['holiday'].fillna(0).astype(int)
    df['temp_c'] = df['temp_c'].fillna(30.0)
    df['rain_mm'] = df['rain_mm'].fillna(0.0)

    # 2. Lag Features (Strictly Historical t-k)
    print("   [2/6] Generating Strict Historical Lags (t-1, t-2, t-7, t-14)...")
    for lag in [1, 2, 7, 14]:
        df[f'demand_lag_{lag}'] = df.groupby(group_keys)['total_quantity_sold'].shift(lag)

    # 3. Rolling Window Statistics (Shifted by 1 to prevent leakage)
    print("   [3/6] Generating Rolling Windows (7-day, 14-day stats)...")
    for window in [7, 14]:
        df[f'rolling_mean_{window}'] = df.groupby(group_keys)['total_quantity_sold'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'rolling_std_{window}'] = df.groupby(group_keys)['total_quantity_sold'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std()
        ).fillna(0.0)
        df[f'rolling_max_{window}'] = df.groupby(group_keys)['total_quantity_sold'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).max()
        )
        df[f'rolling_min_{window}'] = df.groupby(group_keys)['total_quantity_sold'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).min()
        )

    # Short 3-day momentum
    df['rolling_mean_3'] = df.groupby(group_keys)['total_quantity_sold'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    # 4. Inventory Health & Reorder Metrics
    print("   [4/6] Formulating Inventory Metrics (Days of Inventory, Reorder Gap)...")
    df['reorder_gap'] = df['closing_stock'] - df['reorder_level']
    df['stock_below_reorder_flag'] = (df['closing_stock'] < df['reorder_level']).astype(int)
    df['days_of_inventory'] = (df['closing_stock'] / (df['rolling_mean_7'].fillna(1.0) + 0.1)).clip(0, 90).round(1)
    df['inventory_to_demand_ratio'] = (df['closing_stock'] / (df['demand_lag_1'].fillna(1.0) + 0.1)).clip(0, 100).round(2)

    # 5. Pricing, Promotion & Profit Margin
    print("   [5/6] Calculating Price Ratios, Margin & Promo Interactions...")
    df['discount_pct'] = df['max_discount_pct'].fillna(0.0)
    df['promotion_flag'] = df['promotion_flag'].fillna(0).astype(int)
    df['profit_margin'] = ((df['mrp'] - df['cost_price']) / (df['mrp'] + 1e-3)).round(3)
    df['effective_price_ratio'] = (df['avg_selling_price'] / (df['mrp'] + 1e-3)).clip(0.5, 1.2).round(3)
    df['promo_x_weekend'] = df['promotion_flag'] * df['is_weekend']
    df['promo_x_discount'] = df['promotion_flag'] * (df['discount_pct'] / 100.0)

    # 6. Store & Product Encodings
    print("   [6/6] Categorical Encodings & Structural Mapping...")
    store_map = {'Hypermarket': 3, 'Supermarket': 2, 'Express': 1}
    df['store_type_code'] = df['store_type'].map(store_map).fillna(2).astype(int)

    cat_freq = df['category'].value_counts(normalize=True).to_dict()
    df['category_freq_enc'] = df['category'].map(cat_freq)

    brand_freq = df['brand'].value_counts(normalize=True).to_dict()
    df['brand_freq_enc'] = df['brand'].map(brand_freq)

    # Backfill missing lags at start of series
    lag_cols = [c for c in df.columns if 'lag' in c or 'rolling' in c]
    for c in lag_cols:
        df[c] = df.groupby(group_keys)[c].transform(lambda x: x.bfill().fillna(0))

    df = df.fillna(0)

    feat_path = os.path.join(PROCESSED_DIR, 'feature_matrix.csv')
    df.to_csv(feat_path, index=False)
    print(f" Feature Matrix ready: {feat_path} ({df.shape[0]:,} rows x {df.shape[1]} cols)")
    return df


def get_model_feature_columns():
    """Returns strict feature column list for model training."""
    features = [
        # Temporal
        'day_of_week', 'is_weekend', 'day_of_month', 'week_of_year', 'month', 'quarter',
        'is_month_start', 'is_month_end', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
        'festival_flag', 'holiday_flag', 'temp_c', 'rain_mm',
        # Lags
        'demand_lag_1', 'demand_lag_2', 'demand_lag_7', 'demand_lag_14',
        # Rolling stats
        'rolling_mean_3', 'rolling_mean_7', 'rolling_std_7', 'rolling_max_7', 'rolling_min_7',
        'rolling_mean_14', 'rolling_std_14', 'rolling_max_14', 'rolling_min_14',
        # Inventory
        'closing_stock', 'reorder_level', 'reorder_gap', 'stock_below_reorder_flag',
        'days_of_inventory', 'inventory_to_demand_ratio',
        # Pricing & Promo
        'discount_pct', 'promotion_flag', 'profit_margin', 'effective_price_ratio',
        'promo_x_weekend', 'promo_x_discount',
        # Store & Product specs
        'floor_area_sqft', 'avg_daily_customers', 'shelf_life_days', 'lead_time_days',
        'store_type_code', 'category_freq_enc', 'brand_freq_enc'
    ]
    return features


if __name__ == '__main__':
    master_path = os.path.join(PROCESSED_DIR, 'master_table.csv')
    if os.path.exists(master_path):
        master = pd.read_csv(master_path)
        feat_df = engineer_features(master)
        cols = get_model_feature_columns()
        print(f"\nModel Features count: {len(cols)}")
    else:
        print("master_table.csv not found!")
