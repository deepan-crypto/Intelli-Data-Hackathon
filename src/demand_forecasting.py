"""
=============================================================================
IntelliData 2026 — Demand Forecasting Module
=============================================================================
XGBoost-based 7-day demand forecasting per Store × Product combination.
Includes:
  • Train/test split (last 14 days for testing)
  • Model training with hyperparameter tuning
  • 7-day forward forecasting
  • Model evaluation (MAE, RMSE, MAPE)
  • Feature importance analysis
=============================================================================
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')


def prepare_train_test(features_df, feature_cols, target_col='quantity_sold', test_days=14):
    """Split data into train and test sets using temporal split."""
    print("\n🔀 Preparing train/test split...")

    max_date = features_df['date'].max()
    test_start = max_date - pd.Timedelta(days=test_days)

    train_df = features_df[features_df['date'] <= test_start].copy()
    test_df = features_df[features_df['date'] > test_start].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    print(f"   Train: {len(train_df):,} rows ({train_df['date'].min().strftime('%Y-%m-%d')} to {train_df['date'].max().strftime('%Y-%m-%d')})")
    print(f"   Test:  {len(test_df):,} rows ({test_df['date'].min().strftime('%Y-%m-%d')} to {test_df['date'].max().strftime('%Y-%m-%d')})")

    return X_train, y_train, X_test, y_test, train_df, test_df


def train_demand_model(X_train, y_train, X_test=None, y_test=None):
    """Train XGBoost model for demand forecasting."""
    print("\n🤖 Training XGBoost Demand Forecasting Model...")

    model = XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    eval_set = [(X_train, y_train)]
    if X_test is not None and y_test is not None:
        eval_set.append((X_test, y_test))

    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False,
    )

    print("   ✅ Model training complete")
    return model


def evaluate_model(model, X_test, y_test, test_df):
    """Evaluate model performance."""
    print("\n📏 Evaluating model performance...")

    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)  # no negative predictions

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / np.clip(y_test, 1, None))) * 100

    metrics = {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'MAPE': round(mape, 2),
    }

    print(f"   MAE:  {mae:.2f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   MAPE: {mape:.2f}%")

    # Store-level evaluation
    test_df = test_df.copy()
    test_df['predicted'] = y_pred
    store_metrics = test_df.groupby('store_id').apply(
        lambda g: pd.Series({
            'MAE': mean_absolute_error(g['quantity_sold'], g['predicted']),
            'RMSE': np.sqrt(mean_squared_error(g['quantity_sold'], g['predicted'])),
        })
    ).round(2)

    print("\n   Store-level performance:")
    print(store_metrics.to_string(index=True))

    return metrics, y_pred, store_metrics


def get_feature_importance(model, feature_cols, top_n=20):
    """Get top N important features."""
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return importance.head(top_n)


def forecast_7_days(model, features_df, feature_cols, stores_df, products_df):
    """
    Generate 7-day forward demand forecast for every Store × Product combination.
    Uses the most recent data to create features for future dates.
    """
    print("\n🔮 Generating 7-day demand forecast...")

    max_date = features_df['date'].max()
    forecast_dates = [max_date + pd.Timedelta(days=d) for d in range(1, 8)]

    all_forecasts = []

    store_product_combos = features_df.groupby(['store_id', 'product_id']).size().reset_index()[['store_id', 'product_id']]

    for _, combo in store_product_combos.iterrows():
        sid, pid = combo['store_id'], combo['product_id']

        # Get historical data for this combo
        hist = features_df[
            (features_df['store_id'] == sid) &
            (features_df['product_id'] == pid)
        ].sort_values('date').tail(30).copy()

        if len(hist) < 7:
            continue

        last_row = hist.iloc[-1]

        for i, fdate in enumerate(forecast_dates):
            # Build feature row from last known data + adjustments
            feat_row = {}
            for col in feature_cols:
                if col in last_row.index:
                    feat_row[col] = last_row[col]
                else:
                    feat_row[col] = 0

            # Update temporal features for forecast date
            feat_row['day_of_week'] = fdate.weekday()
            feat_row['day_of_month'] = fdate.day
            feat_row['week_of_year'] = fdate.isocalendar()[1]
            feat_row['month'] = fdate.month
            feat_row['quarter'] = (fdate.month - 1) // 3 + 1
            feat_row['is_weekend'] = 1 if fdate.weekday() >= 5 else 0
            feat_row['is_month_start'] = 1 if fdate.day <= 5 else 0
            feat_row['is_month_end'] = 1 if fdate.day >= 25 else 0
            feat_row['dow_sin'] = np.sin(2 * np.pi * fdate.weekday() / 7)
            feat_row['dow_cos'] = np.cos(2 * np.pi * fdate.weekday() / 7)
            feat_row['month_sin'] = np.sin(2 * np.pi * fdate.month / 12)
            feat_row['month_cos'] = np.cos(2 * np.pi * fdate.month / 12)
            feat_row['dom_sin'] = np.sin(2 * np.pi * fdate.day / 31)
            feat_row['dom_cos'] = np.cos(2 * np.pi * fdate.day / 31)

            # Shift lag features forward
            if i == 0:
                if 'quantity_sold_lag_1' in feat_row:
                    feat_row['quantity_sold_lag_1'] = last_row.get('quantity_sold', 0)
            elif i >= 1 and len(all_forecasts) > 0:
                prev_forecast = [f for f in all_forecasts if f['store_id'] == sid and f['product_id'] == pid]
                if prev_forecast:
                    if 'quantity_sold_lag_1' in feat_row:
                        feat_row['quantity_sold_lag_1'] = prev_forecast[-1]['forecasted_demand']

            feat_df = pd.DataFrame([feat_row])[feature_cols]
            pred = max(0, round(model.predict(feat_df)[0]))

            all_forecasts.append({
                'date': fdate,
                'store_id': sid,
                'product_id': pid,
                'forecasted_demand': pred,
                'forecast_day': i + 1,
            })

    forecast_df = pd.DataFrame(all_forecasts)

    # Enrich with names
    forecast_df = forecast_df.merge(stores_df[['store_id', 'store_name']], on='store_id', how='left')
    forecast_df = forecast_df.merge(products_df[['product_id', 'product_name', 'category']], on='product_id', how='left')

    print(f"   ✅ Forecast generated: {len(forecast_df):,} predictions")
    print(f"   📅 Forecast period: {forecast_dates[0].strftime('%Y-%m-%d')} to {forecast_dates[-1].strftime('%Y-%m-%d')}")

    return forecast_df


def save_model(model, feature_cols, metrics, filename='demand_model.pkl'):
    """Save trained model to disk."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_cols': feature_cols,
            'metrics': metrics,
        }, f)
    print(f"   💾 Model saved to {filepath}")


def load_model(filename='demand_model.pkl'):
    """Load trained model from disk."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['feature_cols'], data['metrics']
