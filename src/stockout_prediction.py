"""
=============================================================================
IntelliData 2026 — Stock-Out Prediction Module
=============================================================================
Predicts the probability of stock-out and classifies risk level:
  • Uses Random Forest + XGBoost ensemble
  • Risk classification: High (≥70%) / Medium (40-70%) / Low (<40%)
  • Generates risk scores for each Store × Product combination
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier


def create_stockout_target(features_df, forecast_df=None, horizon=3):
    """
    Create stock-out risk target variable.
    A product is at risk if it stocks out within the next `horizon` days.
    """
    print(f"\n🎯 Creating stock-out target (horizon={horizon} days)...")

    df = features_df.copy()
    group_cols = ['store_id', 'product_id']

    # Forward-looking: will this product stock out in the next N days?
    df['stockout_next_n'] = (
        df.groupby(group_cols)['stock_out_flag']
        .transform(lambda x: x.shift(-1).rolling(horizon, min_periods=1).max())
    ).fillna(0).astype(int)

    print(f"   Positive rate: {df['stockout_next_n'].mean()*100:.1f}%")
    return df


def prepare_stockout_features(df, feature_cols):
    """Prepare features specifically for stock-out prediction."""
    # Add stock-out specific features
    extra_features = []

    if 'closing_stock' in df.columns:
        extra_features.append('closing_stock')
    if 'days_of_stock' in df.columns:
        extra_features.append('days_of_stock')
    if 'stock_below_reorder' in df.columns:
        extra_features.append('stock_below_reorder')
    if 'stock_to_demand_ratio' in df.columns:
        extra_features.append('stock_to_demand_ratio')
    if 'dos_roll_mean_7' in df.columns:
        extra_features.append('dos_roll_mean_7')

    all_features = list(set(feature_cols + extra_features))
    valid_features = [f for f in all_features if f in df.columns]
    return valid_features


def train_stockout_model(X_train, y_train, X_test=None, y_test=None):
    """Train stock-out prediction model using XGBoost."""
    print("\n🤖 Training Stock-Out Prediction Model...")

    # Handle class imbalance
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / max(pos_count, 1)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=min(scale_pos_weight, 10),
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        eval_metric='auc',
    )

    eval_set = [(X_train, y_train)]
    if X_test is not None and y_test is not None:
        eval_set.append((X_test, y_test))

    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    print("   ✅ Stock-out model training complete")
    return model


def evaluate_stockout_model(model, X_test, y_test):
    """Evaluate stock-out prediction performance."""
    print("\n📏 Stock-Out Model Evaluation:")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Classification report
    report = classification_report(y_test, y_pred, target_names=['No Stockout', 'Stockout'],
                                    output_dict=True)
    print(classification_report(y_test, y_pred, target_names=['No Stockout', 'Stockout']))

    # AUC
    try:
        auc = roc_auc_score(y_test, y_prob)
        print(f"   AUC-ROC: {auc:.4f}")
    except:
        auc = 0.0

    metrics = {
        'accuracy': report['accuracy'],
        'precision_stockout': report['Stockout']['precision'],
        'recall_stockout': report['Stockout']['recall'],
        'f1_stockout': report['Stockout']['f1-score'],
        'auc_roc': auc,
    }

    return metrics, y_prob


def classify_risk(probability):
    """Classify stock-out risk based on probability."""
    if probability >= 0.70:
        return 'High'
    elif probability >= 0.40:
        return 'Medium'
    else:
        return 'Low'


def generate_risk_assessment(features_df, forecast_df, model, feature_cols, stores_df, products_df):
    """
    Generate stock-out risk assessment for each Store × Product combination.
    Combines ML predictions with rule-based signals.
    """
    print("\n⚠️  Generating Stock-Out Risk Assessment...")

    # Get the latest data point for each store-product combo
    latest = features_df.sort_values('date').groupby(['store_id', 'product_id']).last().reset_index()

    stockout_features = prepare_stockout_features(latest, feature_cols)
    valid_features = [f for f in stockout_features if f in latest.columns]

    X = latest[valid_features].fillna(0)

    # Predict probabilities
    probs = model.predict_proba(X)[:, 1]

    risk_df = latest[['store_id', 'product_id', 'date']].copy()
    risk_df['stockout_probability'] = probs
    risk_df['risk_level'] = risk_df['stockout_probability'].apply(classify_risk)

    # Add inventory context
    if 'closing_stock' in latest.columns:
        risk_df['current_stock'] = latest['closing_stock'].values
    if 'days_of_stock' in latest.columns:
        risk_df['days_of_stock'] = latest['days_of_stock'].values
    if 'quantity_sold_roll_mean_7' in latest.columns:
        risk_df['avg_daily_demand'] = latest['quantity_sold_roll_mean_7'].values

    # Add 7-day total forecast demand
    if forecast_df is not None:
        total_forecast = forecast_df.groupby(['store_id', 'product_id'])['forecasted_demand'].sum().reset_index()
        total_forecast.rename(columns={'forecasted_demand': 'forecast_7day_total'}, inplace=True)
        risk_df = risk_df.merge(total_forecast, on=['store_id', 'product_id'], how='left')

        # Calculate stock coverage
        risk_df['stock_coverage_days'] = (
            risk_df['current_stock'] / risk_df['avg_daily_demand'].replace(0, 1)
        ).round(1)

        # Override risk if stock coverage < 3 days
        risk_df.loc[risk_df['stock_coverage_days'] < 3, 'risk_level'] = 'High'
        risk_df.loc[risk_df['stock_coverage_days'] < 3, 'stockout_probability'] = \
            risk_df.loc[risk_df['stock_coverage_days'] < 3, 'stockout_probability'].clip(lower=0.7)

    # Enrich with names
    risk_df = risk_df.merge(stores_df[['store_id', 'store_name']], on='store_id', how='left')
    risk_df = risk_df.merge(products_df[['product_id', 'product_name', 'category',
                                          'supplier_lead_time_days']], on='product_id', how='left')

    # Summary
    risk_counts = risk_df['risk_level'].value_counts()
    print(f"\n   Risk Distribution:")
    for level in ['High', 'Medium', 'Low']:
        count = risk_counts.get(level, 0)
        pct = count / len(risk_df) * 100
        emoji = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}[level]
        print(f"   {emoji} {level}: {count:,} ({pct:.1f}%)")

    return risk_df
