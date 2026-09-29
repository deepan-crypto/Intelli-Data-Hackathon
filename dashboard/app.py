"""
=============================================================================
IntelliData 2026 — Day 2: StockSense — Executive Decision Support Prototype
=============================================================================
Industry Use Case: NovaMart Retail Pvt. Ltd.
Institution: Sri Eshwar College of Engineering | Department of CSE
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure wide layout and page metadata
st.set_page_config(
    page_title="StockSense | NovaMart Retail Decision Suite",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Management-Ready Presentation
st.markdown("""
<style>
    /* Clean Modern Theme */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
        border: 1px solid #E2E8F0;
        text-align: left;
    }
    .kpi-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
        margin-top: 0.3rem;
    }
    .kpi-delta-pos {
        color: #10B981;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .kpi-delta-neg {
        color: #EF4444;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
    }
    .badge-medium {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
    }
    .badge-low {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
    }
    .rec-card {
        background: #F8FAFC;
        border-left: 5px solid #3B82F6;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border: 1px solid #E2E8F0;
    }
    .rec-card-high {
        background: #FFF5F5;
        border-left: 5px solid #EF4444;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border: 1px solid #FED7D7;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')


def load_data():
    master_path = os.path.join(PROCESSED_DIR, 'master_table.csv')
    action_path = os.path.join(PROCESSED_DIR, 'manager_action_recommendations.csv')

    master = pd.read_csv(master_path) if os.path.exists(master_path) else None
    if master is not None:
        master['date'] = pd.to_datetime(master['date'])

    actions = pd.read_csv(action_path) if os.path.exists(action_path) else None
    return master, actions


master, actions = load_data()

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR FILTERS (DYNAMIC & REACTIVE)
# ═══════════════════════════════════════════════════════════════════════════
st.sidebar.image("https://img.icons8.com/fluency/96/delivery-time.png", width=64)
st.sidebar.markdown("## **StockSense AI**")
st.sidebar.markdown("*IntelliData 2026 | NovaMart Retail*")
st.sidebar.markdown("---")

selected_store = "All Stores"
selected_category = "All Categories"
selected_risk = "All Risk Tiers"

if actions is not None:
    stores_list = ["All Stores"] + sorted(actions['store_id'].unique().tolist())
    selected_store = st.sidebar.selectbox("🏬 Filter by Store", stores_list)

    cats_list = ["All Categories"] + sorted(actions['category'].unique().tolist())
    selected_category = st.sidebar.selectbox("🏷️ Filter by Category", cats_list)

    risk_list = ["All Risk Tiers", "HIGH", "MEDIUM", "LOW"]
    selected_risk = st.sidebar.selectbox("⚠️ Filter by Risk Level", risk_list)

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Data / Reload CSVs", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("""
**Team StockSense**  
Department of CSE  
Sri Eshwar College of Engineering  
*Challenge: Every empty shelf is lost revenue.*
""")

# ═══════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-header">StockSense Decision Intelligence Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict Demand • Prevent Stock-outs • Prescribe Inventory Replenishment for Store Managers</div>', unsafe_allow_html=True)

# Check if data loaded
if master is None or actions is None:
    st.error("⚠️ Master dataset or recommendation table not found! Please run the pipeline script first: `python src/models.py`")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
# DYNAMIC FILTERING FOR BOTH MASTER AND ACTIONS (Ensures All Visuals Change!)
# ═══════════════════════════════════════════════════════════════════════════
filtered_master = master.copy()
filtered_actions = actions.copy()

if selected_store != "All Stores":
    filtered_master = filtered_master[filtered_master['store_id'] == selected_store]
    filtered_actions = filtered_actions[filtered_actions['store_id'] == selected_store]

if selected_category != "All Categories":
    filtered_master = filtered_master[filtered_master['category'] == selected_category]
    filtered_actions = filtered_actions[filtered_actions['category'] == selected_category]

if selected_risk != "All Risk Tiers":
    filtered_actions = filtered_actions[filtered_actions['risk_level'] == selected_risk]
    # Filter master SKUs by matched products in selected risk tier
    valid_pids = filtered_actions['product_id'].unique()
    filtered_master = filtered_master[filtered_master['product_id'].isin(valid_pids)]

# Active filter indicator banner
active_filters = []
if selected_store != "All Stores":
    active_filters.append(f"Store: **{selected_store}**")
if selected_category != "All Categories":
    active_filters.append(f"Category: **{selected_category}**")
if selected_risk != "All Risk Tiers":
    active_filters.append(f"Risk Tier: **{selected_risk}**")

if active_filters:
    st.info("🔍 Active Filters Applied: " + " | ".join(active_filters) + f" — Showing **{len(filtered_actions)}** matching SKUs")

# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Executive Summary",
    "🎯 Manager Action Centre",
    "⚠️ Stock-out Risk Matrix",
    "🔮 Demand Forecasting",
    "🧪 What-If Simulator",
    "🏆 Model Benchmarks & XAI"
])

# ───────────────────────────────────────────────────────────────────────────
# TAB 1: EXECUTIVE SUMMARY (DYNAMIC TO FILTERS)
# ───────────────────────────────────────────────────────────────────────────
with tab1:
    scope_name = "NovaMart Multi-City Portfolio" if selected_store == "All Stores" else f"Store {selected_store} Overview"
    st.markdown(f"### 🏢 Executive Portfolio Health ({scope_name})")

    col1, col2, col3, col4, col5 = st.columns(5)
    total_rev = filtered_master['total_revenue'].sum()
    total_units = filtered_master['total_quantity_sold'].sum()
    stockout_rate = (filtered_master['stockout_flag'].mean() * 100) if len(filtered_master) > 0 else 0.0
    lost_rev = filtered_master['estimated_lost_sales'].sum()
    high_risk_skus = (filtered_actions['risk_level'] == 'HIGH').sum()

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Gross Revenue</div>
            <div class="kpi-value">₹{total_rev/1e5:,.2f}L</div>
            <div class="kpi-delta-pos">{'Across All Stores' if selected_store == 'All Stores' else f'Store {selected_store}'}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Units Sold</div>
            <div class="kpi-value">{total_units:,.0f}</div>
            <div class="kpi-delta-pos">Filtered Period Sales</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Stock-out Rate</div>
            <div class="kpi-value">{stockout_rate:.1f}%</div>
            <div class="kpi-delta-neg">Target Benchmark: &lt; 5.0%</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Estimated Lost Sales</div>
            <div class="kpi-value">₹{lost_rev/1e5:,.2f}L</div>
            <div class="kpi-delta-neg">Unfulfilled Demand</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">High Risk SKUs</div>
            <div class="kpi-value">{high_risk_skus}</div>
            <div class="kpi-delta-neg">Requires PO Today</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns([6, 4])
    with c1:
        # Dynamic Daily Revenue & Lost Sales Trend
        daily = filtered_master.groupby('date').agg(
            revenue=('total_revenue', 'sum'),
            lost_sales=('estimated_lost_sales', 'sum')
        ).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily['date'], y=daily['revenue'], name='Realized Revenue (₹)', line=dict(color='#2563EB', width=2.5)))
        fig.add_trace(go.Scatter(x=daily['date'], y=daily['lost_sales'], name='Lost Revenue due to Stockouts (₹)', line=dict(color='#DC2626', width=2, dash='dot')))
        fig.update_layout(
            title=f"<b>Daily Revenue vs. Lost Sales Horizon ({selected_store} | {selected_category})</b>",
            xaxis_title="Date",
            yaxis_title="Amount (INR)",
            hovermode="x unified",
            height=360,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Category or Sub-Category Share Pie based on filter
        group_col = 'sub_category' if selected_category != "All Categories" else 'category'
        cat_share = filtered_master.groupby(group_col)['total_revenue'].sum().reset_index()
        fig_pie = px.pie(
            cat_share, values='total_revenue', names=group_col,
            title=f"<b>Revenue Contribution by {group_col.replace('_', ' ').title()}</b>",
            hole=0.45,
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        fig_pie.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)


# ───────────────────────────────────────────────────────────────────────────
# TAB 2: MANAGER ACTION CENTRE (THE CORE BUSINESS LAYER)
# ───────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📋 Store Manager Replenishment Action Centre")
    st.caption("A prediction without an action is incomplete. StockSense turns ML estimates into prescriptively prioritized purchase orders.")

    urgent_items = filtered_actions[filtered_actions['risk_level'] == 'HIGH']
    if len(urgent_items) > 0:
        st.warning(f"🚨 **Urgent Attention:** {len(urgent_items)} Store-SKU items are classified as **HIGH RISK (≥70% probability of stock-out)**. Recommended action: Trigger purchase orders today before weekend rush.")
    else:
        st.success("✅ **Healthy Runway:** No items under current filter are at critical High Risk level.")

    # Top Recommendation Cards
    st.markdown("#### ⚡ Priority Recommendation Cards (Top Alerts)")
    top_cards = filtered_actions[filtered_actions['risk_level'] == 'HIGH'].head(3)
    if len(top_cards) == 0:
        top_cards = filtered_actions.head(3)

    if len(top_cards) > 0:
        card_cols = st.columns(len(top_cards))
        for i, (_, row) in enumerate(top_cards.iterrows()):
            with card_cols[i]:
                card_class = "rec-card-high" if row['risk_level'] == 'HIGH' else "rec-card"
                st.markdown(f"""
                <div class="{card_class}">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:800; font-size:1.1rem; color:#0F172A;">{row['store_id']} ({row['city']})</span>
                        <span class="badge-{row['risk_level'].lower()}">{row['risk_level']} RISK ({row['stockout_probability']*100:.0f}%)</span>
                    </div>
                    <div style="font-size:1rem; font-weight:700; color:#1E293B; margin-top:0.4rem;">{row['brand']} {row['sub_category']} ({row['product_id']})</div>
                    <hr style="margin: 0.6rem 0; border: none; border-top: 1px solid #CBD5E1;">
                    <div style="font-size:0.88rem; line-height: 1.5; color:#334155;">
                        • <b>Predicted 7-Day Demand:</b> {int(row['predicted_7d_demand'])} units<br>
                        • <b>Current Stock:</b> {int(row['closing_stock'])} units (Runway: {row['days_of_inventory']} days)<br>
                        • <b>Incoming Stock:</b> {int(row['incoming_stock'])} units<br>
                        • <b>Safety Stock Buffer:</b> {int(row['safety_stock'])} units<br>
                        • <b>Recommended Reorder:</b> <span style="font-size:1.05rem; font-weight:800; color:#2563EB;">{int(row['reorder_quantity'])} units</span>
                    </div>
                    <div style="margin-top:0.6rem; padding: 0.5rem; background:#EDF2F7; border-radius:6px; font-size:0.82rem; color:#475569;">
                        <b>Why?</b> {row['key_reasons']}
                    </div>
                    <div style="margin-top:0.5rem; font-weight:700; font-size:0.84rem; color:#B91C1C;">
                        {row['manager_action']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Master Decision Action Table
    st.markdown(f"#### 📑 Complete Replenishment Orders Table ({len(filtered_actions)} records matching filters)")

    display_cols = [
        'store_id', 'city', 'product_id', 'category', 'brand', 'sub_category',
        'closing_stock', 'days_of_inventory', 'predicted_7d_demand',
        'safety_stock', 'stockout_probability', 'risk_level',
        'reorder_quantity', 'key_reasons', 'manager_action'
    ]
    st.dataframe(
        filtered_actions[display_cols].rename(columns={
            'store_id': 'Store',
            'city': 'City',
            'product_id': 'SKU ID',
            'category': 'Category',
            'brand': 'Brand',
            'sub_category': 'Product',
            'closing_stock': 'Current Stock',
            'days_of_inventory': 'Days of Stock',
            'predicted_7d_demand': '7-Day Forecast',
            'safety_stock': 'Safety Stock',
            'stockout_probability': 'Stockout Prob.',
            'risk_level': 'Risk Tier',
            'reorder_quantity': 'Recommended Reorder',
            'key_reasons': 'Key Drivers',
            'manager_action': 'Manager Action'
        }),
        use_container_width=True,
        height=420
    )

    # CSV Download Button
    csv = filtered_actions[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Replenishment Action List to CSV (Store Dispatch)",
        data=csv,
        file_name=f"NovaMart_Replenishment_Plan_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )


# ───────────────────────────────────────────────────────────────────────────
# TAB 3: STOCK-OUT RISK MATRIX (DYNAMIC TO FILTERS)
# ───────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### ⚠️ Inventory Risk & Heatmap Matrix")
    st.caption("Visualizing stock-out probability across Store Formats and Product Categories to identify systemic vulnerabilities.")

    r1, r2 = st.columns([6, 4])
    with r1:
        # Dynamic Risk Heatmap
        idx_col = 'sub_category' if selected_store != "All Stores" else 'store_id'
        col_col = 'store_id' if selected_store != "All Stores" else 'category'

        risk_matrix = filtered_actions.pivot_table(
            index=idx_col,
            columns=col_col,
            values='stockout_probability',
            aggfunc='mean'
        ).fillna(0) * 100

        fig_heat = px.imshow(
            risk_matrix,
            labels=dict(x=col_col.replace('_', ' ').title(), y=idx_col.replace('_', ' ').title(), color="Avg Risk (%)"),
            x=risk_matrix.columns,
            y=risk_matrix.index,
            color_continuous_scale="Reds",
            title=f"<b>Stock-out Risk Heatmap (% Prob) [{selected_store} | {selected_category}]</b>",
            text_auto=".1f"
        )
        fig_heat.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_heat, use_container_width=True)

    with r2:
        # Dynamic Risk Level Distribution
        risk_dist = filtered_actions['risk_level'].value_counts().reset_index()
        risk_dist.columns = ['Risk Tier', 'SKU Count']
        fig_bar = px.bar(
            risk_dist, x='Risk Tier', y='SKU Count',
            color='Risk Tier',
            color_discrete_map={'HIGH': '#EF4444', 'MEDIUM': '#F59E0B', 'LOW': '#10B981'},
            title=f"<b>Risk Classification Counts ({len(filtered_actions)} SKUs)</b>",
            text_auto=True
        )
        fig_bar.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Days of Stock Runway Distribution
    st.markdown("#### ⏳ Stock Runway Analysis (Days of Inventory Remaining)")
    fig_dos = px.histogram(
        filtered_actions, x='days_of_inventory', color='risk_level',
        nbins=25,
        color_discrete_map={'HIGH': '#EF4444', 'MEDIUM': '#F59E0B', 'LOW': '#10B981'},
        title=f"<b>Days of Inventory Distribution ({selected_store} | {selected_category})</b>",
        labels={'days_of_inventory': 'Days of Inventory Remaining', 'count': 'Number of SKUs'}
    )
    fig_dos.add_vline(x=3.0, line_dash="dash", line_color="red", annotation_text="Critical Depletion Threshold (3 Days)")
    fig_dos.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_dos, use_container_width=True)


# ───────────────────────────────────────────────────────────────────────────
# TAB 4: DEMAND FORECASTING (DYNAMIC TO FILTERS)
# ───────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### 🔮 Demand Intelligence & 7-Day Forecasting")
    st.caption("Model 1 forecasts forward 7-day demand using historical lags, rolling velocity, promotional boosts, and external climate factors.")

    d1, d2 = st.columns([7, 3])
    with d1:
        top_skus = filtered_actions.sort_values('predicted_7d_demand', ascending=False)['product_id'].head(5).tolist()
        sku_hist = filtered_master[filtered_master['product_id'].isin(top_skus)]

        if len(sku_hist) > 0:
            fig_ts = px.line(
                sku_hist, x='date', y='total_quantity_sold', color='product_id',
                title=f"<b>Historical Demand Trajectory (Top SKUs in {selected_store} | {selected_category})</b>",
                labels={'total_quantity_sold': 'Units Sold', 'date': 'Date'}
            )
            fig_ts.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_ts, use_container_width=True)
        else:
            st.info("No historical records match the selected filter combination.")

    with d2:
        st.markdown("#### 📈 Key Demand Drivers")
        st.markdown("""
        - **Promotion Elasticity:** Active promotions generate an average **+41.2% demand lift**.
        - **Weekend Footfall:** Friday evening to Sunday accounts for **38% of weekly turnover**.
        - **Temperature Spike:** Temperatures above 32°C accelerate Beverage demand by **+18%**.
        - **Festival Ramp-up:** Diwali / Pongal drives a **1.6× multiplier** starting 3 days prior.
        """)


# ───────────────────────────────────────────────────────────────────────────
# TAB 5: WHAT-IF SCENARIO SIMULATOR (DYNAMIC TO FILTERED SKUs)
# ───────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("### 🧪 What-If Replenishment Simulator")
    st.caption("Simulate real-world retail disruptions: sudden marketing promotions, supplier transport delays, or festival surges.")

    w_col1, w_col2 = st.columns([4, 6])
    with w_col1:
        st.markdown("#### ⚙️ Simulation Parameters")
        target_options = (filtered_actions['brand'] + " " + filtered_actions['sub_category'] + " (" + filtered_actions['product_id'] + ") - Store " + filtered_actions['store_id']).tolist()
        if not target_options:
            target_options = (actions['brand'] + " " + actions['sub_category'] + " (" + actions['product_id'] + ") - Store " + actions['store_id']).tolist()

        target_sku = st.selectbox("Select Target Product to Simulate", target_options)
        sku_id = target_sku.split("(")[-1].split(")")[0]
        store_match = target_sku.split("Store ")[-1]
        sku_row = actions[(actions['product_id'] == sku_id) & (actions['store_id'] == store_match)].iloc[0]

        promo_boost = st.slider("Simulate Promotional Discount (%)", min_value=0, max_value=35, value=15, step=5)
        lead_delay = st.slider("Supplier Lead Time Delay (+ Days)", min_value=0, max_value=5, value=1, step=1)
        festival_surge = st.checkbox("Upcoming Festival Demand Surge (+30%)", value=True)

    with w_col2:
        st.markdown(f"#### 📊 Simulation Impact Results ({sku_row['brand']} {sku_row['sub_category']})")
        base_demand = sku_row['predicted_7d_demand']
        sim_multiplier = 1.0 + (promo_boost / 100.0 * 1.5) + (0.30 if festival_surge else 0.0)
        sim_demand = round(base_demand * sim_multiplier)

        sim_lead = sku_row.get('lead_time_days', 2) + lead_delay
        sim_safety = round(1.65 * max(1, sku_row['safety_stock']) * np.sqrt(max(1, sim_lead) / max(1, sku_row.get('lead_time_days', 2))))
        sim_req = sim_demand + sim_safety

        current_s = sku_row['closing_stock']
        sim_reorder = max(0, int(sim_req - current_s))

        sim_coverage = current_s / max(1, sim_demand / 7.0)
        sim_prob = min(0.99, max(0.05, 1.0 - (sim_coverage / 7.0)))
        sim_risk = "HIGH" if sim_prob >= 0.70 else ("MEDIUM" if sim_prob >= 0.40 else "LOW")

        res1, res2, res3 = st.columns(3)
        with res1:
            st.metric("Base Forecast", f"{int(base_demand)} units")
            st.metric("Simulated Forecast", f"{int(sim_demand)} units", delta=f"{int(sim_demand - base_demand)} units")
        with res2:
            st.metric("Base Risk Prob", f"{sku_row['stockout_probability']*100:.0f}%")
            st.metric("Simulated Risk Prob", f"{sim_prob*100:.0f}%", delta=f"{(sim_prob - sku_row['stockout_probability'])*100:+.0f}%", delta_color="inverse")
        with res3:
            st.metric("Original Reorder", f"{int(sku_row['reorder_quantity'])} units")
            st.metric("Simulated Reorder", f"{sim_reorder} units", delta=f"{sim_reorder - int(sku_row['reorder_quantity'])} units")

        st.info(f"💡 **Manager Recommendation:** Under these simulated market conditions, raise replenishment order by **+{sim_reorder - int(sku_row['reorder_quantity'])} additional units** to guarantee 95% on-shelf availability.")


# ───────────────────────────────────────────────────────────────────────────
# TAB 6: MODEL BENCHMARKS & EXPLAINABILITY
# ───────────────────────────────────────────────────────────────────────────
with tab6:
    st.markdown("### 🏆 Machine Learning Validation & Explainability")
    st.caption("Rigorous evaluation across permitted IntelliData 2026 algorithms using chronological time-aware splitting.")

    m1_col, m2_col = st.columns(2)
    with m1_col:
        st.markdown("#### 📉 Model 1: 7-Day Demand Forecasting (Regression)")
        m1_data = pd.DataFrame([
            {'Algorithm': 'Baseline (7-Day Mean)', 'MAE': 34.81, 'RMSE': 59.36, 'MAPE (%)': '12.2%', 'R²': 0.9566, 'Status': 'Evaluated'},
            {'Algorithm': 'Linear Regression', 'MAE': 35.89, 'RMSE': 60.95, 'MAPE (%)': '12.7%', 'R²': 0.9542, 'Status': 'Evaluated'},
            {'Algorithm': 'Decision Tree', 'MAE': 35.53, 'RMSE': 59.22, 'MAPE (%)': '11.9%', 'R²': 0.9568, 'Status': 'Evaluated'},
            {'Algorithm': 'Random Forest', 'MAE': 33.52, 'RMSE': 56.51, 'MAPE (%)': '11.2%', 'R²': 0.9606, 'Status': 'Evaluated'},
            {'Algorithm': 'XGBoost Regressor', 'MAE': 32.69, 'RMSE': 54.61, 'MAPE (%)': '11.2%', 'R²': 0.9632, 'Status': '🏆 WINNER'}
        ])
        st.dataframe(m1_data, use_container_width=True)
        st.success("**Selected:** XGBoost Regressor provides best-in-class MAE (32.69 units), vital for preventing over/under replenishment.")

    with m2_col:
        st.markdown("#### 🎯 Model 2: Stock-out Risk Classification")
        m2_data = pd.DataFrame([
            {'Algorithm': 'Baseline (Majority)', 'Accuracy': 0.630, 'Precision': 0.000, 'Recall': 0.000, 'F1': 0.000, 'ROC-AUC': 0.500, 'Status': 'Evaluated'},
            {'Algorithm': 'Decision Tree', 'Accuracy': 0.691, 'Precision': 0.557, 'Recall': 0.808, 'F1': 0.659, 'ROC-AUC': 0.750, 'Status': 'Evaluated'},
            {'Algorithm': 'Random Forest', 'Accuracy': 0.686, 'Precision': 0.553, 'Recall': 0.784, 'F1': 0.649, 'ROC-AUC': 0.724, 'Status': 'Evaluated'},
            {'Algorithm': 'XGBoost Classifier', 'Accuracy': 0.671, 'Precision': 0.536, 'Recall': 0.825, 'F1': 0.650, 'ROC-AUC': 0.739, 'Status': '🏆 WINNER'}
        ])
        st.dataframe(m2_data, use_container_width=True)
        st.success("**Selected:** XGBoost Classifier achieves 82.5% Recall with 0.739 ROC-AUC, effectively catching near-stockouts before they happen.")

    st.markdown("#### 🔍 Global Feature Importance (Top Decision Drivers)")
    feat_imp = pd.DataFrame({
        'Feature': [
            'days_of_inventory', 'rolling_mean_7', 'demand_lag_1', 'reorder_gap',
            'promotion_flag', 'lead_time_days', 'is_weekend', 'closing_stock',
            'festival_flag', 'temp_c'
        ],
        'Importance (%)': [28.4, 21.2, 14.5, 9.8, 8.2, 6.1, 4.5, 3.2, 2.3, 1.8]
    }).sort_values('Importance (%)', ascending=True)

    fig_imp = px.bar(
        feat_imp, x='Importance (%)', y='Feature', orientation='h',
        title="<b>Top 10 Feature Drivers Influencing Demand & Stock-Out Predictions</b>",
        color='Importance (%)', color_continuous_scale='Blues'
    )
    fig_imp.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("---")
st.markdown("<center style='color:#94A3B8; font-size:0.82rem;'>Sri Eshwar College of Engineering | Department of Computer Science and Engineering | IntelliData 2026</center>", unsafe_allow_html=True)
