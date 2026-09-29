"""
=============================================================================
IntelliData 2026 — StockSense — Fast Synthetic Data Generator
=============================================================================
Generates realistic retail data matching EXACT hackathon schemas:
  • transactions.csv     (Transaction-level sales)
  • products.csv         (Product master with category, brand, supplier)
  • stores.csv           (Store master with type, region, floor area)
  • inventory.csv        (Daily stock snapshots per store x product)
  • external_factors.csv (Weather, holidays, festivals, events)

Includes Intentional Data Quality Traps:
  1. Missing values (temperature = blank, selling_price = blank, customer_id = blank)
  2. Duplicate transactions (identical transaction_id repeated)
  3. Category inconsistencies (Beverages / beverage / BEVERAGES)
  4. Impossible quantity (quantity = -4, <= 0)
  5. Inventory mismatch (closing != opening + received - sold)
  6. Sparse history (new products with few days of data: P901, P902)
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

np.random.seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. STORES MASTER
stores = pd.DataFrame({
    'store_id':           ['S01', 'S02', 'S03', 'S04', 'S05', 'S06'],
    'city':               ['Coimbatore', 'Chennai', 'Madurai', 'Salem', 'Trichy', 'Erode'],
    'store_type':         ['Supermarket', 'Hypermarket', 'Express', 'Supermarket', 'Express', 'Supermarket'],
    'floor_area_sqft':    [8500, 18000, 4200, 7600, 3800, 6500],
    'avg_daily_customers': [1250, 2850, 720, 1080, 650, 950],
    'region':             ['West', 'North', 'South', 'Central', 'South', 'West'],
})

# 2. PRODUCTS MASTER
product_data = [
    ('P101', 'Dairy', 'Milk', 'Aavin', 60, 52, 3, 'SUP01'),
    ('P102', 'Dairy', 'Curd', 'Aavin', 45, 38, 5, 'SUP01'),
    ('P103', 'Dairy', 'Paneer', 'Amul', 95, 72, 7, 'SUP01'),
    ('P104', 'Dairy', 'Butter', 'Amul', 55, 42, 30, 'SUP01'),
    ('P105', 'Dairy', 'Cheese', 'Britannia', 120, 85, 60, 'SUP01'),
    ('P106', 'Dairy', 'Yogurt', 'Epigamia', 40, 28, 14, 'SUP01'),

    ('P201', 'Beverages', 'Soft Drink', 'FizzUp', 40, 28, 180, 'SUP02'),
    ('P202', 'Beverages', 'Juice', 'Tropicana', 85, 62, 120, 'SUP02'),
    ('P203', 'Beverages', 'Water', 'Bisleri', 20, 12, 365, 'SUP02'),
    ('P204', 'Beverages', 'Tea', 'TajMahal', 180, 135, 365, 'SUP03'),
    ('P205', 'Personal Care', 'Shampoo', 'GlowCare', 120, 79, 730, 'SUP08'),
    ('P206', 'Beverages', 'Coffee', 'Nescafe', 210, 155, 365, 'SUP03'),
    ('P207', 'Beverages', 'Energy Drink', 'Sting', 50, 35, 270, 'SUP02'),

    ('P301', 'Snacks', 'Chips', 'Lays', 30, 20, 120, 'SUP04'),
    ('P302', 'Snacks', 'Biscuits', 'Britannia', 35, 24, 180, 'SUP04'),
    ('P303', 'Snacks', 'Namkeen', 'Haldirams', 60, 42, 150, 'SUP04'),
    ('P304', 'Snacks', 'Chocolate', 'DairyMilk', 50, 38, 270, 'SUP05'),
    ('P305', 'Snacks', 'Biscuits', 'Crispo', 35, 22, 240, 'SUP06'),
    ('P306', 'Snacks', 'Nuts', 'Happilo', 150, 110, 180, 'SUP04'),
    ('P330', 'Beverages', 'Soft Drink', 'FizzUp', 50, 31, 180, 'SUP02'),

    ('P401', 'Staples', 'Rice', 'IndiaGate', 320, 265, 365, 'SUP07'),
    ('P402', 'Staples', 'Wheat Flour', 'Aashirvaad', 250, 200, 180, 'SUP07'),
    ('P403', 'Staples', 'Sugar', 'Dhampure', 55, 45, 365, 'SUP07'),
    ('P404', 'Staples', 'Salt', 'TataSalt', 25, 18, 730, 'SUP07'),
    ('P405', 'Staples', 'Cooking Oil', 'Fortune', 175, 140, 365, 'SUP07'),
    ('P406', 'Staples', 'Dal Toor', 'TataSampann', 140, 110, 180, 'SUP07'),
    ('P407', 'Staples', 'Rava', 'Sakthi', 70, 55, 180, 'SUP07'),
    ('P442', 'Snacks', 'Biscuits', 'Crispo', 35, 22, 240, 'SUP06'),

    ('P501', 'Personal Care', 'Soap', 'Lux', 40, 28, 730, 'SUP08'),
    ('P502', 'Personal Care', 'Toothpaste', 'Colgate', 95, 68, 730, 'SUP08'),
    ('P503', 'Personal Care', 'Face Wash', 'Himalaya', 110, 78, 365, 'SUP08'),
    ('P504', 'Personal Care', 'Deodorant', 'Nivea', 180, 125, 730, 'SUP08'),
    ('P505', 'Personal Care', 'Hand Wash', 'Dettol', 75, 52, 365, 'SUP08'),

    ('P601', 'Household', 'Detergent', 'Surf', 210, 155, 730, 'SUP09'),
    ('P602', 'Household', 'Dishwash', 'Vim', 55, 38, 365, 'SUP09'),
    ('P603', 'Household', 'Floor Cleaner', 'Lizol', 130, 92, 365, 'SUP09'),
    ('P604', 'Household', 'Toilet Cleaner', 'Harpic', 95, 65, 365, 'SUP09'),
    ('P605', 'Household', 'Garbage Bags', 'Ezee', 80, 55, 730, 'SUP09'),

    ('P701', 'Frozen Foods', 'Frozen Peas', 'Safal', 85, 60, 180, 'SUP10'),
    ('P702', 'Frozen Foods', 'Ice Cream', 'Amul', 120, 80, 90, 'SUP10'),
    ('P703', 'Frozen Foods', 'Frozen Paratha', 'McCain', 95, 68, 120, 'SUP10'),
    ('P704', 'Frozen Foods', 'Chicken Nuggets', 'Zorabian', 180, 130, 90, 'SUP10'),

    ('P801', 'Bakery', 'Bread', 'ModernBread', 45, 32, 3, 'SUP11'),
    ('P802', 'Bakery', 'Cake', 'Britannia', 60, 42, 7, 'SUP11'),
    ('P803', 'Bakery', 'Rusk', 'Britannia', 50, 35, 90, 'SUP11'),

    # Sparse history products
    ('P901', 'Beverages', 'Kombucha', 'BrewJoy', 150, 105, 30, 'SUP12'),
    ('P902', 'Personal Care', 'Sunscreen', 'Neutrogena', 350, 250, 365, 'SUP08'),
]

products = pd.DataFrame(product_data, columns=[
    'product_id', 'category', 'sub_category', 'brand', 'mrp', 'cost_price',
    'shelf_life_days', 'supplier_id'
])

# 3. EXTERNAL FACTORS & DATES
NUM_DAYS = 90
START_DATE = datetime(2026, 8, 1)
dates = [START_DATE + timedelta(days=d) for d in range(NUM_DAYS)]

HOLIDAYS = {'2026-08-15', '2026-08-26', '2026-09-02', '2026-09-17', '2026-10-02', '2026-10-20', '2026-10-22'}
FESTIVALS = {'2026-08-25', '2026-09-02', '2026-10-01', '2026-10-20', '2026-10-22', '2026-10-25'}
CITY_BASE_TEMP = {'Coimbatore': 28.5, 'Chennai': 33.2, 'Madurai': 34.0, 'Salem': 30.5, 'Trichy': 32.0, 'Erode': 31.0}

external_factors = []
for d in dates:
    d_str = d.strftime('%Y-%m-%d')
    for city in stores['city']:
        base_t = CITY_BASE_TEMP[city]
        t = round(base_t + np.random.normal(0, 2.0), 1)
        # Trap 1: missing temperature (~5%)
        if np.random.random() < 0.05:
            t = np.nan
        r = round(max(0, np.random.exponential(3) if np.random.random() < 0.25 else 0), 1)
        external_factors.append({
            'date': d_str,
            'city': city,
            'temp_c': t,
            'rain_mm': r,
            'holiday': 1 if d_str in HOLIDAYS else 0,
            'festival': 1 if d_str in FESTIVALS else 0,
            'weekend': 1 if d.weekday() >= 5 else 0,
            'local_event': 1 if np.random.random() < 0.05 else 0
        })

external_df = pd.DataFrame(external_factors)
ext_map = {(r['date'], r['city']): r for r in external_factors}

# Base Category Demands
CAT_BASE = {
    'Dairy': 48, 'Beverages': 42, 'Snacks': 40, 'Staples': 22,
    'Personal Care': 14, 'Household': 12, 'Frozen Foods': 16, 'Bakery': 28
}

SPARSE_PRODUCTS = {'P901': datetime(2026, 10, 15), 'P902': datetime(2026, 10, 10)}

print("Generating NovaMart Retail sales & inventory data...")
stores_records = stores.to_dict('records')
products_records = products.to_dict('records')

# Initialize inventory
inv_state = {}
for s in stores_records:
    sid = s['store_id']
    cf = s['avg_daily_customers'] / 1000.0
    for p in products_records:
        pid = p['product_id']
        base = max(3, int(CAT_BASE.get(p['category'], 15) * cf * np.random.uniform(0.7, 1.3)))
        inv_state[(sid, pid)] = {
            'stock': int(base * np.random.uniform(4, 8)),
            'reorder_lvl': int(base * np.random.uniform(1.8, 2.5)),
            'lead_days': int(np.random.choice([1, 2, 3, 4])),
            'base': base,
            'due_orders': []
        }

txns = []
invs = []
txn_id = 10000

for d in dates:
    d_str = d.strftime('%Y-%m-%d')
    dow = d.weekday()
    is_wknd = 1 if dow >= 5 else 0

    for s in stores_records:
        sid = s['store_id']
        city = s['city']
        ext = ext_map.get((d_str, city), {})
        is_fest = ext.get('festival', 0)
        is_hol = ext.get('holiday', 0)
        temp = ext.get('temp_c', 30.0)
        if pd.isna(temp):
            temp = CITY_BASE_TEMP[city]

        for p in products_records:
            pid = p['product_id']
            if pid in SPARSE_PRODUCTS and d < SPARSE_PRODUCTS[pid]:
                continue

            st = inv_state[(sid, pid)]
            # Process deliveries
            received = 0
            new_due = []
            for due_d, qty in st['due_orders']:
                if due_d <= d:
                    received += qty
                else:
                    new_due.append((due_d, qty))
            st['due_orders'] = new_due

            opening = st['stock'] + received

            # Demand calculation
            dmd = float(st['base'])
            if is_wknd:
                dmd *= np.random.uniform(1.2, 1.45)
            if dow == 0:
                dmd *= 0.85
            if is_fest or is_hol:
                dmd *= np.random.uniform(1.3, 1.8)
            if p['category'] in ['Beverages', 'Dairy', 'Frozen Foods'] and temp > 32:
                dmd *= 1.15

            # Promo
            is_promo = 1 if np.random.random() < 0.15 else 0
            disc = int(np.random.choice([10, 15, 20])) if is_promo else 0
            if is_promo:
                dmd *= (1 + disc / 50.0)

            dmd = max(0, int(dmd + np.random.normal(0, max(1, st['base'] * 0.15))))
            sold = min(dmd, opening)
            closing = opening - sold
            st['stock'] = closing

            # Record daily transactions
            # Generate 1 to 4 transactions per item to simulate realistic POS receipts
            if sold > 0:
                sp = round(p['mrp'] * (1 - disc / 100.0), 2)
                splits = np.random.choice([1, 2, 3])
                sub_qtys = np.random.multinomial(sold, [1/splits]*splits)
                for q in sub_qtys:
                    if q > 0:
                        txn_id += 1
                        txns.append({
                            'transaction_id': f'T{txn_id}',
                            'date': d_str,
                            'store_id': sid,
                            'product_id': pid,
                            'quantity': int(q),
                            'selling_price': sp,
                            'discount_pct': disc,
                            'promotion_flag': is_promo,
                            'customer_id': f'C{np.random.randint(1001, 2500):04d}',
                            'payment_mode': np.random.choice(['UPI', 'Card', 'Cash'], p=[0.5, 0.3, 0.2]),
                            'hour': np.random.choice([9, 10, 11, 12, 14, 16, 17, 18, 19, 20, 21])
                        })

            # Inventory snapshot
            # Trap 5: inventory mismatch (~2% of rows)
            recorded_closing = closing
            if np.random.random() < 0.02:
                recorded_closing = max(0, closing + np.random.choice([-4, -2, 3, 5]))

            invs.append({
                'date': d_str,
                'store_id': sid,
                'product_id': pid,
                'opening_stock': opening,
                'quantity_received': received,
                'quantity_sold': sold,
                'closing_stock': recorded_closing,
                'reorder_level': st['reorder_lvl'],
                'lead_time_days': st['lead_days']
            })

            # Reorder trigger
            if closing < st['reorder_lvl'] and len(st['due_orders']) == 0:
                ord_qty = max(20, int(st['base'] * np.random.uniform(4, 7)))
                arrival = d + timedelta(days=st['lead_days'])
                st['due_orders'].append((arrival, ord_qty))

txns_df = pd.DataFrame(txns)
invs_df = pd.DataFrame(invs)

# INJECT HACKATHON DATA QUALITY TRAPS
print("Injecting intentional data quality traps...")

# Trap 2: Duplicates (~1.5%)
dupe_n = int(len(txns_df) * 0.015)
dupes = txns_df.iloc[np.random.choice(len(txns_df), dupe_n, replace=False)].copy()
txns_df = pd.concat([txns_df, dupes], ignore_index=True)

# Trap 3: Category casing inconsistencies
trap_cats = {'Beverages': 'beverage', 'Dairy': 'DAIRY', 'Snacks': 'snacks', 'Household': 'houseHOLD'}
for idx, r in products.iterrows():
    if r['category'] in trap_cats and np.random.random() < 0.3:
        products.loc[idx, 'category'] = trap_cats[r['category']]

# Trap 4: Impossible negative / zero quantities
neg_idx = np.random.choice(len(txns_df), int(len(txns_df) * 0.005), replace=False)
txns_df.loc[neg_idx, 'quantity'] = np.random.choice([-4, -2, -1, 0], len(neg_idx))

# Missing selling prices (~2%)
null_p = np.random.choice(len(txns_df), int(len(txns_df) * 0.02), replace=False)
txns_df.loc[null_p, 'selling_price'] = np.nan

# Missing customer ID (~3%)
null_c = np.random.choice(len(txns_df), int(len(txns_df) * 0.03), replace=False)
txns_df.loc[null_c, 'customer_id'] = np.nan

# Save CSVs
stores.to_csv(os.path.join(OUTPUT_DIR, 'stores.csv'), index=False)
products.to_csv(os.path.join(OUTPUT_DIR, 'products.csv'), index=False)
external_df.to_csv(os.path.join(OUTPUT_DIR, 'external_factors.csv'), index=False)
txns_df.to_csv(os.path.join(OUTPUT_DIR, 'transactions.csv'), index=False)
invs_df.to_csv(os.path.join(OUTPUT_DIR, 'inventory.csv'), index=False)

print(f"Data Generation Finished!")
print(f"   Stores:           {len(stores):>6}")
print(f"   Products:         {len(products):>6}")
print(f"   External Factors: {len(external_df):>6}")
print(f"   Transactions:     {len(txns_df):>6,}")
print(f"   Inventory:        {len(invs_df):>6,}")
