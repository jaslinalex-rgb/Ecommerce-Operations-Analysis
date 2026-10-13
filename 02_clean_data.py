"""
02_clean_data.py
Cleans data/raw/ecommerce_orders_raw.csv -> data/cleaned/ecommerce_orders_clean.csv
Every step logs how many rows/values it touched, saved to data/cleaned/cleaning_log.md
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "ecommerce_orders_raw.csv"
OUT = ROOT / "data" / "cleaned" / "ecommerce_orders_clean.csv"
LOG = ROOT / "data" / "cleaned" / "cleaning_log.md"

log = []
def note(step, detail):
    log.append((step, detail))
    print(f"[{step}] {detail}")

df = pd.read_csv(RAW, dtype=str)
note("Load", f"{len(df):,} rows, {df.shape[1]} columns")

# 1. exact duplicates ---------------------------------------------------------
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
note("Duplicates", f"removed {before - len(df):,} exact duplicate rows")
assert df["order_id"].is_unique, "order_id should be unique after de-duplication"

# 2. whitespace + placeholder nulls -------------------------------------------
for c in df.columns:
    df[c] = df[c].str.strip()
df = df.replace({"N/A": np.nan, "": np.nan})

# 3. numeric types ------------------------------------------------------------
for c in ["quantity", "unit_price", "discount_pct", "shipping_cost"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# 4. dates (3 mixed formats) --------------------------------------------------
def parse_dates(s):
    out = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    for fmt in ("%d/%m/%Y", "%b %d, %Y"):
        miss = out.isna() & s.notna()
        out[miss] = pd.to_datetime(s[miss], format=fmt, errors="coerce")
    return out

df["order_date"] = parse_dates(df["order_date"])
df["delivery_date"] = parse_dates(df["delivery_date"])
unparsed = df["order_date"].isna().sum()
note("Dates", f"standardised 3 date formats to YYYY-MM-DD; unparseable order dates: {unparsed}")
df = df.dropna(subset=["order_date"])

# 5. text standardisation -----------------------------------------------------
city_map = {"Bangalore": "Bengaluru", "Madras": "Chennai", "Bombay": "Mumbai"}
df["city"] = df["city"].str.title().replace(city_map)
n = df["city"].isna().sum()
df["city"] = df["city"].fillna("Unknown")
note("City", f"fixed case/spacing, mapped old city names (Bangalore, Madras, Bombay); {n:,} missing -> 'Unknown'")

pay_map = {"Upi": "UPI", "Cod": "Cash on Delivery"}
df["payment_method"] = df["payment_method"].str.title().replace(pay_map)
df["payment_method"] = df["payment_method"].replace({"Cash On Delivery": "Cash on Delivery"})
n = df["payment_method"].isna().sum()
df["payment_method"] = df["payment_method"].fillna("Unknown")
note("Payment", f"unified spellings (UPI / COD variants); {n:,} missing -> 'Unknown'")

df["order_status"] = df["order_status"].str.title()
df["category"] = df["category"].str.replace(" And ", " & ", regex=False).str.replace(" and ", " & ", regex=False)

# category: recover from product (each product belongs to exactly one category)
prod_cat = df.dropna(subset=["category"]).groupby("product")["category"].agg(lambda s: s.mode()[0])
miss = df["category"].isna()
df.loc[miss, "category"] = df.loc[miss, "product"].map(prod_cat)
note("Category", f"recovered {miss.sum() - df['category'].isna().sum():,} missing/'N/A' categories from product lookup; "
                 f"{df['category'].isna().sum()} still missing")

# customer name: recover from customer_id, else Unknown
df["customer_name"] = df["customer_name"].str.title()
cust_name = df.dropna(subset=["customer_name"]).groupby("customer_id")["customer_name"].agg(lambda s: s.mode()[0])
miss = df["customer_name"].isna()
df.loc[miss, "customer_name"] = df.loc[miss, "customer_id"].map(cust_name)
left = df["customer_name"].isna().sum()
df["customer_name"] = df["customer_name"].fillna("Unknown")
note("Customer name", f"recovered {miss.sum() - left:,} from customer_id; {left} -> 'Unknown'")

# 6. invalid / missing numerics -----------------------------------------------
# quantity: missing or <= 0 cannot be recovered -> drop
bad_q = df["quantity"].isna() | (df["quantity"] <= 0)
note("Quantity", f"dropped {bad_q.sum():,} rows with missing or negative quantity")
df = df[~bad_q].copy()
df["quantity"] = df["quantity"].astype(int)

# unit_price: outliers (>5x product median, typo of x100) and missing -> product median
med = df.groupby("product")["unit_price"].transform("median")
outlier = df["unit_price"] > 5 * med
missing_price = df["unit_price"].isna()
df.loc[outlier | missing_price, "unit_price"] = med[outlier | missing_price]
note("Unit price", f"{outlier.sum():,} extreme outliers and {missing_price.sum():,} missing prices replaced with product median")
df["unit_price"] = df["unit_price"].round(2)

# discount: invalid (>100) or missing -> 0
bad_d = (df["discount_pct"] > 100) | (df["discount_pct"] < 0)
miss_d = df["discount_pct"].isna()
df.loc[bad_d | miss_d, "discount_pct"] = 0
note("Discount", f"{bad_d.sum():,} invalid (>100%) and {miss_d.sum():,} missing discounts set to 0 (assumed no discount)")

# delivery before order date -> invalid, blank it out
bad_dd = df["delivery_date"].notna() & (df["delivery_date"] < df["order_date"])
df.loc[bad_dd, "delivery_date"] = pd.NaT
note("Delivery date", f"{bad_dd.sum():,} delivery dates earlier than order date set to blank")

# 7. derived columns ----------------------------------------------------------
df["gross_amount"] = (df["quantity"] * df["unit_price"]).round(2)
df["discount_amount"] = (df["gross_amount"] * df["discount_pct"] / 100).round(2)
df["net_revenue"] = (df["gross_amount"] - df["discount_amount"]).round(2)
df["realised_revenue"] = np.where(df["order_status"] == "Delivered", df["net_revenue"], 0.0)
df["delivery_days"] = (df["delivery_date"] - df["order_date"]).dt.days
df["order_year"] = df["order_date"].dt.year
df["order_month"] = df["order_date"].dt.to_period("M").astype(str)
df["order_quarter"] = df["order_date"].dt.year.astype(str) + "-Q" + df["order_date"].dt.quarter.astype(str)
note("Derived", "added gross_amount, discount_amount, net_revenue, realised_revenue (Delivered only), "
                "delivery_days, order_year, order_month, order_quarter")

df = df.sort_values("order_date").reset_index(drop=True)
for c in ["order_date", "delivery_date"]:
    df[c] = df[c].dt.strftime("%Y-%m-%d")

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
note("Save", f"{len(df):,} clean rows -> {OUT.relative_to(ROOT)}")

# remaining-null check
nulls = df.isna().sum()
nulls = nulls[nulls > 0]
note("Null check", "remaining nulls: " + (", ".join(f"{k}={v:,}" for k, v in nulls.items()) or "none")
     + " (delivery_date blank is expected for Cancelled/Shipped orders)")

with open(LOG, "w") as f:
    f.write("# Cleaning log\n\n| Step | What was done |\n|---|---|\n")
    for s, d in log:
        f.write(f"| {s} | {d} |\n")
