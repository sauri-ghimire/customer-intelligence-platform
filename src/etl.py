# src/etl.py
# ETL Pipeline — Customer Intelligence Platform
# Ingests all 9 Olist CSVs, cleans them, merges into master tables, loads to SQLite

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
import logging

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = r"D:\customer_intelligence_platform"
RAW_DIR     = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR    = os.path.join(BASE_DIR, "data", "processed")
DB_PATH     = os.path.join(PROC_DIR, "olist.db")

os.makedirs(PROC_DIR, exist_ok=True)

# ── 1. LOAD ────────────────────────────────────────────────────────────────────
def load_raw_data():
    log.info("Loading raw CSVs...")
    files = {
        "orders":         "olist_orders_dataset.csv",
        "order_items":    "olist_order_items_dataset.csv",
        "order_payments": "olist_order_payments_dataset.csv",
        "order_reviews":  "olist_order_reviews_dataset.csv",
        "customers":      "olist_customers_dataset.csv",
        "products":       "olist_products_dataset.csv",
        "sellers":        "olist_sellers_dataset.csv",
        "category_trans": "product_category_name_translation.csv",
        "geolocation":    "olist_geolocation_dataset.csv",
    }
    dfs = {}
    for name, fname in files.items():
        path = os.path.join(RAW_DIR, fname)
        dfs[name] = pd.read_csv(path)
        log.info(f"  {name:20s} {dfs[name].shape}")
    return dfs

# ── 2. CLEAN ───────────────────────────────────────────────────────────────────
def clean_orders(df):
    log.info("Cleaning orders...")
    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Keep only meaningful statuses
    valid = ["delivered", "shipped", "canceled", "invoiced",
             "processing", "unavailable", "approved"]
    df = df[df["order_status"].isin(valid)].copy()

    # Drop rows where purchase timestamp is missing
    df.dropna(subset=["order_purchase_timestamp"], inplace=True)
    log.info(f"  Orders after cleaning: {len(df):,}")
    return df


def clean_order_items(df):
    log.info("Cleaning order_items...")
    df.dropna(subset=["order_id", "product_id", "seller_id"], inplace=True)
    df["price"]         = df["price"].clip(lower=0)
    df["freight_value"] = df["freight_value"].clip(lower=0)
    return df


def clean_order_payments(df):
    log.info("Cleaning order_payments...")
    df.dropna(subset=["order_id"], inplace=True)
    df["payment_value"] = df["payment_value"].clip(lower=0)
    return df


def clean_order_reviews(df):
    log.info("Cleaning order_reviews...")
    df.dropna(subset=["order_id", "review_score"], inplace=True)
    df["review_score"] = df["review_score"].astype(int)
    # Fill missing text fields
    df["review_comment_title"]   = df["review_comment_title"].fillna("")
    df["review_comment_message"] = df["review_comment_message"].fillna("")
    return df


def clean_products(df, category_trans):
    log.info("Cleaning products...")
    # Merge English category names
    df = df.merge(category_trans, on="product_category_name", how="left")
    df["product_category_name_english"] = (
        df["product_category_name_english"]
        .fillna(df["product_category_name"])
        .fillna("unknown")
    )
    # Fill numeric nulls with median
    num_cols = [
        "product_name_lenght", "product_description_lenght",
        "product_photos_qty", "product_weight_g",
        "product_length_cm", "product_height_cm", "product_width_cm",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df


def clean_geolocation(df):
    log.info("Cleaning geolocation...")
    # Remove obvious outliers for Brazil bounding box
    df = df[
        (df["geolocation_lat"].between(-35, 5)) &
        (df["geolocation_lng"].between(-75, -30))
    ].copy()
    # Keep one row per zip prefix (average lat/lng)
    df = (
        df.groupby("geolocation_zip_code_prefix")[
            ["geolocation_lat", "geolocation_lng"]
        ]
        .mean()
        .reset_index()
    )
    return df

# ── 3. BUILD MASTER TABLES ─────────────────────────────────────────────────────
def build_master_orders(dfs):
    """
    One row per order — joins orders, payments, reviews, customer info.
    """
    log.info("Building master_orders...")

    # Aggregate payments per order
    payments_agg = (
        dfs["order_payments"]
        .groupby("order_id")
        .agg(
            total_payment   = ("payment_value", "sum"),
            payment_installments = ("payment_installments", "max"),
            payment_type    = ("payment_type", lambda x: x.mode()[0])
        )
        .reset_index()
    )

    # Aggregate reviews per order (take worst score if multiple)
    reviews_agg = (
        dfs["order_reviews"]
        .groupby("order_id")
        .agg(review_score=("review_score", "min"))
        .reset_index()
    )

    master = (
        dfs["orders"]
        .merge(dfs["customers"],  on="customer_id",  how="left")
        .merge(payments_agg,      on="order_id",      how="left")
        .merge(reviews_agg,       on="order_id",      how="left")
    )

    # Derived columns
    master["delivery_days"] = (
        master["order_delivered_customer_date"] -
        master["order_purchase_timestamp"]
    ).dt.days

    master["estimated_days"] = (
        master["order_estimated_delivery_date"] -
        master["order_purchase_timestamp"]
    ).dt.days

    master["delivery_delay_days"] = (
        master["delivery_days"] - master["estimated_days"]
    )

    master["purchase_year"]  = master["order_purchase_timestamp"].dt.year
    master["purchase_month"] = master["order_purchase_timestamp"].dt.month
    master["purchase_dow"]   = master["order_purchase_timestamp"].dt.day_name()

    log.info(f"  master_orders shape: {master.shape}")
    return master


def build_master_items(dfs):
    """
    One row per order-item — joins items, products (with English names), sellers.
    """
    log.info("Building master_items...")
    master = (
        dfs["order_items"]
        .merge(dfs["products"], on="product_id", how="left")
        .merge(dfs["sellers"],  on="seller_id",  how="left")
    )
    master["item_revenue"] = master["price"] + master["freight_value"]
    log.info(f"  master_items shape: {master.shape}")
    return master

# ── 4. DATA QUALITY REPORT ─────────────────────────────────────────────────────
def data_quality_report(dfs: dict, processed: dict):
    log.info("Generating data quality report...")
    lines = ["=" * 60, "DATA QUALITY REPORT", "=" * 60]

    lines.append("\n--- RAW TABLES ---")
    for name, df in dfs.items():
        null_pct = (df.isnull().sum().sum() / df.size * 100)
        lines.append(f"{name:25s}  rows={len(df):>7,}  null%={null_pct:.1f}")

    lines.append("\n--- PROCESSED TABLES ---")
    for name, df in processed.items():
        null_pct = (df.isnull().sum().sum() / df.size * 100)
        lines.append(f"{name:25s}  rows={len(df):>7,}  null%={null_pct:.1f}")

    report = "\n".join(lines)
    print("\n" + report)

    report_path = os.path.join(PROC_DIR, "data_quality_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    log.info(f"  Report saved → {report_path}")

# ── 5. SAVE TO SQLITE ──────────────────────────────────────────────────────────
def save_to_sqlite(processed: dict):
    log.info(f"Saving to SQLite → {DB_PATH}")
    engine = create_engine(f"sqlite:///{DB_PATH}")
    for name, df in processed.items():
        df.to_sql(name, con=engine, if_exists="replace", index=False)
        log.info(f"  ✔ Table '{name}' saved  ({len(df):,} rows)")
    log.info("All tables saved successfully.")
    return engine

# ── 6. QUICK VALIDATION QUERIES ───────────────────────────────────────────────
def run_validation_queries(engine):
    log.info("Running validation queries...")
    queries = {
        "Total orders":          "SELECT COUNT(*) FROM master_orders",
        "Delivered orders":      "SELECT COUNT(*) FROM master_orders WHERE order_status='delivered'",
        "Total revenue":         "SELECT ROUND(SUM(total_payment),2) FROM master_orders",
        "Avg review score":      "SELECT ROUND(AVG(review_score),2) FROM master_orders",
        "Unique customers":      "SELECT COUNT(DISTINCT customer_unique_id) FROM master_orders",
        "Total order items":     "SELECT COUNT(*) FROM master_items",
        "Product categories":    "SELECT COUNT(DISTINCT product_category_name_english) FROM master_items",
    }
    print("\n" + "=" * 45)
    print("VALIDATION QUERIES")
    print("=" * 45)
    with engine.connect() as conn:
        for label, sql in queries.items():
            result = conn.execute(__import__("sqlalchemy").text(sql)).scalar()
            print(f"  {label:30s}: {result:>15,}")
    print("=" * 45)

# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    log.info("Starting ETL Pipeline")

    # 1. Load
    dfs = load_raw_data()

    # 2. Clean each table
    dfs["orders"]         = clean_orders(dfs["orders"])
    dfs["order_items"]    = clean_order_items(dfs["order_items"])
    dfs["order_payments"] = clean_order_payments(dfs["order_payments"])
    dfs["order_reviews"]  = clean_order_reviews(dfs["order_reviews"])
    dfs["products"]       = clean_products(dfs["products"], dfs["category_trans"])
    dfs["geolocation"]    = clean_geolocation(dfs["geolocation"])

    # 3. Build master tables
    master_orders = build_master_orders(dfs)
    master_items  = build_master_items(dfs)

    # 4. Package processed tables
    processed = {
        "master_orders":    master_orders,
        "master_items":     master_items,
        "customers":        dfs["customers"],
        "sellers":          dfs["sellers"],
        "products":         dfs["products"],
        "order_payments":   dfs["order_payments"],
        "order_reviews":    dfs["order_reviews"],
        "geolocation":      dfs["geolocation"],
    }

    # 5. Data quality report
    data_quality_report(dfs, processed)

    # 6. Save to SQLite
    engine = save_to_sqlite(processed)

    # 7. Validate
    run_validation_queries(engine)

    log.info("ETL Pipeline complete!")
    log.info(f"   Database → {DB_PATH}")

if __name__ == "__main__":
    main()