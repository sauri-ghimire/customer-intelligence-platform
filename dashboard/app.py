# dashboard/app.py
#
# Customer Intelligence Platform — Interactive Streamlit Dashboard
#
# This dashboard pulls from the SQLite database built by the ETL pipeline
# and presents all analysis modules in one place:
#   - Business Overview: top-level KPIs and revenue trends
#   - Geographic Analysis: state-level performance across Brazil
#   - Customer Segments: RFM scoring and K-Means cluster profiles
#   - Churn Risk: XGBoost model output and revenue impact
#   - Demand Forecast: Prophet forecast and seasonality patterns
#
# Author: Sauri Ghimire
# MS Data Analytics & Information Systems — Texas State University

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sqlalchemy import create_engine, text
import os
import warnings
warnings.filterwarnings("ignore")

pio.templates.default = "simple_white"

# Page configuration — must be the first Streamlit call in the script
st.set_page_config(
    page_title            = "Customer Intelligence Platform",
    layout                = "wide",
    initial_sidebar_state = "expanded"
)

# Database path, relative to this file so it works locally and when deployed
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "data", "processed", "olist_dashboard.db")

# Cache the database engine so we dont reconnect on every interaction
@st.cache_resource
def get_engine():
    return create_engine(f"sqlite:///{DB_PATH}")

# Cache query results so the dashboard stays responsive
@st.cache_data
def run_query(query):
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)

# Sidebar navigation
st.sidebar.title("Customer Intelligence Platform")

page = st.sidebar.radio("Navigate", [
    "Business Overview",
    "Geographic Analysis",
    "Customer Segments",
    "Churn Risk",
    "Demand Forecast",
])

st.sidebar.markdown("**Dataset:** Olist E-Commerce")
st.sidebar.markdown("**Records:** 99,436 orders")
st.sidebar.markdown("**Period:** 2016 - 2018")
st.sidebar.markdown("**Built by:** Sauri Ghimire")


# ==============================================================================
# PAGE 1 — BUSINESS OVERVIEW
# Top-level KPIs, revenue and order trends, category breakdown,
# payment methods, and review score distribution.
# ==============================================================================
if page == "Business Overview":
    st.title("Business Overview")
    st.markdown("Key performance indicators across the entire Olist marketplace.")

    # Pull all top-level KPIs in a single query
    kpi = run_query("""
        SELECT
            COUNT(DISTINCT order_id)                    AS total_orders,
            COUNT(DISTINCT customer_unique_id)          AS unique_customers,
            ROUND(SUM(total_payment), 0)                AS total_revenue,
            ROUND(AVG(total_payment), 2)                AS avg_order_value,
            ROUND(AVG(review_score), 2)                 AS avg_review,
            ROUND(AVG(delivery_days), 1)                AS avg_delivery_days,
            SUM(CASE WHEN delivery_delay_days > 0
                     THEN 1 ELSE 0 END) * 100.0
                / COUNT(*)                              AS pct_late
        FROM master_orders
        WHERE order_status = 'delivered'
    """).iloc[0]

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Total Orders",      f"{kpi.total_orders:,.0f}")
    c2.metric("Unique Customers",  f"{kpi.unique_customers:,.0f}")
    c3.metric("Total Revenue",     f"${kpi.total_revenue/1e6:.2f}M")
    c4.metric("Avg Order Value",   f"${kpi.avg_order_value:.2f}")
    c5.metric("Avg Review Score",  f"{kpi.avg_review:.2f} / 5.0")
    c6.metric("Avg Delivery Days", f"{kpi.avg_delivery_days:.1f}")
    c7.metric("Pct Late Orders",   f"{kpi.pct_late:.1f}%")


    # Monthly revenue and order volume — filtered to 2017-2018
    # because 2016 only has a partial year of data
    monthly = run_query("""
        SELECT
            purchase_year                           AS year,
            purchase_month                          AS month,
            ROUND(SUM(total_payment), 2)            AS revenue,
            COUNT(order_id)                         AS num_orders
        FROM master_orders
        WHERE order_status = 'delivered'
          AND purchase_year IN (2017, 2018)
        GROUP BY purchase_year, purchase_month
        ORDER BY purchase_year, purchase_month
    """)
    monthly["period"] = (
        monthly["year"].astype(str) + "-" +
        monthly["month"].astype(str).str.zfill(2)
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Monthly Revenue")
        fig = px.bar(
            monthly, x="period", y="revenue",
            color_discrete_sequence=["#2196F3"],
            labels={"revenue": "Revenue ($)", "period": "Month"}
        )
        fig.update_layout(
            xaxis_tickangle=-45, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Monthly Order Volume")
        fig2 = px.line(
            monthly, x="period", y="num_orders",
            markers=True,
            color_discrete_sequence=["#1565C0"],
            labels={"num_orders": "Orders", "period": "Month"}
        )
        fig2.update_layout(
            xaxis_tickangle=-45, height=350)
        st.plotly_chart(fig2, use_container_width=True)

    # Top 15 product categories by total revenue
    st.subheader("Top 15 Product Categories by Revenue")
    cat = run_query("""
        SELECT
            mi.product_category_name_english    AS category,
            ROUND(SUM(mi.price), 2)             AS revenue,
            COUNT(*)                            AS num_items,
            ROUND(AVG(mi.price), 2)             AS avg_price
        FROM master_items mi
        JOIN master_orders mo ON mi.order_id = mo.order_id
        WHERE mo.order_status = 'delivered'
        GROUP BY category
        ORDER BY revenue DESC
        LIMIT 15
    """)
    fig3 = px.bar(
        cat, x="revenue", y="category",
        orientation="h",
        color="avg_price",
        color_continuous_scale="Blues",
        labels={
            "revenue":   "Revenue ($)",
            "category":  "Category",
            "avg_price": "Avg Price ($)"
        },
        height=450
    )
    fig3.update_layout(
        yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig3, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Payment Methods")
        pay = run_query("""
            SELECT payment_type,
                   COUNT(*) AS num_orders
            FROM master_orders
            WHERE order_status = 'delivered'
            GROUP BY payment_type
            ORDER BY num_orders DESC
        """)
        fig4 = px.pie(
            pay, values="num_orders", names="payment_type",
            color_discrete_sequence=px.colors.sequential.Blues_r,
            hole=0.4
        )
        fig4.update_layout(height=300)
        st.plotly_chart(fig4, use_container_width=True)

    with col4:
        st.subheader("Review Score Distribution")
        rev = run_query("""
            SELECT review_score,
                   COUNT(*) AS count
            FROM master_orders
            WHERE order_status = 'delivered'
              AND review_score IS NOT NULL
            GROUP BY review_score
            ORDER BY review_score
        """)
        fig5 = px.bar(
            rev, x="review_score", y="count",
            color="review_score",
            color_continuous_scale="Blues",
            labels={"review_score": "Score", "count": "Orders"}
        )
        fig5.update_layout(
            height=300, showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)


# ==============================================================================
# PAGE 2 — GEOGRAPHIC ANALYSIS
# Order volume, revenue, delivery days, and review scores by Brazilian state.
# ==============================================================================
elif page == "Geographic Analysis":
    st.title("Geographic Analysis")
    st.markdown("Order volume, revenue, and delivery performance by Brazilian state.")

    state = run_query("""
        SELECT
            customer_state                          AS state,
            COUNT(order_id)                         AS num_orders,
            ROUND(SUM(total_payment), 2)            AS revenue,
            ROUND(AVG(total_payment), 2)            AS avg_order_value,
            ROUND(AVG(delivery_days), 1)            AS avg_delivery_days,
            ROUND(AVG(delivery_delay_days), 1)      AS avg_delay,
            ROUND(AVG(review_score), 2)             AS avg_review
        FROM master_orders
        WHERE order_status = 'delivered'
        GROUP BY customer_state
        ORDER BY revenue DESC
    """)

    # Let the user pick which metric to color the map by
    metric = st.selectbox(
        "Select metric to visualize on map:",
        ["num_orders", "revenue", "avg_delivery_days", "avg_review"]
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        # Choropleth map of Brazil colored by selected metric
        fig = px.choropleth(
            state,
            locations              = "state",
            locationmode           = "geojson-id",
            color                  = metric,
            scope                  = "south america",
            color_continuous_scale = "Blues",
            hover_data             = ["num_orders", "revenue",
                                      "avg_delivery_days", "avg_review"],
            labels                 = {
                "num_orders":        "Orders",
                "revenue":           "Revenue ($)",
                "avg_delivery_days": "Avg Delivery Days",
                "avg_review":        "Avg Review Score",
            },
            title = f"Brazil — {metric.replace('_', ' ').title()} by State"
        )
        fig.update_geos(fitbounds="locations", visible=True)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("State Rankings")
        display = state[["state", "num_orders", "revenue",
                          "avg_delivery_days", "avg_review"]].copy()
        display["revenue"] = display["revenue"].apply(
            lambda x: f"${x:,.0f}")
        st.dataframe(display, use_container_width=True, height=480)

    # Scatter plot: does slower delivery lead to worse reviews?
    st.subheader("Delivery Speed vs Customer Satisfaction")
    fig2 = px.scatter(
        state,
        x="avg_delivery_days", y="avg_review",
        size="num_orders", color="revenue",
        hover_name="state",
        color_continuous_scale="Blues",
        labels={
            "avg_delivery_days": "Avg Delivery Days",
            "avg_review":        "Avg Review Score",
            "num_orders":        "Order Volume",
            "revenue":           "Revenue ($)"
        },
        title="Delivery Days vs Review Score by State (bubble size = order volume)"
    )
    fig2.update_layout(
        height=400)
    st.plotly_chart(fig2, use_container_width=True)


# ==============================================================================
# PAGE 3 — CUSTOMER SEGMENTS
# RFM scoring results and K-Means cluster profiles.
# ==============================================================================
elif page == "Customer Segments":
    st.title("Customer Segmentation")
    st.markdown("Customers grouped by purchase behavior using RFM analysis and K-Means clustering.")

    rfm = run_query("""
        SELECT customer_id, recency_days, frequency, monetary,
               avg_review, segment, cluster_name
        FROM rfm_segments
    """)

    # Summarize each segment by key metrics
    seg = (
        rfm.groupby("segment")
        .agg(
            customers     = ("customer_id",  "count"),
            avg_recency   = ("recency_days", "mean"),
            avg_frequency = ("frequency",    "mean"),
            avg_monetary  = ("monetary",     "mean"),
            total_revenue = ("monetary",     "sum")
        )
        .round(2)
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customers per Segment")
        fig = px.bar(
            seg, x="segment", y="customers",
            color="customers",
            color_continuous_scale="Blues",
            labels={"segment": "Segment", "customers": "Customers"}
        )
        fig.update_layout(
            xaxis_tickangle=-20, height=350,
            showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Revenue per Segment")
        fig2 = px.bar(
            seg, x="segment", y="total_revenue",
            color="total_revenue",
            color_continuous_scale="Blues",
            labels={"segment": "Segment", "total_revenue": "Revenue ($)"}
        )
        fig2.update_layout(
            xaxis_tickangle=-20, height=350,
            showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Scatter plot — recency vs spend colored by segment
    # Sample 5000 rows so the chart renders quickly
    st.subheader("Recency vs Total Spend by Segment")
    sample = rfm.sample(min(5000, len(rfm)), random_state=42)
    fig3 = px.scatter(
        sample,
        x="recency_days",
        y=sample["monetary"].clip(upper=2000),
        color="segment",
        opacity=0.5,
        labels={
            "recency_days": "Days Since Last Order",
            "y":            "Total Spend (capped at $2,000)"
        },
        height=450
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Full segment summary table
    st.subheader("Segment Summary Table")
    seg_display = seg.copy()
    seg_display["avg_monetary"]  = seg_display["avg_monetary"].apply(
        lambda x: f"${x:,.2f}")
    seg_display["total_revenue"] = seg_display["total_revenue"].apply(
        lambda x: f"${x:,.0f}")
    seg_display["avg_recency"]   = seg_display["avg_recency"].apply(
        lambda x: f"{x:.0f} days")
    st.dataframe(seg_display, use_container_width=True)


# ==============================================================================
# PAGE 4 — CHURN RISK
# XGBoost churn model output, risk tiers, and revenue at stake.
# ==============================================================================
elif page == "Churn Risk":
    st.title("Churn Risk Analysis")
    st.markdown("XGBoost churn prediction model — customer risk scoring and revenue impact.")

    churn = run_query("""
        SELECT customer_unique_id, total_orders, total_spend,
               avg_review_score, recency_days, churned,
               churn_probability, risk_tier, customer_state
        FROM churn_scores
    """)

    # High and critical risk customers are the priority for retention efforts
    high_risk = churn[churn["risk_tier"].isin(["High Risk", "Critical Risk"])]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers Scored", f"{len(churn):,}")
    c2.metric("High and Critical Risk", f"{len(high_risk):,}")
    c3.metric("Revenue at Stake",       f"${high_risk['total_spend'].sum():,.0f}")
    c4.metric("Model AUC",              "0.7476")


    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customers by Risk Tier")
        tier_counts = churn["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["risk_tier", "count"]
        fig = px.bar(
            tier_counts, x="risk_tier", y="count",
            color="risk_tier",
            color_discrete_map={
                "Low Risk":      "#4CAF50",
                "Medium Risk":   "#FFC107",
                "High Risk":     "#FF9800",
                "Critical Risk": "#F44336",
            },
            labels={"risk_tier": "Risk Tier", "count": "Customers"}
        )
        fig.update_layout(
            height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Revenue at Stake by Tier")
        tier_rev = (
            churn.groupby("risk_tier")["total_spend"]
            .sum().reset_index()
        )
        fig2 = px.bar(
            tier_rev, x="risk_tier", y="total_spend",
            color="risk_tier",
            color_discrete_map={
                "Low Risk":      "#4CAF50",
                "Medium Risk":   "#FFC107",
                "High Risk":     "#FF9800",
                "Critical Risk": "#F44336",
            },
            labels={"risk_tier": "Risk Tier", "total_spend": "Revenue ($)"}
        )
        fig2.update_layout(
            height=350, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Distribution of churn probabilities across all customers
    st.subheader("Churn Probability Distribution")
    fig3 = px.histogram(
        churn, x="churn_probability",
        color="risk_tier",
        nbins=50,
        color_discrete_map={
            "Low Risk":      "#4CAF50",
            "Medium Risk":   "#FFC107",
            "High Risk":     "#FF9800",
            "Critical Risk": "#F44336",
        },
        labels={
            "churn_probability": "Churn Probability",
            "count":             "Customers"
        },
        height=350
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Interactive filter table so a business user can drill into
    # specific risk tiers or states
    st.subheader("High Risk Customer Table")
    st.markdown("Use the filters below to explore customers most likely to churn.")

    col3, col4 = st.columns(2)
    with col3:
        tier_filter = st.multiselect(
            "Risk Tier",
            options=churn["risk_tier"].unique().tolist(),
            default=["High Risk", "Critical Risk"]
        )
    with col4:
        state_filter = st.multiselect(
            "State",
            options=sorted(churn["customer_state"].unique().tolist()),
            default=[]
        )

    filtered = churn[churn["risk_tier"].isin(tier_filter)]
    if state_filter:
        filtered = filtered[filtered["customer_state"].isin(state_filter)]

    filtered_display = (
        filtered[[
            "customer_unique_id", "customer_state", "total_orders",
            "total_spend", "avg_review_score", "recency_days",
            "churn_probability", "risk_tier"
        ]]
        .sort_values("churn_probability", ascending=False)
        .head(500)
        .copy()
    )

    filtered_display["churn_probability"] = filtered_display[
        "churn_probability"].apply(lambda x: f"{x:.1%}")
    filtered_display["total_spend"] = filtered_display[
        "total_spend"].apply(lambda x: f"${x:,.2f}")

    st.dataframe(filtered_display, use_container_width=True, height=400)
    st.caption(f"Showing top 500 of {len(filtered):,} filtered customers")


# ==============================================================================
# PAGE 5 — DEMAND FORECAST
# Prophet 90-day forecast, seasonality breakdown, and model comparison.
# ==============================================================================
elif page == "Demand Forecast":
    st.title("Demand Forecast")
    st.markdown("Facebook Prophet 90-day demand forecast with seasonality analysis.")

    # Pull daily order and revenue data for the main analysis period
    daily = run_query("""
        SELECT
            DATE(order_purchase_timestamp)      AS ds,
            COUNT(DISTINCT order_id)            AS num_orders,
            ROUND(SUM(total_payment), 2)        AS revenue
        FROM master_orders
        WHERE order_status = 'delivered'
          AND order_purchase_timestamp >= '2017-01-01'
          AND order_purchase_timestamp <  '2018-09-01'
        GROUP BY DATE(order_purchase_timestamp)
        ORDER BY ds
    """)
    daily["ds"] = pd.to_datetime(daily["ds"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Daily Orders",  f"{daily.num_orders.mean():.0f}")
    c2.metric("Peak Day Orders",   f"{daily.num_orders.max():,}")
    c3.metric("Forecast (90 days)","5,392 orders")
    c4.metric("Prophet MAPE",      "67.0%")


    # Historical order volume with 7-day rolling average to smooth noise
    st.subheader("Historical Order Volume with 7-Day Moving Average")
    daily["ma7"] = daily["num_orders"].rolling(7).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["ds"], y=daily["num_orders"],
        name="Daily Orders", opacity=0.4,
        line=dict(color="#90CAF9", width=1)
    ))
    fig.add_trace(go.Scatter(
        x=daily["ds"], y=daily["ma7"],
        name="7-Day Moving Average",
        line=dict(color="#1565C0", width=2.5)
    ))
    fig.update_layout(
        height=380,
        xaxis_title="Date", yaxis_title="Orders",
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Seasonality breakdown — day of week and month of year
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Average Orders by Day of Week")
        daily["dow"] = daily["ds"].dt.day_name()
        dow_order = ["Monday","Tuesday","Wednesday",
                     "Thursday","Friday","Saturday","Sunday"]
        dow_avg = (
            daily.groupby("dow")["num_orders"]
            .mean()
            .reindex(dow_order)
            .reset_index()
        )
        fig2 = px.bar(
            dow_avg, x="dow", y="num_orders",
            color_discrete_sequence=["#2196F3"],
            labels={"dow": "Day", "num_orders": "Avg Orders"}
        )
        fig2.update_layout(
            height=300)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Average Orders by Month")
        daily["month"] = daily["ds"].dt.month
        month_avg = (
            daily.groupby("month")["num_orders"]
            .mean()
            .reset_index()
        )
        month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        month_avg["month_name"] = month_avg["month"].apply(
            lambda x: month_names[x - 1])
        fig3 = px.bar(
            month_avg, x="month_name", y="num_orders",
            color_discrete_sequence=["#1565C0"],
            labels={"month_name": "Month", "num_orders": "Avg Orders"}
        )
        fig3.update_layout(
            height=300)
        st.plotly_chart(fig3, use_container_width=True)

    # 90-day forecast summary table from the Prophet model
    st.subheader("90-Day Forecast Summary")
    forecast_summary = pd.DataFrame({
        "Month":             ["Sep 2018", "Oct 2018", "Nov 2018"],
        "Forecasted Orders": [1764, 1820, 1808],
        "Low Estimate":      [1456, 1498, 1481],
        "High Estimate":     [2071, 2141, 2134],
    })
    st.dataframe(forecast_summary, use_container_width=True)

    # Note explaining the MAPE result honestly
    st.info(
        "Forecast note: A MAPE of 67% is expected at the daily granularity "
        "for marketplace data due to high day-to-day noise. Weekly aggregation "
        "typically brings MAPE down to 15-25%. Prophet outperformed ARIMA on "
        "both MAPE (67.0% vs 68.6%) and RMSE (73.7 vs 83.0 orders per day)."
    )