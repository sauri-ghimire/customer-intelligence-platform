# Customer Intelligence Platform

An end-to-end analytics project on the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). It covers ETL, SQL exploration, customer segmentation, churn prediction, demand forecasting, and an interactive Streamlit dashboard.

**Author:** Sauri Ghimire, MS Data Analytics & Information Systems, Texas State University

## What it does

| Module | Techniques | Notebook / File |
|---|---|---|
| ETL pipeline | Ingest 9 CSVs, clean, merge into master tables, load to SQLite | `src/etl.py` |
| Exploratory analysis | SQL queries on revenue, categories, geography, delivery vs. reviews, sellers, timing | `notebooks/02_eda_sql_analysis.ipynb` |
| Customer segmentation | RFM scoring, K-Means clustering | `notebooks/03_customer_segmentation.ipynb` |
| Churn prediction | XGBoost, SHAP explainability, risk tiers | `notebooks/04_churn_prediction.ipynb` |
| Demand forecasting | Prophet vs. ARIMA, seasonality analysis | `notebooks/05_demand_forecasting.ipynb` |
| Dashboard | Streamlit + Plotly | `dashboard/app.py` |

The dashboard has five views: Business Overview, Geographic Analysis, Customer Segments, Churn Risk, and Demand Forecast.

## Project structure

```
customer_intelligence_platform/
├── dashboard/app.py          # Streamlit dashboard
├── data/
│   ├── raw/                  # Olist CSVs (not tracked, see Setup)
│   └── processed/            # olist.db (generated) and data_quality_report.txt
├── notebooks/                # EDA, segmentation, churn, forecasting
├── outputs/                  # Charts (PNG) and forecast_report.xlsx
├── src/etl.py                # ETL pipeline
└── requirements.txt
```

## Setup

1. Clone the repo and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and place the 9 CSV files in `data/raw/`.

3. `src/etl.py` has a hardcoded `BASE_DIR` pointing to `D:\customer_intelligence_platform`. Change it to where you cloned the project. The dashboard uses relative paths and reads `data/processed/olist_dashboard.db`, a slimmed copy of the database that is included in the repo.

## Usage

Build the database (also writes the data quality report):

```bash
python src/etl.py
```

Run the notebooks in order (`02` to `05`) with Jupyter, then launch the dashboard:

```bash
streamlit run dashboard/app.py
```

## Outputs

Charts for each stage are saved in `outputs/`:

- `01`–`06`: revenue trend, categories, geography, delivery and reviews, sellers, timing
- `07`–`09`: RFM segmentation, optimal k, K-Means clusters
- `10`–`13`: churn distribution, model evaluation, SHAP, risk tiers
- `14`–`17`: time series, seasonality, Prophet forecast, Prophet vs. ARIMA

`forecast_report.xlsx` holds the forecast results.

## Tech stack

Python, pandas, NumPy, SQLAlchemy, SQLite, scikit-learn, XGBoost, SHAP, Prophet, statsmodels, matplotlib, seaborn, Plotly, Streamlit.

## Data

Olist Brazilian E-Commerce Public Dataset (CC BY-NC-SA 4.0), about 100k orders from 2016 to 2018. The raw data is not included in this repo.
