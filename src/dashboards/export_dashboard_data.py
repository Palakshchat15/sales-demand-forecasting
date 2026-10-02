"""
Exports the tables the Excel and Tableau dashboards read into data/processed/.

forecast_vs_actual.csv is written by src/ml_pipeline/forecast_prophet.py; this
script adds the warehouse aggregates and a display-ready model comparison.
"""
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "ml_pipeline"))
from db_utils import get_engine  # noqa: E402

PROCESSED_DIR = Path(os.environ.get("DASHBOARD_DATA_DIR", PROJECT_ROOT / "data" / "processed"))
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    engine = get_engine()

    monthly = pd.read_sql(
        """
        SELECT date_trunc('month', ds)::date AS sales_month, product_category,
               SUM(y) AS units_sold, ROUND(SUM(revenue)::numeric, 2) AS revenue
        FROM warehouse.mart_category_daily_sales
        GROUP BY 1, 2 ORDER BY 1, 2
        """,
        engine,
    )
    monthly.to_csv(PROCESSED_DIR / "monthly_category_sales.csv", index=False)

    next_90 = pd.read_sql(
        """
        SELECT product_category,
               ROUND(SUM(yhat) FILTER (WHERE horizon_days = 30)::numeric) AS forecast_units_30d,
               ROUND(SUM(yhat) FILTER (WHERE horizon_days <= 60)::numeric) AS forecast_units_60d,
               ROUND(SUM(yhat)::numeric) AS forecast_units_90d
        FROM warehouse.demand_forecast
        WHERE model_name = 'prophet'
        GROUP BY product_category ORDER BY product_category
        """,
        engine,
    )
    next_90.to_csv(PROCESSED_DIR / "next_90_day_forecast.csv", index=False)

    comparison = pd.read_csv(OUTPUTS_DIR / "model_comparison.csv")
    comparison["model"] = comparison["model"].map(
        {"prophet": "Prophet", "lightgbm": "LightGBM", "seasonal_naive": "Seasonal Naive"})
    if comparison["model"].isna().any():
        raise ValueError("model_comparison.csv has a model name with no display label")
    comparison.round({"mape": 2, "rmse": 1}).to_csv(PROCESSED_DIR / "model_comparison.csv", index=False)

    for name, df in (("monthly_category_sales", monthly), ("next_90_day_forecast", next_90)):
        print(f"{name}.csv: {len(df)} rows")
    print(f"model_comparison.csv: {len(comparison)} rows; forecast_vs_actual.csv present: "
          f"{(PROCESSED_DIR / 'forecast_vs_actual.csv').exists()}")


if __name__ == "__main__":
    main()
