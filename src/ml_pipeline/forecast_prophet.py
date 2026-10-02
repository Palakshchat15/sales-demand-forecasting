"""
Per-category Prophet forecasting.

Reads warehouse.mart_category_daily_sales (produced by dbt). Per category:
  1. Picks the trend flexibility (changepoint_prior_scale) by time-series
     cross-validation on the training data only.
  2. Trains Prophet on everything except the last 90 days and scores that
     holdout (real MAPE/RMSE, out-of-sample).
  3. Refits on the full history and forecasts the next 30/60/90 days, saved
     to warehouse.demand_forecast (model_name='prophet').
Also writes a PNG per category to outputs/forecast_plots/ and
data/processed/forecast_vs_actual.csv (recent actuals, holdout predictions and
forward forecast, long format) for the Excel/Tableau dashboards.
"""

import os
import warnings

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from db_utils import get_engine

warnings.filterwarnings("ignore")

HOLDOUT_DAYS = 90
# Prophet draws its uncertainty intervals by sampling from numpy's global RNG;
# seeding before each predict() makes yhat_lower/yhat_upper reproducible too.
RNG_SEED = 42
FORWARD_HORIZONS = [30, 60, 90]
# At 0.1 the trend absorbed the autumn ramp-up on spiky series (Toys forecast
# +51% YoY vs +5% historically), so the value is chosen per category by CV.
CHANGEPOINT_PRIOR_GRID = [0.001, 0.005, 0.01, 0.05, 0.1]
CV_INITIAL, CV_PERIOD = "540 days", "90 days"
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
OUTPUT_DIR = os.environ.get("FORECAST_PLOTS_DIR", os.path.join(_PROJECT_ROOT, "outputs", "forecast_plots"))
DASHBOARD_DATA_DIR = os.environ.get("DASHBOARD_DATA_DIR", os.path.join(_PROJECT_ROOT, "data", "processed"))
DASHBOARD_HISTORY_DAYS = 90  # actuals shown before the holdout window starts

SERIES_ACTUAL = "Actual"
SERIES_HOLDOUT = "Backtest"  # predictions for the 90-day holdout, never seen in training
SERIES_FORWARD = "Forecast"  # next 90 days, from the model refit on all data


# Holidays: the standard US public-holiday calendar (Prophet's built-in
# add_country_holidays), i.e. information any forecaster would have. An earlier
# version copied the synthetic generator's own spike dates (Black Friday,
# Christmas rush, back-to-school Sundays...), which handed Prophet the answer.
# forecast_compare.py gives LightGBM the same calendar as a feature.
HOLIDAY_COUNTRY = "US"


def mape(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)


def rmse(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def make_model(changepoint_prior_scale):
    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=changepoint_prior_scale,
    )
    model.add_country_holidays(country_name=HOLIDAY_COUNTRY)
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    return model


def select_changepoint_prior(train):
    """Rolling-origin CV on the training window only; returns (best value, {value: cv MAPE})."""
    scores = {}
    for cps in CHANGEPOINT_PRIOR_GRID:
        model = make_model(cps)
        model.fit(train)
        cv = cross_validation(model, initial=CV_INITIAL, period=CV_PERIOD,
                              horizon=f"{HOLDOUT_DAYS} days", parallel="processes", disable_tqdm=True)
        scores[cps] = mape(cv["y"], cv["yhat"])
    return min(scores, key=scores.get), scores


def dashboard_rows(frame, category, series, value_col, lower_col=None, upper_col=None):
    return [
        {
            "ds": r["ds"].date(),
            "product_category": category,
            "series": series,
            "units": max(float(r[value_col]), 0.0),
            "units_lower": max(float(r[lower_col]), 0.0) if lower_col else None,
            "units_upper": max(float(r[upper_col]), 0.0) if upper_col else None,
        }
        for _, r in frame.iterrows()
    ]


def run_for_category(cat_df, category):
    cat_df = cat_df.sort_values("ds").reset_index(drop=True)
    last_actual_date = cat_df["ds"].max()
    cutoff = last_actual_date - pd.Timedelta(days=HOLDOUT_DAYS)
    train = cat_df[cat_df["ds"] <= cutoff][["ds", "y"]]
    test = cat_df[cat_df["ds"] > cutoff][["ds", "y"]]

    cps, cv_scores = select_changepoint_prior(train)
    print(f"{category}: CV MAPE by changepoint_prior_scale "
          + ", ".join(f"{k}={v:.2f}%" for k, v in cv_scores.items()) + f" -> using {cps}")

    # 1. Out-of-sample evaluation: the model never sees the holdout window.
    eval_model = make_model(cps)
    eval_model.fit(train)
    np.random.seed(RNG_SEED)
    holdout_forecast = eval_model.predict(test[["ds"]])
    merged = test.merge(holdout_forecast[["ds", "yhat"]], on="ds", how="inner")
    cat_mape = mape(merged["y"], merged["yhat"])
    cat_rmse = rmse(merged["y"], merged["yhat"])

    # 2. Forward forecast: refit on the full history so the latest 90 days inform it.
    full_model = make_model(cps)
    full_model.fit(cat_df[["ds", "y"]])
    max_horizon = max(FORWARD_HORIZONS)
    np.random.seed(RNG_SEED)
    sub = full_model.predict(full_model.make_future_dataframe(periods=max_horizon, include_history=False))

    history_start = cutoff - pd.Timedelta(days=DASHBOARD_HISTORY_DAYS)
    recent_actuals = cat_df[cat_df["ds"] > history_start]
    dash_rows = (
        dashboard_rows(recent_actuals, category, SERIES_ACTUAL, "y")
        + dashboard_rows(holdout_forecast, category, SERIES_HOLDOUT, "yhat", "yhat_lower", "yhat_upper")
        + dashboard_rows(sub, category, SERIES_FORWARD, "yhat", "yhat_lower", "yhat_upper")
    )

    # Each forward date gets exactly one row, tagged with the smallest horizon
    # bucket (30/60/90) it falls into, so (category, date, model) stays unique.
    forward_rows = []
    for _, r in sub.iterrows():
        days_out = (r["ds"] - last_actual_date).days
        horizon_bucket = next(h for h in sorted(FORWARD_HORIZONS) if days_out <= h)
        forward_rows.append({
            "product_category": category,
            "forecast_date": r["ds"].date(),
            "model_name": "prophet",
            "yhat": max(r["yhat"], 0),
            "yhat_lower": max(r["yhat_lower"], 0),
            "yhat_upper": max(r["yhat_upper"], 0),
            "horizon_days": horizon_bucket,
        })

    # Plot: history, holdout forecast (evaluation model), forward forecast (refit model)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(cat_df["ds"], cat_df["y"], label="Actual", color="black", linewidth=0.8, alpha=0.6)
    for frame, label, color in ((holdout_forecast, "Holdout forecast", "tab:blue"),
                                (sub, "Forecast: next 90 days", "tab:orange")):
        ax.plot(frame["ds"], frame["yhat"], label=label, color=color, linewidth=1.2)
        ax.fill_between(frame["ds"], frame["yhat_lower"], frame["yhat_upper"], color=color, alpha=0.2)
    ax.axvline(cutoff, color="red", linestyle="--", linewidth=1, label="Train/holdout split")
    ax.set_title(f"{category} — Prophet forecast (holdout MAPE={cat_mape:.1f}%, RMSE={cat_rmse:.1f})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units sold")
    ax.legend(loc="upper left")
    fig.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe_name = category.replace(" ", "_").replace("&", "and")
    fig.savefig(os.path.join(OUTPUT_DIR, f"{safe_name}_prophet_forecast.png"), dpi=110)
    plt.close(fig)

    metrics = {"product_category": category, "model": "prophet", "mape": cat_mape, "rmse": cat_rmse}
    tuning = {"product_category": category, "changepoint_prior_scale": cps,
              **{f"cv_mape_cps_{k}": round(v, 3) for k, v in cv_scores.items()}}
    return forward_rows, metrics, dash_rows, tuning


def main():
    engine = get_engine()
    df = pd.read_sql("SELECT ds, product_category, y FROM warehouse.mart_category_daily_sales ORDER BY product_category, ds", engine)
    df["ds"] = pd.to_datetime(df["ds"])

    categories = sorted(df["product_category"].unique())
    print(f"Running Prophet forecasts for {len(categories)} categories: {categories}")

    all_forward_rows = []
    all_dash_rows = []
    metrics = []
    tuning = []
    for cat in categories:
        print(f"\n--- {cat} ---")
        cat_df = df[df["product_category"] == cat][["ds", "y"]]
        forward_rows, m, dash_rows, t = run_for_category(cat_df, cat)
        all_forward_rows.extend(forward_rows)
        all_dash_rows.extend(dash_rows)
        metrics.append(m)
        tuning.append(t)
        print(f"{cat}: holdout MAPE={m['mape']:.2f}%  RMSE={m['rmse']:.2f}")

    forecast_df = pd.DataFrame(all_forward_rows)
    with engine.begin() as conn:
        conn.exec_driver_sql("DELETE FROM warehouse.demand_forecast WHERE model_name = 'prophet'")
    forecast_df.to_sql(
        "demand_forecast", engine, schema="warehouse", if_exists="append", index=False,
        method="multi", chunksize=1000,
    )
    print(f"\nWrote {len(forecast_df)} forecast rows to warehouse.demand_forecast (model_name=prophet)")

    os.makedirs(DASHBOARD_DATA_DIR, exist_ok=True)
    dash_path = os.path.join(DASHBOARD_DATA_DIR, "forecast_vs_actual.csv")
    pd.DataFrame(all_dash_rows).to_csv(dash_path, index=False)
    print(f"Wrote {len(all_dash_rows)} rows to {dash_path}")

    metrics_df = pd.DataFrame(metrics)
    outputs_dir = os.path.join(_PROJECT_ROOT, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    metrics_df.to_csv(os.path.join(outputs_dir, "prophet_holdout_metrics.csv"), index=False)
    pd.DataFrame(tuning).to_csv(os.path.join(outputs_dir, "prophet_tuning.csv"), index=False)
    print("\nProphet holdout metrics:")
    print(metrics_df.to_string(index=False))
    print(f"\nOverall mean MAPE: {metrics_df['mape'].mean():.2f}%")


if __name__ == "__main__":
    main()
