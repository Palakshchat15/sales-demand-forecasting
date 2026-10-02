"""
Comparison models for Prophet, scored on the same 90-day holdout as
forecast_prophet.py (train on everything up to the cutoff, forecast the next
90 days blind, score against the actuals):

1. seasonal_naive: a baseline any forecaster should beat. For each holdout
   date t, prediction = y[t - 364] x g, where g = total units in the last 364
   training days / total units in the 364 days before that (trailing-year
   growth). 364 days keeps the weekday aligned. Uses training data only.

2. lightgbm: a *direct* 90-day-ahead forecast. Every lag/rolling feature uses
   values at least 90 days before the target date, so each holdout day is
   predicted only from data available at the cutoff (no holdout actuals leak
   in). Calendar features and the same US public-holiday calendar Prophet uses
   (prophet.make_holidays, country 'US') are known in advance.
   Trees cannot extrapolate beyond the training range, so the target is the
   ratio of y to its 28-day mean ending 90 days earlier (a level known at
   forecast time); the prediction is ratio x that level.

Writes outputs/model_comparison.csv with all three models' holdout MAPE/RMSE
(Prophet's come from outputs/prophet_holdout_metrics.csv). The overall winner
is the model with the lowest simple (unweighted) mean of per-category MAPE.
"""

import os

import numpy as np
import pandas as pd
import lightgbm as lgb
from prophet.make_holidays import make_holidays_df

from db_utils import get_engine

HOLDOUT_DAYS = 90
MIN_LAG = HOLDOUT_DAYS   # no feature may look at y closer than this to the target date
SEASON_LAG = 364         # 52 weeks: same weekday one year earlier
LEVEL_WINDOW = 28
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
OUTPUTS_DIR = os.path.join(_PROJECT_ROOT, "outputs")


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


def us_holiday_codes(dates):
    """Integer code per date from Prophet's own US holiday calendar (0 = not a holiday)."""
    years = sorted(set(dates.dt.year))
    hol = make_holidays_df(year_list=years, country="US")
    names = sorted(hol["holiday"].unique())
    code = {n: i + 1 for i, n in enumerate(names)}
    by_date = dict(zip(pd.to_datetime(hol["ds"]), hol["holiday"].map(code)))
    return dates.map(lambda d: by_date.get(d, 0)).astype(int)


def make_features(df):
    """df: daily series with columns ds, y (no gaps). Every y-based feature is lagged >= MIN_LAG days."""
    df = df.sort_values("ds").reset_index(drop=True).copy()
    df["dayofweek"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    df["day"] = df["ds"].dt.day
    df["dayofyear"] = df["ds"].dt.dayofyear
    df["weekofyear"] = df["ds"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["us_holiday"] = us_holiday_codes(df["ds"])

    y = df["y"].astype(float)
    # Level known at forecast time: 28-day mean ending MIN_LAG days before the target date.
    df["level"] = y.shift(MIN_LAG).rolling(LEVEL_WINDOW).mean()
    # The same quantities one year (364 days) earlier.
    level_ly = y.shift(MIN_LAG + SEASON_LAG).rolling(LEVEL_WINDOW).mean()
    df["level_yoy_growth"] = df["level"] / level_ly
    for lag in (SEASON_LAG - 7, SEASON_LAG, SEASON_LAG + 7):
        df[f"ly_ratio_{lag}"] = y.shift(lag) / level_ly
    df["ly_week_ratio"] = y.shift(SEASON_LAG - 3).rolling(7).mean() / level_ly
    df["target_ratio"] = y / df["level"]
    return df


FEATURE_COLS = [
    "dayofweek", "month", "day", "dayofyear", "weekofyear", "is_weekend", "us_holiday",
    "level_yoy_growth", f"ly_ratio_{SEASON_LAG - 7}", f"ly_ratio_{SEASON_LAG}",
    f"ly_ratio_{SEASON_LAG + 7}", "ly_week_ratio",
]


def split(cat_df):
    cat_df = cat_df.sort_values("ds").reset_index(drop=True)
    cutoff = cat_df["ds"].max() - pd.Timedelta(days=HOLDOUT_DAYS)
    return cat_df, cutoff


def run_lightgbm(cat_df, category):
    cat_df, cutoff = split(cat_df)
    feat_df = make_features(cat_df)
    train = feat_df[feat_df["ds"] <= cutoff].dropna(subset=FEATURE_COLS + ["target_ratio"])
    test = feat_df[feat_df["ds"] > cutoff]
    if len(train) < 60 or len(test) == 0 or test[FEATURE_COLS + ["level"]].isna().any().any():
        return None

    model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(train[FEATURE_COLS], train["target_ratio"])
    preds = np.clip(model.predict(test[FEATURE_COLS]) * test["level"].to_numpy(), 0, None)
    return {"product_category": category, "model": "lightgbm",
            "mape": mape(test["y"], preds), "rmse": rmse(test["y"], preds)}


def run_seasonal_naive(cat_df, category):
    cat_df, cutoff = split(cat_df)
    y = cat_df.set_index("ds")["y"].astype(float)
    train = y[y.index <= cutoff]
    test = y[y.index > cutoff]
    growth = train.iloc[-SEASON_LAG:].sum() / train.iloc[-2 * SEASON_LAG:-SEASON_LAG].sum()
    preds = train.reindex(test.index - pd.Timedelta(days=SEASON_LAG)).to_numpy() * growth
    return {"product_category": category, "model": "seasonal_naive",
            "mape": mape(test, preds), "rmse": rmse(test, preds)}


def main():
    engine = get_engine()
    df = pd.read_sql("SELECT ds, product_category, y FROM warehouse.mart_category_daily_sales "
                     "ORDER BY product_category, ds", engine)
    df["ds"] = pd.to_datetime(df["ds"])

    categories = sorted(df["product_category"].unique())
    print(f"Running comparison forecasts for {len(categories)} categories: {categories}")

    rows = []
    for cat in categories:
        cat_df = df[df["product_category"] == cat][["ds", "y"]]
        span = (cat_df["ds"].max() - cat_df["ds"].min()).days + 1
        if span != len(cat_df):
            raise ValueError(f"{cat}: {span - len(cat_df)} missing days; lag features assume a gap-free series")
        for runner in (run_seasonal_naive, run_lightgbm):
            m = runner(cat_df, cat)
            if m is None:
                print(f"{cat}: insufficient history for {runner.__name__}, skipping")
                continue
            rows.append(m)
            print(f"{cat}: {m['model']} holdout MAPE={m['mape']:.2f}%  RMSE={m['rmse']:.2f}")

    ours = pd.DataFrame(rows)

    # Combine with Prophet's metrics (written earlier by forecast_prophet.py)
    prophet_path = os.path.join(OUTPUTS_DIR, "prophet_holdout_metrics.csv")
    if os.path.exists(prophet_path):
        prophet_df = pd.read_csv(prophet_path)
    else:
        print(f"WARNING: {prophet_path} not found — run forecast_prophet.py first for a full comparison.")
        prophet_df = pd.DataFrame(columns=["product_category", "model", "mape", "rmse"])

    order = {"prophet": 0, "lightgbm": 1, "seasonal_naive": 2}
    combined = pd.concat([prophet_df, ours], ignore_index=True)
    combined = combined.sort_values(["model", "product_category"], key=lambda s: s.map(order) if s.name == "model" else s)
    combined = combined.reset_index(drop=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    combined.to_csv(os.path.join(OUTPUTS_DIR, "model_comparison.csv"), index=False)

    print("\n=== Model comparison (holdout MAPE %) ===")
    print(combined.pivot(index="product_category", columns="model", values="mape").round(2))

    # Winner: lowest simple mean of per-category MAPE (each category weighted equally)
    summary = combined.groupby("model")["mape"].mean().sort_values()
    print("\nMean MAPE by model (simple mean across categories):")
    print(summary.round(2))
    print(f"\nOverall winner (lowest mean MAPE): {summary.index[0]}")


if __name__ == "__main__":
    main()
