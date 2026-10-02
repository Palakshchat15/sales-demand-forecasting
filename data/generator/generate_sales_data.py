"""
Synthetic daily sales data generator for the Sales/Demand Forecasting portfolio project.

Generates 3 years of daily sales data (2023-01-01 through 2025-12-31) across
8 product categories and 5 store regions, with strong, deliberately-coded
deterministic signal (trend + weekly seasonality + yearly seasonality +
holiday spikes + promotions) plus modest noise on top, so that models like
Prophet can recover the signal cleanly and produce strong accuracy metrics.

Output: data/raw/sales_history.csv
Columns: date, product_category, units_sold, revenue, avg_unit_price,
         store_region, promotion_flag
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
START_DATE = "2023-01-01"
END_DATE = "2025-12-31"

REGIONS = ["North", "South", "East", "West", "Online"]
REGION_WEIGHTS = {
    "North": 0.20,
    "South": 0.20,
    "East": 0.18,
    "West": 0.17,
    "Online": 0.25,
}

# Category configuration:
#   base: baseline daily units sold (before seasonality)
#   annual_growth: fractional growth per year (compounding), e.g. 0.15 = +15%/yr
#   yearly_amplitude: strength of yearly (day-of-year) seasonal cycle, as fraction of base
#   yearly_peak_doy: day-of-year (approx) where the yearly cycle peaks
#   weekday_profile: multiplier for Mon..Sun (7 values)
#   noise_frac: relative std-dev of Gaussian noise added on top of signal
#   unit_price: base average unit price (with small daily jitter)
#   promo_boost: multiplicative boost applied to units_sold on promo days
CATEGORIES = {
    "Electronics": dict(
        base=420, annual_growth=0.18, yearly_amplitude=0.35, yearly_peak_doy=330,  # late Nov/Dec
        weekday_profile=[0.90, 0.88, 0.90, 0.95, 1.05, 1.35, 1.25],
        noise_frac=0.06, unit_price=180.0, promo_boost=1.6,
    ),
    "Apparel": dict(
        base=650, annual_growth=0.08, yearly_amplitude=0.25, yearly_peak_doy=260,  # back to school/fall
        weekday_profile=[0.85, 0.85, 0.90, 0.95, 1.10, 1.45, 1.30],
        noise_frac=0.07, unit_price=42.0, promo_boost=1.5,
    ),
    "Home & Garden": dict(
        base=380, annual_growth=0.06, yearly_amplitude=0.45, yearly_peak_doy=140,  # spring/summer
        weekday_profile=[0.95, 0.92, 0.93, 0.97, 1.05, 1.25, 1.20],
        noise_frac=0.07, unit_price=65.0, promo_boost=1.4,
    ),
    "Groceries": dict(
        base=1400, annual_growth=0.03, yearly_amplitude=0.10, yearly_peak_doy=350,  # holiday cooking
        weekday_profile=[0.95, 0.90, 0.90, 0.95, 1.10, 1.40, 1.30],
        noise_frac=0.04, unit_price=8.5, promo_boost=1.2,
    ),
    "Toys": dict(
        base=180, annual_growth=0.05, yearly_amplitude=0.90, yearly_peak_doy=345,  # huge Nov-Dec spike
        weekday_profile=[0.88, 0.86, 0.88, 0.92, 1.05, 1.40, 1.30],
        noise_frac=0.08, unit_price=28.0, promo_boost=1.8,
    ),
    "Sporting Goods": dict(
        base=260, annual_growth=0.07, yearly_amplitude=0.40, yearly_peak_doy=1,  # New Year resolutions (Jan) & summer(180)
        weekday_profile=[0.92, 0.90, 0.92, 0.95, 1.08, 1.30, 1.25],
        noise_frac=0.07, unit_price=55.0, promo_boost=1.4,
    ),
    "Books": dict(
        base=310, annual_growth=0.02, yearly_amplitude=0.20, yearly_peak_doy=250,  # back to school
        weekday_profile=[0.97, 0.95, 0.96, 0.98, 1.05, 1.15, 1.18],
        noise_frac=0.06, unit_price=15.0, promo_boost=1.3,
    ),
    "Beauty": dict(
        base=340, annual_growth=0.12, yearly_amplitude=0.30, yearly_peak_doy=340,  # holiday gifting
        weekday_profile=[0.90, 0.88, 0.90, 0.94, 1.08, 1.35, 1.28],
        noise_frac=0.07, unit_price=32.0, promo_boost=1.5,
    ),
}

# Fixed holiday spike dates (month, day) -> (multiplier, applies_to categories or None=all)
HOLIDAYS = {
    # Black Friday (day after US Thanksgiving, 4th Thu of Nov) - approximate fixed dates per year
    "2023-11-24": 3.2, "2024-11-29": 3.2, "2025-11-28": 3.2,  # Black Friday
    "2023-11-27": 1.8, "2024-12-02": 1.8, "2025-12-01": 1.8,  # Cyber Monday
    "2023-12-23": 2.2, "2024-12-23": 2.2, "2025-12-23": 2.2,  # Christmas rush
    "2023-12-24": 1.6, "2024-12-24": 1.6, "2025-12-24": 1.6,
    "2023-08-20": 1.7, "2024-08-18": 1.7, "2025-08-17": 1.7,  # Back to school (Books/Apparel heavy)
    "2023-08-27": 1.5, "2024-08-25": 1.5, "2025-08-24": 1.5,
    "2023-01-01": 1.4, "2024-01-01": 1.4, "2025-01-01": 1.4,  # New Year (Sporting Goods)
    "2023-05-26": 1.5, "2024-05-24": 1.5, "2025-05-23": 1.5,  # Memorial Day weekend (Home & Garden)
    "2023-07-04": 1.6, "2024-07-04": 1.6, "2025-07-04": 1.6,  # July 4th
}

BACK_TO_SCHOOL_CATS = {"Books", "Apparel", "Electronics"}


def build_calendar():
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    return dates


def yearly_seasonal_factor(doy, peak_doy, amplitude, year_length=365.25):
    """Smooth cosine-based yearly cycle peaking at peak_doy, returns multiplicative factor."""
    phase = 2 * np.pi * (doy - peak_doy) / year_length
    return 1.0 + amplitude * np.cos(phase)


def generate():
    rng = np.random.default_rng(RNG_SEED)
    dates = build_calendar()
    n_days = len(dates)

    rows = []

    for cat, cfg in CATEGORIES.items():
        base = cfg["base"]
        growth = cfg["annual_growth"]
        amp = cfg["yearly_amplitude"]
        peak_doy = cfg["yearly_peak_doy"]
        weekday_profile = cfg["weekday_profile"]
        noise_frac = cfg["noise_frac"]
        unit_price = cfg["unit_price"]
        promo_boost = cfg["promo_boost"]

        # Promotion days: random ~3% of days, but never on top of a major holiday (avoid double counting)
        promo_days = rng.random(n_days) < 0.03

        for i, d in enumerate(dates):
            doy = d.dayofyear
            year_frac = (d - pd.Timestamp(START_DATE)).days / 365.25
            trend_factor = (1.0 + growth) ** year_frac

            seasonal_factor = yearly_seasonal_factor(doy, peak_doy, amp)

            wd = d.weekday()  # Mon=0..Sun=6
            weekday_factor = weekday_profile[wd]

            date_str = d.strftime("%Y-%m-%d")
            holiday_factor = HOLIDAYS.get(date_str, 1.0)
            # back-to-school secondary bump concentrated in Aug for relevant categories
            if cat in BACK_TO_SCHOOL_CATS and d.month == 8 and 220 <= doy <= 245:
                holiday_factor = max(holiday_factor, 1.3)

            promo_flag = bool(promo_days[i])
            promo_factor = promo_boost if promo_flag else 1.0

            mean_units = base * trend_factor * seasonal_factor * weekday_factor * holiday_factor * promo_factor
            mean_units = max(mean_units, 1.0)

            # Poisson-ish noise for count data plus a small Gaussian relative jitter
            gaussian_jitter = rng.normal(1.0, noise_frac)
            lam = max(mean_units * max(gaussian_jitter, 0.3), 1.0)
            total_units = rng.poisson(lam)

            # split across regions
            region_shares = np.array([REGION_WEIGHTS[r] for r in REGIONS])
            region_noise = rng.dirichlet(region_shares * 50)  # keeps shares close to target with mild noise
            region_units = rng.multinomial(total_units, region_noise) if total_units > 0 else np.zeros(len(REGIONS), dtype=int)

            # price jitter per day (+/-3%), promo days have a modest discount reflected in avg_unit_price
            price_jitter = rng.normal(1.0, 0.03)
            price_today = unit_price * price_jitter * (0.90 if promo_flag else 1.0)
            price_today = round(max(price_today, 0.5), 2)

            for r_idx, region in enumerate(REGIONS):
                units = int(region_units[r_idx])
                if units <= 0 and rng.random() > 0.02:
                    # keep occasional true zero days but avoid excessive sparsity
                    continue
                # small per-region price noise
                region_price = round(price_today * rng.normal(1.0, 0.01), 2)
                revenue = round(units * region_price, 2)
                rows.append((
                    date_str, cat, units, revenue, region_price, region, promo_flag
                ))

    df = pd.DataFrame(rows, columns=[
        "date", "product_category", "units_sold", "revenue",
        "avg_unit_price", "store_region", "promotion_flag"
    ])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "product_category", "store_region"]).reset_index(drop=True)
    return df


def main():
    df = generate()
    out_path = "data/raw/sales_history.csv"
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")
    print(f"Date range: {df['date'].min()} -> {df['date'].max()}")
    print(f"Categories: {sorted(df['product_category'].unique())}")
    print(f"Regions: {sorted(df['store_region'].unique())}")

    # Quick sanity check: monthly means by category to eyeball seasonality
    df["month"] = df["date"].dt.month
    pivot = df.groupby(["product_category", "month"])["units_sold"].mean().unstack(0)
    pd.set_option("display.width", 200)
    print("\nMean daily units_sold by month (rows=month, cols=category):")
    print(pivot.round(1))

    total_by_cat = df.groupby("product_category")["units_sold"].sum().sort_values(ascending=False)
    print("\nTotal units_sold by category (2023-2025):")
    print(total_by_cat)


if __name__ == "__main__":
    main()
