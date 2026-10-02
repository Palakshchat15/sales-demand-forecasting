# Sales / Demand Forecasting Pipeline

End-to-end data engineering + ML + GenAI portfolio project simulating a
retailer's real-time sales stream, demand forecasting, and automated
demand-planning reporting.

> All numbers in this document are real outputs from triggered
> `sales_pipeline_dag` runs, not placeholders. After an audit (section 12,
> "Audit fixes") the model results changed substantially; the current numbers
> come from run `manual__audit_fixes_2` (all 11 tasks succeeded).

## 1. Architecture

```
 data/generator/generate_sales_data.py
              |
              v
 data/raw/sales_history.csv  (3 years, 8 categories, 5 regions)
              |
              v
 +-------------------------+        +----------------------+
 |  Kafka producer.py      |------->|  Kafka topic          |
 |  (replays CSV rows)     |        |  sales-events         |
 +-------------------------+        +----------+-----------+
                                                |
                                                v
                                     +----------------------+
                                     |  Kafka consumer.py    |
                                     |  micro-batch writer   |
                                     +----------+-----------+
                                                |
                                                v
                                  Postgres: raw.sales_events
                                                |
                     +--------------------------+-------------------------+
                     |                                                    |
                     v                                                    v
        Great Expectations validation                     Spark (spark-submit, JDBC)
        (raw.sales_events schema/range checks)             aggregate_sales.py
                     |                                      7d/30d rolling avg, WoW growth
                     |                                                    |
                     |                                                    v
                     |                                Postgres: staging.sales_rollups
                     |                                                    |
                     +-------------------+--------------------------------+
                                         v
                                    dbt run / dbt test
                        staging (stg_*) -> intermediate (int_*) -> marts
                                         |
                                         v
                     warehouse.mart_category_daily_sales (forecast-ready)
                     warehouse.dim_product_category / fact_daily_sales
                                         |
                +------------------------+------------------------+
                v                                                  v
   src/ml_pipeline/forecast_prophet.py               src/ml_pipeline/forecast_compare.py
   Prophet per category, 30/60/90d forecast          LightGBM (direct 90-day-ahead) + seasonal
   -> warehouse.demand_forecast                      naive baseline -> outputs/model_comparison.csv
   -> outputs/forecast_plots/*.png
                |                                                  |
                +------------------------+-------------------------+
                                         v
                        src/ml_pipeline/report_generator.py
              Groq / OpenAI / local Ollama (llama3.2) -> business report
                                         v
                        outputs/demand_planning_report.pdf

  Orchestration: airflow/dags/sales_pipeline_dag.py (Airflow, LocalExecutor)
  All services: docker-compose.yml (Postgres, Zookeeper, Kafka, Spark
  master/worker, Airflow webserver + scheduler)
```

### Reconciling Kafka streaming with Airflow batch scheduling
Kafka topics model an always-on stream, but Airflow DAG runs are bounded
batch executions. Rather than assuming a permanently-running consumer, the
DAG's `kafka_replay_ingest` task triggers a **bounded replay**: the producer
sends up to a fixed number of rows starting from an on-disk offset-state
file (`kafka/.producer_state.json`), and the consumer runs with an idle
timeout so it naturally exits once the topic is drained. Each DAG run thus
performs a real producer -> broker -> consumer -> Postgres round trip, while
remaining deterministic and finite, as required for Airflow scheduling.

## 2. Folder structure

```
sales_forecast_project/
├── data/
│   ├── generator/generate_sales_data.py
│   ├── raw/sales_history.csv
│   └── processed/  (dashboard inputs: forecast_vs_actual, model_comparison,
│                    monthly_category_sales, next_90_day_forecast)
├── kafka/
│   ├── producer.py
│   └── consumer.py
├── spark/
│   ├── aggregate_sales.py
│   └── jars/postgresql-42.7.3.jar
├── airflow/
│   ├── Dockerfile
│   └── dags/sales_pipeline_dag.py
├── dbt/
│   ├── profiles.yml
│   └── sales_dbt/
│       ├── dbt_project.yml
│       ├── models/
│           ├── staging/  (stg_sales_events, stg_sales_rollups, sources.yml, schema.yml)
│           ├── intermediate/  (int_daily_category_sales)
│           └── marts/  (dim_product_category, fact_daily_sales, mart_category_daily_sales, schema.yml)
│       └── tests/generic/unique_combination_of_columns.sql  (natural-key uniqueness test)
├── great_expectations/
│   └── validate_sales_events.py
├── src/ml_pipeline/
│   ├── db_utils.py
│   ├── forecast_prophet.py
│   ├── forecast_compare.py
│   └── report_generator.py
├── src/dashboards/
│   ├── export_dashboard_data.py
│   ├── build_excel_dashboard.py
│   └── build_tableau_workbook.py
├── tableau/Sales_Forecast.twbx
├── outputs/
│   ├── forecast_plots/*.png
│   ├── prophet_holdout_metrics.csv
│   ├── prophet_tuning.csv
│   ├── model_comparison.csv
│   ├── Sales_Forecast_Dashboard.xlsx
│   ├── demand_planning_report.pdf
│   └── demand_planning_report.txt
├── postgres_init/01_schemas.sql
├── docker-compose.yml
├── requirements-airflow.txt
├── .env.example
└── README_PIPELINE.md
```

## 3. Synthetic data assumptions

- 3 years daily data: 2023-01-01 through 2025-12-31, 8 product categories
  (Electronics, Apparel, Home & Garden, Groceries, Toys, Sporting Goods,
  Books, Beauty), 5 store regions (North, South, East, West, Online).
- Each category has a hand-coded yearly trend (compounding annual growth
  rate), a yearly seasonal cosine cycle with its own amplitude and peak
  day-of-year (e.g. Toys peak ~Nov/Dec with amplitude 0.9; Home & Garden
  peaks in spring/summer), a per-weekday multiplier profile (weekend
  spikes), fixed holiday-date multipliers (Black Friday ~3.2x, Cyber
  Monday, Christmas rush, back-to-school, New Year, Memorial Day, July 4th),
  random promotion days (~3% of days, 1.2-1.8x boost, with a modest price
  discount), and Poisson noise around the deterministic mean plus a small
  Gaussian relative jitter — deliberately dominated by the coded signal.
- **This structure favours Prophet.** The generator's formula is
  trend x weekday x yearly-cosine x holiday multipliers, which is exactly
  Prophet's multiplicative model form. Prophet is no longer given the
  generator's holiday dates (see "Audit fixes"), but its model form still
  matches how the data was made, so a Prophet win here says little about how
  it would do on real retail data, where demand is not generated by Prophet's
  own equation. Treat the model ranking as a test of the pipeline, not as
  evidence about which model is better in general.
- Regional split uses a Dirichlet-perturbed multinomial draw around fixed
  target shares (Online largest at 25%) so each day's total splits
  realistically but noisily across regions.
- See `data/generator/generate_sales_data.py` for the exact per-category
  parameters (base level, growth rate, seasonal amplitude/peak, weekday
  profile, noise fraction, unit price, promo boost).

## 4. How to run from scratch

```bash
cp .env.example .env          # fill in GROQ_API_KEY/OPENAI_API_KEY if you have one; Ollama works with no key
python data/generator/generate_sales_data.py   # writes data/raw/sales_history.csv

docker compose up -d postgres
docker compose up -d zookeeper
docker compose up -d kafka
docker compose up -d spark-master spark-worker
docker compose up -d airflow-init      # runs migrations + creates admin user, then exits
docker compose up -d airflow-webserver airflow-scheduler

# Airflow UI: http://localhost:8082  (admin/admin)
# Trigger the pipeline:
docker compose exec airflow-webserver airflow dags trigger sales_pipeline_dag
docker compose exec airflow-webserver airflow tasks states-for-dag-run sales_pipeline_dag <run_id>
```

## 5. Libraries

See `requirements-airflow.txt` for the full pinned list (installed inside
the Airflow image, used by the Airflow tasks that call the Kafka/Spark
orchestration and Python ML scripts directly). Key pins and why:
- `sqlalchemy>=1.4.36,<2.0` — required by both apache-airflow 2.9.3 and dbt-postgres 1.7.13
- `numpy==1.26.4`, `pandas==2.1.4` — required for prophet/cmdstanpy compatibility
- `cmdstanpy==1.2.0`, `prophet==1.1.5` — pinned together per known compatibility
- `openai==1.54.4` — newer releases ship a broken vendored httpx_aiohttp transport

## 6. Model comparison results

Holdout evaluation: the last 90 days (2025-10-03 to 2025-12-31) are held out.
Every model is trained only on data up to the cutoff (2025-10-02) and
forecasts all 90 days blind; no model sees any holdout actual. The three models:

- **Prophet**: multiplicative trend + weekly + yearly + monthly seasonality,
  US public holidays (`add_country_holidays('US')`), trend flexibility chosen by
  cross-validation on the training data (below).
- **LightGBM**: a direct 90-day-ahead model. Every lag/rolling feature is at
  least 90 days old at the target date (so all of them are known at the cutoff):
  the 28-day level ending 90 days earlier, its year-over-year growth, and last
  year's values 357/364/371 days back relative to last year's level; plus
  calendar features and the same US holiday calendar Prophet uses. Trees cannot
  extrapolate past the training range, so it predicts the ratio of sales to the
  28-day level and multiplies back. This design was fixed before scoring and
  not tuned on the holdout.
- **Seasonal Naive** (baseline): the value 364 days earlier (same weekday)
  times the trailing-year growth (units in the last 364 training days / units
  in the 364 days before that).

| Category | Prophet MAPE | Prophet RMSE | LightGBM MAPE | LightGBM RMSE | Seasonal Naive MAPE | Seasonal Naive RMSE |
|---|---|---|---|---|---|---|
| Apparel | 12.60% | 273.3 | 16.33% | 318.5 | 13.47% | 218.5 |
| Beauty | 10.90% | 181.6 | 12.81% | 193.0 | 11.78% | 129.8 |
| Books | 11.67% | 99.1 | 15.25% | 117.3 | **10.87%** | 60.0 |
| Electronics | 11.09% | 272.4 | 15.90% | 332.8 | 12.59% | 196.6 |
| Groceries | 9.72% | 490.4 | 12.55% | 530.7 | **7.94%** | 291.0 |
| Home & Garden | 15.55% | 87.9 | 14.63% | 85.9 | **14.29%** | 66.5 |
| Sporting Goods | 11.87% | 122.8 | 15.67% | 137.5 | 11.97% | 91.4 |
| Toys | 15.77% | 123.2 | 16.08% | 127.2 | 16.01% | 120.3 |
| **Mean** | **12.40%** | **206.4** | **14.90%** | **230.4** | **12.37%** | **146.8** |

**Selection rule:** the winner is the model with the lowest *simple mean of the
8 per-category MAPEs* (each category weighted equally, regardless of volume).

**Result: no model beats the naive baseline in a meaningful way.** By the rule,
Seasonal Naive "wins" with 12.37% against Prophet's 12.40%, a gap of 0.03
percentage points, which is a tie on one 90-day holdout. Prophet has the lower
MAPE in 5 of 8 categories (Apparel, Beauty, Electronics, Sporting Goods, Toys),
Seasonal Naive in 3 (Books, Groceries, Home & Garden). On RMSE the baseline is
clearly better in every category (mean 146.8 vs 206.4): RMSE is dominated by
the big spike days (Black Friday is 3.2x normal in the generator), and last
year's value on the same weekday lands on those spikes, while Prophet's US
public-holiday calendar has Thanksgiving but not Black Friday, Cyber Monday or
Christmas Eve. **LightGBM is the worst of the three and does not beat the
baseline in any category.**

What this means: the earlier headline (Prophet 8.57% vs LightGBM 13.53%) came
from Prophet being handed the generator's exact holiday dates and from LightGBM
being scored one step ahead; neither was a fair measure. Honestly measured,
the pipeline's models add nothing over "last year x growth" on this data. The
obvious next step is to give every model retail events derivable from the
public calendar (Black Friday = Thanksgiving + 1, Cyber Monday, Christmas Eve),
chosen up front rather than after looking at the holdout; that has not been
done here.

### How Prophet's trend flexibility was chosen

An early version hard-coded `changepoint_prior_scale=0.1`; on the spiky series
the trend absorbed the autumn ramp-up as permanent growth (Toys forecast
+51% YoY for Jan–Mar 2026 vs +5% historically). `changepoint_prior_scale` is
now chosen per category by rolling-origin cross-validation (Prophet's
`cross_validation`, 540-day initial window, 90-day steps and horizon) **on the
training data only**, over `[0.001, 0.005, 0.01, 0.05, 0.1]`. The holdout is
never used for tuning. With the US holiday calendar the picks are 0.005
(Apparel, Sporting Goods, Toys), 0.01 (Books, Electronics, Home & Garden) and
0.05 (Beauty, Groceries). The CV error is fairly flat between 0.005 and 0.05
for most categories (e.g. Groceries 7.25% / 7.21% / 7.21%) and clearly worse
at both ends of the grid. Full CV scores: `outputs/prophet_tuning.csv`.

### Does the Jan–Mar 2026 forecast look like the history?

Prophet's forward forecast (refit on all data) for 2026-01-01..2026-03-31,
average units/day, compared with actuals on the same dates of 2025. For
context: the growth the same quarter showed a year earlier (Q1 2025 vs Q1
2024), the full-year growth 2025 vs 2024, and the growth rate coded in the
generator (the true underlying trend, known only because the data is synthetic).

| Category | Forecast vs Q1 2025 (before fixes) | Forecast vs Q1 2025 (now) | Q1 2025 vs Q1 2024 | 2025 vs 2024 | Generator growth |
|---|---|---|---|---|---|
| Apparel | +10.9% | +11.2% | +3.4% | +8.2% | 8% |
| Beauty | +13.2% | +14.3% | +10.5% | +12.8% | 12% |
| Books | +4.3% | +5.3% | +0.8% | +1.8% | 2% |
| Electronics | +14.4% | +14.7% | +21.9% | +18.2% | 18% |
| Groceries | +2.1% | +2.6% | +3.2% | +3.1% | 3% |
| Home & Garden | +5.6% | +6.0% | +5.7% | +6.3% | 6% |
| Sporting Goods | +5.3% | +6.1% | +5.3% | +6.0% | 7% |
| Toys | +2.3% | +2.0% | +5.1% | +3.7% | 5% |

In line with the history (within about half a point of the 2025 growth):
Groceries, Home & Garden, Sporting Goods. Slightly high: Beauty (+14.3% vs
+12.8%). **Not in line:** Apparel (+11.2% vs +8.2% full-year and +3.4% for the same
quarter) and Books (+5.3% vs +1.8%) are forecast to grow noticeably faster than
they ever have; Electronics (+14.7% vs +18.2%) and Toys (+2.0% vs +3.7%) are
forecast to grow more slowly. The earlier README claimed every category was
in line; that was wrong before the fixes and is still wrong now.

### Forward forecast is refit on all data

The holdout model is trained without the last 90 days so it can be scored. The
next-90-days forecast now comes from a second fit on the **full** history, so it
uses the most recent quarter. Previously the forward forecast came from the
holdout model, which meant it was really forecasting 91–180 days past its
training data.

## 7. Sample forecast (Toys, 30-day horizon, first 10 days)

From `warehouse.demand_forecast` (model_name='prophet'), forecast dates
starting the day after the last historical date (2025-12-31):

| forecast_date | yhat | yhat_lower | yhat_upper |
|---|---|---|---|
| 2026-01-01 | 407.5 | 313.8 | 504.0 |
| 2026-01-02 | 449.8 | 360.1 | 537.2 |
| 2026-01-03 | 515.9 | 427.8 | 609.2 |
| 2026-01-04 | 490.8 | 395.6 | 589.7 |
| 2026-01-05 | 389.3 | 289.0 | 480.0 |
| 2026-01-06 | 368.2 | 279.1 | 457.7 |
| 2026-01-07 | 357.1 | 270.9 | 453.8 |
| 2026-01-08 | 362.5 | 270.8 | 455.6 |
| 2026-01-09 | 410.0 | 316.2 | 499.4 |
| 2026-01-10 | 477.8 | 386.3 | 567.0 |

(Before the audit fixes, 1 January was 508.7 because the generator's own
New Year spike date was in Prophet's holiday list.) Weekends are higher, as in
the history. See `outputs/forecast_plots/Toys_prophet_forecast.png`: the
forecast declines to about 180 units/day in the last week of March, similar to
the same weeks in earlier years. The plot also shows the holdout forecast
missing the Black Friday spike (about 1,340 units actual), which is not in the
US public-holiday calendar.

## 8. GenAI report excerpt

Generated by local Ollama (`llama3.2`, via the OpenAI-compatible endpoint
at `http://host.docker.internal:11434/v1`) during the real DAG run — no
GROQ_API_KEY or OPENAI_API_KEY was configured in this environment, so the
pipeline correctly fell through to Ollama per the documented preference
order, verifying reachability first via a real GET to `/api/tags`. Full
text saved at `outputs/demand_planning_report.txt`; rendered PDF at
`outputs/demand_planning_report.pdf` (includes the model comparison table as
an appendix). Excerpt from run `manual__audit_fixes_2`:

> **Model Performance & Recommendation**
>
> Our models were evaluated on their performance using a past backtest window
> from October 3rd to December 31st, 2025. The results show:
>
> * Seasonal Naive has the lowest mean holdout MAPE (12.37%), followed closely
>   by Prophet (12.40%).
> * LightGBM has the highest mean holdout MAPE (14.90%).
> * Prophet outperformed Seasonal Naive in 5 categories, while LightGBM
>   performed poorly in all categories.
> * Seasonal Naive outperformed Prophet in 3 categories, and LightGBM beat the
>   Seasonal Naive baseline in none.
>
> Given the small gap between the models, we recommend keeping Prophet as our
> forward forecasting model for the next 90 days. We will also maintain
> Seasonal Naive as a benchmark model [...]

**Checked against the data.** Every number in that report matches the data:
the 8 forecast averages (e.g. Electronics 792.7/day, Groceries 1,741.5/day),
the 8 year-over-year growth figures (Electronics +16.7% ... Books +1.7%), the
three mean MAPEs and the win counts. Wording problems that remain: it calls
the forecast "our forecast models" when only Prophet produces it; "past year"
for what is really the last 90 days compared with a year earlier; "LightGBM
performed poorly in all categories" overstates things (it beat Prophet on Home
& Garden, 14.63% vs 15.55%, though never the baseline); and it left out the
Action Items section it was asked for. The recommendation itself (keep Prophet,
use Seasonal Naive as the benchmark, don't switch over a 0.03-point gap) is
decided in pandas and handed to the model; an earlier draft, before that
change, wrongly told readers to "continue using Seasonal Naive as our forward
forecast model" and gave the backtest window as the forecast period.

**Keeping a small local model factual.** Given the raw 16-row comparison table,
llama3.2 invented results ("LightGBM performed better on Electronics and
Groceries": false, its error there was about double Prophet's) and called
Groceries the slowest-growing category when three categories declined faster.
`report_generator.py` now computes the per-category winners and the growth and
decline rankings in pandas, and passes them as explicit facts the model must
restate rather than derive. The per-window (days 1-30 / 31-60 / 61-90) table
is no longer sent, because llama3.2 quoted the days 1-30 value as the 90-day
average. The trend facts are now year over year (last 90
days vs the same dates a year earlier) and the forecast facts compare the
Jan–Mar 2026 forecast with the same dates of 2025, each under its own heading;
the earlier "Category Forecast Highlights" section actually showed a
last-30-vs-prior-30-days *actual* trend (see "Audit fixes", item 4).

## 9. Dashboards

Both are rebuilt by the DAG on every run (`export_dashboard_data` →
`build_excel_dashboard` + `build_tableau_workbook`, in parallel with the AI report).

**Tableau: `tableau/Sales_Forecast.twbx`** (open in Tableau Public). One
dashboard, "Sales Demand Forecasting Overview", with automatic sizing. It uses the same layout as the
Retail Analytics dashboard (8 KPI tiles, then 6 charts in a 3 x 2 grid) with a
dark green theme shared with the Fraud dashboard: background `#0e1f16`, text
`#9ccfae`, white titles, KPI labels `#7fd49a`, fixed mark colour `#3fa968`.
Measures on colour use a green gradient (light `#c7ebd2` for low values → dark
`#2e8b57` for high). There are no gridlines, zero lines, axis lines, dividers or zone borders.

KPI tiles (centred): total revenue **$249.1M**, units sold **5.26M** (3 years);
**Q4 units YoY +7.6%** (Q4 2025 vs Q4 2024); **90-day forecast 477,848 units**,
**+7.3% vs Q1 2025** (the same calendar quarter a year earlier); holdout mean MAPE for
the **seasonal-naive baseline 12.37%**, **Prophet 12.40%**, **LightGBM 14.90%**.

| Chart | Type | What it shows |
|---|---|---|
| Daily Units: Actual, Backtest, Forecast | Line + **category dropdown** | One category at a time: the last 180 days of actuals, Prophet's 90-day holdout backtest, and the next 90 days. The title doubles as the colour legend (Actual green, Forecast light green, Backtest near-white, drawn on top of Actual). |
| Holdout MAPE % (darker = worse) | Heatmap | MAPE for each category and model. LightGBM is the darkest column. Prophet and the baseline ("Naive") trade wins: Prophet is better in 5 of 8 categories, but the baseline is much better on Groceries (7.9 vs 9.7), so the means tie. |
| Next 90 Days Forecast (Units, vs Q1 2025) | Bar, sorted | Forecast units per category, labelled with growth vs the same quarter last year. Groceries is the largest (156,739, +2.6%). Electronics and Beauty grow fastest (+14.7%, +14.3%). |
| Monthly Revenue, All Categories ($M) | Area | 36 months of revenue: a strong Q4 peak each year and year-on-year growth. |
| Seasonality Index (100 = category's avg day) | Heatmap | Average daily units per calendar month relative to each category's average. Toys swings from 11 (Jun) to 207 (Dec). Home & Garden peaks in spring (Apr–Jun). Groceries is nearly flat. This explains why the seasonal-naive baseline is hard to beat. |
| Share of Revenue by Category (3 Years) | Treemap | Electronics alone is 48% of revenue ($118.6M), then Apparel 15% and Home & Garden 13%. |

The builder (`src/dashboards/build_tableau_workbook.py`) precomputes small
`dash_*.csv` tables in `data/processed/` from the files already there
(`forecast_vs_actual.csv`, `model_comparison.csv`, `monthly_category_sales.csv`,
`next_90_day_forecast.csv`). It writes each table to a `.hyper` extract (Tableau
Public only opens extract-based workbooks) and packages everything into the `.twbx`. The DAG task
`build_tableau_workbook` runs it unchanged.

Tableau quirks found while building this (in Tableau Public 2026.2):
- A categorical colour map written into the XML is silently ignored. We tried it at
  worksheet level, at data-source level (before and after the extract) and with a named
  palette, and Tableau kept its default blue/orange/red. So the forecast chart colours its
  series through a numeric `series_code` on a custom green ramp, and puts the legend in the
  title. A separate legend zone would show a meaningless 0–2 gradient, so there isn't one.
- The green gradient uses a `custom-interpolated` colour palette. Tableau's built-in
  sequential palettes default to blue.
- Mark-label colour is set with a `customized-label` run (dark text inside heatmap and
  treemap cells). The `label` style element colours row and column headers, not mark labels.
- The category dropdown keeps Tableau's default grey box: `quick-filter` style rules were
  ignored.

**Excel: `outputs/Sales_Forecast_Dashboard.xlsx`.** Dashboard sheet with six KPI
cards (total revenue $249M, units sold 5,258,922, Prophet mean MAPE 12.4%,
LightGBM mean MAPE 14.9%, Seasonal Naive mean MAPE 12.4%, next-90-day Prophet
forecast 477,848 units) and four charts: backtest (Prophet) vs actual, all
categories; MAPE by category for all three models; monthly revenue; and
next-90-day forecast by category. The KPI cards are formulas over
the data sheets (`ForecastVsActual`, `ModelComparison`, `MonthlySales`,
`Next90Days`), so they recalculate if the data is refreshed.

**How they were verified.** The workbooks the pipeline produced were opened in
the real applications. For Tableau, a script waits for Tableau's log to show
the workbook finished loading (or an error dialog), then screenshots it. The
redesigned green dashboard went through several build → open → screenshot passes
until every chart rendered in green with readable labels, centred and untruncated
KPI values, and no gridlines or borders. The final pass loaded with no error dialog. For
Excel, the file is opened through Excel's COM interface, fully recalculated,
checked for formula errors (none), and each chart is exported to an image and
inspected. The KPI totals match the raw data exactly ($249,071,160 and
5,258,922 units).

## 10. End-to-end verification evidence

All of the following were actually executed and verified in this
environment (not just written):

- `docker compose up -d` brought up postgres, zookeeper, kafka, spark-master,
  spark-worker, airflow-webserver, airflow-scheduler — all reported healthy.
- `airflow dags list-import-errors` → **No data found** (DAG parses cleanly).
- Kafka producer replayed all 43,823 CSV rows onto the `sales-events` topic
  in two passes (5,000 then 38,823, exercising the offset-resume logic);
  the consumer wrote all 43,823 rows into `raw.sales_events`, verified via
  `SELECT count(*)`.
- Great Expectations validated all 9 expectations (8 originally, plus the
  natural-key uniqueness check added by the audit fixes) against the full
  43,823-row table — all passed.
- The real dockerized Spark job (`apache/spark:3.5.9`, master+worker,
  submitted via `spark-submit --master spark://spark-master:7077` with the
  Postgres JDBC driver jar) read 43,823 rows via JDBC and wrote 8,768
  aggregated rows (7d/30d rolling averages, WoW growth) to
  `staging.sales_rollups` — no fallback needed, the real Spark+JDBC path
  worked on the first architecture (after swapping `bitnami/spark:3.5`,
  which has been pulled from Docker Hub, for `apache/spark:3.5.9`).
- `dbt run` built all 6 models (3 staging views + dim/fact/mart tables);
  `dbt test` passed all 23 tests (18 originally; the audit fixes added 4
  natural-key uniqueness tests and a not-null test on `store_region`).
- Prophet trained per-category models with the US public-holiday calendar and
  wrote 720 forecast rows (8 categories x 90 days) to `warehouse.demand_forecast`.
- The comparison step scored LightGBM and the seasonal-naive baseline and
  produced `outputs/model_comparison.csv` (24 rows: 8 categories x 3 models).
- The GenAI report step verified no GROQ/OPENAI key was usable, confirmed
  Ollama reachability with a real HTTP call, generated a real report via
  llama3.2, and rendered a real 2-page PDF.
- Finally, the **entire pipeline was triggered for real** via
  `airflow dags trigger sales_pipeline_dag` and all 8 tasks
  (`kafka_replay_ingest`, `great_expectations_validate`,
  `spark_aggregate_sales`, `dbt_run`, `dbt_test`, `prophet_forecast`,
  `comparison_forecast`, `genai_report_generation`) completed with state
  `success`, confirmed via `airflow tasks states-for-dag-run`.
- After adding the tuning, refit and dashboards, the DAG was triggered again
  (`manual__dashboards_1790328653`): all **11** tasks `success`, including
  `export_dashboard_data`, `build_excel_dashboard` and `build_tableau_workbook`.
- After the audit fixes (section 12) the DAG was triggered twice:
  `manual__audit_fixes_1` and, after tightening the AI report's facts,
  `manual__audit_fixes_2`. All **11** tasks `success` in both. Both runs
  produced byte-identical `model_comparison.csv`, `prophet_tuning.csv`,
  `prophet_holdout_metrics.csv` and `forecast_vs_actual.csv`, identical to
  two manual runs of the forecast steps. The metrics in this document come
  from those runs. (The Tableau workbook was rebuilt once more afterwards with
  the same script, only to shorten the dot-plot title so it fits on one line.)

## 11. Known limitations / fallbacks taken

- `bitnami/spark:3.5` is no longer published on Docker Hub (Bitnami removed
  most legacy tags from the free tier in 2025); this project uses
  `apache/spark:3.5.9-scala2.12-java11-python3-r-ubuntu` instead. This is
  the real, fully-dockerized Spark standalone cluster (master + worker)
  requested — not a local-mode or pandas fallback.
- The Airflow containers run as root (`user: "0:0"` in docker-compose) and
  mount the host's Docker socket so the `spark_aggregate_sales` task can
  `docker exec` into the sibling `sales_spark_master` container to run
  `spark-submit`. This is a pragmatic choice for a local portfolio/demo
  environment; a production setup would use the Spark REST/Livy API or a
  dedicated Airflow Spark provider instead of Docker-socket access.
- Neither Prophet nor LightGBM beats the seasonal-naive baseline in a
  meaningful way (section 6). Toys and Home & Garden are the hardest categories
  (about 14-16% MAPE for every model). The most likely gain is giving the
  models retail events derived from the public calendar (Black Friday, Cyber
  Monday, Christmas Eve), chosen up front; not done here.
- The data is synthetic and generated with Prophet's own multiplicative model
  form (section 3), so it favours Prophet, and the error levels are better than
  real retail data would usually give. The ranking is a test of the pipeline,
  not of the models.
- Only one 90-day holdout window is scored. Differences of a fraction of a
  point (e.g. Seasonal Naive vs Prophet, 0.03 points) are within the noise of a
  single window; a rolling multi-origin backtest of all three models would be
  needed to rank them properly.
- Only Prophet produces a forward forecast; LightGBM and the baseline are
  scored on the holdout only.
- The AI report comes from a small local model. It is kept factual by passing
  precomputed facts, but its prose is basic, and a larger hosted model would
  write a better report.

## 12. Audit fixes

An audit of the project found six logic problems (three of them inflated the
headline result) plus some minor ones. Each is listed with what changed and
the numbers before and after. The "after" numbers come from the DAG runs
`manual__audit_fixes_1` and `_2` (identical metrics); the forecast steps were also run twice by hand and
produced byte-identical metrics, forecasts and intervals.

**Headline, before → after (mean holdout MAPE, simple mean of 8 categories):**

| Model | Before | After |
|---|---|---|
| Prophet | 8.57% | 12.40% |
| LightGBM | 13.53% | 14.90% |
| Seasonal Naive (new) | not in comparison | 12.37% |
| Winner | Prophet, by 4.96 points | Seasonal Naive, by 0.03 points (a tie with Prophet) |

Mean RMSE before → after: Prophet 82.1 → 206.4, LightGBM 212.6 → 230.4,
Seasonal Naive (new) 146.8. Per-category before → after:

| Category | Prophet MAPE | LightGBM MAPE | Naive MAPE | Prophet RMSE | LightGBM RMSE | Naive RMSE |
|---|---|---|---|---|---|---|
| Apparel | 9.21 → 12.60% | 13.74 → 16.33% | 13.47% | 152.6 → 273.3 | 281.7 → 318.5 | 218.5 |
| Beauty | 7.75 → 10.90% | 15.67 → 12.81% | 11.78% | 78.2 → 181.6 | 215.5 → 193.0 | 129.8 |
| Books | 6.72 → 11.67% | 13.10 → 15.25% | 10.87% | 32.7 → 99.1 | 97.9 → 117.3 | 60.0 |
| Electronics | 7.77 → 11.09% | 16.31 → 15.90% | 12.59% | 110.8 → 272.4 | 326.9 → 332.8 | 196.6 |
| Groceries | 4.40 → 9.72% | 9.01 → 12.55% | 7.94% | 101.9 → 490.4 | 452.8 → 530.7 | 291.0 |
| Home & Garden | 11.12 → 15.55% | 14.59 → 14.63% | 14.29% | 42.7 → 87.9 | 81.7 → 85.9 | 66.5 |
| Sporting Goods | 8.21 → 11.87% | 13.24 → 15.67% | 11.97% | 55.0 → 122.8 | 124.7 → 137.5 | 91.4 |
| Toys | 13.42 → 15.77% | 12.61 → 16.08% | 16.01% | 82.5 → 123.2 | 119.3 → 127.2 | 120.3 |

1. **LightGBM was scored one step ahead (HIGH).** Its lag-1/7/14/28 and
   rolling features were built over the whole series including the holdout,
   so predicting 31 December used the actual for 30 December, while Prophet
   forecast 90 days blind. *Fix:* `forecast_compare.py` now builds a direct
   90-day-ahead model: every sales-based feature is at least 90 days old at
   the target date, so each holdout day uses only data up to the cutoff
   (details in section 6). *Effect:* LightGBM 13.53% → 14.90%. Interestingly
   it got better in Beauty and Electronics, but it is now worse than the naive
   baseline in every category.

2. **No naive baseline (HIGH).** *Fix:* added `seasonal_naive` (value 364
   days earlier x trailing-year growth, training data only) to
   `model_comparison.csv`, the Excel and Tableau dashboards (MAPE chart, dot
   plot, a KPI card) and the AI report. The AI report's recommendation is now
   decided in pandas: gaps under 0.5 points in mean MAPE are reported as a tie. *Effect:* 12.37% mean MAPE, lowest of
   the three by a hair and clearly lowest on RMSE.

3. **Prophet was handed the answer (HIGH).** `forecast_prophet.py` used a
   custom holiday list copied from the generator's own spike dates (including
   Black Friday, Christmas rush, back-to-school Sundays). *Fix:* removed; Prophet
   now uses `add_country_holidays(country_name='US')`, and LightGBM gets the
   same calendar (`prophet.make_holidays`) as a feature. The README now says
   that the generator's formula is Prophet's own model form (section 3).
   *Effect:* Prophet 8.57% → 12.40%, RMSE 82.1 → 206.4; the US calendar has
   Thanksgiving and Christmas Day but not Black Friday (3.2x in the data),
   Cyber Monday or Christmas Eve. Prophet's CV-chosen `changepoint_prior_scale`
   also changed (Apparel, Sporting Goods, Toys 0.01 → 0.005; Beauty
   0.01 → 0.05; Groceries 0.005 → 0.05).

4. **AI report called seasonal swings "declines" (MEDIUM).** It compared the
   last 30 days with the previous 30 (December vs a November containing Black
   Friday), and presented that under a "Category Forecast Highlights" heading.
   *Fix:* `report_generator.py` now computes (in pandas) the last 90 days
   vs the same dates a year earlier, and separately the Jan–Mar 2026 forecast
   vs the same dates of 2025. Section headings are now "Forecast for the Next
   90 Days" and "Recent Demand Trend (Year over Year)". *Effect:* before,
   5 categories were reported as declining (Apparel −15.8%, Books −11.1%,
   Electronics −6.1%, Groceries −5.7%, Beauty −2.8%); after, all 8 grow year over
   year (Electronics +16.7%, Beauty +13.9%, Apparel +10.2%, Sporting Goods
   +6.5%, Home & Garden +5.0%, Toys +3.8%, Groceries +2.4%, Books +1.7%).

5. **README claimed every forecast was "in line with its history" (MEDIUM).**
   *Fix:* recomputed and stated per category (section 6 table). Forecast YoY
   before → after: Apparel +10.9 → +11.2%, Beauty +13.2 → +14.3%, Books
   +4.3 → +5.3%, Electronics +14.4 → +14.7%, Groceries +2.1 → +2.6%, Home &
   Garden +5.6 → +6.0%, Sporting Goods +5.3 → +6.1%, Toys +2.3 → +2.0%.
   Apparel and Books are forecast well above their history, Electronics and
   Toys below it.

6. **Nothing deduplicated (MEDIUM).** `stg_sales_events.sql` said "dedupe" but
   only filtered; the consumer auto-committed offsets and the producer saved
   its offset only at the end, so a partial rerun would double-count. *Fix:*
   - consumer: offsets committed manually only after the Postgres commit; a
     unique index on `(event_date, product_category, store_region)` plus
     `INSERT ... ON CONFLICT DO NOTHING`, so redelivered events are ignored
     (index also added to `postgres_init/01_schemas.sql`);
   - producer: offset checkpointed after every flushed batch;
   - dbt `stg_sales_events`: keeps one row per natural key (latest
     `ingested_at`); the Spark job applies the same rule before aggregating;
   - tests: a local generic test `unique_combination_of_columns` on
     `stg_sales_events` (date, category, region) and on `stg_sales_rollups`,
     `fact_daily_sales`, `mart_category_daily_sales` (date, category), plus a
     Great Expectations `expect_compound_columns_to_be_unique` check on
     `raw.sales_events`.
   *Verified:* re-sending the first 5,000 CSV rows through Kafka
   (`--reset-state`, separate state file) → the consumer reported 5,000
   consumed, 0 new rows, 5,000 duplicates skipped; `raw.sales_events` stayed at
   43,823 rows / 43,823 distinct keys. The staging dedupe query run against a
   deliberately doubled copy of raw returned 43,823 rows with the later copy
   kept. The existing data had no duplicates, so no metric changed.

**Minor fixes.**
- The report's "30/60/90-day average forecast" was three separate windows;
  it is now labelled "days 1-30 / 31-60 / 61-90 (not cumulative)".
- `avg_unit_price` in `int_daily_category_sales.sql` was a plain mean of the
  regional prices; it is now revenue ÷ units (units-weighted). Not used by any
  model, so no metric changed.
- The winner rule (simple mean of per-category MAPE) is now stated in
  section 6 and in the report.
- Prophet's uncertainty intervals were not reproducible (it samples from
  numpy's global RNG); `np.random.seed(42)` before each `predict()` fixed that.
