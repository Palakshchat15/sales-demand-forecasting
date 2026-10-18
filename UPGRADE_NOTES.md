# Upgrade notes: MLflow experiment tracking (2026-10-17, code; run 2026-10-18)

**Run on 2026-10-18: done.** DAG run `mlflow_upgrade_20261018_015101`: all 11 tasks succeeded; runs
`prophet_forecast` (mean MAPE 12.40) and `comparison_forecast` (winner seasonal_naive) logged with the same
`airflow_run_id`; `sales-prophet-<category>` v1 registered for all 8 categories. Forecasts and metrics
byte-identical to the pre-upgrade backup; only `demand_planning_report.txt` (LLM-written, temperature 0.4)
changed. UI http://localhost:5003. Container memory not measured. Results: `README_PIPELINE.md`, section 13.

## What was added
- An `mlflow` service (tracking server + model registry) in the Docker Compose stack:
  official image `ghcr.io/mlflow/mlflow:v2.16.2`, SQLite metadata store and artifact store on
  the named volume `mlflow_data`, one gunicorn worker, `mem_limit: 512m`, a `/health`
  healthcheck, host port **5003** (UI http://localhost:5003).
- `MLFLOW_TRACKING_URI=http://mlflow:5000` in the shared Airflow environment.
- Logging in both forecasting scripts, after their outputs are written, into experiment
  `sales_demand_forecasting`:
  - `forecast_prophet.py`, run `prophet_forecast`: Prophet settings (holdout, seed, grid, CV
    window, holidays, seasonalities) and each category's chosen `changepoint_prior_scale` as
    params; per-category holdout `mape`/`rmse`, CV MAPE per grid value, `mean_mape` and
    `mean_rmse` as metrics; `prophet_holdout_metrics.csv`, `prophet_tuning.csv`,
    `forecast_plots/` and `data/processed/forecast_vs_actual.csv` as artifacts; each category's
    refit (full-history) model via `mlflow.prophet.log_model`, registered as
    **`sales-prophet-<category>`** (one registered model per category, a new version per run).
  - `forecast_compare.py`, run `comparison_forecast`: comparison settings and LightGBM params;
    every model's per-category `mape`/`rmse` and `<model>.mean_mape` (Prophet, LightGBM,
    seasonal-naive, read from `model_comparison.csv`); the CSV; the per-category LightGBM models
    via `mlflow.lightgbm.log_model` (logged, not registered); tag `winner`.
  - Tags on both: `airflow_dag_id`, `airflow_run_id`, `airflow_task_id`, `airflow_try_number`
    (Airflow's `AIRFLOW_CTX_*` variables; `manual` outside Airflow) and `data_version` (mart rows,
    date range, number of categories: it grows over the first 3 DAG runs on a fresh database).
- `mlflow==2.16.2` in `requirements-airflow.txt` (same version as the server image).

## Files changed
- `docker-compose.yml`: `mlflow` service, `MLFLOW_TRACKING_URI`, `mlflow_data` volume.
- `requirements-airflow.txt`: `mlflow==2.16.2`.
- `src/ml_pipeline/mlflow_tracking.py` (new): the guarded logging helper (identical file in the
  Flight and Fraud projects).
- `src/ml_pipeline/forecast_prophet.py`: module dict `FITTED_MODELS` keeps each category's refit
  model (one added line in `run_for_category`); `main()` returns a data summary; new
  `log_to_mlflow()` after `main()`.
- `src/ml_pipeline/forecast_compare.py`: the same pattern for the LightGBM models and
  `model_comparison.csv`.
- `README_PIPELINE.md`: new section "13. Experiment tracking (MLflow)" with `not run yet`
  placeholders (filled in after the 2026-10-18 run).

## Why the results cannot change
The logging runs after `main()` has written every output, only reads those files and the
already-fitted models, and touches no random state (Prophet's `np.random.seed` calls before each
`predict()` are untouched). Model code, seeds, data and outputs are unchanged. If
`MLFLOW_TRACKING_URI` is unset, `mlflow` cannot be imported, the server does not answer `/health`
within 5 s, or any MLflow call fails, the script prints `WARNING [mlflow]: ...` and the task still
succeeds (HTTP retries capped at 2).

## How to run (in order)
1. Back up the current outputs to compare against (PowerShell, from this folder):
   `New-Item -ItemType Directory -Force ..\..\_work_in_progress\upgrades\baselines; Copy-Item outputs ..\..\_work_in_progress\upgrades\baselines\sales_outputs_pre_mlflow -Recurse; Copy-Item data\processed\forecast_vs_actual.csv ..\..\_work_in_progress\upgrades\baselines\sales_forecast_vs_actual_pre_mlflow.csv`
2. Rebuild the Airflow image (new package): `docker compose build`
3. Start the stack as in README_PIPELINE.md section 4, plus the server:
   `docker compose up -d mlflow` then `curl http://localhost:5003/health` (expect `OK`).
4. Check the image: `docker compose exec airflow-scheduler pip check` and
   `docker compose exec airflow-scheduler python -c "import mlflow, prophet; print(mlflow.__version__)"` (2.16.2).
5. Trigger: `docker compose exec airflow-webserver airflow dags trigger sales_pipeline_dag`; on a
   fresh database this needs 3 runs (CLAUDE.md / README section 4: run 1 stops at
   `prophet_forecast` by design). Compare outputs only after a run on all 43,823 rows.
6. `docker compose stop` when done.

## What to verify
- The `prophet_forecast` and `comparison_forecast` logs end with `[mlflow] logged run ...` and
  have no `WARNING [mlflow]`.
- MLflow UI (http://localhost:5003): experiment `sales_demand_forecasting` has both runs with the
  same `airflow_run_id`; `mean_mape` (Prophet run) and `seasonal_naive.mean_mape` /
  `prophet.mean_mape` / `lightgbm.mean_mape` (comparison run) match `outputs/model_comparison.csv`.
- Models tab: one `sales-prophet-<category>` per category, each with a new version.
- Outputs byte-identical to the backup (full-data run): `Get-FileHash` on
  `prophet_holdout_metrics.csv`, `prophet_tuning.csv`, `model_comparison.csv` and
  `data/processed/forecast_vs_actual.csv`.
- Guard: `docker compose stop mlflow`, clear and re-run `prophet_forecast`; it must succeed with
  one `WARNING [mlflow]: tracking server ... is unreachable` line.
- Memory: `docker stats --no-stream` during the run; record the `mlflow` container and the total.
- Fill in the table in README_PIPELINE.md section 13.

## Expected run time (estimates, not measured)
Image rebuild a few minutes; first pull of the MLflow image a few minutes; logging adds seconds
to each of the two forecasting tasks (a handful of small Prophet/LightGBM models and plots).

## Risks
- **Dependencies:** the Dockerfile installs without Airflow's constraints file, so pip resolves
  `mlflow` against the image. Checked against PyPI metadata (not installed): mlflow 2.16.2 needs
  `sqlalchemy>=1.4,<3`, `protobuf>=3.12,<6`, `packaging<25`, `Flask<4`, `alembic<2`,
  `gunicorn<24`, `pyarrow>=4,<18`, `docker>=4,<8`, `importlib-metadata<9`,
  `opentelemetry-api/sdk<3`, `sqlparse>=0.4,<1`, `numpy<3`, `pandas<3`, `matplotlib<4`,
  `scikit-learn<2`, `scipy<2`. Airflow 2.9.3's Python 3.11 constraints (SQLAlchemy 1.4.52,
  protobuf 4.25.3, packaging 24.1, Flask 2.2.5, alembic 1.13.2, gunicorn 22.0.0, pyarrow 16.1.0,
  docker 7.1.0, importlib-metadata 6.11.0, opentelemetry 1.25.0, sqlparse 0.5.0), dbt-core
  1.7.13 (`protobuf<5`, `sqlparse<0.6`) and this project's `numpy==1.26.4` / `pandas==2.1.4`
  (needed by Prophet 1.1.5) are all inside those bounds, so no downgrade is expected. The
  openai/groq/httpx pins are not touched by mlflow. `pip check` in step 4 confirms it. Fallback
  if the build conflicts: pin `mlflow-skinny==2.16.2` instead (client only).
- **Image:** needs network for the first pull from ghcr.io (tag `v2.16.2` exists). Server and
  client versions must stay equal.
- **Memory:** the server is capped at 512 MB; the real use is not measured yet. Sales peaked at
  2.5 GB in a 4 GB VM; with Kafka/Spark also running, start `mlflow` only when tracking is wanted.
  If it is OOM-killed (healthcheck fails), raise `mem_limit`.
- **Registry names:** 8 registered models, category names turned into slugs (`Home & Garden` ->
  `sales-prophet-home-and-garden`). Each DAG run adds versions; `docker compose down -v` deletes
  all runs.
- Prophet is registered because it is the project's forecasting model; it does **not** beat the
  seasonal-naive baseline (README section 6), and the comparison run records that via `winner`.
