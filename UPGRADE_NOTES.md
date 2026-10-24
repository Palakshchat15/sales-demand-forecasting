# Upgrade notes: Delta Lake lakehouse (2026-10-24, code; verified on a minimal stack)

**Verified on 2026-10-24 via direct `spark-submit` on a minimal stack** (Postgres + Spark master/worker,
`_work_in_progress/upgrades/run_sales_delta_min.sh`): first build, parity, Delta demo, second build, parity again,
all exit 0. Bronze got 43,823 rows; silver got 43,823 (MERGE inserted 43,823); gold has 8,768 rows. Parity was
**26/26 PASS** both times, with the maximum difference 0 in every column. The re-run appended 0 rows and the
MERGE inserted/updated/deleted 0. Time travel showed 0 rows before the first MERGE and 43,823 after it. The 4 bad
writes were all rejected, OPTIMIZE/ZORDER kept the data, and the VACUUM dry run listed 0 files. The Postgres
fingerprints and `outputs/` hashes were identical before and after. Peak `docker stats` was 1.9 GB (Spark master
1.0 GB, worker 0.9 GB). Full results: `README_PIPELINE.md`, section 14.

**Not verified: a full 13-task DAG run.** Five attempts of `run_sales_delta.sh` on 2026-10-24 failed or hung on
this 8 GB laptop before the lakehouse tasks started:
- attempts 1, 3 and 4: the *existing* `spark_aggregate_sales` task (unchanged code) failed 3 times with
  `UnknownHostException: postgres`, and the scheduler heartbeat could not resolve `postgres`;
- attempt 2: Docker's bind mounts returned `OSError: [Errno 12] Cannot allocate memory` (Windows commit
  was nearly exhausted; the Ollama tray app held about 4 GB);
- attempt 5: after the static-IP fix below, Docker itself stopped responding a minute after the trigger
  (host free RAM about 400 MB).

So the DAG wiring (13 tasks, parses with no import errors) is in place but has not completed a run here, and
README.md still describes the verified 11-task run. In every attempt, `outputs/`, `data/processed/` and the
Postgres fingerprints stayed unchanged and `dbt test` passed (PASS=23; it could not run in attempt 2).
Logs: `_work_in_progress/upgrades/attempt1_*` .. `attempt4_*`, `run_sales_delta.log` (attempt 5),
`sales_delta_min_*` (the verification run).

## What was added
- A Delta Lake 3.2.1 lakehouse (open source, local) in the medallion layout, built by Spark on
  the existing `apache/spark:3.5.9` cluster from `raw.sales_events`:
  - **bronze** `data/lakehouse/bronze/sales_events`: append-only copy of the Kafka-ingested
    events with `_bronze_batch_id` / `_bronze_loaded_at` / `_source`; only new rows (by `id`,
    `ingested_at`) are appended; `delta.appendOnly = true`.
  - **silver** `data/lakehouse/silver/sales_events`: typed, cleaned, one row per natural key
    `(event_date, product_category, store_region)`, upserted with `MERGE INTO` (insert new
    keys, update only strictly newer rows, so a re-run changes 0 rows); explicit DDL with
    `NOT NULL` columns and `CHECK` constraints `units_sold >= 0`, `revenue >= 0`.
  - **gold** `data/lakehouse/gold/daily_category_rollups`: the rollups of
    `spark/aggregate_sales.py` (daily totals, 7/30-day rolling averages, WoW growth) computed
    on silver, cast to the types of `staging.sales_rollups`, partitioned by
    `product_category`, rebuilt with an atomic overwrite.
- DAG tasks `spark_lakehouse_build` -> `lakehouse_parity` on a branch after
  `comparison_forecast` (13 tasks). Parity: gold vs `staging.sales_rollups` (counts, sums,
  full outer join row by row), bronze/silver vs `raw.sales_events`; writes
  `results/lakehouse_parity.csv`, fails on any mismatch (no retries).
- `lakehouse_demo.py` (run by hand or by the run script): history, time travel, MERGE
  idempotency, schema enforcement and constraints, OPTIMIZE + ZORDER, VACUUM dry run ->
  `results/lakehouse_delta_demo.txt`, `results/lakehouse_history.csv`.

## Design and why
- **Additional, read-only towards Postgres.** The existing `spark_aggregate_sales` -> Postgres
  write is untouched (`aggregate_sales.py` not edited); the lakehouse only reads
  `raw.sales_events` and `staging.sales_rollups`. Gold re-implements the same aggregation
  independently, so the parity check is a real cross-check of two code paths.
- **Branch after `comparison_forecast`, not next to `spark_aggregate_sales`.** Nothing
  downstream reads the lakehouse, so it should not gate dbt or the forecasts; placed earlier,
  its Spark driver (768 MB) and executor (1 GB) would overlap with Prophet/LightGBM in the
  4 GB Docker VM. After `comparison_forecast` it overlaps only with the light dashboard
  tasks and the report task (whose LLM runs on the host). Cost: a lakehouse failure marks the
  DAG run failed even though the forecasts are fine.
- **Bronze source is `raw.sales_events`, not the Kafka topic.** The topic is drained by the
  consumer each run (and the checkpoint is at 43,823, so new runs send nothing); reading the
  Kafka sink keeps one ingestion path and needs no Spark-Kafka connector jars.
- **Bind mount `./data/lakehouse`** (gitignored) on master and worker at `/opt/lakehouse`, so
  the driver and executor see the same tables and the `_delta_log` is visible on the host.
  `./results` is mounted on the master only.
- **Jars instead of `--packages`:** `spark/get_delta_jars.sh` downloads
  `delta-spark_2.12-3.2.1.jar` and `delta-storage-3.2.1.jar` once from Maven Central into
  `spark/jars/` (SHA-1 checked, gitignored), passed with `--jars`; the Delta extension and
  catalog are set in `lakehouse_common.get_spark()`. No network or ivy cache at run time.
  Checked on Maven Central: delta-spark_2.12 3.2.1 is the latest 3.2.x, built against Spark
  3.5.3 / Scala 2.12 (the image is `3.5.9-scala2.12`); its only runtime dependencies besides
  Spark are delta-storage 3.2.1 and antlr4-runtime 4.9.3, which ships with Spark 3.5. Both
  jars were downloaded on 2026-10-24.
- **Small Spark settings:** `spark.sql.shuffle.partitions = 4` (the default 200 would write 200
  tiny files per MERGE), session time zone UTC (the rolling windows cast dates to timestamps).

## Files changed
- `spark/lakehouse/` (new): `build_lakehouse.py`, `lakehouse_parity.py`, `lakehouse_demo.py`,
  `lakehouse_common.py` (paths, JDBC, Delta session), `lakehouse_logic.py` (pure Python: MERGE
  SQL builder, parity rules, history helpers, VACUUM guard).
- `spark/get_delta_jars.sh` (new).
- `tests/test_lakehouse_logic.py` (new): 23 pytest tests for `lakehouse_logic.py`, no Spark
  needed (`python -m pytest tests -q`; all pass on the host with Python 3.11 + pytest).
- `airflow/dags/sales_pipeline_dag.py`: `lakehouse_submit()`, tasks `spark_lakehouse_build`
  and `lakehouse_parity`, dependency `comparison_forecast >> spark_lakehouse_build >>
  lakehouse_parity`; docstring. Existing tasks unchanged.
- `docker-compose.yml`: mounts `./data/lakehouse:/opt/lakehouse` (spark-master, spark-worker)
  and `./results:/opt/results` (spark-master).
  After the DNS failures: a fixed subnet `172.28.0.0/16` with fixed IPs for `postgres` (.10), `kafka` (.11),
  `spark-master` (.12, also `hostname: spark-master`) and `mlflow` (.13), plus `extra_hosts` entries for them
  in the Airflow and Spark containers, so they resolve without Docker's embedded DNS.
- `_work_in_progress/upgrades/run_sales_delta_min.sh` (new, outside the repo): the minimal-stack verification run.
- `.gitignore`: `data/lakehouse/`, `spark/jars/delta-*.jar`.
- `README_PIPELINE.md`: folder structure; new section 14 with `not run yet` placeholders.
- Not changed: `requirements-airflow.txt` and the Airflow image (no new Python package),
  `aggregate_sales.py`, dbt, the forecasting and report code.

## Why the existing results cannot change
The lakehouse tasks only read Postgres (JDBC) and write under `data/lakehouse/` and
`results/`. They start after `comparison_forecast`, when every Postgres write of the run is
done, and no existing task reads `data/lakehouse/` or `results/`. `outputs/` and
`data/processed/` are not touched. Only `demand_planning_report.txt` (LLM-written,
temperature 0.4) may differ, as in every run.

## How to run (in order)
One command does all of it, unattended (Git Bash):
`bash "/c/Users/HP/Downloads/Projects (1)/Projects/_work_in_progress/upgrades/run_sales_delta.sh"`
(or double-click `run_sales_delta_launch.cmd` next to it). It:
1. waits while `_work_in_progress/upgrades/docker_run.lock` exists (another project's run;
   checks every 5 minutes), then creates it and removes it on exit;
2. downloads the Delta jars if missing; backs up `outputs/` and `data/processed/` to
   `_work_in_progress/upgrades/baselines/sales_delta/` and hashes their json/csv/txt files;
3. starts Docker Desktop if needed, `docker compose up -d` (no rebuild needed), waits for
   the webserver, checks the DAG has 13 tasks and no import errors;
4. fingerprints the Postgres tables (row count + md5 of all values, without run timestamps:
   `raw.sales_events`, `staging.sales_rollups`, the 3 marts, `warehouse.demand_forecast`);
5. triggers `sales_pipeline_dag` once (`delta_upgrade_<timestamp>`), waits (up to 60 min),
   samples `docker stats` every 30 s;
6. compares output hashes and Postgres fingerprints; runs `dbt test`;
7. runs `lakehouse_demo.py`, then `lakehouse_parity.py` again;
8. writes `_work_in_progress/upgrades/sales_delta_summary.log`, then `docker compose stop`.

Manual equivalent: README_PIPELINE.md section 14.

## What to verify
- `sales_delta_summary.log`: DAG run success, 13/13 tasks; outputs changed only
  `demand_planning_report.txt`; data/processed and Postgres tables changed: none; dbt test
  `PASS=23`; every demo CHECK line PASS; parity rc=0.
- `results/lakehouse_parity.csv`: all PASS; gold 8,768 rows, silver 43,823.
- `results/lakehouse_delta_demo.txt`: silver MERGE re-run inserted 0 / updated 0; silver time
  travel 0 rows before the first MERGE, 43,823 after; the 4 bad writes rejected.
- `data/lakehouse/*/_delta_log/*.json` on the host: the commits.
- Fill in the table in README_PIPELINE.md section 14 (and update README.md: 13 tasks, the
  lakehouse in the pipeline line) only from these results.

## Expected run time (estimates, not measured)
Compose up about 2-3 min (images present; the Spark containers are recreated); DAG about
5 min (3.3 min before, plus the lakehouse branch of about 1.5-2.5 min, partly in parallel with
the report task); dbt test under 1 min; demo 2-3 min (several Spark jobs, OPTIMIZE; Spark
retries each rejected write's task 4 times); parity about 1 min; stop under 1 min. About
12-15 min in total, plus any wait for the lock.

## Risks
- **Delta on a Windows bind mount:** Delta's commits rely on rename, and Hadoop sets file
  permissions; Docker Desktop bind mounts normally allow both for root, but this is not
  tested here, and many small files on a Windows mount are slow. If a commit or chmod fails,
  switch the two `./data/lakehouse` mounts to a named volume (e.g. `sales_lakehouse`, declared
  under `volumes:`); the tables are then only visible inside the containers.
- **Version compatibility:** Delta 3.2.1 is built against Spark 3.5.3; the image is 3.5.9
  (same minor version, normally binary compatible). If the session fails to start with a
  `NoSuchMethodError` / `ClassNotFoundException`, switch to Delta 3.3.2 (also Spark 3.5 /
  Scala 2.12; SHA-1 `07fff11b74e7284eec27fe106f31c95b3acc8018` for delta-spark,
  `e24b01486ee7a26df8a2735eb186f4ee1821fc9e` for delta-storage) in `get_delta_jars.sh`, the
  DAG and the run script.
- **Spark code not executed yet:** written without a Spark run (no Docker in this phase, no
  local Java). Only the pure-Python logic is unit-tested. Likely first-run issues are SQL
  details (path-based `ALTER TABLE ... ADD CONSTRAINT`, `VACUUM ... DRY RUN`); they fail
  loudly in the task log without touching Postgres.
- **Memory:** driver 768 MB + executor 1 GB on top of the stack (2.5 GB peak before, 4 GB
  cap), next to the report and dashboard tasks. If the VM runs out, stop `mlflow` and
  `airflow-webserver` during the run (as for the Flight project).
- **Parity rounding:** gold casts the same doubles to `DECIMAL(14,4)` / `DECIMAL(10,6)` in
  Spark that Postgres rounds from float8; a value exactly on a half can round differently, so
  the rolling averages / growth have one unit of tolerance; units and revenue must be exact.
- **Fresh database:** after `docker compose down -v`, delete `data/lakehouse/` too; otherwise
  bronze/silver keep the old events and parity fails until all 43,823 rows are reloaded.
- **Growth:** each DAG run adds a few Delta commits (a bronze append only when there are new
  rows, a silver MERGE, a gold overwrite); replaced files stay for 7 days (time travel). One
  full copy of the data is about 1-2 MB, so this is negligible.

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
