"""
Sales Demand Forecasting pipeline DAG.

Orchestrates: bounded Kafka replay (producer+consumer) -> Great Expectations
validation on raw.sales_events -> Spark aggregation (spark-submit) ->
dbt run -> dbt test -> Prophet forecasting -> comparison model, then in
parallel: GenAI report generation, and dashboard export -> Excel + Tableau.

Reconciling streaming vs batch scheduling:
Kafka is normally an always-on stream, but Airflow schedules bounded batch
runs. Rather than pretending we have a 24/7 consumer, each DAG run triggers
a BOUNDED "replay remaining historical data" pass: the producer sends up to
--limit rows starting from wherever a small on-disk offset-state file
(kafka/.producer_state.json) left off, and the consumer runs with an idle
timeout so it naturally stops once it has drained the topic. This makes the
Kafka step deterministic and DAG-friendly while still exercising a real
producer/consumer/broker round trip on every run.
"""

import logging
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

logger = logging.getLogger(__name__)

PROJECT_ROOT = "/opt/airflow"
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "sales-events")

default_args = {
    "owner": "sales-forecast-pipeline",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


def on_failure_callback(context):
    ti = context.get("task_instance")
    logger.error(
        "Task failed: dag=%s task=%s execution_date=%s try=%s",
        context.get("dag").dag_id if context.get("dag") else "?",
        ti.task_id if ti else "?",
        context.get("execution_date"),
        ti.try_number if ti else "?",
    )


def run_kafka_replay(**kwargs):
    """Bounded replay: run producer (limited rows) then consumer (idle-timeout bounded)."""
    import subprocess

    producer_cmd = [
        "python", f"{PROJECT_ROOT}/kafka/producer.py",
        "--csv", f"{PROJECT_ROOT}/data/raw/sales_history.csv",
        "--bootstrap-servers", KAFKA_BOOTSTRAP,
        "--topic", KAFKA_TOPIC,
        "--speed", "20000",
        "--batch-size", "1000",
        "--limit", "20000",
        "--offset-state-file", f"{PROJECT_ROOT}/kafka/.producer_state.json",
    ]
    logger.info("Running producer: %s", " ".join(producer_cmd))
    result = subprocess.run(producer_cmd, capture_output=True, text=True, timeout=600)
    logger.info(result.stdout)
    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError(f"Kafka producer failed: {result.stderr}")

    consumer_cmd = [
        "python", f"{PROJECT_ROOT}/kafka/consumer.py",
        "--bootstrap-servers", KAFKA_BOOTSTRAP,
        "--topic", KAFKA_TOPIC,
        "--group-id", "sales-consumer-group",
        "--batch-size", "1000",
        "--max-idle-seconds", "20",
    ]
    logger.info("Running consumer: %s", " ".join(consumer_cmd))
    result = subprocess.run(consumer_cmd, capture_output=True, text=True, timeout=600)
    logger.info(result.stdout)
    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError(f"Kafka consumer failed: {result.stderr}")


def run_ge_validation(**kwargs):
    import subprocess

    cmd = ["python", f"{PROJECT_ROOT}/great_expectations/validate_sales_events.py"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    logger.info(result.stdout)
    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError("Great Expectations validation failed")


with DAG(
    dag_id="sales_pipeline_dag",
    default_args=default_args,
    description="Kafka -> Spark -> dbt -> Prophet/LightGBM -> GenAI demand planning report",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["sales-forecasting", "portfolio"],
    on_failure_callback=on_failure_callback,
) as dag:

    kafka_replay = PythonOperator(
        task_id="kafka_replay_ingest",
        python_callable=run_kafka_replay,
    )

    ge_validate = PythonOperator(
        task_id="great_expectations_validate",
        python_callable=run_ge_validation,
    )

    spark_aggregate = BashOperator(
        task_id="spark_aggregate_sales",
        bash_command=(
            "docker exec "
            "-e POSTGRES_HOST=postgres -e POSTGRES_PORT=5432 "
            "-e POSTGRES_USER=$POSTGRES_USER -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD -e POSTGRES_DB=$POSTGRES_DB "
            "sales_spark_master "
            "/opt/spark/bin/spark-submit --master spark://spark-master:7077 "
            "--jars /opt/spark/extra-jars/postgresql-42.7.3.jar "
            "/opt/spark_jobs/aggregate_sales.py"
        ),
        env=os.environ,
    )

    dbt_run = BashOperator(
        task_id="dbt_run",
        bash_command=(
            f"cd {PROJECT_ROOT}/dbt/sales_dbt && "
            f"DBT_PROFILES_DIR={PROJECT_ROOT}/dbt dbt run"
        ),
    )

    dbt_test = BashOperator(
        task_id="dbt_test",
        bash_command=(
            f"cd {PROJECT_ROOT}/dbt/sales_dbt && "
            f"DBT_PROFILES_DIR={PROJECT_ROOT}/dbt dbt test"
        ),
    )

    prophet_forecast = BashOperator(
        task_id="prophet_forecast",
        bash_command=f"cd {PROJECT_ROOT}/src/ml_pipeline && python forecast_prophet.py",
    )

    comparison_forecast = BashOperator(
        task_id="comparison_forecast",
        bash_command=f"cd {PROJECT_ROOT}/src/ml_pipeline && python forecast_compare.py",
    )

    genai_report = BashOperator(
        task_id="genai_report_generation",
        bash_command=f"cd {PROJECT_ROOT}/src/ml_pipeline && python report_generator.py",
    )

    export_dashboard_data = BashOperator(
        task_id="export_dashboard_data",
        bash_command=f"cd {PROJECT_ROOT}/src/dashboards && python export_dashboard_data.py",
    )

    build_excel_dashboard = BashOperator(
        task_id="build_excel_dashboard",
        bash_command=f"cd {PROJECT_ROOT}/src/dashboards && python build_excel_dashboard.py",
    )

    build_tableau_workbook = BashOperator(
        task_id="build_tableau_workbook",
        bash_command=f"cd {PROJECT_ROOT}/src/dashboards && python build_tableau_workbook.py",
    )

    kafka_replay >> ge_validate >> spark_aggregate >> dbt_run >> dbt_test >> prophet_forecast >> comparison_forecast
    comparison_forecast >> genai_report
    comparison_forecast >> export_dashboard_data >> [build_excel_dashboard, build_tableau_workbook]
