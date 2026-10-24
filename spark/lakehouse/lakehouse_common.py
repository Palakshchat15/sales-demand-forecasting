"""
Shared settings for the Delta Lake lakehouse jobs (bronze/silver/gold).

The jobs run with spark-submit in the sales_spark_master container (driver) and the
sales_spark_worker container (executor). Both containers bind-mount the host folder
data/lakehouse at /opt/lakehouse, so the driver (which commits the _delta_log) and the
executor (which writes the Parquet files) see the same tables. Results go to
/opt/results (host folder results/, mounted on the master only: only the driver writes).

Jars (spark/jars/, mounted at /opt/spark/extra-jars): postgresql-42.7.3.jar,
delta-spark_2.12-3.2.1.jar, delta-storage-3.2.1.jar (spark/get_delta_jars.sh).
"""

import os
from datetime import datetime, timezone

from pyspark.sql import SparkSession

LAKEHOUSE_ROOT = os.environ.get("LAKEHOUSE_ROOT", "/opt/lakehouse")
RESULTS_DIR = os.environ.get("LAKEHOUSE_RESULTS_DIR", "/opt/results")

BRONZE = f"{LAKEHOUSE_ROOT}/bronze/sales_events"
SILVER = f"{LAKEHOUSE_ROOT}/silver/sales_events"
GOLD = f"{LAKEHOUSE_ROOT}/gold/daily_category_rollups"
TABLES = (("bronze", BRONZE), ("silver", SILVER), ("gold", GOLD))

PG_HOST = os.environ.get("POSTGRES_HOST", "postgres")
PG_PORT = os.environ.get("POSTGRES_PORT", "5432")
PG_DB = os.environ.get("POSTGRES_DB", "sales_forecast")
JDBC_URL = f"jdbc:postgresql://{PG_HOST}:{PG_PORT}/{PG_DB}"
JDBC_PROPS = {
    "user": os.environ.get("POSTGRES_USER", "sales_user"),
    "password": os.environ.get("POSTGRES_PASSWORD", "sales_pass"),
    "driver": "org.postgresql.Driver",
}


def get_spark(app_name):
    """SparkSession with Delta Lake enabled. Small data (43,823 events, 8,768 rollups), so
    few shuffle partitions: the default 200 would write 200 tiny files per MERGE."""
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def run_id():
    """Batch id stamped on bronze/silver rows: the Airflow run id when called from the DAG."""
    rid = os.environ.get("LAKEHOUSE_RUN_ID", "").strip()
    return rid or "manual_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def is_delta_table(path):
    return os.path.isdir(os.path.join(path, "_delta_log"))


def read_pg(spark, table):
    return spark.read.jdbc(url=JDBC_URL, table=table, properties=JDBC_PROPS)


def delta_ref(path):
    from lakehouse_logic import quote_path
    return quote_path(path)


def history(spark, path, limit=None):
    """DESCRIBE HISTORY as a list of dicts, newest first."""
    sql = f"DESCRIBE HISTORY {delta_ref(path)}" + (f" LIMIT {int(limit)}" if limit else "")
    out = []
    for r in spark.sql(sql).collect():
        d = r.asDict()
        out.append({
            "version": d["version"],
            "timestamp": d["timestamp"],
            "operation": d["operation"],
            "operationParameters": dict(d.get("operationParameters") or {}),
            "operationMetrics": dict(d.get("operationMetrics") or {}),
        })
    return out


def current_version(spark, path):
    return history(spark, path, limit=1)[0]["version"]
