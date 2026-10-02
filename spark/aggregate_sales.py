"""
Spark job that reads raw.sales_events from Postgres via JDBC, computes
daily/category-level aggregations plus 7-day and 30-day rolling average
demand and week-over-week growth rate, and writes the result to
staging.sales_rollups in Postgres (also via JDBC).

Run via spark-submit inside the spark-master/worker containers, e.g.:

    spark-submit --master spark://spark-master:7077 \
        --jars /opt/spark/extra-jars/postgresql-42.7.3.jar \
        /opt/spark_jobs/aggregate_sales.py

Falls back gracefully with a clear log message and non-zero style warning
if the JDBC driver jar cannot be found, though this should not occur once
spark/jars/postgresql-42.7.3.jar is downloaded (see spark/README note / the
Dockerfile-less bootstrap step run before this job is invoked).
"""

import os
import sys

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

PG_HOST = os.environ.get("POSTGRES_HOST", "postgres")
PG_PORT = os.environ.get("POSTGRES_PORT", "5432")
PG_DB = os.environ.get("POSTGRES_DB", "sales_forecast")
PG_USER = os.environ.get("POSTGRES_USER", "sales_user")
PG_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "sales_pass")

JDBC_URL = f"jdbc:postgresql://{PG_HOST}:{PG_PORT}/{PG_DB}"
JDBC_PROPS = {
    "user": PG_USER,
    "password": PG_PASSWORD,
    "driver": "org.postgresql.Driver",
}


def main():
    spark = (
        SparkSession.builder
        .appName("sales-aggregation")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print(f"Reading raw.sales_events from {JDBC_URL} ...")
    raw = spark.read.jdbc(url=JDBC_URL, table="raw.sales_events", properties=JDBC_PROPS)
    row_count = raw.count()
    print(f"Read {row_count} rows from raw.sales_events")

    if row_count == 0:
        print("No rows found in raw.sales_events — nothing to aggregate. Exiting.")
        spark.stop()
        sys.exit(0)

    # Dedupe on the natural key (same rule as dbt's stg_sales_events: keep the
    # most recently ingested row) so a replayed event can't be double-counted.
    latest_first = (
        Window.partitionBy("event_date", "product_category", "store_region")
        .orderBy(F.col("ingested_at").desc(), F.col("id").desc())
    )
    deduped = (
        raw.withColumn("_rn", F.row_number().over(latest_first))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )
    dup_count = row_count - deduped.count()
    print(f"Dropped {dup_count} duplicate rows on (event_date, product_category, store_region)")

    # Daily totals per category
    daily = (
        deduped.groupBy("event_date", "product_category")
        .agg(
            F.sum("units_sold").alias("daily_units"),
            F.sum("revenue").alias("daily_revenue"),
        )
    )

    cat_window_7 = (
        Window.partitionBy("product_category")
        .orderBy(F.col("event_date").cast("timestamp").cast("long"))
        .rangeBetween(-6 * 86400, 0)
    )
    cat_window_30 = (
        Window.partitionBy("product_category")
        .orderBy(F.col("event_date").cast("timestamp").cast("long"))
        .rangeBetween(-29 * 86400, 0)
    )
    week_lag_window = Window.partitionBy("product_category").orderBy("event_date")

    result = (
        daily
        .withColumn("rolling_avg_units_7d", F.avg("daily_units").over(cat_window_7))
        .withColumn("rolling_avg_units_30d", F.avg("daily_units").over(cat_window_30))
        .withColumn("units_7d_ago", F.lag("daily_units", 7).over(week_lag_window))
        .withColumn(
            "wow_growth_rate",
            F.when(
                (F.col("units_7d_ago").isNotNull()) & (F.col("units_7d_ago") != 0),
                (F.col("daily_units") - F.col("units_7d_ago")) / F.col("units_7d_ago"),
            ).otherwise(F.lit(None)),
        )
        .select(
            "event_date", "product_category", "daily_units", "daily_revenue",
            "rolling_avg_units_7d", "rolling_avg_units_30d", "wow_growth_rate",
        )
    )

    out_count = result.count()
    print(f"Computed {out_count} aggregated rows. Writing to staging.sales_rollups ...")

    # Truncate + append (rather than Spark's `overwrite`, which would DROP/CREATE
    # the table and lose the PRIMARY KEY defined in postgres_init) for idempotency.
    truncate_props = dict(JDBC_PROPS)
    truncate_props["truncate"] = "true"
    result.write.jdbc(
        url=JDBC_URL,
        table="staging.sales_rollups",
        mode="overwrite",
        properties=truncate_props,
    )

    print(f"Wrote {out_count} rows to staging.sales_rollups")
    spark.stop()


if __name__ == "__main__":
    main()
