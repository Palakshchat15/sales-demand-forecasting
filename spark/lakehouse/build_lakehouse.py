"""
Build the Delta Lake lakehouse (medallion layers) from raw.sales_events.

    bronze/sales_events            append-only copy of the Kafka-ingested events
                                   (+ _bronze_batch_id, _bronze_loaded_at, _source);
                                   only rows not yet in bronze are appended
    silver/sales_events            typed, cleaned, one row per natural key
                                   (event_date, product_category, store_region),
                                   maintained with MERGE INTO; NOT NULL + CHECK constraints
    gold/daily_category_rollups    daily category totals, 7/30-day rolling averages and
                                   week-over-week growth (the logic of spark/aggregate_sales.py),
                                   partitioned by product_category, rebuilt with an atomic overwrite

Read-only towards Postgres: the existing staging.sales_rollups, dbt models and forecasts
are not touched. Re-running is idempotent: bronze appends 0 rows, the silver MERGE changes
0 rows, gold is rewritten with identical contents (a new version, same data).

    spark-submit --master spark://spark-master:7077 \
        --jars /opt/spark/extra-jars/postgresql-42.7.3.jar,/opt/spark/extra-jars/delta-spark_2.12-3.2.1.jar,/opt/spark/extra-jars/delta-storage-3.2.1.jar \
        /opt/spark_jobs/lakehouse/build_lakehouse.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyspark.sql import Window  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

import lakehouse_common as C  # noqa: E402
import lakehouse_logic as L  # noqa: E402

BRONZE_DDL = """
CREATE TABLE IF NOT EXISTS {ref} (
    id BIGINT,
    event_date DATE,
    product_category STRING,
    units_sold INT,
    revenue DECIMAL(14,2),
    avg_unit_price DECIMAL(10,2),
    store_region STRING,
    promotion_flag BOOLEAN,
    ingested_at TIMESTAMP,
    _bronze_batch_id STRING,
    _bronze_loaded_at TIMESTAMP,
    _source STRING
) USING DELTA
COMMENT 'Bronze: raw sales events copied from Postgres raw.sales_events (Kafka sink), append-only'
TBLPROPERTIES ('delta.appendOnly' = 'true')
"""

SILVER_DDL = """
CREATE TABLE IF NOT EXISTS {ref} (
    event_date DATE NOT NULL,
    product_category STRING NOT NULL,
    store_region STRING NOT NULL,
    units_sold INT NOT NULL,
    revenue DECIMAL(14,2) NOT NULL,
    avg_unit_price DECIMAL(10,2) NOT NULL,
    promotion_flag BOOLEAN NOT NULL,
    source_id BIGINT NOT NULL,
    source_ingested_at TIMESTAMP NOT NULL,
    _silver_batch_id STRING,
    _silver_updated_at TIMESTAMP
) USING DELTA
COMMENT 'Silver: cleaned sales events, one row per (event_date, product_category, store_region), upserted with MERGE'
"""

SILVER_CONSTRAINTS = (
    ("units_sold_non_negative", "units_sold >= 0"),
    ("revenue_non_negative", "revenue >= 0"),
)


def ensure_tables(spark):
    """Create bronze and silver with explicit schemas (schema enforcement from the first write)."""
    if not C.is_delta_table(C.BRONZE):
        spark.sql(BRONZE_DDL.format(ref=C.delta_ref(C.BRONZE)))
        print(f"Created bronze table {C.BRONZE} (delta.appendOnly = true)")
    if not C.is_delta_table(C.SILVER):
        spark.sql(SILVER_DDL.format(ref=C.delta_ref(C.SILVER)))
        for name, expr in SILVER_CONSTRAINTS:
            spark.sql(f"ALTER TABLE {C.delta_ref(C.SILVER)} ADD CONSTRAINT {name} CHECK ({expr})")
        print(f"Created silver table {C.SILVER} with CHECK constraints "
              f"{', '.join(n for n, _ in SILVER_CONSTRAINTS)}")


def ingest_bronze(spark, batch_id):
    """Append the raw.sales_events rows that bronze does not hold yet.

    A row is identified by (id, ingested_at): the consumer only inserts (ON CONFLICT DO
    NOTHING), so every new Postgres row has a new id; a re-created database gets new
    ingested_at values, so its rows are appended as a new batch instead of being skipped.
    Returns (raw_rows, appended_rows)."""
    raw = C.read_pg(spark, "raw.sales_events").cache()
    raw_rows = raw.count()
    seen = spark.read.format("delta").load(C.BRONZE).select("id", "ingested_at")
    new = (
        raw.join(seen, ["id", "ingested_at"], "left_anti")
        .select(
            F.col("id").cast("bigint").alias("id"),
            F.col("event_date").cast("date").alias("event_date"),
            F.col("product_category").cast("string").alias("product_category"),
            F.col("units_sold").cast("int").alias("units_sold"),
            F.col("revenue").cast("decimal(14,2)").alias("revenue"),
            F.col("avg_unit_price").cast("decimal(10,2)").alias("avg_unit_price"),
            F.col("store_region").cast("string").alias("store_region"),
            F.col("promotion_flag").cast("boolean").alias("promotion_flag"),
            F.col("ingested_at").cast("timestamp").alias("ingested_at"),
            F.lit(batch_id).alias("_bronze_batch_id"),
            F.current_timestamp().alias("_bronze_loaded_at"),
            F.lit("postgres:raw.sales_events").alias("_source"),
        )
        .cache()
    )
    appended = new.count()
    if appended:
        # coalesce(1): one Parquet file per batch (43,823 rows is about 1 MB)
        new.coalesce(1).write.format("delta").mode("append").save(C.BRONZE)
    new.unpersist()
    raw.unpersist()
    print(f"Bronze: raw.sales_events has {raw_rows} rows; appended {appended} new rows "
          f"(batch {batch_id})")
    return raw_rows, appended


def silver_updates(spark, batch_id):
    """Bronze -> typed, cleaned, one row per natural key (latest ingested_at, then highest id:
    the rule of dbt's stg_sales_events and spark/aggregate_sales.py)."""
    bronze = spark.read.format("delta").load(C.BRONZE)
    cleaned = (
        bronze
        .filter(F.col("event_date").isNotNull()
                & F.col("product_category").isNotNull()
                & F.col("store_region").isNotNull()
                & F.col("units_sold").isNotNull() & (F.col("units_sold") >= 0)
                & F.col("revenue").isNotNull() & (F.col("revenue") >= 0)
                & F.col("avg_unit_price").isNotNull())
        .select(
            F.col("event_date").cast("date").alias("event_date"),
            F.trim("product_category").alias("product_category"),
            F.trim("store_region").alias("store_region"),
            F.col("units_sold").cast("int").alias("units_sold"),
            F.col("revenue").cast("decimal(14,2)").alias("revenue"),
            F.col("avg_unit_price").cast("decimal(10,2)").alias("avg_unit_price"),
            F.coalesce(F.col("promotion_flag"), F.lit(False)).alias("promotion_flag"),
            F.col("id").alias("source_id"),
            F.col("ingested_at").alias("source_ingested_at"),
        )
    )
    latest_first = (
        Window.partitionBy(*L.SILVER_KEY)
        .orderBy(F.col("source_ingested_at").desc(), F.col("source_id").desc())
    )
    return (
        cleaned.withColumn("_rn", F.row_number().over(latest_first))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
        .withColumn("_silver_batch_id", F.lit(batch_id))
        .withColumn("_silver_updated_at", F.current_timestamp())
    )


def merge_silver(spark, batch_id):
    """MERGE the deduplicated bronze rows into silver.

    Returns (version, metrics, committed). If Delta found nothing to change and wrote no
    commit, the version stays the same and the metrics are all zero."""
    before = C.current_version(spark, C.SILVER)
    updates = silver_updates(spark, batch_id)
    updates.createOrReplaceTempView("silver_updates")
    spark.sql(L.build_merge_sql(C.SILVER, "silver_updates"))
    last = C.history(spark, C.SILVER, limit=1)[0]
    committed = last["version"] != before
    if committed and last["operation"] != "MERGE":
        raise RuntimeError(f"unexpected commit after MERGE: {last['operation']} v{last['version']}")
    metrics = L.merge_metrics(last["operationMetrics"] if committed else {})
    print(f"Silver: MERGE -> version {last['version']}"
          f"{'' if committed else ' (no commit: nothing to change)'}: "
          f"source rows {metrics['source_rows']}, inserted {metrics['inserted']}, "
          f"updated {metrics['updated']}, deleted {metrics['deleted']}")
    return last["version"], metrics, committed


def gold_rollups(spark):
    """Same aggregation as spark/aggregate_sales.py, on silver instead of raw.sales_events,
    cast to the column types of staging.sales_rollups."""
    silver = spark.read.format("delta").load(C.SILVER)
    daily = silver.groupBy("event_date", "product_category").agg(
        F.sum("units_sold").alias("daily_units"),
        F.sum("revenue").alias("daily_revenue"),
    )
    day_seconds = F.col("event_date").cast("timestamp").cast("long")
    w7 = Window.partitionBy("product_category").orderBy(day_seconds).rangeBetween(-6 * 86400, 0)
    w30 = Window.partitionBy("product_category").orderBy(day_seconds).rangeBetween(-29 * 86400, 0)
    lag_w = Window.partitionBy("product_category").orderBy("event_date")
    return (
        daily
        .withColumn("rolling_avg_units_7d", F.avg("daily_units").over(w7))
        .withColumn("rolling_avg_units_30d", F.avg("daily_units").over(w30))
        .withColumn("units_7d_ago", F.lag("daily_units", 7).over(lag_w))
        .withColumn(
            "wow_growth_rate",
            F.when(F.col("units_7d_ago").isNotNull() & (F.col("units_7d_ago") != 0),
                   (F.col("daily_units") - F.col("units_7d_ago")) / F.col("units_7d_ago")),
        )
        .select(
            F.col("event_date"),
            F.col("product_category"),
            F.col("daily_units").cast("int").alias("daily_units"),
            F.col("daily_revenue").cast("decimal(14,2)").alias("daily_revenue"),
            F.col("rolling_avg_units_7d").cast("decimal(14,4)").alias("rolling_avg_units_7d"),
            F.col("rolling_avg_units_30d").cast("decimal(14,4)").alias("rolling_avg_units_30d"),
            F.col("wow_growth_rate").cast("decimal(10,6)").alias("wow_growth_rate"),
            F.year("event_date").alias("event_year"),
            F.current_timestamp().alias("_gold_built_at"),
        )
    )


def build_gold(spark):
    """Atomic full rebuild (rolling windows depend on every earlier day, so an incremental
    MERGE would have to recompute the same windows anyway). One file per category partition.
    No overwriteSchema: a schema change fails instead of silently replacing the table."""
    gold = gold_rollups(spark)
    (gold.repartition("product_category")
         .write.format("delta").mode("overwrite")
         .partitionBy("product_category")
         .save(C.GOLD))
    last = C.history(spark, C.GOLD, limit=1)[0]
    rows = spark.read.format("delta").load(C.GOLD).count()
    print(f"Gold: {rows} rows -> version {last['version']} ({last['operation']}), "
          f"partitioned by product_category")
    return last["version"], rows


def main():
    spark = C.get_spark("sales-lakehouse-build")
    batch_id = C.run_id()
    print(f"Lakehouse root {C.LAKEHOUSE_ROOT}; batch {batch_id}; source {C.JDBC_URL}")
    ensure_tables(spark)
    raw_rows, _ = ingest_bronze(spark, batch_id)
    if raw_rows == 0:
        print("raw.sales_events is empty: nothing to build. Exiting.")
        spark.stop()
        return
    merge_silver(spark, batch_id)
    silver_rows = spark.read.format("delta").load(C.SILVER).count()
    print(f"Silver: {silver_rows} rows")
    build_gold(spark)
    for name, path in C.TABLES:
        print(f"{name}: version {C.current_version(spark, path)}  {path}")
    spark.stop()


if __name__ == "__main__":
    main()
