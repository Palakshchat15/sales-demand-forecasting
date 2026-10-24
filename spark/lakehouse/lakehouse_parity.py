"""
Parity check: Delta gold table vs Postgres staging.sales_rollups (written by
spark/aggregate_sales.py), plus bronze/silver vs raw.sales_events.

Checks (results/lakehouse_parity.csv, one line each, status PASS/FAIL):
- bronze: every raw.sales_events row is in bronze;
- silver: one row per natural key, same number of keys as raw.sales_events;
- gold vs staging.sales_rollups: row count, distinct keys, categories, date range, column
  sums, null growth rates; full outer join on (event_date, product_category): rows only on
  one side, rows whose values differ beyond the tolerance, maximum absolute difference.

Exit code 1 if any check fails. Read-only on both sides.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyspark.sql import functions as F  # noqa: E402

import lakehouse_common as C  # noqa: E402
import lakehouse_logic as L  # noqa: E402

OUT_CSV = os.path.join(C.RESULTS_DIR, "lakehouse_parity.csv")


def summarize(df):
    aggs = [
        F.count(F.lit(1)).alias("row_count"),
        F.countDistinct(*L.GOLD_KEY).alias("distinct_keys"),
        F.countDistinct("product_category").alias("categories"),
        F.min("event_date").alias("min_date"),
        F.max("event_date").alias("max_date"),
        F.sum(F.when(F.col("wow_growth_rate").isNull(), 1).otherwise(0)).alias("null_wow_growth_rate"),
    ] + [F.sum(c).alias(f"sum_{c}") for c in L.GOLD_VALUE_COLS]
    return df.agg(*aggs).collect()[0].asDict()


def rowlevel(gold, pg):
    g = gold.select(*L.GOLD_KEY, *[F.col(c).alias(f"g_{c}") for c in L.GOLD_VALUE_COLS]) \
            .withColumn("_in_g", F.lit(1))
    p = pg.select(*L.GOLD_KEY, *[F.col(c).alias(f"p_{c}") for c in L.GOLD_VALUE_COLS]) \
          .withColumn("_in_p", F.lit(1))
    j = g.join(p, list(L.GOLD_KEY), "full_outer")
    both = F.col("_in_g").isNotNull() & F.col("_in_p").isNotNull()
    aggs = [
        F.sum(F.when(F.col("_in_p").isNull(), 1).otherwise(0)).alias("only_in_lakehouse"),
        F.sum(F.when(F.col("_in_g").isNull(), 1).otherwise(0)).alias("only_in_postgres"),
        F.sum(F.when(both, 1).otherwise(0)).alias("matched"),
    ]
    for c in L.GOLD_VALUE_COLS:
        diff = F.abs(F.col(f"g_{c}") - F.col(f"p_{c}"))
        tol = F.lit(str(L.ROW_TOLERANCE[c])).cast("decimal(20,10)")
        differs = ~F.col(f"g_{c}").eqNullSafe(F.col(f"p_{c}")) & (diff.isNull() | (diff > tol))
        aggs.append(F.sum(F.when(both & differs, 1).otherwise(0)).alias(f"mismatch_{c}"))
        aggs.append(F.max(F.when(both, diff)).alias(f"max_diff_{c}"))
    return j.agg(*aggs).collect()[0].asDict()


def upstream(spark):
    raw = C.read_pg(spark, "raw.sales_events").select("id", "ingested_at", *L.SILVER_KEY).cache()
    bronze = spark.read.format("delta").load(C.BRONZE).select("id", "ingested_at")
    silver = spark.read.format("delta").load(C.SILVER)
    out = {
        "raw_rows": raw.count(),
        "raw_distinct_keys": raw.select(*L.SILVER_KEY).distinct().count(),
        "raw_rows_missing_from_bronze": raw.join(bronze, ["id", "ingested_at"], "left_anti").count(),
        "bronze_rows": bronze.count(),
        "silver_rows": silver.count(),
        "silver_distinct_keys": silver.select(*L.SILVER_KEY).distinct().count(),
    }
    raw.unpersist()
    return out


def main():
    spark = C.get_spark("sales-lakehouse-parity")
    for name, path in C.TABLES:
        if not C.is_delta_table(path):
            print(f"ERROR: {name} table missing at {path}; run build_lakehouse.py first")
            spark.stop()
            sys.exit(1)

    counts = upstream(spark)
    gold = spark.read.format("delta").load(C.GOLD).cache()
    pg = C.read_pg(spark, "staging.sales_rollups").cache()
    rows = (L.upstream_parity_rows(counts)
            + L.summary_parity_rows(summarize(gold), summarize(pg))
            + L.rowlevel_parity_rows(rowlevel(gold, pg)))
    L.write_parity_csv(rows, OUT_CSV)

    versions = {name: C.current_version(spark, path) for name, path in C.TABLES}
    print(f"raw.sales_events rows {counts['raw_rows']}, bronze rows {counts['bronze_rows']}, "
          f"silver rows {counts['silver_rows']}; Delta versions {versions}")
    print(L.format_table(rows, L.PARITY_COLUMNS))
    failed = [r for r in rows if r["status"] != "PASS"]
    print(f"\nParity: {len(rows) - len(failed)}/{len(rows)} checks PASS -> {OUT_CSV}")
    spark.stop()
    if failed:
        print("FAILED: " + ", ".join(f"{r['layer']}.{r['check']}" for r in failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
