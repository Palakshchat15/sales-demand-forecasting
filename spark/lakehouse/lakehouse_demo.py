"""
Delta Lake feature demo on the sales lakehouse (run after build_lakehouse.py).

Writes results/lakehouse_delta_demo.txt (report with PASS/FAIL checks) and
results/lakehouse_history.csv (DESCRIBE HISTORY of the three tables).

1. Table details: files, size, partitioning, properties (constraints, appendOnly).
2. Idempotent re-run: bronze ingest appends 0 rows; the silver MERGE inserts/updates/deletes
   0 rows; a gold rebuild writes a new version with identical contents.
3. Time travel: silver before / at its first MERGE / now (versionAsOf and VERSION AS OF).
4. Schema enforcement and constraints: an extra column, a negative units_sold (CHECK), a
   NULL store_region (NOT NULL) and a DELETE on append-only bronze must all be rejected
   without a new version (if one ever got through, the table is RESTOREd to the version
   before and the check fails).
5. OPTIMIZE (silver bin-packing, gold ZORDER BY event_date); the data must not change.
   Skip with LAKEHOUSE_SKIP_OPTIMIZE=1.
6. VACUUM ... DRY RUN at the default 7-day retention (lists, deletes nothing). Shorter
   retention is refused here: it would break time travel to older versions.

Changes the tables only through steps 2 and 5 (new versions, same data). Exit code 1 if a
check fails.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyspark.sql import functions as F  # noqa: E402

import build_lakehouse as B  # noqa: E402
import lakehouse_common as C  # noqa: E402
import lakehouse_logic as L  # noqa: E402

OUT_TXT = os.path.join(C.RESULTS_DIR, "lakehouse_delta_demo.txt")
OUT_HISTORY = os.path.join(C.RESULTS_DIR, "lakehouse_history.csv")

lines = []
checks = []


def say(text=""):
    print(text)
    lines.append(text)


def check(name, ok, detail):
    checks.append((name, bool(ok)))
    say(f"CHECK {name}: {'PASS' if ok else 'FAIL'} - {detail}")


def load(spark, path, version=None):
    reader = spark.read.format("delta")
    if version is not None:
        reader = reader.option("versionAsOf", int(version))
    return reader.load(path)


def short_error(exc):
    first = (str(exc).strip().splitlines() or [""])[0]
    return f"{type(exc).__name__}: {first[:220]}"


def gold_data_cols(spark):
    return [c for c in load(spark, C.GOLD).columns if c != "_gold_built_at"]


def symmetric_diff(spark, path, cols, v_old, v_new=None):
    old = load(spark, path, v_old).select(*cols)
    new = load(spark, path, v_new).select(*cols)
    return old.exceptAll(new).count() + new.exceptAll(old).count()


# ------------------------------------------------------------------ 1. details
def table_details(spark):
    say("== 1. Tables")
    for name, path in C.TABLES:
        d = spark.sql(f"DESCRIBE DETAIL {C.delta_ref(path)}").collect()[0].asDict()
        props = {k: v for k, v in (d.get("properties") or {}).items()
                 if k.startswith("delta.constraints") or k == "delta.appendOnly"}
        say(f"{name:6s} {path}")
        say(f"       version {C.current_version(spark, path)}, files {d['numFiles']}, "
            f"bytes {d['sizeInBytes']}, partitioned by {list(d['partitionColumns']) or '-'}, "
            f"protocol r{d.get('minReaderVersion')}/w{d.get('minWriterVersion')}")
        for k, v in sorted(props.items()):
            say(f"       {k} = {v}")
    say()


# ------------------------------------------------------------------ 2. idempotency
def idempotent_rerun(spark):
    say("== 2. Idempotent re-run (same input, second pass)")
    batch = C.run_id() + "_demo_rerun"
    v_bronze = C.current_version(spark, C.BRONZE)
    _, appended = B.ingest_bronze(spark, batch)
    check("bronze_reingest_appends_0_rows",
          appended == 0 and C.current_version(spark, C.BRONZE) == v_bronze,
          f"appended {appended} rows; version {v_bronze} -> {C.current_version(spark, C.BRONZE)}")

    n_before = load(spark, C.SILVER).count()
    version, m, committed = B.merge_silver(spark, batch)
    n_after = load(spark, C.SILVER).count()
    check("silver_merge_rerun_changes_0_rows",
          L.merge_is_idempotent(m) and n_before == n_after,
          f"MERGE {'version ' + str(version) if committed else '(no commit)'}: source rows "
          f"{m['source_rows']}, inserted {m['inserted']}, updated {m['updated']}, "
          f"deleted {m['deleted']}; rows {n_before} -> {n_after}")

    v_gold_old = C.current_version(spark, C.GOLD)
    v_gold_new, rows = B.build_gold(spark)
    diff = symmetric_diff(spark, C.GOLD, gold_data_cols(spark), v_gold_old, v_gold_new)
    check("gold_rebuild_is_deterministic",
          v_gold_new > v_gold_old and diff == 0,
          f"overwrite version {v_gold_old} -> {v_gold_new} ({rows} rows); rows that differ "
          f"between the two versions (ignoring _gold_built_at): {diff}")
    say()


# ------------------------------------------------------------------ 3. time travel
def time_travel(spark):
    say("== 3. Time travel")
    hist = C.history(spark, C.SILVER)
    merges = L.versions_with_operation(hist, "MERGE")
    if not merges:
        check("silver_time_travel", False, "silver has no MERGE commit")
        return
    first = merges[0]
    before = L.version_before(hist, first)
    n_before = load(spark, C.SILVER, before).count() if before is not None else 0
    n_first = load(spark, C.SILVER, first).count()
    n_sql = spark.sql(f"SELECT count(*) AS n FROM {C.delta_ref(C.SILVER)} VERSION AS OF {first}").collect()[0]["n"]
    n_now = load(spark, C.SILVER).count()
    say(f"silver versions with MERGE: {merges}")
    say(f"silver versionAsOf {before} (before the first MERGE): {n_before} rows")
    say(f"silver versionAsOf {first} (first MERGE): {n_first} rows; "
        f"SQL 'VERSION AS OF {first}': {n_sql} rows")
    say(f"silver now (version {C.current_version(spark, C.SILVER)}): {n_now} rows")
    check("silver_time_travel", n_before == 0 and n_first == n_sql and n_first > 0,
          f"{n_before} rows before the first MERGE, {n_first} after it")

    bhist = C.history(spark, C.BRONZE)
    writes = L.versions_with_operation(bhist, "WRITE")
    sizes = [(v, load(spark, C.BRONZE, v).count()) for v in writes]
    say(f"bronze appends (version, rows in table after it): {sizes}")
    say()


# ------------------------------------------------------------------ 4. enforcement
def rejected(spark, name, path, action, expectation):
    """Run `action`; it must raise and leave the table version unchanged."""
    v0 = C.current_version(spark, path)
    try:
        action()
        error = None
    except Exception as exc:  # Py4J / AnalysisException / DeltaInvariantViolationException
        error = short_error(exc)
    v1 = C.current_version(spark, path)
    if v1 != v0:
        spark.sql(f"RESTORE TABLE {C.delta_ref(path)} TO VERSION AS OF {v0}")
        say(f"       table changed ({v0} -> {v1}); RESTOREd to version {v0}")
    check(name, error is not None and v1 == v0,
          f"{expectation}: {'rejected, ' + error if error else 'NOT rejected'}; version {v0} -> {v1}")


def schema_enforcement(spark):
    say("== 4. Schema enforcement and constraints")
    one = load(spark, C.SILVER).limit(1)

    rejected(spark, "silver_rejects_unknown_column", C.SILVER,
             lambda: one.withColumn("unexpected_column", F.lit(1))
                        .write.format("delta").mode("append").save(C.SILVER),
             "append with an extra column (no mergeSchema)")
    rejected(spark, "silver_check_units_sold_non_negative", C.SILVER,
             lambda: one.withColumn("units_sold", F.lit(-1))
                        .write.format("delta").mode("append").save(C.SILVER),
             "append with units_sold = -1")
    rejected(spark, "silver_not_null_store_region", C.SILVER,
             lambda: one.withColumn("store_region", F.lit(None).cast("string"))
                        .write.format("delta").mode("append").save(C.SILVER),
             "append with store_region = NULL")
    rejected(spark, "bronze_is_append_only", C.BRONZE,
             lambda: spark.sql(f"DELETE FROM {C.delta_ref(C.BRONZE)} WHERE id < 0"),
             "DELETE on bronze (delta.appendOnly = true)")
    say()


# ------------------------------------------------------------------ 5. optimize
def optimize(spark):
    say("== 5. OPTIMIZE")
    if os.environ.get("LAKEHOUSE_SKIP_OPTIMIZE") == "1":
        say("skipped (LAKEHOUSE_SKIP_OPTIMIZE=1)\n")
        return
    for name, path, sql_suffix in (("silver", C.SILVER, ""), ("gold", C.GOLD, " ZORDER BY (event_date)")):
        v0 = C.current_version(spark, path)
        files0 = spark.sql(f"DESCRIBE DETAIL {C.delta_ref(path)}").collect()[0]["numFiles"]
        spark.sql(f"OPTIMIZE {C.delta_ref(path)}{sql_suffix}")
        v1 = C.current_version(spark, path)
        files1 = spark.sql(f"DESCRIBE DETAIL {C.delta_ref(path)}").collect()[0]["numFiles"]
        if v1 == v0:
            check(f"{name}_optimize", True, f"OPTIMIZE{sql_suffix}: nothing to compact (no commit), files {files0}")
            continue
        cols = gold_data_cols(spark) if name == "gold" else list(load(spark, path).columns)
        diff = symmetric_diff(spark, path, cols, v0, v1)
        check(f"{name}_optimize_keeps_data", diff == 0,
              f"OPTIMIZE{sql_suffix}: version {v0} -> {v1}, files {files0} -> {files1}, "
              f"rows that differ: {diff}")
    say()


# ------------------------------------------------------------------ 6. vacuum
def vacuum(spark):
    say("== 6. VACUUM (dry run)")
    hours = os.environ.get("LAKEHOUSE_VACUUM_RETAIN_HOURS", str(L.DEFAULT_VACUUM_RETENTION_HOURS))
    if not L.vacuum_retention_is_safe(hours):
        say(f"refused: retention {hours} h is below Delta's {L.DEFAULT_VACUUM_RETENTION_HOURS} h default; "
            f"it would delete files that time travel to older versions still needs\n")
        return
    for name, path in C.TABLES:
        listed = spark.sql(f"VACUUM {C.delta_ref(path)} RETAIN {int(float(hours))} HOURS DRY RUN").count()
        say(f"{name:6s} files older than {hours} h and no longer referenced: {listed} (dry run, nothing deleted)")
    say("Files replaced by MERGE/overwrite/OPTIMIZE stay on disk for 7 days so the older versions "
        "above remain readable; VACUUM after that only removes files no current version uses.\n")


# ------------------------------------------------------------------ history
def write_history(spark):
    say("== History (newest first, last 12 commits per table)")
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    with open(OUT_HISTORY, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["table", "version", "timestamp", "operation", "operation_metrics"])
        for name, path in C.TABLES:
            hist = C.history(spark, path)
            rows = []
            for h in hist:
                metrics = ";".join(f"{k}={v}" for k, v in sorted(h["operationMetrics"].items()))
                w.writerow([name, h["version"], h["timestamp"], h["operation"], metrics])
                keep = {k: h["operationMetrics"].get(k) for k in (
                    "numOutputRows", "numTargetRowsInserted", "numTargetRowsUpdated",
                    "numFiles", "numAddedFiles", "numRemovedFiles") if h["operationMetrics"].get(k)}
                rows.append({"version": h["version"], "timestamp": h["timestamp"],
                             "operation": h["operation"],
                             "metrics": ", ".join(f"{k}={v}" for k, v in keep.items())})
            say(f"-- {name}")
            say(L.format_table(rows[:12], ("version", "timestamp", "operation", "metrics")))
    say()


def main():
    spark = C.get_spark("sales-lakehouse-demo")
    for name, path in C.TABLES:
        if not C.is_delta_table(path):
            print(f"ERROR: {name} table missing at {path}; run build_lakehouse.py first")
            spark.stop()
            sys.exit(1)
    say(f"Delta Lake feature demo, lakehouse {C.LAKEHOUSE_ROOT}, Spark {spark.version}, run {C.run_id()}")
    say()
    table_details(spark)
    idempotent_rerun(spark)
    time_travel(spark)
    schema_enforcement(spark)
    optimize(spark)
    vacuum(spark)
    write_history(spark)
    failed = [n for n, ok in checks if not ok]
    say(f"Summary: {len(checks) - len(failed)}/{len(checks)} checks PASS"
        + (f"; FAILED: {', '.join(failed)}" if failed else ""))
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    with open(OUT_TXT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {OUT_TXT} and {OUT_HISTORY}")
    spark.stop()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
