"""
Pure-Python helpers for the Delta Lake lakehouse jobs (no pyspark import, so they
can be unit-tested on the host: tests/test_lakehouse_logic.py).

- SQL builders for the silver MERGE (natural-key upsert, "latest ingested wins").
- Parity evaluation: gold Delta table vs Postgres staging.sales_rollups.
- Delta history helpers: MERGE idempotency, version lookup, VACUUM retention guard.
- A tiny fixed-width text table formatter for the demo report.
"""

import csv
import os
from decimal import Decimal, InvalidOperation

# Natural key of one sales event (same as the Postgres unique index and dbt's stg_sales_events).
SILVER_KEY = ("event_date", "product_category", "store_region")
# Key of one daily category rollup (primary key of staging.sales_rollups).
GOLD_KEY = ("event_date", "product_category")

SILVER_VALUE_COLS = (
    "units_sold", "revenue", "avg_unit_price", "promotion_flag",
    "source_id", "source_ingested_at", "_silver_batch_id", "_silver_updated_at",
)

# Value columns of the gold table that are compared with staging.sales_rollups, and the
# allowed absolute difference per row. Postgres stores the rolling averages as NUMERIC(14,4)
# and the growth rate as NUMERIC(10,6), rounded from float8; gold casts the same doubles to
# the same DECIMAL scales in Spark. The two roundings can differ by one unit in the last
# place on an exact half, hence one unit of tolerance; units and revenue must match exactly.
GOLD_VALUE_COLS = (
    "daily_units", "daily_revenue", "rolling_avg_units_7d",
    "rolling_avg_units_30d", "wow_growth_rate",
)
ROW_TOLERANCE = {
    "daily_units": Decimal("0"),
    "daily_revenue": Decimal("0"),
    "rolling_avg_units_7d": Decimal("0.0001"),
    "rolling_avg_units_30d": Decimal("0.0001"),
    "wow_growth_rate": Decimal("0.000001"),
}

# Delta's default VACUUM retention (delta.deletedFileRetentionDuration = 7 days).
DEFAULT_VACUUM_RETENTION_HOURS = 168

PARITY_COLUMNS = ("layer", "check", "lakehouse", "postgres", "abs_diff", "tolerance", "status")


# --------------------------------------------------------------------------- SQL builders
def quote_path(path):
    """Delta path identifier: delta.`/some/path` (backticks inside a path are rejected)."""
    if "`" in path:
        raise ValueError(f"backtick in Delta path: {path!r}")
    return f"delta.`{path}`"


def build_merge_sql(target_path, source_view, key_cols=SILVER_KEY, value_cols=SILVER_VALUE_COLS,
                    order_cols=("source_ingested_at", "source_id")):
    """MERGE INTO <target> USING <source> on the natural key.

    - not matched -> INSERT (new event)
    - matched and the source row is strictly newer by (source_ingested_at, source_id)
      -> UPDATE (a corrected re-delivery wins, the same rule as dbt's stg_sales_events)
    - matched and not newer -> nothing, so re-running the same batch changes 0 rows.
    """
    if not key_cols:
        raise ValueError("key_cols must not be empty")
    on = " AND ".join(f"t.{c} = s.{c}" for c in key_cols)
    first, second = order_cols
    newer = f"(s.{first} > t.{first} OR (s.{first} = t.{first} AND s.{second} > t.{second}))"
    sets = ", ".join(f"t.{c} = s.{c}" for c in value_cols)
    all_cols = list(key_cols) + list(value_cols)
    ins_cols = ", ".join(all_cols)
    ins_vals = ", ".join(f"s.{c}" for c in all_cols)
    return (
        f"MERGE INTO {quote_path(target_path)} AS t\n"
        f"USING {source_view} AS s\n"
        f"ON {on}\n"
        f"WHEN MATCHED AND {newer} THEN UPDATE SET {sets}\n"
        f"WHEN NOT MATCHED THEN INSERT ({ins_cols}) VALUES ({ins_vals})"
    )


# --------------------------------------------------------------------------- Delta history
def _to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def merge_metrics(operation_metrics):
    """Integer view of a MERGE commit's operationMetrics (missing keys -> 0)."""
    m = operation_metrics or {}
    return {
        "source_rows": _to_int(m.get("numSourceRows")),
        "inserted": _to_int(m.get("numTargetRowsInserted")),
        "updated": _to_int(m.get("numTargetRowsUpdated")),
        "deleted": _to_int(m.get("numTargetRowsDeleted")),
        "files_added": _to_int(m.get("numTargetFilesAdded")),
        "files_removed": _to_int(m.get("numTargetFilesRemoved")),
    }


def merge_is_idempotent(operation_metrics):
    """True when a MERGE inserted, updated and deleted nothing."""
    m = merge_metrics(operation_metrics)
    return m["inserted"] == 0 and m["updated"] == 0 and m["deleted"] == 0


def versions_with_operation(history, operation):
    """Sorted versions whose operation equals `operation` (history: dicts with version/operation)."""
    return sorted(int(h["version"]) for h in history if h.get("operation") == operation)


def version_before(history, version):
    """Highest version strictly lower than `version`, or None."""
    lower = [int(h["version"]) for h in history if int(h["version"]) < version]
    return max(lower) if lower else None


def vacuum_retention_is_safe(hours, minimum=DEFAULT_VACUUM_RETENTION_HOURS):
    """VACUUM below Delta's 7-day default can delete files that time travel or a running
    reader still needs; Delta refuses it unless the retention check is switched off, which
    these jobs never do."""
    try:
        return float(hours) >= float(minimum)
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------- parity
def to_decimal(value):
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(f"not a number: {value!r}")


def parity_row(layer, check, lakehouse, postgres, tolerance=0):
    """One parity line. Numbers are compared as Decimals within `tolerance`; anything else
    (dates, strings) must be equal. A missing value on either side fails."""
    tol = to_decimal(tolerance)
    if lakehouse is None or postgres is None:
        ok = lakehouse is None and postgres is None
        diff = ""
    elif isinstance(lakehouse, (int, float, Decimal)) and isinstance(postgres, (int, float, Decimal)) \
            and not isinstance(lakehouse, bool) and not isinstance(postgres, bool):
        d = abs(to_decimal(lakehouse) - to_decimal(postgres))
        ok = d <= tol
        diff = d
    else:
        ok = str(lakehouse) == str(postgres)
        diff = "" if ok else "differs"
    return {
        "layer": layer, "check": check, "lakehouse": lakehouse, "postgres": postgres,
        "abs_diff": diff, "tolerance": tol, "status": "PASS" if ok else "FAIL",
    }


def summary_parity_rows(gold, pg):
    """Table-level checks. `gold`/`pg`: dicts with row_count, distinct_keys, min_date,
    max_date, categories, sum_<col> for every GOLD_VALUE_COLS column and null_<col> for
    the growth rate (null for the first 7 days of each category)."""
    rows = []
    n = max(_to_int(gold.get("row_count")), _to_int(pg.get("row_count")))
    for check in ("row_count", "distinct_keys", "categories", "min_date", "max_date"):
        rows.append(parity_row("gold", check, gold.get(check), pg.get(check)))
    for col in GOLD_VALUE_COLS:
        # A sum over n rounded values can drift by at most n x the per-row tolerance.
        rows.append(parity_row("gold", f"sum_{col}", gold.get(f"sum_{col}"), pg.get(f"sum_{col}"),
                               ROW_TOLERANCE[col] * n))
    rows.append(parity_row("gold", "null_wow_growth_rate",
                           gold.get("null_wow_growth_rate"), pg.get("null_wow_growth_rate")))
    return rows


def rowlevel_parity_rows(rowlevel):
    """Row-level checks from a full outer join on (event_date, product_category).
    `rowlevel`: only_in_lakehouse, only_in_postgres, matched, and per value column
    mismatch_<col> (rows beyond ROW_TOLERANCE) and max_diff_<col>."""
    rows = [
        parity_row("gold", "rows_only_in_lakehouse", rowlevel.get("only_in_lakehouse"), 0),
        parity_row("gold", "rows_only_in_postgres", 0, rowlevel.get("only_in_postgres")),
    ]
    for col in GOLD_VALUE_COLS:
        rows.append(parity_row("gold", f"rows_mismatched_{col}", rowlevel.get(f"mismatch_{col}"), 0))
        max_diff = rowlevel.get(f"max_diff_{col}")
        rows.append(parity_row("gold", f"max_abs_diff_{col}",
                               max_diff if max_diff is not None else 0, 0, ROW_TOLERANCE[col]))
    return rows


def upstream_parity_rows(counts):
    """Bronze/silver vs raw.sales_events. `counts`: raw_rows, raw_distinct_keys,
    raw_rows_missing_from_bronze, silver_rows, silver_distinct_keys."""
    return [
        parity_row("bronze", "raw_rows_missing_from_bronze", counts.get("raw_rows_missing_from_bronze"), 0),
        parity_row("silver", "row_count_vs_raw_distinct_keys",
                   counts.get("silver_rows"), counts.get("raw_distinct_keys")),
        parity_row("silver", "distinct_keys_vs_raw_distinct_keys",
                   counts.get("silver_distinct_keys"), counts.get("raw_distinct_keys")),
    ]


def parity_passed(rows):
    return bool(rows) and all(r["status"] == "PASS" for r in rows)


def _fmt(value):
    if isinstance(value, Decimal):
        # Plain notation, no exponent; trailing zeros of the scale kept as stored.
        return format(value, "f")
    return "" if value is None else str(value)


def write_parity_csv(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(PARITY_COLUMNS)
        for r in rows:
            w.writerow([_fmt(r[c]) for c in PARITY_COLUMNS])


# --------------------------------------------------------------------------- text output
def format_table(rows, columns):
    """Fixed-width text table (rows: list of dicts)."""
    cells = [[_fmt(r.get(c)) for c in columns] for r in rows]
    widths = [max([len(c)] + [len(row[i]) for row in cells]) for i, c in enumerate(columns)]
    line = "  ".join(c.ljust(widths[i]) for i, c in enumerate(columns))
    sep = "  ".join("-" * w for w in widths)
    body = ["  ".join(row[i].ljust(widths[i]) for i in range(len(columns))) for row in cells]
    return "\n".join([line, sep] + body)
