"""Unit tests for spark/lakehouse/lakehouse_logic.py (pure Python, no Spark needed).

    python -m pytest tests -q
"""

import csv
import os
import sys
from datetime import date
from decimal import Decimal

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "spark", "lakehouse"))

import lakehouse_logic as L  # noqa: E402


# ------------------------------------------------------------------ MERGE SQL
def test_merge_sql_joins_on_the_full_natural_key():
    sql = L.build_merge_sql("/opt/lakehouse/silver/sales_events", "silver_updates")
    assert "MERGE INTO delta.`/opt/lakehouse/silver/sales_events` AS t" in sql
    assert "USING silver_updates AS s" in sql
    assert ("ON t.event_date = s.event_date AND t.product_category = s.product_category "
            "AND t.store_region = s.store_region") in sql


def test_merge_sql_updates_only_strictly_newer_rows():
    sql = L.build_merge_sql("/x", "v")
    matched = [line for line in sql.splitlines() if line.startswith("WHEN MATCHED")][0]
    assert ("(s.source_ingested_at > t.source_ingested_at OR (s.source_ingested_at = "
            "t.source_ingested_at AND s.source_id > t.source_id))") in matched
    assert ">=" not in matched  # an equal row must not be rewritten, or re-runs would update rows


def test_merge_sql_inserts_every_column_and_never_deletes():
    sql = L.build_merge_sql("/x", "v")
    cols = list(L.SILVER_KEY) + list(L.SILVER_VALUE_COLS)
    assert f"INSERT ({', '.join(cols)})" in sql
    assert f"VALUES ({', '.join('s.' + c for c in cols)})" in sql
    assert "DELETE" not in sql
    for c in L.SILVER_VALUE_COLS:
        assert f"t.{c} = s.{c}" in sql
    for c in L.SILVER_KEY:  # key columns are never updated
        assert f"t.{c} = s.{c}," not in sql.split("UPDATE SET", 1)[1]


def test_quote_path_rejects_backticks():
    assert L.quote_path("/a/b") == "delta.`/a/b`"
    with pytest.raises(ValueError):
        L.quote_path("/a`; DROP")


def test_merge_sql_needs_a_key():
    with pytest.raises(ValueError):
        L.build_merge_sql("/x", "v", key_cols=())


# ------------------------------------------------------------------ history helpers
def test_merge_metrics_and_idempotency():
    first = {"numSourceRows": "43823", "numTargetRowsInserted": "43823",
             "numTargetRowsUpdated": "0", "numTargetRowsDeleted": "0"}
    rerun = {"numSourceRows": "43823", "numTargetRowsInserted": "0",
             "numTargetRowsUpdated": "0", "numTargetRowsDeleted": "0"}
    assert L.merge_metrics(first)["inserted"] == 43823
    assert not L.merge_is_idempotent(first)
    assert L.merge_is_idempotent(rerun)
    assert L.merge_is_idempotent({})          # no commit at all: nothing changed
    assert L.merge_is_idempotent(None)
    assert not L.merge_is_idempotent({"numTargetRowsUpdated": "1"})
    assert L.merge_metrics({"numTargetRowsInserted": "n/a"})["inserted"] == 0


def test_version_lookup():
    hist = [
        {"version": 4, "operation": "MERGE"},
        {"version": 3, "operation": "MERGE"},
        {"version": 2, "operation": "ADD CONSTRAINT"},
        {"version": 1, "operation": "ADD CONSTRAINT"},
        {"version": 0, "operation": "CREATE TABLE"},
    ]
    assert L.versions_with_operation(hist, "MERGE") == [3, 4]
    assert L.versions_with_operation(hist, "WRITE") == []
    assert L.version_before(hist, 3) == 2
    assert L.version_before(hist, 0) is None


@pytest.mark.parametrize("hours,safe", [(168, True), (200, True), ("168", True),
                                        (167.9, False), (0, False), ("x", False), (None, False)])
def test_vacuum_retention_guard(hours, safe):
    assert L.vacuum_retention_is_safe(hours) is safe


# ------------------------------------------------------------------ parity
def test_parity_row_numbers_within_tolerance():
    assert L.parity_row("gold", "c", Decimal("1.2345"), Decimal("1.2346"), "0.0001")["status"] == "PASS"
    assert L.parity_row("gold", "c", Decimal("1.2345"), Decimal("1.2347"), "0.0001")["status"] == "FAIL"
    assert L.parity_row("gold", "c", 8768, 8768)["status"] == "PASS"
    assert L.parity_row("gold", "c", 8768, 8767)["status"] == "FAIL"
    row = L.parity_row("gold", "c", 10, 7)
    assert row["abs_diff"] == Decimal(3)


def test_parity_row_non_numbers_and_missing_values():
    assert L.parity_row("gold", "d", date(2023, 1, 1), date(2023, 1, 1))["status"] == "PASS"
    assert L.parity_row("gold", "d", date(2023, 1, 1), date(2023, 1, 2))["status"] == "FAIL"
    assert L.parity_row("gold", "x", None, 5)["status"] == "FAIL"
    assert L.parity_row("gold", "x", None, None)["status"] == "PASS"
    assert L.parity_row("gold", "b", True, 1)["status"] == "FAIL"  # bools are not numbers here


def _summary(**over):
    s = {"row_count": 8768, "distinct_keys": 8768, "categories": 8,
         "min_date": date(2023, 1, 1), "max_date": date(2025, 12, 31),
         "null_wow_growth_rate": 56,
         "sum_daily_units": 4_000_000, "sum_daily_revenue": Decimal("123456789.12"),
         "sum_rolling_avg_units_7d": Decimal("3999000.1234"),
         "sum_rolling_avg_units_30d": Decimal("3998000.5678"),
         "sum_wow_growth_rate": Decimal("12.345678")}
    s.update(over)
    return s


def test_summary_parity_identical_tables_pass():
    rows = L.summary_parity_rows(_summary(), _summary())
    assert L.parity_passed(rows)
    assert {r["check"] for r in rows} >= {"row_count", "sum_daily_revenue", "null_wow_growth_rate"}


def test_summary_parity_sum_tolerance_scales_with_rows():
    # 8768 rows x 0.0001 per row: a sum drift of 0.5 is allowed, 1.0 is not
    ok = L.summary_parity_rows(_summary(sum_rolling_avg_units_7d=Decimal("3999000.6234")), _summary())
    bad = L.summary_parity_rows(_summary(sum_rolling_avg_units_7d=Decimal("3999001.1234")), _summary())
    assert L.parity_passed(ok)
    assert not L.parity_passed(bad)


def test_summary_parity_revenue_and_units_must_be_exact():
    assert not L.parity_passed(L.summary_parity_rows(
        _summary(sum_daily_revenue=Decimal("123456789.13")), _summary()))
    assert not L.parity_passed(L.summary_parity_rows(_summary(sum_daily_units=4_000_001), _summary()))
    assert not L.parity_passed(L.summary_parity_rows(_summary(row_count=8767), _summary()))


def _rowlevel(**over):
    r = {"only_in_lakehouse": 0, "only_in_postgres": 0, "matched": 8768}
    for c in L.GOLD_VALUE_COLS:
        r[f"mismatch_{c}"] = 0
        r[f"max_diff_{c}"] = Decimal(0)
    r.update(over)
    return r


def test_rowlevel_parity():
    assert L.parity_passed(L.rowlevel_parity_rows(_rowlevel()))
    assert L.parity_passed(L.rowlevel_parity_rows(_rowlevel(max_diff_rolling_avg_units_7d=Decimal("0.0001"))))
    assert not L.parity_passed(L.rowlevel_parity_rows(_rowlevel(max_diff_rolling_avg_units_7d=Decimal("0.0002"))))
    assert not L.parity_passed(L.rowlevel_parity_rows(_rowlevel(only_in_postgres=3)))
    assert not L.parity_passed(L.rowlevel_parity_rows(_rowlevel(mismatch_daily_units=1)))
    # max diff is NULL when no row matched at all (e.g. empty tables): treated as 0
    assert L.parity_passed(L.rowlevel_parity_rows(_rowlevel(max_diff_wow_growth_rate=None)))


def test_upstream_parity():
    good = {"raw_rows": 43823, "raw_distinct_keys": 43823, "raw_rows_missing_from_bronze": 0,
            "silver_rows": 43823, "silver_distinct_keys": 43823}
    assert L.parity_passed(L.upstream_parity_rows(good))
    assert not L.parity_passed(L.upstream_parity_rows(dict(good, raw_rows_missing_from_bronze=5)))
    assert not L.parity_passed(L.upstream_parity_rows(dict(good, silver_rows=43824)))
    assert not L.parity_passed([])


def test_write_parity_csv(tmp_path):
    rows = L.summary_parity_rows(_summary(), _summary())
    out = tmp_path / "sub" / "lakehouse_parity.csv"
    L.write_parity_csv(rows, str(out))
    with open(out, newline="", encoding="utf-8") as f:
        data = list(csv.reader(f))
    assert data[0] == list(L.PARITY_COLUMNS)
    assert len(data) == len(rows) + 1
    revenue = [d for d in data if d[1] == "sum_daily_revenue"][0]
    assert revenue[2] == "123456789.12" and revenue[6] == "PASS"
    assert "E" not in "".join(d[5] for d in data[1:])  # tolerances in plain notation


def test_format_table():
    text = L.format_table([{"a": 1, "b": "xyz"}, {"a": 22, "b": None}], ("a", "b"))
    lines = text.splitlines()
    assert lines[0].split() == ["a", "b"]
    assert lines[2].split() == ["1", "xyz"]
    assert lines[3].split() == ["22"]
