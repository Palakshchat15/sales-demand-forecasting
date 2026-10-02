"""
Great Expectations validation of raw.sales_events in Postgres.

Uses GE's lightweight pandas-based validation (reads the table into a
DataFrame via SQLAlchemy, then runs an in-memory Expectation Suite) rather
than a full GE Data Context / checkpoint project, since that proved more
robust across environments in prior portfolio builds.

Exits non-zero if any expectation fails, so it can be used directly as an
Airflow BashOperator/PythonOperator gate before downstream Spark/dbt steps.
"""

import os
import sys

import great_expectations as ge
import pandas as pd
import sqlalchemy


def get_engine():
    user = os.environ.get("POSTGRES_USER", "sales_user")
    password = os.environ.get("POSTGRES_PASSWORD", "sales_pass")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5433")
    db = os.environ.get("POSTGRES_DB", "sales_forecast")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return sqlalchemy.create_engine(url)


VALID_CATEGORIES = [
    "Electronics", "Apparel", "Home & Garden", "Groceries",
    "Toys", "Sporting Goods", "Books", "Beauty",
]
VALID_REGIONS = ["North", "South", "East", "West", "Online"]


def main():
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM raw.sales_events", engine)
    print(f"Loaded {len(df)} rows from raw.sales_events for validation")

    if len(df) == 0:
        print("FAIL: raw.sales_events is empty — nothing to validate.")
        sys.exit(1)

    gdf = ge.from_pandas(df)

    results = []
    results.append(gdf.expect_column_values_to_not_be_null("event_date"))
    results.append(gdf.expect_column_values_to_not_be_null("product_category"))
    results.append(gdf.expect_column_values_to_be_in_set("product_category", VALID_CATEGORIES))
    results.append(gdf.expect_column_values_to_be_in_set("store_region", VALID_REGIONS))
    results.append(gdf.expect_column_values_to_be_between("units_sold", min_value=0, max_value=100000))
    results.append(gdf.expect_column_values_to_be_between("revenue", min_value=0, max_value=1e8))
    results.append(gdf.expect_column_values_to_be_between("avg_unit_price", min_value=0, max_value=10000))
    results.append(gdf.expect_column_values_to_not_be_null("promotion_flag"))
    # One row per natural key; a duplicate means a replayed event slipped past the consumer's upsert.
    results.append(gdf.expect_compound_columns_to_be_unique(["event_date", "product_category", "store_region"]))

    failed = [r for r in results if not r.success]

    print(f"\nRan {len(results)} expectations, {len(results) - len(failed)} passed, {len(failed)} failed.")
    for r in results:
        exp_type = r.expectation_config.expectation_type
        status = "PASS" if r.success else "FAIL"
        print(f"  [{status}] {exp_type}")
        if not r.success:
            print(f"        details: {r.result}")

    if failed:
        print("\nGreat Expectations validation FAILED.")
        sys.exit(1)

    print("\nGreat Expectations validation PASSED.")


if __name__ == "__main__":
    main()
