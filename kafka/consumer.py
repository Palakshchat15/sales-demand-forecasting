"""
Kafka consumer that reads sales events from the `sales-events` topic and
writes them in micro-batches into Postgres `raw.sales_events`.

Usage:
    python kafka/consumer.py --bootstrap-servers localhost:9094 \
        --topic sales-events --group-id sales-consumer-group \
        --batch-size 500 --max-idle-seconds 15 [--max-messages 10000]

Runs until either --max-messages have been consumed, or the consumer sees
no new messages for --max-idle-seconds (bounded run, suitable for being
invoked from an Airflow task rather than running forever).

Delivery semantics: offsets are committed manually, only after a batch is
committed to Postgres (at-least-once). Redelivered or re-sent events are made
harmless by a unique index on the natural key (event_date, product_category,
store_region) and INSERT ... ON CONFLICT DO NOTHING, so a partial rerun cannot
double-count. dbt's stg_sales_events also dedupes on the same key.
"""

import argparse
import json
import os
import time

import psycopg2
import psycopg2.extras
from kafka import KafkaConsumer


def get_pg_conn():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5433"),
        user=os.environ.get("POSTGRES_USER", "sales_user"),
        password=os.environ.get("POSTGRES_PASSWORD", "sales_pass"),
        dbname=os.environ.get("POSTGRES_DB", "sales_forecast"),
    )


NATURAL_KEY_INDEX_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS uq_sales_events_natural_key
    ON raw.sales_events (event_date, product_category, store_region)
"""

INSERT_SQL = """
INSERT INTO raw.sales_events
    (event_date, product_category, units_sold, revenue, avg_unit_price, store_region, promotion_flag)
VALUES %s
ON CONFLICT (event_date, product_category, store_region) DO NOTHING
"""


def flush_batch(conn, batch):
    if not batch:
        return 0
    rows = [
        (e["date"], e["product_category"], e["units_sold"], e["revenue"],
         e["avg_unit_price"], e["store_region"], e["promotion_flag"])
        for e in batch
    ]
    with conn.cursor() as cur:
        # page_size=len(rows) so rowcount covers the whole batch (one statement)
        psycopg2.extras.execute_values(cur, INSERT_SQL, rows, page_size=len(rows))
        inserted = cur.rowcount
    conn.commit()
    return inserted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap-servers", default=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9094"))
    ap.add_argument("--topic", default=os.environ.get("KAFKA_TOPIC", "sales-events"))
    ap.add_argument("--group-id", default="sales-consumer-group")
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--max-idle-seconds", type=float, default=15)
    ap.add_argument("--max-messages", type=int, default=None)
    args = ap.parse_args()

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.bootstrap_servers,
        group_id=args.group_id,
        auto_offset_reset="earliest",
        enable_auto_commit=False,  # commit offsets only after the rows are in Postgres
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        consumer_timeout_ms=int(args.max_idle_seconds * 1000),
    )

    conn = get_pg_conn()
    with conn.cursor() as cur:
        cur.execute(NATURAL_KEY_INDEX_SQL)
    conn.commit()
    total_written = 0
    total_consumed = 0
    batch = []

    print(f"Consuming from '{args.topic}' via {args.bootstrap_servers} "
          f"(group={args.group_id}, idle timeout={args.max_idle_seconds}s)")

    try:
        for msg in consumer:
            batch.append(msg.value)
            total_consumed += 1
            if len(batch) >= args.batch_size:
                n = flush_batch(conn, batch)
                consumer.commit()
                total_written += n
                print(f"Flushed batch of {len(batch)} events, {n} new rows (total written: {total_written})")
                batch = []

            if args.max_messages is not None and total_consumed >= args.max_messages:
                break
        # Only reached on a clean exit: write and commit the final partial batch.
        # On an exception, uncommitted offsets are redelivered next run (and deduped).
        if batch:
            n = flush_batch(conn, batch)
            consumer.commit()
            total_written += n
            print(f"Flushed final batch of {len(batch)} events, {n} new rows (total written: {total_written})")
    finally:
        conn.close()
        consumer.close(autocommit=False)

    skipped = total_consumed - total_written
    print(f"Done. Consumed {total_consumed} events; {total_written} new rows written to raw.sales_events; "
          f"{skipped} duplicates skipped.")


if __name__ == "__main__":
    main()
