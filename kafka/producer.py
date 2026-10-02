"""
Kafka producer that replays data/raw/sales_history.csv rows as a simulated
real-time event stream onto the `sales-events` topic.

Usage:
    python kafka/producer.py --csv data/raw/sales_history.csv \
        --bootstrap-servers localhost:9094 --topic sales-events \
        --speed 5000 --batch-size 500 \
        [--start-date 2023-01-01] [--end-date 2025-12-31] [--limit 10000]

--speed: number of rows "sent" per second worth of simulated throughput
         (used only to pace the replay a little; since this is historical
         replay, we do not literally wait real elapsed time between days).
--batch-size: number of rows produced before a flush + short pause.
--limit: optional cap on total rows produced (useful for smoke tests / for
         a bounded "replay remaining historical data" pass triggered by Airflow).
--offset-state-file: path to a small JSON file tracking how many rows have
         already been replayed, so repeated/DAG-triggered runs can resume
         from where they left off rather than re-sending everything.
"""

import argparse
import json
import os
import time

import pandas as pd
from kafka import KafkaProducer
from kafka.errors import KafkaError


def get_state(state_file):
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            return json.load(f).get("rows_sent", 0)
    return 0


def set_state(state_file, rows_sent):
    os.makedirs(os.path.dirname(state_file) or ".", exist_ok=True)
    with open(state_file, "w") as f:
        json.dump({"rows_sent": rows_sent}, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/sales_history.csv")
    ap.add_argument("--bootstrap-servers", default=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9094"))
    ap.add_argument("--topic", default=os.environ.get("KAFKA_TOPIC", "sales-events"))
    ap.add_argument("--speed", type=int, default=5000, help="approx rows/sec pacing")
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None, help="cap total rows produced this run")
    ap.add_argument("--reset-state", action="store_true", help="ignore prior offset state, start from row 0")
    ap.add_argument("--offset-state-file", default="kafka/.producer_state.json")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    total_rows = len(df)

    start_idx = 0 if args.reset_state else get_state(args.offset_state_file)
    if start_idx >= total_rows:
        print(f"All {total_rows} rows already replayed (offset={start_idx}). Nothing to do.")
        return

    end_idx = total_rows
    if args.limit is not None:
        end_idx = min(total_rows, start_idx + args.limit)

    print(f"Producing rows [{start_idx}:{end_idx}] of {total_rows} to topic '{args.topic}' "
          f"via {args.bootstrap_servers} (speed~{args.speed} rows/sec)")

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        retries=5,
        acks="all",
    )

    subset = df.iloc[start_idx:end_idx]
    sent = 0
    t0 = time.time()
    pause_per_batch = args.batch_size / max(args.speed, 1)

    for i, row in enumerate(subset.itertuples(index=False)):
        event = {
            "date": str(row.date),
            "product_category": row.product_category,
            "units_sold": int(row.units_sold),
            "revenue": float(row.revenue),
            "avg_unit_price": float(row.avg_unit_price),
            "store_region": row.store_region,
            "promotion_flag": bool(row.promotion_flag),
        }
        try:
            producer.send(args.topic, value=event)
        except KafkaError as e:
            print(f"Send failed at row {start_idx + i}: {e}")
            raise
        sent += 1

        if sent % args.batch_size == 0:
            producer.flush()
            # Checkpoint after every acknowledged batch, so a crash mid-run
            # re-sends at most one batch (which the consumer's upsert ignores).
            set_state(args.offset_state_file, start_idx + sent)
            time.sleep(pause_per_batch)

    producer.flush()
    elapsed = time.time() - t0
    new_offset = start_idx + sent
    set_state(args.offset_state_file, new_offset)

    print(f"Done. Sent {sent} events in {elapsed:.2f}s "
          f"({sent / max(elapsed, 1e-6):.1f} events/sec). New offset={new_offset}/{total_rows}")


if __name__ == "__main__":
    main()
