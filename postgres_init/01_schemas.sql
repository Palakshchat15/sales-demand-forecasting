-- Schemas for the Sales Demand Forecasting project
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS warehouse;

-- raw.sales_events: written by the Kafka consumer, one row per (date, category, region) event replayed
CREATE TABLE IF NOT EXISTS raw.sales_events (
    id BIGSERIAL PRIMARY KEY,
    event_date DATE NOT NULL,
    product_category TEXT NOT NULL,
    units_sold INTEGER NOT NULL,
    revenue NUMERIC(14,2) NOT NULL,
    avg_unit_price NUMERIC(10,2) NOT NULL,
    store_region TEXT NOT NULL,
    promotion_flag BOOLEAN NOT NULL DEFAULT FALSE,
    ingested_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_sales_events_date ON raw.sales_events(event_date);
CREATE INDEX IF NOT EXISTS idx_sales_events_category ON raw.sales_events(product_category);
-- Natural key: the consumer inserts with ON CONFLICT DO NOTHING so replays can't double-count.
CREATE UNIQUE INDEX IF NOT EXISTS uq_sales_events_natural_key
    ON raw.sales_events (event_date, product_category, store_region);

-- staging.sales_rollups: written by the Spark aggregation job
CREATE TABLE IF NOT EXISTS staging.sales_rollups (
    event_date DATE NOT NULL,
    product_category TEXT NOT NULL,
    daily_units INTEGER,
    daily_revenue NUMERIC(14,2),
    rolling_avg_units_7d NUMERIC(14,4),
    rolling_avg_units_30d NUMERIC(14,4),
    wow_growth_rate NUMERIC(10,6),
    computed_at TIMESTAMP NOT NULL DEFAULT now(),
    PRIMARY KEY (event_date, product_category)
);

-- warehouse.demand_forecast: written by Prophet / comparison model
CREATE TABLE IF NOT EXISTS warehouse.demand_forecast (
    id BIGSERIAL PRIMARY KEY,
    product_category TEXT NOT NULL,
    forecast_date DATE NOT NULL,
    model_name TEXT NOT NULL,
    yhat NUMERIC(14,4),
    yhat_lower NUMERIC(14,4),
    yhat_upper NUMERIC(14,4),
    horizon_days INTEGER,
    generated_at TIMESTAMP NOT NULL DEFAULT now(),
    UNIQUE (product_category, forecast_date, model_name)
);
