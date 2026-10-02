-- Forecast-ready mart: one row per category per day with all signal columns
-- Prophet/other models will consume this directly (ds = event_date, y = units_sold).
select
    event_date                     as ds,
    product_category,
    units_sold                     as y,
    revenue,
    avg_unit_price,
    promotion_rate,
    rolling_avg_units_7d,
    rolling_avg_units_30d,
    wow_growth_rate
from {{ ref('fact_daily_sales') }}
order by product_category, event_date
