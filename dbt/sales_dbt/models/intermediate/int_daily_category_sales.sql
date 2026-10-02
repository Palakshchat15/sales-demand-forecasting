-- Blend raw events (aggregated to date/category grain, including region mix and
-- promotion rate) with the Spark-computed rolling aggregations.
with events_agg as (
    select
        event_date,
        product_category,
        sum(units_sold)                                            as units_sold,
        sum(revenue)                                               as revenue,
        -- revenue-weighted: a plain mean of regional prices over-weights small regions
        sum(revenue) / nullif(sum(units_sold), 0)                 as avg_unit_price,
        avg(case when promotion_flag then 1.0 else 0.0 end)        as promotion_rate,
        count(distinct store_region)                               as region_count
    from {{ ref('stg_sales_events') }}
    group by 1, 2
),

rollups as (
    select * from {{ ref('stg_sales_rollups') }}
)

select
    e.event_date,
    e.product_category,
    e.units_sold,
    e.revenue,
    e.avg_unit_price,
    e.promotion_rate,
    e.region_count,
    r.rolling_avg_units_7d,
    r.rolling_avg_units_30d,
    r.wow_growth_rate
from events_agg e
left join rollups r
  on e.event_date = r.event_date
 and e.product_category = r.product_category
