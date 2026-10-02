-- Fact table: one row per category per day, keyed to dim_product_category
select
    d.category_id,
    s.product_category,
    s.event_date,
    s.units_sold,
    s.revenue,
    s.avg_unit_price,
    s.promotion_rate,
    s.rolling_avg_units_7d,
    s.rolling_avg_units_30d,
    s.wow_growth_rate
from {{ ref('int_daily_category_sales') }} s
left join {{ ref('dim_product_category') }} d
  on s.product_category = d.product_category
