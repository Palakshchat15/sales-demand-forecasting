-- Pass-through cleanup of the Spark-computed rollups
with source as (
    select * from {{ source('staging_src', 'sales_rollups') }}
)

select
    event_date::date            as event_date,
    trim(product_category)      as product_category,
    daily_units,
    daily_revenue,
    rolling_avg_units_7d,
    rolling_avg_units_30d,
    wow_growth_rate,
    computed_at
from source
