-- Clean and dedupe raw.sales_events.
-- The natural key is (event_date, product_category, store_region): the source
-- CSV has at most one row per key. A replayed Kafka message would otherwise be
-- double-counted, so if the same key arrives more than once only the most
-- recently ingested row is kept. (The consumer also inserts with
-- ON CONFLICT DO NOTHING against a unique index; this is the second line of defence.)
with source as (
    select * from {{ source('raw', 'sales_events') }}
),

cleaned as (
    select
        id,
        event_date::date                       as event_date,
        trim(product_category)                 as product_category,
        trim(store_region)                     as store_region,
        units_sold::int                        as units_sold,
        revenue::numeric(14,2)                  as revenue,
        avg_unit_price::numeric(10,2)           as avg_unit_price,
        coalesce(promotion_flag, false)        as promotion_flag,
        ingested_at
    from source
    where event_date is not null
      and product_category is not null
      and units_sold >= 0
),

ranked as (
    select
        *,
        row_number() over (
            partition by event_date, product_category, store_region
            order by ingested_at desc, id desc
        ) as row_num
    from cleaned
)

select
    event_date,
    product_category,
    store_region,
    units_sold,
    revenue,
    avg_unit_price,
    promotion_flag,
    ingested_at
from ranked
where row_num = 1
