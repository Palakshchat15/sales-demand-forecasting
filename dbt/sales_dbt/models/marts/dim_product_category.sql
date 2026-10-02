-- Dimension: one row per product category
with categories as (
    select distinct product_category
    from {{ ref('int_daily_category_sales') }}
)

select
    row_number() over (order by product_category) as category_id,
    product_category,
    case
        when product_category in ('Toys') then 'Highly seasonal (holiday-driven)'
        when product_category in ('Home & Garden', 'Sporting Goods') then 'Seasonal (weather-driven)'
        when product_category in ('Apparel', 'Books', 'Electronics') then 'Back-to-school / holiday influenced'
        else 'Steady demand'
    end as seasonality_profile
from categories
