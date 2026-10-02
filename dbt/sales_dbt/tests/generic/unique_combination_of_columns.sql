{#
  Generic test: fails if any combination of the given columns appears more
  than once (the grain / natural-key check). Equivalent to
  dbt_utils.unique_combination_of_columns, written locally to avoid a package dependency.
#}
{% test unique_combination_of_columns(model, combination_of_columns) %}

select
    {{ combination_of_columns | join(', ') }},
    count(*) as n_rows
from {{ model }}
group by {{ combination_of_columns | join(', ') }}
having count(*) > 1

{% endtest %}
