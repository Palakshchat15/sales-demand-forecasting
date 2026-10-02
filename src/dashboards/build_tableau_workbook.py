"""
Generates tableau/Sales_Forecast.twbx: a dark green dashboard with a row of
eight KPI tiles and six charts (3 x 2 grid), built from data/processed/.

Same layout and styling approach as the Retail Analytics dashboard: text
worksheets as big-number tiles, no gridlines/axis lines/dividers, borderless
zones, automatic sizing. The forecast chart keeps a category dropdown and a
colour legend.

Tableau Public only opens workbooks whose data sources are extracts, so every
input table is written to a .hyper file (Tableau Hyper API) and packaged with
the workbook XML into a .twbx.

Runs on the Windows host or inside the Airflow container. The original CSV
path recorded in the workbook (used only for "Refresh Extract") should be the
host path: set TABLEAU_DATA_DIR when running in Docker.
"""
import calendar
import csv
import os
import re
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

import pandas as pd
from tableauhyperapi import (Connection, CreateMode, HyperProcess, SqlType, TableDefinition,
                             TableName, Telemetry, escape_string_literal)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = Path(os.environ.get("PROCESSED_DIR", PROJECT_ROOT / "data" / "processed"))
TABLEAU_DIR = Path(os.environ.get("TABLEAU_DIR", PROJECT_ROOT / "tableau"))
WORKBOOK_NAME = "Sales_Forecast"
TWBX_PATH = TABLEAU_DIR / f"{WORKBOOK_NAME}.twbx"
DATA_DIR_IN_WORKBOOK = os.environ.get("TABLEAU_DATA_DIR", str(PROCESSED_DIR))

THEME = {
    "background": "#0e1f16",
    "text": "#9ccfae",
    "title": "#ffffff",
    "kpi_label": "#7fd49a",
    "kpi_value": "#ffffff",
    "mark": "#3fa968",
    # Sequential gradient for measures on Color: light green (low) -> dark green (high).
    # The dark end keeps dark mark labels readable (about 4:1) and stays visible on the background.
    "gradient": ("#c7ebd2", "#2e8b57"),
    "label_on_mark": "#0e1f16",  # label colour inside heatmap / treemap cells
    # Forecast-chart series, in series_code order (0, 1, 2). Actual and Forecast never
    # overlap in time; Backtest overlaps Actual, so it gets the strongest contrast (near white).
    "series": {"Actual": "#3fa968", "Forecast": "#a8e0bb", "Backtest": "#f0fff4"},
}

HYPER_TYPES = {
    "integer": SqlType.big_int(),
    "real": SqlType.double(),
    "date": SqlType.date(),
    "datetime": SqlType.timestamp(),
    "string": SqlType.text(),
}
EXTRACT_TABLE = TableName("Extract", "Extract")

# prefix -> (derivation, column-instance type)
AGGREGATIONS = {
    "none": ("None", "nominal"),
    "sum": ("Sum", "quantitative"),
    "avg": ("Avg", "quantitative"),
    "cnt": ("Count", "quantitative"),
    "tmn": ("Month-Trunc", "quantitative"),
    "twk": ("Week-Trunc", "quantitative"),
    "tdy": ("Day-Trunc", "quantitative"),
}

CAPTIONS = {
    "ds": "Date",
    "units": "Units / Day",
    "series": "Series",
    "product_category": "Category",
    "mape": "Holdout MAPE (%)",
    "model": "Model",
    "forecast_units_90d": "Forecast Units, Next 90 Days",
    "sales_month": "Date",
    "revenue_m": "Revenue ($M)",
    "month": "Month of Year",
    "seasonal_index": "Seasonal Index (100 = avg)",
    "revenue": "Revenue ($)",
    "revenue_label": "Revenue",
    "growth_label": "Forecast (vs Q1 2025)",
}

MODEL_ORDER = ["Naive", "Prophet", "LightGBM"]  # "Naive" = the seasonal-naive baseline
MONTH_ORDER = [calendar.month_abbr[m] for m in range(1, 13)]

KPIS = [  # (column in dash_kpis.csv, tile label)
    ("total_revenue", "REVENUE, 3 YRS"),
    ("units_sold", "UNITS SOLD, 3 YRS"),
    ("q4_yoy", "Q4 UNITS YOY"),
    ("forecast_90d", "90-DAY FORECAST"),
    ("forecast_growth", "FCST VS Q1 2025"),
    ("mape_seasonal_naive", "BASELINE MAPE"),  # seasonal naive
    ("mape_prophet", "PROPHET MAPE"),
    ("mape_lightgbm", "LIGHTGBM MAPE"),
]

DATASOURCES = {
    "kpi": {"caption": "KPIs", "csv": "dash_kpis.csv"},
    "fva": {"caption": "Forecast vs Actual", "csv": "dash_forecast_vs_actual.csv"},
    "comp": {"caption": "Model Comparison", "csv": "dash_model_mape.csv"},
    "next": {"caption": "Next 90 Day Forecast", "csv": "dash_next_90_days.csv"},
    "monthly": {"caption": "Monthly Revenue", "csv": "dash_monthly_revenue.csv"},
    "season": {"caption": "Seasonality", "csv": "dash_seasonality.csv"},
    "share": {"caption": "Revenue Share", "csv": "dash_revenue_share.csv"},
}

FORECAST_CHART = "Daily Units: Actual, Backtest, Forecast"

CHARTS = [
    # Line: last 180 days of actuals, the holdout backtest, and the forward forecast
    # for one category at a time (dropdown).
    # Tableau ignores a categorical colour map written into the XML (tried at worksheet and
    # data source level), so the series are coloured through a numeric code on a custom
    # green ramp, and the title doubles as the legend (each series name in its colour).
    {"name": FORECAST_CHART, "ds": "fva", "mark": "Line",
     "rows": [("sum", "units")], "cols": [("tdy", "ds")], "detail": ("none", "series"),
     "color": ("avg", "series_code"), "ramp": list(THEME["series"].values()),
     "title_legend": THEME["series"], "pick_one": ("product_category", "Toys"),
     # Tableau draws the first series on top, so Backtest goes first to stay visible over Actual.
     "manual_order": [("series", list(THEME["series"])[::-1])]},
    # Heatmap: holdout error per category and model; shows the Prophet / baseline tie.
    {"name": "Holdout MAPE % (darker = worse)", "ds": "comp", "mark": "Square",
     "rows": [("none", "product_category")], "cols": [("none", "model")],
     "color": ("sum", "mape"), "text": [("sum", "mape")], "labels": True,
     "manual_order": [("model", MODEL_ORDER)]},
    # Bar: forward forecast per category, labelled with growth vs the same quarter last year.
    {"name": "Next 90 Days Forecast (Units, vs Q1 2025)", "ds": "next", "mark": "Bar",
     "rows": [("none", "product_category")], "cols": [("sum", "forecast_units_90d")],
     "color": ("sum", "forecast_units_90d"), "text": [("none", "growth_label")], "labels": True,
     "sort_desc_by": ("product_category", ("sum", "forecast_units_90d"))},
    # Area: total monthly revenue, three years.
    {"name": "Monthly Revenue, All Categories ($M)", "ds": "monthly", "mark": "Area",
     "rows": [("sum", "revenue_m")], "cols": [("tmn", "sales_month")]},
    # Heatmap: seasonal index (avg daily units in that month / category average).
    {"name": "Seasonality Index (100 = category's avg day)", "ds": "season", "mark": "Square",
     "rows": [("none", "product_category")], "cols": [("none", "month")],
     "color": ("sum", "seasonal_index"), "text": [("sum", "seasonal_index")], "labels": True,
     "label_size": 8, "header_width": [("product_category", 100)], "header_size": 8,
     "manual_order": [("month", MONTH_ORDER)]},
    # Treemap: each category's share of three-year revenue.
    {"name": "Share of Revenue by Category (3 Years)", "ds": "share", "mark": "Square",
     "rows": [], "cols": [], "size": ("sum", "revenue"), "color": ("sum", "revenue"),
     "text": [("none", "product_category"), ("none", "revenue_label")], "labels": True},
]

DASHBOARD_NAME = "Sales Demand Forecasting Overview"
KPI_HEIGHT, GAP = 12000, 600
CHART_W = (100000 - 4 * GAP) // 3
CHART_H = (100000 - KPI_HEIGHT - 4 * GAP) // 2
KPI_W = (100000 - (len(KPIS) + 1) * GAP) // len(KPIS)
CONTROL_H = 8600  # category dropdown above the forecast chart

CLEAN_RULES = """
          <style-rule element='gridline'>
            <format attr='line-visibility' scope='rows' value='off' />
            <format attr='line-visibility' scope='cols' value='off' />
          </style-rule>
          <style-rule element='zeroline'>
            <format attr='line-visibility' value='off' />
          </style-rule>
          <style-rule element='axis'>
            <format attr='line-visibility' value='off' />
          </style-rule>
          <style-rule element='table-div'>
            <format attr='line-visibility' scope='rows' value='off' />
            <format attr='line-visibility' scope='cols' value='off' />
          </style-rule>"""

FORMATS = {"mape": "n#,##0.0", "revenue_m": "n#,##0.0"}  # Tableau number formats by column

ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ---------------------------------------------------------------- dashboard data

def fmt_money(value):
    return f"${value / 1e6:.1f}M" if abs(value) >= 1e6 else f"${value:,.0f}"


def fmt_pct_change(value):
    return f"{value:+.1%}"


def prepare_dashboard_data():
    """Builds the small tables the dashboard plots from the CSVs export_dashboard_data.py wrote."""
    monthly = pd.read_csv(PROCESSED_DIR / "monthly_category_sales.csv", parse_dates=["sales_month"])
    comp = pd.read_csv(PROCESSED_DIR / "model_comparison.csv")
    nxt = pd.read_csv(PROCESSED_DIR / "next_90_day_forecast.csv")

    fva = pd.read_csv(PROCESSED_DIR / "forecast_vs_actual.csv")
    fva["series_code"] = fva["series"].map({name: i for i, name in enumerate(THEME["series"])})
    fva.to_csv(PROCESSED_DIR / "dash_forecast_vs_actual.csv", index=False)

    comp["mape"] = comp["mape"].round(1)
    # Short display name so the heatmap column header isn't truncated ("Seasonal Nai..").
    comp["model"] = comp["model"].replace({"Seasonal Naive": "Naive"})
    comp[["product_category", "model", "mape"]].to_csv(PROCESSED_DIR / "dash_model_mape.csv", index=False)

    # Forecast horizon (next 90 days) starts the day after the last actual; compare with
    # the same calendar months a year earlier.
    monthly["year"] = monthly["sales_month"].dt.year
    monthly["moy"] = monthly["sales_month"].dt.month
    last = monthly["sales_month"].max()
    horizon = pd.date_range(last + pd.offsets.MonthBegin(1), periods=3, freq="MS")
    prior = monthly[monthly["sales_month"].isin(horizon - pd.DateOffset(years=1))]
    prior_units = prior.groupby("product_category")["units_sold"].sum()
    nxt["growth"] = nxt["product_category"].map(nxt.set_index("product_category")["forecast_units_90d"]
                                                / prior_units - 1)
    nxt["growth_label"] = [f"{u:,.0f} ({fmt_pct_change(g)})" for u, g in zip(nxt["forecast_units_90d"], nxt["growth"])]
    nxt[["product_category", "forecast_units_90d", "growth_label"]].to_csv(
        PROCESSED_DIR / "dash_next_90_days.csv", index=False)

    rev = monthly.groupby("sales_month", as_index=False)["revenue"].sum()
    rev["revenue_m"] = (rev["revenue"] / 1e6).round(2)
    rev[["sales_month", "revenue_m"]].to_csv(PROCESSED_DIR / "dash_monthly_revenue.csv", index=False)

    # Seasonal index: average daily units in each calendar month relative to the
    # category's overall average (per day, so February isn't penalised for being short).
    monthly["daily_units"] = monthly["units_sold"] / monthly["sales_month"].dt.days_in_month
    season = monthly.groupby(["product_category", "moy"])["daily_units"].mean().rename("avg").reset_index()
    season["seasonal_index"] = (100 * season["avg"]
                                / season.groupby("product_category")["avg"].transform("mean")).round().astype(int)
    season["month"] = season["moy"].map(lambda m: calendar.month_abbr[m])
    season[["product_category", "month", "seasonal_index"]].to_csv(
        PROCESSED_DIR / "dash_seasonality.csv", index=False)

    share = monthly.groupby("product_category", as_index=False)["revenue"].sum()
    share["revenue_label"] = [f"{fmt_money(r)} ({r / share['revenue'].sum():.0%})" for r in share["revenue"]]
    share["revenue"] = share["revenue"].round(2)
    share.to_csv(PROCESSED_DIR / "dash_revenue_share.csv", index=False)

    q4 = lambda year: monthly[(monthly["year"] == year) & (monthly["moy"] >= 10)]["units_sold"].sum()  # noqa: E731
    last_year = last.year
    # Mean MAPE from the unrounded per-category values (comp["mape"] was rounded for labels).
    mean_mape = pd.read_csv(PROCESSED_DIR / "model_comparison.csv").groupby("model")["mape"].mean()
    kpis = {
        "total_revenue": fmt_money(monthly["revenue"].sum()),
        "units_sold": f"{monthly['units_sold'].sum() / 1e6:.2f}M",
        "q4_yoy": fmt_pct_change(q4(last_year) / q4(last_year - 1) - 1),
        "forecast_90d": f"{nxt['forecast_units_90d'].sum():,.0f}",
        "forecast_growth": fmt_pct_change(nxt["forecast_units_90d"].sum() / prior_units.sum() - 1),
        "mape_seasonal_naive": f"{mean_mape['Seasonal Naive']:.2f}%",
        "mape_prophet": f"{mean_mape['Prophet']:.2f}%",
        "mape_lightgbm": f"{mean_mape['LightGBM']:.2f}%",
    }
    pd.DataFrame([kpis]).to_csv(PROCESSED_DIR / "dash_kpis.csv", index=False)
    print("KPIs:", kpis)


# ---------------------------------------------------------------- workbook XML

def infer_type(values):
    """Returns (datatype, role, type) for a CSV column from sample values."""
    vals = [v for v in values if v not in ("", None)]
    if vals and all(re.fullmatch(r"-?\d+", v) for v in vals):
        return "integer", "measure", "quantitative"
    if vals and all(re.fullmatch(r"-?\d+(\.\d+)?([eE][-+]?\d+)?", v) for v in vals):
        return "real", "measure", "quantitative"
    if vals and all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", v) for v in vals):
        return "date", "dimension", "ordinal"
    if vals and all(re.fullmatch(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?", v) for v in vals):
        return "datetime", "dimension", "ordinal"
    return "string", "dimension", "nominal"


def read_schema(csv_path, sample_rows=100000):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        samples = [row for _, row in zip(range(sample_rows), reader)]
    schema = []
    for i, name in enumerate(header):
        if not ID_RE.match(name):
            raise ValueError(f"{csv_path.name}: column {name!r} needs quoting support")
        schema.append((name, i, *infer_type([r[i] for r in samples if i < len(r)])))
    return schema


def caption(col):
    return CAPTIONS.get(col, col.replace("_", " ").title())


def ds_name(key):
    return f"federated.{key}"


def instance_name(prefix, col):
    kind = "nk" if AGGREGATIONS[prefix][1] == "nominal" else "qk"
    return f"[{prefix}:{col}:{kind}]"


def field_ref(ds_key, prefix, col):
    return f"[{ds_name(ds_key)}].{instance_name(prefix, col)}"


def write_hyper(csv_path, schema, hyper_path, hyper):
    """Loads csv_path into Extract.Extract in a new .hyper file; returns row count."""
    table = TableDefinition(EXTRACT_TABLE, [
        TableDefinition.Column(name, HYPER_TYPES[dt]) for name, _, dt, _, _ in schema
    ])
    with Connection(hyper.endpoint, hyper_path, CreateMode.CREATE_AND_REPLACE) as conn:
        conn.catalog.create_schema(EXTRACT_TABLE.schema_name)
        conn.catalog.create_table(table)
        return conn.execute_command(
            f"COPY {EXTRACT_TABLE} FROM {escape_string_literal(str(csv_path))} "
            "WITH (format csv, header, NULL '')"
        )


def extract_xml(key, row_count):
    now = datetime.now()
    return f"""      <extract count='-1' enabled='true' units='records'>
        <connection access_mode='readonly' authentication='auth-none' author-locale='en_US' class='hyper' dbname='Data/Extracts/{key}.hyper' default-settings='hyper' schema='Extract' sslmode='' tablename='Extract' update-time={quoteattr(now.strftime('%m/%d/%Y %I:%M:%S %p'))} username='tableau'>
          <relation name='Extract' table='[Extract].[Extract]' type='table' />
          <refresh>
            <refresh-event add-from-file-path={quoteattr(key)} increment-value='%null%' refresh-type='create' rows-inserted='{row_count}' timestamp-start={quoteattr(now.strftime('%Y-%m-%d %H:%M:%S.000'))} />
          </refresh>
        </connection>
      </extract>"""


def gradient_xml(field, colors=None):
    """Custom colour ramp (default: the theme's green gradient) for a continuous measure on Color.
    Tableau's built-in sequential palettes default to blue; a custom-interpolated palette is
    honoured, whereas categorical colour maps written into the XML are silently ignored."""
    stops = "\n".join(f"                <color>{c}</color>" for c in (colors or THEME["gradient"]))
    return f"""
          <style-rule element='mark'>
            <encoding attr='color' field='{field}' type='custom-interpolated'>
              <color-palette custom='true' name='' type='ordered-sequential'>
{stops}
              </color-palette>
            </encoding>
          </style-rule>"""


def datasource_xml(key, spec, schema, row_count):
    csv_file = spec["csv"]
    stem = Path(csv_file).stem
    relation_cols = "\n".join(
        f"            <column datatype='{dt}' name={quoteattr(n)} ordinal='{i}' />"
        for n, i, dt, _, _ in schema
    )
    ds_cols = "\n".join(
        f"      <column caption={quoteattr(caption(n))} datatype='{dt}'"
        + (f" default-format='{FORMATS[n]}'" if n in FORMATS else "")
        + f" name='[{n}]' role='{role}' type='{typ}' />"
        for n, _, dt, role, typ in schema
    )
    return f"""    <datasource caption={quoteattr(spec['caption'])} inline='true' name='{ds_name(key)}' version='18.1'>
      <connection class='federated'>
        <named-connections>
          <named-connection caption={quoteattr(stem)} name='textscan.{key}'>
            <connection class='textscan' directory={quoteattr(DATA_DIR_IN_WORKBOOK.replace(chr(92), '/'))} filename={quoteattr(csv_file)} password='' server='' />
          </named-connection>
        </named-connections>
        <relation connection='textscan.{key}' name={quoteattr(csv_file)} table='[{stem}#csv]' type='table'>
          <columns character-set='UTF-8' header='yes' locale='en_US' separator=','>
{relation_cols}
          </columns>
        </relation>
      </connection>
      <aliases enabled='yes' />
{ds_cols}
{extract_xml(key, row_count)}
    </datasource>"""


def title_xml(text, color, size, align=None, legend=None):
    """legend: {label: colour} -> the title keeps its text before ':' and then lists each
    label in its own colour, so the title doubles as the colour legend."""
    align_attr = " fontalignment='1'" if align == "center" else ""
    runs = [f"<run bold='true' fontcolor='{color}' fontsize='{size}'{align_attr}>{escape(text)}</run>"]
    if legend:
        runs = [f"<run bold='true' fontcolor='{color}' fontsize='{size}'>{escape(text.split(':')[0] + ':')}</run>"]
        runs += [f"<run bold='true' fontcolor='{c}' fontsize='{size}'>{escape('  ' + chr(9632) + ' ' + label)}</run>"
                 for label, c in legend.items()]
    body = "\n".join(f"            {r}" for r in runs)
    return f"""      <layout-options>
        <title>
          <formatted-text>
{body}
          </formatted-text>
        </title>
      </layout-options>"""


def sheet_style_xml(extra_rules=""):
    text = f"""
          <style-rule element='worksheet'>
            <format attr='color' value='{THEME["text"]}' />
          </style-rule>"""
    return f"""        <style>
          <style-rule element='table'>
            <format attr='background-color' value='{THEME["background"]}' />
          </style-rule>{text}{CLEAN_RULES}{extra_rules}
        </style>"""


def pane_style_xml(labels, fixed_color):
    formats = []
    if labels:
        formats.append("<format attr='mark-labels-show' value='true' />")
    if fixed_color:
        formats.append(f"<format attr='mark-color' value='{THEME['mark']}' />")
    if not formats:
        return ""
    body = "\n".join(f"                {f}" for f in formats)
    return f"""
            <style>
              <style-rule element='mark'>
{body}
              </style-rule>
            </style>"""


def shelf(key, fields):
    """Multiple fields on one shelf are nested (Tableau's '/' operator); empty -> ''."""
    refs = [field_ref(key, p, c) for p, c in fields]
    if not refs:
        return ""
    return refs[0] if len(refs) == 1 else "(" + " / ".join(refs) + ")"


def shelf_xml(tag, key, fields):
    value = shelf(key, fields)
    return f"<{tag}>{value}</{tag}>" if value else f"<{tag} />"


def dependencies_xml(key, types, used):
    deps = []
    for col in dict.fromkeys(c for _, c in used):
        dt, role, typ = types[col]
        deps.append(f"            <column caption={quoteattr(caption(col))} datatype='{dt}' "
                    f"name='[{col}]' role='{role}' type='{typ}' />")
        for prefix, c in used:
            if c == col:
                derivation, itype = AGGREGATIONS[prefix]
                deps.append(
                    f"            <column-instance column='[{col}]' derivation='{derivation}' "
                    f"name='{instance_name(prefix, col)}' pivot='key' type='{itype}' />"
                )
    return "\n".join(deps)


def chart_xml(ws, schemas):
    key = ws["ds"]
    types = {n: (dt, role, typ) for n, _, dt, role, typ in schemas[key]}
    extra = [ws[k] for k in ("color", "detail", "size") if k in ws] + ws.get("text", [])
    if "pick_one" in ws:
        extra.append(("none", ws["pick_one"][0]))
    used = list(dict.fromkeys(ws["rows"] + ws["cols"] + extra))
    filter_xml, slices_xml = "", ""
    if "pick_one" in ws:
        field, default = ws["pick_one"]
        member = quoteattr(chr(34) + default + chr(34))
        filter_xml += f"""
          <filter class='categorical' column='{field_ref(key, "none", field)}'>
            <groupfilter function='member' level='{instance_name("none", field)}' member={member} user:ui-domain='database' user:ui-enumeration='inclusive' user:ui-marker='enumerate' />
          </filter>"""
        slices_xml = f"""
          <slices>
            <column>{field_ref(key, "none", field)}</column>
          </slices>"""
    if "sort_desc_by" in ws:
        dim, measure = ws["sort_desc_by"]
        filter_xml += f"""
          <sort class='computed' column='{field_ref(key, "none", dim)}' direction='DESC' using='{field_ref(key, *measure)}' />"""
    for dim, values in ws.get("manual_order", []):
        buckets = "\n".join(f"              <bucket>{escape(chr(34) + v + chr(34))}</bucket>" for v in values)
        filter_xml += f"""
          <sort class='manual' column='{field_ref(key, "none", dim)}' direction='ASC'>
            <dictionary>
{buckets}
            </dictionary>
          </sort>"""
    enc = []
    if "color" in ws:
        enc.append(f"<color column='{field_ref(key, *ws['color'])}' />")
    if "size" in ws:
        enc.append(f"<size column='{field_ref(key, *ws['size'])}' />")
    enc += [f"<text column='{field_ref(key, *t)}' />" for t in ws.get("text", [])]
    if "detail" in ws:
        enc.append(f"<lod column='{field_ref(key, *ws['detail'])}' />")
    encodings = ""
    if enc:
        body = "\n".join(f"              {e}" for e in enc)
        encodings = f"""
            <encodings>
{body}
            </encodings>"""
    if ws["mark"] == "Square" and ws.get("text"):
        # Labels inside light-to-mid green cells (heatmap, treemap) get dark text; elsewhere
        # labels sit on the dark background and keep the theme text colour.
        label = "&#10;".join(f"&lt;{field_ref(key, *t)}&gt;" for t in ws["text"])
        size = f" fontsize='{ws['label_size']}'" if "label_size" in ws else ""
        encodings += f"""
            <customized-label>
              <formatted-text>
                <run fontcolor='{THEME["label_on_mark"]}'{size}>{label}</run>
              </formatted-text>
            </customized-label>"""
    encodings += pane_style_xml(ws.get("labels", False), fixed_color="color" not in ws)
    color_rules = ""
    if "color" in ws and ws["color"][0] != "none":
        color_rules = gradient_xml(field_ref(key, *ws["color"]), ws.get("ramp"))
    for dim, width in ws.get("header_width", []):  # row-header width in pixels (avoids "Electro..")
        color_rules += f"""
          <style-rule element='header'>
            <format attr='width' field='{field_ref(key, "none", dim)}' value='{width}' />
          </style-rule>"""
    if "header_size" in ws:
        color_rules += f"""
          <style-rule element='header'>
            <format attr='font-size' value='{ws["header_size"]}' />
          </style-rule>"""
    return f"""    <worksheet name={quoteattr(ws['name'])}>
{title_xml(ws['name'], THEME['title'], 12, legend=ws.get("title_legend"))}
      <table>
        <view>
          <datasources>
            <datasource caption={quoteattr(DATASOURCES[key]['caption'])} name='{ds_name(key)}' />
          </datasources>
          <datasource-dependencies datasource='{ds_name(key)}'>
{dependencies_xml(key, types, used)}
          </datasource-dependencies>{filter_xml}{slices_xml}
          <aggregation value='true' />
        </view>
{sheet_style_xml(color_rules)}
        <panes>
          <pane selection-relaxation-option='selection-relaxation-allow'>
            <view>
              <breakdown value='auto' />
            </view>
            <mark class='{ws['mark']}' />{encodings}
          </pane>
        </panes>
        {shelf_xml('rows', key, ws['rows'])}
        {shelf_xml('cols', key, ws['cols'])}
      </table>
    </worksheet>"""


def kpi_style_xml():
    return f"""        <style>
          <style-rule element='table'>
            <format attr='background-color' value='{THEME["background"]}' />
          </style-rule>
          <style-rule element='cell'>
            <format attr='font-size' value='20' />
            <format attr='font-weight' value='bold' />
            <format attr='color' value='{THEME["kpi_value"]}' />
            <format attr='text-align' value='center' />
            <format attr='vertical-align' value='center' />
          </style-rule>{CLEAN_RULES}
        </style>"""


def kpi_xml(column, label, schemas):
    """A big-number tile: one text mark showing a preformatted value, titled with the label."""
    types = {n: (dt, role, typ) for n, _, dt, role, typ in schemas["kpi"]}
    ref = field_ref("kpi", "none", column)
    return f"""    <worksheet name={quoteattr(label)}>
{title_xml(label, THEME['kpi_label'], 9, align='center')}
      <table>
        <view>
          <datasources>
            <datasource caption='KPIs' name='{ds_name("kpi")}' />
          </datasources>
          <datasource-dependencies datasource='{ds_name("kpi")}'>
{dependencies_xml("kpi", types, [("none", column)])}
          </datasource-dependencies>
          <aggregation value='true' />
        </view>
{kpi_style_xml()}
        <panes>
          <pane selection-relaxation-option='selection-relaxation-allow'>
            <view>
              <breakdown value='auto' />
            </view>
            <mark class='Text' />
            <encodings>
              <text column='{ref}' />
            </encodings>
            <customized-label>
              <formatted-text>
                <run bold='true' fontalignment='1' fontcolor='{THEME["kpi_value"]}' fontsize='22'>&lt;{ref}&gt;</run>
              </formatted-text>
            </customized-label>
          </pane>
        </panes>
        <rows />
        <cols />
      </table>
    </worksheet>"""


ZONE_STYLE = ("            <zone-style>\n"
              "              <format attr='border-style' value='none' />\n"
              "              <format attr='border-width' value='0' />\n"
              "              <format attr='margin' value='4' />\n"
              "            </zone-style>\n")


def zones():
    """KPI tiles across the top, charts in a 3 x 2 grid below. The forecast chart's
    cell also holds its category dropdown at the top (its legend is in the chart title)."""
    placed, zone_id = [], 2  # (id, name, x, y, w, h, extra attributes)
    for i, (_, label) in enumerate(KPIS):
        x = GAP + i * (KPI_W + GAP)
        placed.append((zone_id, label, x, GAP, KPI_W, KPI_HEIGHT, "")); zone_id += 1
    for i, chart in enumerate(CHARTS):
        col, row = i % 3, i // 3
        x = GAP + col * (CHART_W + GAP)
        y = KPI_HEIGHT + 2 * GAP + row * (CHART_H + GAP)
        if chart["name"] == FORECAST_CHART:
            key, field = chart["ds"], chart["pick_one"][0]
            placed.append((zone_id, chart["name"], x, y, CHART_W, CONTROL_H,
                           f" mode='dropdown' param='{field_ref(key, 'none', field)}' type-v2='filter'"))
            zone_id += 1
            y, h = y + CONTROL_H, CHART_H - CONTROL_H
        else:
            h = CHART_H
        placed.append((zone_id, chart["name"], x, y, CHART_W, h, "")); zone_id += 1
    return "\n".join(
        f"          <zone h='{h}' id='{zid}'{attrs} name={quoteattr(name)} w='{w}' x='{x}' y='{y}'>\n"
        f"{ZONE_STYLE}          </zone>"
        for zid, name, x, y, w, h, attrs in placed
    )


def dashboard_xml():
    return f"""    <dashboard name={quoteattr(DASHBOARD_NAME)}>
      <style>
        <style-rule element='table'>
          <format attr='background-color' value='{THEME["background"]}' />
        </style-rule>      </style>
      <size maxheight='900' maxwidth='1600' minheight='900' minwidth='1600' />
      <zones>
        <zone h='100000' id='1' type-v2='layout-basic' w='100000' x='0' y='0'>
{zones()}
        </zone>
      </zones>
    </dashboard>"""


def windows_xml():
    names = [label for _, label in KPIS] + [c["name"] for c in CHARTS]
    viewpoints = "\n".join(
        f"        <viewpoint name={quoteattr(n)}>\n          <zoom type='entire-view' />\n        </viewpoint>"
        for n in names
    )
    return f"""  <windows source-height='30'>
    <window class='dashboard' maximized='true' name={quoteattr(DASHBOARD_NAME)}>
      <viewpoints>
{viewpoints}
      </viewpoints>
      <active id='-1' />
    </window>
  </windows>"""


def build():
    prepare_dashboard_data()
    TABLEAU_DIR.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="twbx_"))
    schemas, row_counts, hyper_files = {}, {}, {}
    # log_dir keeps hyperd.log out of the working directory; work_dir is deleted below.
    with HyperProcess(Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU,
                      parameters={"log_dir": str(work_dir)}) as hyper:
        for key, spec in DATASOURCES.items():
            path = PROCESSED_DIR / spec["csv"]
            if not path.exists():
                raise FileNotFoundError(f"{path} not found; run export_dashboard_data.py first")
            schemas[key] = read_schema(path)
            hyper_files[key] = work_dir / f"{key}.hyper"
            row_counts[key] = write_hyper(path, schemas[key], hyper_files[key], hyper)
            print(f"  {spec['csv']}: {row_counts[key]} rows -> {key}.hyper")
    datasources = "\n".join(
        datasource_xml(k, s, schemas[k], row_counts[k]) for k, s in DATASOURCES.items()
    )
    worksheets = "\n".join(
        [kpi_xml(col, label, schemas) for col, label in KPIS] + [chart_xml(c, schemas) for c in CHARTS]
    )
    xml = f"""<?xml version='1.0' encoding='utf-8' ?>
<workbook original-version='18.1' source-build='2026.2.0 (20262.26.0819.2015)' source-platform='win' version='18.1' xmlns:user='http://www.tableausoftware.com/xml/user'>
  <preferences>
    <preference name='ui.encoding.shelf.height' value='24' />
    <preference name='ui.shelf.height' value='26' />
  </preferences>
  <datasources>
{datasources}
  </datasources>
  <worksheets>
{worksheets}
  </worksheets>
  <dashboards>
{dashboard_xml()}
  </dashboards>
{windows_xml()}
</workbook>
"""
    with zipfile.ZipFile(TWBX_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{WORKBOOK_NAME}.twb", xml)
        for key, hyper_path in hyper_files.items():
            z.write(hyper_path, f"Data/Extracts/{key}.hyper")
    shutil.rmtree(work_dir, ignore_errors=True)
    print(f"Wrote {TWBX_PATH}")


if __name__ == "__main__":
    build()
