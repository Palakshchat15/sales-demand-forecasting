"""
Generate a demand-planning business report using an LLM, then render it to
PDF via reportlab.

Provider preference order (checked for real reachability, no mock fallback):
    1. GROQ_API_KEY env var, if set (verified with a real test call)
    2. OPENAI_API_KEY env var, if set (verified with a real test call)
    3. Local Ollama server at OLLAMA_BASE_URL, if reachable (GET /api/tags style check)
    4. Otherwise: raise RuntimeError. No mock/fake report is ever generated.
"""

import os
import sys
from datetime import date

import pandas as pd
import requests
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

from db_utils import get_engine

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
OUTPUTS_DIR = os.path.join(_PROJECT_ROOT, "outputs")

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434/v1")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


def check_groq():
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        return None
    try:
        from groq import Groq
        client = Groq(api_key=key)
        # cheap real call to validate the key actually works
        client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        print("GROQ_API_KEY is set and verified working.")
        return ("groq", client, "llama-3.1-8b-instant")
    except Exception as e:
        print(f"GROQ_API_KEY present but failed verification ({e}); not using Groq.")
        return None


def check_openai():
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        print("OPENAI_API_KEY is set and verified working.")
        return ("openai", client, "gpt-4o-mini")
    except Exception as e:
        print(f"OPENAI_API_KEY present but failed verification ({e}); not using OpenAI.")
        return None


def check_ollama():
    try:
        base = OLLAMA_BASE_URL.replace("/v1", "")
        resp = requests.get(f"{base}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        print(f"Ollama reachable at {base}. Models available: {models}")
        from openai import OpenAI
        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        return ("ollama", client, OLLAMA_MODEL)
    except Exception as e:
        print(f"Ollama not reachable at {OLLAMA_BASE_URL} ({e}); not using Ollama.")
        return None


def get_llm_client():
    for check in (check_groq, check_openai, check_ollama):
        result = check()
        if result is not None:
            return result
    raise RuntimeError(
        "No usable LLM provider found: GROQ_API_KEY/OPENAI_API_KEY not set or invalid, "
        "and local Ollama is not reachable. Refusing to generate a mock report."
    )


MODEL_LABELS = {"prophet": "Prophet", "lightgbm": "LightGBM", "seasonal_naive": "Seasonal Naive"}
YOY_WINDOW_DAYS = 90
# Mean-MAPE gaps below this are reported as a tie: per-category errors differ by
# several points, so a fraction of a point on one 90-day holdout isn't a real ranking.
TIE_MARGIN_PP = 0.5


def pct(new, old):
    return (new - old) / old * 100 if old else float("nan")


def window_mean(g, start, end):
    """Mean units/day over [start, end] (inclusive); per-day so leap years compare fairly."""
    return g[(g["ds"] >= start) & (g["ds"] <= end)]["y"].mean()


def build_context():
    engine = get_engine()

    forecast = pd.read_sql(
        "SELECT product_category, forecast_date, model_name, yhat, horizon_days "
        "FROM warehouse.demand_forecast WHERE model_name = 'prophet' ORDER BY product_category, forecast_date",
        engine,
    )
    forecast["forecast_date"] = pd.to_datetime(forecast["forecast_date"])
    forecast["yhat"] = forecast["yhat"].astype(float)
    history = pd.read_sql(
        "SELECT ds, product_category, y FROM warehouse.mart_category_daily_sales ORDER BY product_category, ds",
        engine,
    )
    history["ds"] = pd.to_datetime(history["ds"])
    last_actual = history["ds"].max()
    one_year = pd.DateOffset(years=1)

    comparison_path = os.path.join(OUTPUTS_DIR, "model_comparison.csv")
    comparison = pd.read_csv(comparison_path) if os.path.exists(comparison_path) else pd.DataFrame()

    # Average forecast units/day in three separate windows after the last actual
    # (horizon_days is the bucket: 30 = days 1-30, 60 = days 31-60, 90 = days 61-90).
    horizon_summary = (
        forecast.groupby(["product_category", "horizon_days"])["yhat"]
        .mean().round(1).unstack("horizon_days")
        .rename(columns={30: "days 1-30", 60: "days 31-60", 90: "days 61-90"})
    )

    # Recent demand trend, year over year: last 90 days of actuals vs the same
    # calendar dates a year earlier. (Comparing with the previous 30 days
    # mistook the normal December-vs-November swing for a decline.)
    recent_end = last_actual
    recent_start = last_actual - pd.Timedelta(days=YOY_WINDOW_DAYS - 1)
    # Forecast check: next 90 days forecast vs the same dates a year earlier,
    # alongside the growth those same dates showed the year before.
    f_start, f_end = forecast["forecast_date"].min(), forecast["forecast_date"].max()
    trend_rows, fc_rows = [], []
    for cat, g in history.groupby("product_category"):
        recent = window_mean(g, recent_start, recent_end)
        recent_ly = window_mean(g, recent_start - one_year, recent_end - one_year)
        trend_rows.append({"product_category": cat, "recent_avg": round(recent, 1),
                           "year_earlier_avg": round(recent_ly, 1), "yoy_pct": round(pct(recent, recent_ly), 1)})
        fc = forecast[forecast["product_category"] == cat]["yhat"].mean()
        ly = window_mean(g, f_start - one_year, f_end - one_year)
        ly2 = window_mean(g, f_start - 2 * one_year, f_end - 2 * one_year)
        fc_rows.append({"product_category": cat, "forecast_avg": round(fc, 1),
                        "same_dates_last_year_avg": round(ly, 1),
                        "forecast_yoy_pct": round(pct(fc, ly), 1), "prior_year_yoy_pct": round(pct(ly, ly2), 1)})
    trend_df = pd.DataFrame(trend_rows).sort_values("yoy_pct", ascending=False)
    forecast_yoy = pd.DataFrame(fc_rows).sort_values("forecast_yoy_pct", ascending=False)

    def d(ts):
        return ts.strftime("%Y-%m-%d")

    def fmt(rows):
        return ", ".join(f"{r.product_category} ({r.yoy_pct:+.1f}%)" for r in rows.itertuples())

    growing, declining = trend_df[trend_df["yoy_pct"] > 0], trend_df[trend_df["yoy_pct"] < 0]
    trend_facts = [
        f"Actual units sold per day, {d(recent_start)} to {d(recent_end)}, versus the same dates one year "
        f"earlier ({d(recent_start - one_year)} to {d(recent_end - one_year)}).",
        "Growing year over year, fastest first: " + (fmt(growing) or "none") + ".",
        "Declining year over year: " + (fmt(declining.iloc[::-1]) or "none") + ".",
    ]
    forecast_facts = [
        f"Prophet forecast for {d(f_start)} to {d(f_end)}, average units per day, versus actuals on the same "
        f"dates one year earlier; 'prior year' is the growth those dates showed the year before that.",
    ] + [
        f"{r.product_category}: {r.forecast_avg:.1f}/day forecast vs {r.same_dates_last_year_avg:.1f}/day "
        f"last year ({r.forecast_yoy_pct:+.1f}%; prior year {r.prior_year_yoy_pct:+.1f}%)."
        for r in forecast_yoy.itertuples()
    ]

    # Model comparison is computed here and handed to the LLM as plain facts:
    # a small local model asked to compare a long table per category invented
    # wins that didn't exist.
    winner, per_category, facts = None, pd.DataFrame(), []
    if not comparison.empty:
        means = comparison.groupby("model")["mape"].mean().sort_values()
        winner = means.index[0]
        per_category = comparison.pivot(index="product_category", columns="model", values="mape").round(2)
        models = [m for m in MODEL_LABELS if m in per_category.columns]
        per_category = per_category[models].rename(columns=MODEL_LABELS)
        labels = [MODEL_LABELS[m] for m in models]
        per_category["best_model"] = per_category[labels].idxmin(axis=1)
        cutoff = last_actual - pd.Timedelta(days=YOY_WINDOW_DAYS)
        facts.append(f"Model accuracy was measured on a past backtest window, {d(cutoff + pd.Timedelta(days=1))} "
                     f"to {d(last_actual)}: each model forecast those 90 days using only data up to {d(cutoff)}, "
                     "then was compared with what actually sold. Seasonal Naive is a simple baseline: the value "
                     "364 days earlier times the trailing-year growth.")
        facts.append("Mean holdout MAPE (simple average of the 8 categories, lower is better): "
                     + ", ".join(f"{MODEL_LABELS[m]} {v:.2f}%" for m, v in means.items()) + ".")
        gap = means.iloc[1] - means.iloc[0]
        facts.append(f"{MODEL_LABELS[winner]} has the lowest mean error, "
                     f"{gap:.2f} percentage points below {MODEL_LABELS[means.index[1]]}"
                     + (", so on this holdout the two are effectively tied." if gap < TIE_MARGIN_PP else "."))
        facts.append(f"The forward forecast in this report ({d(f_start)} to {d(f_end)}) comes from Prophet, the "
                     "only model the pipeline currently uses for forward forecasts.")
        # The recommendation is decided here, not by the LLM.
        if gap < TIE_MARGIN_PP:
            recommendation = (f"No model clearly wins: {MODEL_LABELS[winner]} and {MODEL_LABELS[means.index[1]]} "
                              "are effectively tied. Keep Prophet for forward forecasts for now, keep Seasonal "
                              "Naive as the benchmark every model must beat, and do not switch models over a gap "
                              "this small.")
        elif winner == "prophet":
            recommendation = "Use Prophet as the primary forecasting model."
        else:
            recommendation = (f"{MODEL_LABELS[winner]} is the most accurate model; the pipeline should be changed "
                              f"to produce forward forecasts with it (today they come from Prophet).")
        facts.append("Recommendation: " + recommendation)
        for label in labels:
            wins = per_category.index[per_category["best_model"] == label].tolist()
            facts.append(f"{label} had the lowest error in {len(wins)} of {len(per_category)} categories"
                         + (": " + ", ".join(wins) if wins else "") + ".")
        if {"LightGBM", "Seasonal Naive"} <= set(labels):
            beat = per_category.index[per_category["LightGBM"] < per_category["Seasonal Naive"]].tolist()
            facts.append(f"LightGBM beat the Seasonal Naive baseline in {len(beat)} of {len(per_category)} "
                         "categories" + (": " + ", ".join(beat) if beat else "") + ".")

    return {
        "horizon_summary": horizon_summary,
        "trend_df": trend_df,
        "forecast_yoy": forecast_yoy,
        "comparison": comparison,
        "per_category": per_category,
        "facts": facts,
        "trend_facts": trend_facts,
        "forecast_facts": forecast_facts,
        "winner": winner,
        "forecast_period": f"{d(f_start)} to {d(f_end)}",
    }


def build_prompt(ctx):
    lines = []
    lines.append("You are a senior demand-planning analyst. Write a concise, professional business report "
                 "for retail leadership summarizing sales demand forecasts and model performance. "
                 "Structure it with exactly these sections: Executive Summary, Forecast for the Next 90 Days, "
                 "Recent Demand Trend (Year over Year), Model Performance & Recommendation, and Action Items. "
                 "Use plain language suitable for non-technical executives. Keep it to about 400-600 words. "
                 "The models are Prophet, LightGBM and a Seasonal Naive baseline, all built in-house. "
                 "Restate only the 'Forecast facts', 'Trend facts' and 'Model performance facts' below; do not "
                 "compute new percentages, rank categories or pick winners yourself, and do not make claims "
                 "(such as volatility or causes) that the data below does not contain. "
                 "The 'Trend facts' describe past actual sales, not the forecast. The forecast period is "
                 f"{ctx['forecast_period']}; the backtest window in the model facts is in the past and is not "
                 "the forecast period. State the recommendation exactly as given in the model performance facts.")
    # The per-window table (days 1-30 / 31-60 / 61-90) is not sent: llama3.2
    # quoted the days 1-30 figure as the 90-day average. The forecast facts
    # below give the 90-day averages directly.
    lines.append("\n=== Forecast facts (use exactly these) ===")
    lines.extend(f"- {fact}" for fact in ctx["forecast_facts"])
    lines.append("\n=== Trend facts (use exactly these) ===")
    lines.extend(f"- {fact}" for fact in ctx["trend_facts"])
    if not ctx["per_category"].empty:
        lines.append("\n=== Holdout MAPE % by category (lower is better) ===")
        lines.append(ctx["per_category"].to_string())
        lines.append("\n=== Model performance facts (use exactly these) ===")
        lines.extend(f"- {fact}" for fact in ctx["facts"])
    return "\n".join(lines)


def call_llm(provider, client, model, prompt):
    print(f"Generating report using provider={provider}, model={model} ...")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.4,
    )
    return resp.choices[0].message.content


def render_pdf(report_text, ctx, out_path=None):
    if out_path is None:
        out_path = os.path.join(OUTPUTS_DIR, "demand_planning_report.pdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    doc = SimpleDocTemplate(out_path, pagesize=letter,
                             topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], fontSize=18)
    body_style = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10, leading=14, spaceAfter=8)
    heading_style = ParagraphStyle("Heading", parent=styles["Heading2"], spaceBefore=12, spaceAfter=6)

    story = []
    story.append(Paragraph("Demand Planning Report", title_style))
    story.append(Paragraph(f"Generated {date.today().isoformat()} — Sales Demand Forecasting Pipeline", styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))

    for block in report_text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith("#") or (len(block) < 80 and block.endswith(":")):
            story.append(Paragraph(block.lstrip("# ").strip(), heading_style))
        else:
            story.append(Paragraph(block.replace("\n", "<br/>"), body_style))

    # Append the model comparison table for reference
    if not ctx["comparison"].empty:
        story.append(Paragraph("Appendix: Model Comparison Table", heading_style))
        comp = ctx["comparison"].round(2)
        table_data = [list(comp.columns)] + comp.values.tolist()
        table_data = [[str(c) for c in row] for row in table_data]
        t = Table(table_data, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(t)

    doc.build(story)
    print(f"Wrote PDF report to {out_path}")


def main():
    provider, client, model = get_llm_client()
    ctx = build_context()
    prompt = build_prompt(ctx)
    report_text = call_llm(provider, client, model, prompt)

    print("\n=== GENERATED REPORT (first 1000 chars) ===")
    print(report_text[:1000])

    render_pdf(report_text, ctx)

    # Also save the raw text for inspection
    with open(os.path.join(OUTPUTS_DIR, "demand_planning_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)


if __name__ == "__main__":
    main()
