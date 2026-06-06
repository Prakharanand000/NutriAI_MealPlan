"""
Capability 5 – Macro & Micronutrient Analysis.

Computes per-meal and per-day nutrient totals and compares them against
NIH Recommended Dietary Allowances (RDA) personalised by age and sex.
Flags any day where a tracked nutrient falls below 80 % of RDA.
"""
import pandas as pd
import numpy as np
from config import RDA_TABLE, NUTRIENT_UNITS, NUTRIENT_DISPLAY, RDA_THRESHOLD


TRACKED_MACROS = ["calories", "protein", "carbs", "fat", "fiber"]
TRACKED_MICROS = ["iron", "calcium", "vitB12", "vitD", "zinc",
                  "sodium", "potassium", "magnesium"]
ALL_NUTRIENTS  = TRACKED_MACROS + TRACKED_MICROS


def get_rda(sex: str, age: int) -> dict:
    """Return the RDA row matching the given sex and age."""
    sex_key = sex[0].upper() if sex else "M"
    for (s, lo, hi), vals in RDA_TABLE.items():
        if s == sex_key and lo <= age <= hi:
            return dict(vals)
    # Fallback to 31–50 adult values
    fallback_key = (sex_key, 31, 50)
    return dict(RDA_TABLE.get(fallback_key, list(RDA_TABLE.values())[0]))


def compute_meal_nutrients(
    foods: list[dict],          # each dict has "food" (pd.Series) and "grams"
) -> dict:
    """
    Sum nutrients across all foods in a meal, scaled to their portion weights.

    Parameters
    ----------
    foods : list of {"food": pd.Series, "grams": float}
    """
    totals = {n: 0.0 for n in ALL_NUTRIENTS}
    for item in foods:
        food_row = item["food"]
        grams    = float(item.get("grams", 100))
        scale    = grams / 100.0
        for nutrient in ALL_NUTRIENTS:
            val = food_row.get(nutrient, 0.0)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                val = 0.0
            totals[nutrient] += float(val) * scale
    return {k: round(v, 2) for k, v in totals.items()}


def compute_daily_totals(meals: list[dict]) -> dict:
    """Sum per-meal nutrient dicts into a daily total."""
    totals = {n: 0.0 for n in ALL_NUTRIENTS}
    for meal_nutrients in meals:
        for n in ALL_NUTRIENTS:
            totals[n] += meal_nutrients.get(n, 0.0)
    return {k: round(v, 2) for k, v in totals.items()}


def flag_rda_gaps(daily_totals: dict, rda: dict) -> dict:
    """
    Compare daily totals to RDA.

    Returns dict: nutrient → {"actual": X, "rda": Y, "pct": Z, "ok": bool}
    Note: sodium is a UL (upper limit) — flag if EXCEEDING; others flag if BELOW.
    """
    results = {}
    for nutrient, rda_val in rda.items():
        if nutrient not in daily_totals:
            continue
        actual = daily_totals[nutrient]
        pct    = actual / rda_val if rda_val > 0 else 1.0

        if nutrient == "sodium":
            # Flag if sodium EXCEEDS the UL
            ok = actual <= rda_val
            status = "ok" if ok else "high"
        else:
            ok = pct >= RDA_THRESHOLD
            status = "ok" if ok else ("low" if pct < RDA_THRESHOLD else "ok")

        results[nutrient] = {
            "actual":  round(actual, 2),
            "rda":     rda_val,
            "pct":     round(pct * 100, 1),
            "ok":      ok,
            "status":  status,
            "unit":    NUTRIENT_UNITS.get(nutrient, ""),
            "display": NUTRIENT_DISPLAY.get(nutrient, nutrient),
        }
    return results


def analyze_plan(plan: list[list[dict]], rda: dict) -> dict:
    """
    Full analysis for a 7-day plan.

    Parameters
    ----------
    plan : list of 7 days, each day = list of 3 meal-nutrient dicts
    rda  : RDA values from get_rda()

    Returns
    -------
    {
        "daily":  [7 daily total dicts],
        "rda_analysis": [7 flag dicts],
        "weekly_avg":   dict,
        "flagged_days": [(day_index, nutrient, pct), ...],
        "overall_ok":   bool,
    }
    """
    daily_totals = [compute_daily_totals(day_meals) for day_meals in plan]
    rda_analysis = [flag_rda_gaps(dt, rda) for dt in daily_totals]

    # Weekly average
    weekly_avg = {}
    for n in ALL_NUTRIENTS:
        vals = [dt.get(n, 0) for dt in daily_totals]
        weekly_avg[n] = round(sum(vals) / len(vals), 2) if vals else 0

    # Collect flagged days
    flagged = []
    for day_i, analysis in enumerate(rda_analysis):
        for nutrient, info in analysis.items():
            if not info["ok"]:
                flagged.append({
                    "day":      day_i + 1,
                    "nutrient": info["display"],
                    "pct":      info["pct"],
                    "status":   info["status"],
                    "actual":   info["actual"],
                    "rda":      info["rda"],
                    "unit":     info["unit"],
                })

    return {
        "daily":        daily_totals,
        "rda_analysis": rda_analysis,
        "weekly_avg":   weekly_avg,
        "flagged_days": flagged,
        "overall_ok":   len(flagged) == 0,
    }


def format_nutrient_table(rda_analysis_day: dict) -> pd.DataFrame:
    """Convert a single day's RDA analysis into a display DataFrame."""
    rows = []
    for nutrient, info in rda_analysis_day.items():
        rows.append({
            "Nutrient":   info["display"],
            "Actual":     f"{info['actual']:.1f} {info['unit']}",
            "RDA":        f"{info['rda']:.1f} {info['unit']}",
            "% of RDA":   f"{info['pct']:.0f}%",
            "Status":     "✅ OK" if info["ok"] else ("⚠️ Low" if info["status"] == "low" else "🔴 High"),
        })
    return pd.DataFrame(rows)
