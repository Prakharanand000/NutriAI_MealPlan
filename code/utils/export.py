"""
Export utilities – PDF and CSV export of the 7-day meal plan.
Uses fpdf2 for pure-Python PDF generation (no system dependencies).
"""
import io
import csv
from datetime import date


def plan_to_csv(plan_result: dict) -> bytes:
    """
    Export the 7-day meal plan to a UTF-8 CSV and return bytes.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)

    profile = plan_result.get("profile", {})
    writer.writerow(["NutriAI – 7-Day Meal Plan"])
    writer.writerow(["User", profile.get("name", ""), "Date", str(date.today())])
    writer.writerow(["Conditions", ", ".join(profile.get("conditions", []))])
    writer.writerow(["Allergens",  ", ".join(profile.get("allergens", []))])
    writer.writerow(["Diet Type",  profile.get("diet_type", "")])
    writer.writerow([])

    writer.writerow(["Day", "Meal", "Name", "Foods (g)", "Calories", "Protein(g)",
                     "Carbs(g)", "Fat(g)", "Fiber(g)", "Iron(mg)", "Calcium(mg)",
                     "VitB12(µg)", "VitD(µg)", "Zinc(mg)"])

    for day_obj in plan_result.get("plan", []):
        day_label = f"Day {day_obj['day']} – {day_obj['label']}"
        for meal in day_obj.get("meals", []):
            foods_str = " + ".join(
                f"{f['food_name']} ({f['grams']}g)" for f in meal.get("foods", [])
            )
            n = meal.get("nutrients", {})
            writer.writerow([
                day_label,
                meal["meal_type"].capitalize(),
                meal["name"],
                foods_str,
                round(n.get("calories",  0), 1),
                round(n.get("protein",   0), 1),
                round(n.get("carbs",     0), 1),
                round(n.get("fat",       0), 1),
                round(n.get("fiber",     0), 1),
                round(n.get("iron",      0), 2),
                round(n.get("calcium",   0), 1),
                round(n.get("vitB12",    0), 2),
                round(n.get("vitD",      0), 2),
                round(n.get("zinc",      0), 2),
            ])
        writer.writerow([])

    writer.writerow(["Diversity Score", plan_result.get("diversity_score", "N/A")])
    writer.writerow(["Generation Time (s)", plan_result.get("generation_time", "N/A")])
    writer.writerow([])
    writer.writerow(["Exclusions (first 30)"])
    writer.writerow(["Food", "Reason"])
    for ex in plan_result.get("exclusions", [])[:30]:
        writer.writerow([
            ex.get("food_name", ""),
            ex.get("reason", ex.get("reason", "")),
        ])

    return buf.getvalue().encode("utf-8")


def plan_to_pdf(plan_result: dict) -> bytes:
    """
    Export the 7-day meal plan to a PDF and return bytes.
    Uses fpdf2; falls back to a minimal plain-text PDF if not installed.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        return _plain_text_pdf(plan_result)

    profile = plan_result.get("profile", {})

    def _safe(text: str) -> str:
        """Strip/replace characters outside Latin-1 so Helvetica won't crash."""
        return (
            str(text)
            .replace("–", "-")   # en-dash
            .replace("—", "-")   # em-dash
            .replace("’", "'")   # right single quote
            .replace("‘", "'")   # left single quote
            .replace("“", '"')   # left double quote
            .replace("”", '"')   # right double quote
            .replace("µ", "u")   # micro sign
            .encode("latin-1", errors="replace").decode("latin-1")
        )

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Title ─────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(27, 42, 74)    # dark blue
    pdf.cell(0, 10, _safe("NutriAI - 7-Day Personalised Meal Plan"), ln=True, align="C")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 6, _safe(
             f"Generated: {date.today()} | "
             f"User: {profile.get('name', 'N/A')} | "
             f"Diet: {profile.get('diet_type', 'N/A')} | "
             f"Conditions: {', '.join(profile.get('conditions', [])) or 'None'} | "
             f"Allergens: {', '.join(profile.get('allergens', [])) or 'None'}"),
             ln=True, align="C")

    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # ── Summary stats ─────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Plan Summary", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(60, 6, f"Calorie Target: {profile.get('calorie_target', 'N/A')} kcal/day")
    pdf.cell(60, 6, f"Diversity Score: {plan_result.get('diversity_score', 'N/A')}")
    pdf.cell(60, 6, f"Generation Time: {plan_result.get('generation_time', 'N/A')} s", ln=True)
    pdf.ln(4)

    col_w = [28, 20, 70, 22, 22, 22]
    headers = ["Day", "Meal", "Name / Foods", "Calories", "Protein", "Fiber"]

    for day_obj in plan_result.get("plan", []):
        # Day header
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(27, 42, 74)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(
            sum(col_w), 7,
            _safe(f"Day {day_obj['day']} - {day_obj['label']}"),
            ln=True, fill=True,
        )
        pdf.set_text_color(0, 0, 0)

        # Column headers
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(220, 230, 240)
        for w, h in zip(col_w, headers):
            pdf.cell(w, 6, h, border=1, fill=True)
        pdf.ln()

        # Meal rows
        pdf.set_font("Helvetica", "", 8)
        for meal in day_obj.get("meals", []):
            n = meal.get("nutrients", {})
            foods_str = "; ".join(
                f"{f['food_name']} {f['grams']}g" for f in meal.get("foods", [])
            )[:60]
            values = [
                _safe(day_obj['label'][:6]),
                _safe(meal["meal_type"].capitalize()[:8]),
                _safe(foods_str),
                f"{n.get('calories', 0):.0f} kcal",
                f"{n.get('protein', 0):.1f} g",
                f"{n.get('fiber', 0):.1f} g",
            ]
            fill = False
            pdf.set_fill_color(245, 248, 252)
            for w, v in zip(col_w, values):
                pdf.cell(w, 6, _safe(str(v)), border=1, fill=fill)
            pdf.ln()

        pdf.ln(3)

        if pdf.get_y() > 260:
            pdf.add_page()

    # ── Exclusions ────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Food Exclusion Log (Why Excluded)", ln=True)
    pdf.set_font("Helvetica", "", 8)
    for ex in plan_result.get("exclusions", [])[:40]:
        food   = _safe(ex.get("food_name", "")[:35])
        reason = _safe(ex.get("reason", "")[:80])
        pdf.multi_cell(0, 5, f"- {food}: {reason}")

    return bytes(pdf.output())


def _plain_text_pdf(plan_result: dict) -> bytes:
    """Minimal fallback: embed plan text in a raw PDF byte string."""
    lines = ["NutriAI 7-Day Meal Plan", "=" * 40, ""]
    for day_obj in plan_result.get("plan", []):
        lines.append(f"Day {day_obj['day']} – {day_obj['label']}")
        for meal in day_obj.get("meals", []):
            lines.append(f"  {meal['meal_type'].upper()}: {meal['name']}")
        lines.append("")
    text = "\n".join(lines)
    # Return as plain text bytes (not valid PDF but won't crash the app)
    return text.encode("utf-8")
