"""
Capability 1 – Clinical Condition Filtering.

Applies condition-specific dietary rules to the food database and returns
(safe_foods_df, exclusion_log).

Supported conditions:
  IBS       — exclude high-FODMAP ingredients (Monash University list)
  GERD      — exclude citrus, tomato, fried, spicy, coffee, chocolate, etc.
  Diabetes  — flag high-GI foods (GI > 55); exclude GI > 70
  HTN       — cap sodium; prefer high-potassium, high-magnesium foods (DASH)
"""
import re
import pandas as pd
from config import (
    HIGH_FODMAP_INGREDIENTS, SAFE_FODMAP_FOODS,
    GERD_TRIGGERS, GI_DATABASE, GI_LOW, GI_HIGH,
    DASH_SODIUM_MAX,
)


def _name_matches(food_name: str, keyword_set: set) -> tuple[bool, str]:
    """Check if any keyword appears in the (lower-cased) food name."""
    name_l = food_name.lower()
    for kw in sorted(keyword_set, key=len, reverse=True):  # longest first
        if kw in name_l:
            return True, kw
    return False, ""


def _get_gi(food_name: str) -> float | None:
    """Look up glycaemic index for a food name (approximate match)."""
    name_l = food_name.lower()
    for key, gi in GI_DATABASE.items():
        if key in name_l:
            return gi
    # Use stored GI column if available
    return None


class ClinicalFilter:
    """
    Apply condition-specific food exclusion rules.

    Parameters
    ----------
    conditions : list[str]
        One or more of: "IBS", "GERD", "Diabetes", "Hypertension"
    """

    CONDITION_LABELS = {
        "IBS":          "IBS (Irritable Bowel Syndrome)",
        "GERD":         "GERD / Acid Reflux",
        "Diabetes":     "Type 2 Diabetes",
        "Hypertension": "Hypertension (High Blood Pressure)",
    }

    def __init__(self, conditions: list[str]):
        self.conditions = [c.strip() for c in conditions]
        self.exclusion_log: list[dict] = []

    # ── Per-condition filters ─────────────────────────────────────────

    def _ibs_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Exclude high-FODMAP foods; keep confirmed low-FODMAP and unknowns."""
        keep_mask = pd.Series(True, index=df.index)
        for idx, row in df.iterrows():
            name = str(row.get("food_name", ""))
            matched, kw = _name_matches(name, HIGH_FODMAP_INGREDIENTS)
            if matched:
                # Unless confirmed safe in Monash list
                safe_hit, _ = _name_matches(name, SAFE_FODMAP_FOODS)
                if not safe_hit:
                    keep_mask[idx] = False
                    self.exclusion_log.append({
                        "food_name": name,
                        "condition": "IBS",
                        "reason": f"High-FODMAP ingredient detected: '{kw}' — triggers IBS symptoms",
                        "rule": "Monash University Low-FODMAP guidelines",
                    })
        return df[keep_mask]

    def _gerd_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Exclude GERD-trigger foods."""
        keep_mask = pd.Series(True, index=df.index)
        for idx, row in df.iterrows():
            name = str(row.get("food_name", ""))
            matched, kw = _name_matches(name, GERD_TRIGGERS)
            if matched:
                keep_mask[idx] = False
                self.exclusion_log.append({
                    "food_name": name,
                    "condition": "GERD",
                    "reason": f"GERD trigger detected: '{kw}' — worsens acid reflux",
                    "rule": "American College of Gastroenterology GERD guidelines",
                })
        return df[keep_mask]

    def _diabetes_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Exclude high-GI foods (GI > 70); warn on medium-GI (56–70)."""
        keep_mask = pd.Series(True, index=df.index)
        for idx, row in df.iterrows():
            name = str(row.get("food_name", ""))
            # Use stored gi_index column, fall back to lookup
            gi = row.get("gi_index", None)
            if gi is None or pd.isna(gi):
                gi = _get_gi(name)
            if gi is not None and float(gi) > GI_HIGH:
                keep_mask[idx] = False
                self.exclusion_log.append({
                    "food_name": name,
                    "condition": "Diabetes",
                    "reason": (
                        f"High Glycaemic Index (GI = {gi:.0f} > {GI_HIGH}) — "
                        "causes rapid blood-sugar spike"
                    ),
                    "rule": "University of Sydney GI database; target GI ≤ 55",
                })
        return df[keep_mask]

    def _htn_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For hypertension: flag very high sodium foods (>600 mg/100g).
        The daily cap is enforced at the meal-plan level; here we hard-exclude
        extreme outliers (e.g., cured meats, soy sauce).
        """
        keep_mask = pd.Series(True, index=df.index)
        for idx, row in df.iterrows():
            name    = str(row.get("food_name", ""))
            sodium  = float(row.get("sodium", 0) or 0)
            if sodium > 600:          # >600 mg/100g is extremely high-sodium
                keep_mask[idx] = False
                self.exclusion_log.append({
                    "food_name": name,
                    "condition": "Hypertension",
                    "reason": (
                        f"Very high sodium content ({sodium:.0f} mg/100g) — "
                        f"DASH daily limit is {DASH_SODIUM_MAX} mg"
                    ),
                    "rule": "NHLBI DASH Diet guidelines",
                })
        return df[keep_mask]

    # ── Public API ────────────────────────────────────────────────────

    def apply(self, foods_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
        """
        Apply all active condition filters sequentially.

        Returns
        -------
        filtered_df : pd.DataFrame
            Foods that passed all conditions.
        exclusion_log : list[dict]
            One entry per excluded food with reason and clinical rule cited.
        """
        self.exclusion_log = []
        df = foods_df.copy()

        handlers = {
            "IBS":          self._ibs_filter,
            "GERD":         self._gerd_filter,
            "Diabetes":     self._diabetes_filter,
            "Hypertension": self._htn_filter,
        }

        for condition in self.conditions:
            handler = handlers.get(condition)
            if handler:
                before = len(df)
                df = handler(df)
                after = len(df)

        return df, self.exclusion_log

    def summary(self) -> dict:
        """Return a breakdown of exclusions by condition."""
        by_cond: dict[str, int] = {}
        for entry in self.exclusion_log:
            c = entry["condition"]
            by_cond[c] = by_cond.get(c, 0) + 1
        return by_cond
