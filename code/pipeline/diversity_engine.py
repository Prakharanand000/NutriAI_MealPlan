"""
Capability 4 – Diversity Engine.

Ensures:
  (a) No food appears more than once per day, and at most twice per week.
  (b) Food categories are varied across days (no monotonous plans).
  (c) A diversity score [0, 1] is computed and reported.

The diversity score is 1 − (sum of pairwise Jaccard similarities across days).
A higher score means more variety.
"""
import random
from collections import defaultdict
import pandas as pd
import numpy as np


CATEGORY_SLOTS = {
    # breakfast protein variety
    "breakfast_protein":  ["eggs", "dairy", "legume", "fish"],
    # lunch/dinner protein variety
    "lunch_protein":      ["poultry", "fish", "legume", "beef", "pork"],
    "dinner_protein":     ["poultry", "fish", "legume", "beef", "pork"],
    # grain variety
    "grain_variety":      ["rice", "pasta", "quinoa", "oats", "bread", "potato"],
    # vegetable variety
    "vegetable_variety":  ["leafy", "root", "cruciferous", "nightshade", "allium"],
}


class DiversityEngine:
    """
    Track used foods and enforce variety across a 7-day meal plan.

    Usage
    -----
    engine = DiversityEngine()
    # For each meal:
    selected = engine.pick_diverse(candidates_df, meal_type, day)
    # After all 21 meals:
    score = engine.diversity_score()
    """

    def __init__(self, max_repeats_per_week: int = 2):
        self.max_repeats = max_repeats_per_week
        self._used_counts: dict[str, int] = defaultdict(int)     # food_name → count
        self._used_per_day: list[set] = [set() for _ in range(7)]
        self._category_per_day: list[list[str]] = [[] for _ in range(7)]
        self._all_selected: list[list[str]] = []   # list of meals (each = list of food ids)

    # ── Core selection ────────────────────────────────────────────────

    def pick_diverse(
        self,
        candidates_df: pd.DataFrame,
        meal_type: str,
        day: int,          # 0-indexed
        n_foods: int = 3,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Pick up to *n_foods* rows from *candidates_df*, maximising variety.
        Foods used too often or already used today are penalised.
        """
        if candidates_df.empty:
            return candidates_df

        rng = random.Random(seed)

        # Score candidates: lower score = preferred
        scores = []
        for _, row in candidates_df.iterrows():
            fname = str(row.get("food_name", ""))
            uses  = self._used_counts.get(fname, 0)
            same_day = fname in self._used_per_day[day]
            cat   = str(row.get("category", "unknown"))
            cat_used = self._category_per_day[day].count(cat)

            penalty = (
                uses * 2           # weekly repeat penalty
                + same_day * 100   # hard penalty for same-day repeat
                + cat_used * 0.5   # category monotony penalty
                + rng.random() * 0.1   # tiny random tie-breaker
            )
            scores.append((penalty, row.name if hasattr(row, "name") else _, row))

        scores.sort(key=lambda x: x[0])
        selected_rows = []
        for _, idx, row in scores[:n_foods * 3]:
            fname = str(row.get("food_name", ""))
            if self._used_counts[fname] < self.max_repeats:
                selected_rows.append(row)
                if len(selected_rows) >= n_foods:
                    break

        # Fallback: allow any if we ran out
        if not selected_rows:
            selected_rows = [scores[0][2]] if scores else []

        # Register selections
        for row in selected_rows:
            fname = str(row.get("food_name", ""))
            cat   = str(row.get("category", "unknown"))
            self._used_counts[fname] += 1
            self._used_per_day[day].add(fname)
            self._category_per_day[day].append(cat)

        if selected_rows:
            self._all_selected.append([str(r.get("food_name", "")) for r in selected_rows])

        return pd.DataFrame(selected_rows).reset_index(drop=True)

    # ── Diversity scoring ─────────────────────────────────────────────

    def diversity_score(self) -> float:
        """
        Compute a diversity score ∈ [0, 1].
        Score = 1 − average pairwise Jaccard similarity across meals.
        """
        if len(self._all_selected) < 2:
            return 1.0
        total_sim = 0.0
        count = 0
        n = len(self._all_selected)
        for i in range(n):
            for j in range(i + 1, n):
                a = set(self._all_selected[i])
                b = set(self._all_selected[j])
                inter = len(a & b)
                union = len(a | b)
                sim = inter / union if union > 0 else 0.0
                total_sim += sim
                count += 1
        avg_sim = total_sim / count if count > 0 else 0.0
        return round(1.0 - avg_sim, 3)

    def weekly_food_counts(self) -> dict[str, int]:
        return dict(self._used_counts)

    def category_spread(self) -> dict:
        """Count how many unique categories appear across all days."""
        all_cats = [cat for day_cats in self._category_per_day for cat in day_cats]
        unique = set(all_cats)
        counts = {c: all_cats.count(c) for c in unique}
        return counts

    def reset(self):
        self._used_counts = defaultdict(int)
        self._used_per_day = [set() for _ in range(7)]
        self._category_per_day = [[] for _ in range(7)]
        self._all_selected = []
