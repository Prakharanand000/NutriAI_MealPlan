"""
Capability 3 – Dietary Preference Handling.

Supports vegan, vegetarian, pescatarian, and non-vegetarian modes.
Also handles meal-level mixed households (e.g., vegan breakfast,
non-vegetarian dinner) and cultural/religious constraints.
"""
import pandas as pd
from config import DIET_EXCLUSIONS


DIET_LABEL_MAP = {
    "Non-Vegetarian": "non_vegetarian",
    "Vegetarian": "vegetarian",
    "Vegan": "vegan",
    "Pescatarian": "pescatarian",
}

# Cultural / religious constraint exclusion sets
RELIGIOUS_EXCLUSIONS = {
    "halal":   {"pork", "lard", "gelatin", "alcohol", "wine", "beer"},
    "kosher":  {"pork", "shellfish", "mixing meat and dairy", "lard"},
    "hindu":   {"beef", "veal"},
    "jain":    {"root vegetables", "onion", "garlic", "potato", "carrot",
                "radish", "beet"},
}


class DietFilter:
    """
    Filter foods by dietary preference and optional cultural constraints.

    Parameters
    ----------
    diet_type : str
        One of "Non-Vegetarian", "Vegetarian", "Vegan", "Pescatarian".
    mixed_household : dict, optional
        Per-meal overrides, e.g. {"breakfast": "vegan", "dinner": "non_vegetarian"}.
    cultural_constraints : list[str], optional
        e.g. ["halal", "kosher"]
    """

    def __init__(
        self,
        diet_type: str,
        mixed_household: dict | None = None,
        cultural_constraints: list[str] | None = None,
    ):
        self.diet_key = DIET_LABEL_MAP.get(diet_type, "non_vegetarian")
        self.mixed_household = mixed_household or {}
        self.cultural = [c.lower() for c in (cultural_constraints or [])]
        self.exclusion_log: list[dict] = []

    # ── Internal helpers ──────────────────────────────────────────────

    def _exclusion_set(self, meal_type: str | None = None) -> set:
        if meal_type and meal_type in self.mixed_household:
            diet_key = self.mixed_household[meal_type]
        else:
            diet_key = self.diet_key
        base = set(DIET_EXCLUSIONS.get(diet_key, set()))
        for constraint in self.cultural:
            base |= RELIGIOUS_EXCLUSIONS.get(constraint, set())
        return base

    def _food_violates(self, food_name: str, exclusions: set) -> tuple[bool, str]:
        name_l = food_name.lower()
        for kw in sorted(exclusions, key=len, reverse=True):
            if kw in name_l:
                return True, kw
        return False, ""

    def _stored_diet_ok(self, row: pd.Series, diet_key: str) -> bool:
        """Check pre-computed boolean columns: is_vegan, is_vegetarian, etc."""
        col_map = {
            "vegan":          "is_vegan",
            "vegetarian":     "is_vegetarian",
            "pescatarian":    "is_pescatarian",
            "non_vegetarian": None,  # no restriction
        }
        col = col_map.get(diet_key)
        if col is None:
            return True
        val = row.get(col, None)
        if val is None or str(val).strip() in ("nan", ""):
            return None   # unknown — fall through to text check
        return bool(val) and str(val).strip() not in ("0", "False", "false")

    # ── Public API ────────────────────────────────────────────────────

    def apply(
        self,
        foods_df: pd.DataFrame,
        meal_type: str | None = None,
    ) -> tuple[pd.DataFrame, list[dict]]:
        """
        Filter *foods_df* for the active diet type (and meal-level overrides).

        Returns (filtered_df, exclusion_log_for_this_call).
        """
        self.exclusion_log = []
        exclusions = self._exclusion_set(meal_type)

        if not exclusions:
            return foods_df.copy().reset_index(drop=True), []

        keep_mask = pd.Series(True, index=foods_df.index)

        # Determine effective diet key for label
        if meal_type and meal_type in self.mixed_household:
            eff_key = self.mixed_household[meal_type]
        else:
            eff_key = self.diet_key

        for idx, row in foods_df.iterrows():
            name = str(row.get("food_name", ""))

            # Fast path: stored boolean column
            stored = self._stored_diet_ok(row, eff_key)
            if stored is False:
                keep_mask[idx] = False
                self.exclusion_log.append({
                    "food_name": name,
                    "diet":      eff_key,
                    "reason":    f"Not suitable for {eff_key} diet",
                })
                continue
            if stored is True:
                continue

            # Text-based check
            violated, kw = self._food_violates(name, exclusions)
            if violated:
                keep_mask[idx] = False
                self.exclusion_log.append({
                    "food_name": name,
                    "diet":      eff_key,
                    "reason":    f"Contains '{kw}' — not allowed in {eff_key} diet",
                })

        return (
            foods_df[keep_mask],
            self.exclusion_log,
        )

    def get_per_meal_filter(self) -> dict[str, "DietFilter"]:
        """Return per-meal DietFilter instances for mixed-household support."""
        filters = {}
        for meal_type, diet_key in self.mixed_household.items():
            f = DietFilter(
                diet_type=next(
                    (k for k, v in DIET_LABEL_MAP.items() if v == diet_key),
                    "Non-Vegetarian",
                ),
                cultural_constraints=list(self.cultural),
            )
            filters[meal_type] = f
        return filters
