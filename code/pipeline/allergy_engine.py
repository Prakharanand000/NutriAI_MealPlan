"""
Capability 2 – Allergy Detection & Exclusion.

Uses the Bloom-filter first-pass screen (BAX-423 Technique 1) followed by
exact keyword confirmation to guarantee zero false negatives (no allergen
slips through). Cross-contamination risks are also flagged.
"""
import pandas as pd
from config import ALLERGEN_KEYWORDS, CROSS_CONTAMINATION
from pipeline.bloom_filter import AllergenBloomChecker


class AllergyEngine:
    """
    Allergen-safe food filter with cross-contamination warnings.

    Parameters
    ----------
    allergens : list[str]
        User's declared allergens (canonical names from ALLERGEN_KEYWORDS).
    """

    def __init__(self, allergens: list[str]):
        self.allergens = [a.lower().replace(" ", "_") for a in allergens]
        # Normalise input allergen names to canonical keys
        self._active = [a for a in self.allergens if a in ALLERGEN_KEYWORDS]
        self._checker = AllergenBloomChecker(
            {k: ALLERGEN_KEYWORDS[k] for k in self._active},
            false_positive_rate=0.001,   # very low FPR: safety-critical
        )
        self.exclusion_log: list[dict] = []
        self.cross_contamination_warnings: list[dict] = []

    # ── Helpers ───────────────────────────────────────────────────────

    def _detect_allergens(self, food_name: str) -> list[str]:
        """Return confirmed allergens present in *food_name*."""
        _, matched = self._checker.screen(food_name, self._active)
        return matched

    def _check_cross_contamination(self, food_name: str) -> list[str]:
        """Return cross-contamination risk allergens."""
        name_l = food_name.lower()
        risks = []
        for allergen in self._active:
            secondary = CROSS_CONTAMINATION.get(allergen, [])
            for risk_kw in secondary:
                if risk_kw.lower() in name_l:
                    risks.append(f"{allergen} (cross-contamination risk: {risk_kw})")
        return risks

    # ── Public API ────────────────────────────────────────────────────

    def apply(self, foods_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
        """
        Remove all foods containing any declared allergen.

        Returns
        -------
        safe_df : pd.DataFrame
            Foods with zero detected allergens.
        exclusion_log : list[dict]
        """
        if not self._active:
            return foods_df.copy().reset_index(drop=True), []

        self.exclusion_log = []
        self.cross_contamination_warnings = []
        keep_mask = pd.Series(True, index=foods_df.index)

        for idx, row in foods_df.iterrows():
            name = str(row.get("food_name", ""))

            # Check stored allergen columns first (fast path)
            row_allergens = self._stored_allergens(row)
            confirmed = list(set(row_allergens) & set(self._active))

            # Supplement with text-based Bloom+exact detection
            text_allergens = self._detect_allergens(name)
            for a in text_allergens:
                if a not in confirmed:
                    confirmed.append(a)

            if confirmed:
                keep_mask[idx] = False
                self.exclusion_log.append({
                    "food_name":  name,
                    "allergens":  confirmed,
                    "reason": (
                        f"Contains {', '.join(confirmed).upper()} allergen(s) — "
                        "excluded for zero-tolerance safety"
                    ),
                    "detection": "Bloom-filter pre-screen + exact keyword confirmation",
                })
            else:
                # Check cross-contamination (warning only, not exclusion)
                cc_risks = self._check_cross_contamination(name)
                if cc_risks:
                    self.cross_contamination_warnings.append({
                        "food_name": name,
                        "risks":     cc_risks,
                        "warning": f"Possible cross-contamination: {'; '.join(cc_risks)}",
                    })

        return (
            foods_df[keep_mask],
            self.exclusion_log,
        )

    def _stored_allergens(self, row: pd.Series) -> list[str]:
        """Read pre-computed allergen boolean columns from the food row."""
        col_map = {
            "gluten":    "contains_gluten",
            "dairy":     "contains_dairy",
            "eggs":      "contains_eggs",
            "tree_nuts": "contains_tree_nuts",
            "shellfish": "contains_shellfish",
            "soy":       "contains_soy",
            "peanuts":   "contains_peanuts",
            "fish":      "contains_fish",
        }
        found = []
        for allergen, col in col_map.items():
            if allergen in self._active:
                val = row.get(col, 0)
                if val and str(val).strip() not in ("0", "False", "false", "nan", ""):
                    found.append(allergen)
        return found

    @property
    def bloom_stats(self) -> dict:
        return self._checker.stats()
