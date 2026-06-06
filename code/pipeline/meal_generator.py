"""
Capability 6 – Sub-60-Second Meal Plan Generation.

Orchestrates the full pipeline:
  1. Clinical filter (Capability 1)
  2. Allergy exclusion via Bloom filter (Capability 2 / BAX Technique 1)
  3. Diet preference filter (Capability 3)
  4. FAISS candidate retrieval (BAX Technique 2)
  5. Diversity engine (Capability 4)
  6. Nutrient composition + scaling (Capability 5)
  → 7-day, 3-meal/day plan in < 60 s

Meal composition:
  Each meal = protein source + grain + vegetable + optional fruit/fat.
  Portions are scaled to hit the user's calorie target split (30/35/35).
"""
import time
import random
import pandas as pd
import numpy as np

from config import (
    MEAL_TYPES, MEAL_CALORIE_SPLIT, SERVING_SIZES,
    FAISS_NUTRIENTS, FAISS_NORM,
)
from pipeline.clinical_filter    import ClinicalFilter
from pipeline.allergy_engine     import AllergyEngine
from pipeline.diet_filter        import DietFilter
from pipeline.diversity_engine   import DiversityEngine
from pipeline.nutrient_analyzer  import compute_meal_nutrients, compute_daily_totals
from pipeline.faiss_engine       import build_index, query_top_k, get_meal_query_vector


# Food category tags used to compose meals
PROTEIN_CATS  = {"protein", "fish", "seafood", "poultry", "legume", "egg", "dairy_protein"}
GRAIN_CATS    = {"grain", "bread", "pasta", "rice", "cereal"}
VEG_CATS      = {"vegetable", "leafy", "root", "cruciferous"}
FRUIT_CATS    = {"fruit"}
FAT_CATS      = {"fat", "oil", "nut", "seed"}

MEAL_TEMPLATES = {
    "breakfast": [
        ("grain",   GRAIN_CATS,   0.35),
        ("protein", PROTEIN_CATS, 0.35),
        ("fruit",   FRUIT_CATS,   0.20),
        ("fat",     FAT_CATS,     0.10),
    ],
    "lunch": [
        ("protein",   PROTEIN_CATS, 0.40),
        ("grain",     GRAIN_CATS,   0.30),
        ("vegetable", VEG_CATS,     0.30),
    ],
    "dinner": [
        ("protein",   PROTEIN_CATS, 0.40),
        ("grain",     GRAIN_CATS,   0.25),
        ("vegetable", VEG_CATS,     0.35),
    ],
}

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]


def _filter_by_category(df: pd.DataFrame, cat_set: set) -> pd.DataFrame:
    """Keep rows whose 'category' field overlaps with *cat_set*."""
    if df.empty:
        return df
    mask = df["category"].apply(
        lambda c: bool(set(str(c).lower().split(",")) & {x.lower() for x in cat_set})
    )
    return df[mask]


def _scale_portion(food_row: pd.Series, target_cal: float) -> float:
    """Return grams needed to hit *target_cal* calories from this food."""
    cal_per_100 = float(food_row.get("calories", 100) or 100)
    if cal_per_100 < 5:
        cal_per_100 = 100   # default if missing
    grams = target_cal / cal_per_100 * 100
    return round(max(20, min(grams, 350)), 1)   # clamp to [20, 350] g


def _meal_name(foods: list[dict]) -> str:
    """Generate a human-readable meal name from component foods."""
    names = [item["food"].get("food_name", "").split("(")[0].strip()
             for item in foods]
    if not names:
        return "Mixed Meal"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} with {names[1]}"
    return f"{names[0]} with {names[1]} and {names[2]}"


def generate_plan(
    foods_df:         pd.DataFrame,
    faiss_index,
    faiss_id_map:     list,
    profile:          dict,
    seed:             int = 42,
) -> dict:
    """
    Generate a 7-day meal plan for *profile* using the preloaded food DB
    and FAISS index.

    profile keys: name, age, sex, calorie_target, conditions, allergens,
                  diet_type, cultural_constraints.

    Returns a rich dict (see below).
    """
    t_start = time.perf_counter()
    rng     = random.Random(seed)

    calorie_target = float(profile.get("calorie_target", 2000))

    # ── 1. Clinical filter ────────────────────────────────────────────
    clinical = ClinicalFilter(profile.get("conditions", []))
    safe_df, clinical_excl = clinical.apply(foods_df)

    # ── 2. Allergy exclusion (Bloom filter) ──────────────────────────
    allergy = AllergyEngine(profile.get("allergens", []))
    safe_df, allergy_excl = allergy.apply(safe_df)

    # ── 3. Diet preference filter ─────────────────────────────────────
    diet_filter = DietFilter(
        diet_type            = profile.get("diet_type", "Non-Vegetarian"),
        mixed_household      = profile.get("mixed_household", {}),
        cultural_constraints = profile.get("cultural_constraints", []),
    )

    # ── 4. Diversity engine ────────────────────────────────────────────
    diversity = DiversityEngine(max_repeats_per_week=2)

    all_exclusions = clinical_excl + allergy_excl
    plan_days = []       # list of 7 days; each day = list of 3 meal-nutrient dicts
    plan_meals = []      # structured meal objects for display

    for day_idx in range(7):
        day_meals_nutrients = []
        day_meal_objects    = []

        for meal_type in MEAL_TYPES:
            meal_cal_target = calorie_target * MEAL_CALORIE_SPLIT[meal_type]

            # Diet filter (supports per-meal overrides)
            meal_safe_df, diet_excl = diet_filter.apply(safe_df, meal_type=meal_type)
            if day_idx == 0:    # log diet exclusions only once
                all_exclusions += diet_excl

            if meal_safe_df.empty:
                meal_safe_df = safe_df   # fallback: ignore diet if pool empty

            # ── FAISS candidate retrieval ─────────────────────────────
            # meal_safe_df retains original foods_df indices (no reset_index)
            query_vec    = get_meal_query_vector(meal_cal_target)
            safe_idx_set = set(meal_safe_df.index)    # original df indices
            cand_idxs    = query_top_k(
                faiss_index, faiss_id_map, query_vec,
                k=min(120, max(len(safe_idx_set), 1)),
            )
            # Keep only candidates that passed all filters
            valid_cands = [i for i in cand_idxs if i in safe_idx_set][:50]
            if not valid_cands:
                valid_cands = list(safe_idx_set)[:50]
            candidates_df = foods_df.loc[valid_cands].copy().reset_index()

            # ── Compose meal from template ────────────────────────────
            template = MEAL_TEMPLATES[meal_type]
            meal_foods = []

            # Prepare full safe pool (with reset index) for category fallback
            full_safe_reset = meal_safe_df.reset_index(drop=True)

            for slot_name, cat_set, cal_fraction in template:
                slot_cal  = meal_cal_target * cal_fraction
                slot_pool = _filter_by_category(candidates_df, cat_set)

                if slot_pool.empty:
                    # Fallback 1: category match from full safe pool
                    slot_pool = _filter_by_category(full_safe_reset, cat_set)
                if slot_pool.empty:
                    # Fallback 2: any FAISS candidate
                    slot_pool = candidates_df
                if slot_pool.empty:
                    continue

                # Diversity pick
                chosen = diversity.pick_diverse(
                    slot_pool, meal_type, day_idx, n_foods=1,
                    seed=rng.randint(0, 2**31),
                )
                if chosen.empty:
                    chosen = slot_pool.head(1)

                food_row = chosen.iloc[0]
                grams    = _scale_portion(food_row, slot_cal)
                meal_foods.append({"food": food_row, "grams": grams})

            if not meal_foods:
                # Absolute fallback: random food from safe pool
                food_row = meal_safe_df.sample(1, random_state=rng.randint(0, 9999)).iloc[0]
                meal_foods.append({"food": food_row, "grams": 150})

            # ── Nutrient computation ───────────────────────────────────
            meal_nutrients = compute_meal_nutrients(meal_foods)
            day_meals_nutrients.append(meal_nutrients)

            day_meal_objects.append({
                "meal_type":  meal_type,
                "name":       _meal_name(meal_foods),
                "foods":      [
                    {
                        "food_name": str(f["food"].get("food_name", "")),
                        "category":  str(f["food"].get("category", "")),
                        "grams":     f["grams"],
                        "calories":  round(
                            float(f["food"].get("calories", 0) or 0) * f["grams"] / 100, 1
                        ),
                    }
                    for f in meal_foods
                ],
                "nutrients":  meal_nutrients,
            })

        plan_days.append(day_meals_nutrients)
        plan_meals.append({
            "day":   day_idx + 1,
            "label": DAY_NAMES[day_idx],
            "meals": day_meal_objects,
        })

    t_end = time.perf_counter()
    elapsed_s = round(t_end - t_start, 3)

    return {
        "profile":          profile,
        "plan":             plan_meals,          # for display
        "plan_nutrients":   plan_days,           # for analysis
        "exclusions":       all_exclusions,
        "diversity_score":  diversity.diversity_score(),
        "category_spread":  diversity.category_spread(),
        "generation_time":  elapsed_s,
        "n_foods_available": len(safe_df),
        "clinical_summary": clinical.summary(),
        "bloom_stats":      allergy.bloom_stats,
        "under_60s":        elapsed_s < 60,
    }
