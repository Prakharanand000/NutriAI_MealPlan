"""
Benchmark utilities – measures and compares BAX-423 technique performance.
Used in the Benchmarks tab of the Streamlit app.
"""
import time
import pandas as pd
from config import ALLERGEN_KEYWORDS
from pipeline.bloom_filter import benchmark_bloom_vs_set
from pipeline.faiss_engine import benchmark_faiss_vs_numpy


def run_bloom_benchmark(n_queries: int = 50_000) -> dict:
    """Run and return Bloom filter vs naive linear scan benchmark."""
    result = benchmark_bloom_vs_set(ALLERGEN_KEYWORDS, n_queries=n_queries)
    result["description"] = (
        "Bloom filter (FNV-1a, 3 seeds: 0/42/137) pre-screens allergens "
        "with zero false negatives, then confirms positives with exact keyword lookup. "
        "Baseline: naive linear scan that iterates every allergen keyword per food "
        "with no early-exit — representing the pipeline without a Bloom index. "
        "The Bloom filter also uses a fixed-size bit array regardless of vocabulary size, "
        "keeping memory bounded as the allergen list grows."
    )
    return result


def run_faiss_benchmark(foods_df: pd.DataFrame, n_queries: int = 500) -> dict:
    """Run and return FAISS vs numpy cosine search benchmark."""
    result = benchmark_faiss_vs_numpy(foods_df, n_queries=n_queries)
    result["description"] = (
        "FAISS IndexFlatIP indexes 13-dimensional nutritional profile vectors. "
        "After L2 normalisation, inner product equals cosine similarity. "
        "FAISS avoids brute-force scan of all foods on every meal generation, "
        "enabling sub-second candidate retrieval even over 400k foods."
    )
    return result


def run_pipeline_timing(
    foods_df: pd.DataFrame,
    faiss_index,
    faiss_id_map: list,
    profile: dict,
) -> dict:
    """
    Time each pipeline stage individually and return a breakdown.
    """
    import numpy as np
    from pipeline.clinical_filter  import ClinicalFilter
    from pipeline.allergy_engine   import AllergyEngine
    from pipeline.diet_filter      import DietFilter
    from pipeline.faiss_engine     import query_top_k, get_meal_query_vector

    stages = {}

    # Stage 1: Clinical filter
    t = time.perf_counter()
    clinical = ClinicalFilter(profile.get("conditions", []))
    safe_df, _ = clinical.apply(foods_df)
    stages["1. Clinical Filter"] = round((time.perf_counter() - t) * 1000, 2)

    # Stage 2: Allergy Bloom filter
    t = time.perf_counter()
    allergy = AllergyEngine(profile.get("allergens", []))
    safe_df, _ = allergy.apply(safe_df)
    stages["2. Allergy Bloom Filter"] = round((time.perf_counter() - t) * 1000, 2)

    # Stage 3: Diet filter
    t = time.perf_counter()
    diet_filter = DietFilter(profile.get("diet_type", "Non-Vegetarian"))
    safe_df, _ = diet_filter.apply(safe_df)
    stages["3. Diet Filter"] = round((time.perf_counter() - t) * 1000, 2)

    # Stage 4: FAISS retrieval (21 queries = 7 days × 3 meals)
    t = time.perf_counter()
    for _ in range(21):
        qv = get_meal_query_vector(np.random.uniform(400, 800))
        query_top_k(faiss_index, faiss_id_map, qv, k=30)
    stages["4. FAISS Retrieval (21×)"] = round((time.perf_counter() - t) * 1000, 2)

    # Stage 5: Nutrient computation (proxy)
    from pipeline.nutrient_analyzer import compute_meal_nutrients
    sample_foods = [
        {"food": safe_df.iloc[i % max(len(safe_df), 1)], "grams": 150}
        for i in range(9)
    ] if not safe_df.empty else []
    t = time.perf_counter()
    for i in range(0, len(sample_foods), 3):
        compute_meal_nutrients(sample_foods[i:i+3])
    stages["5. Nutrient Computation"] = round((time.perf_counter() - t) * 1000, 2)

    total = sum(stages.values())
    return {
        "stages":       stages,
        "total_ms":     round(total, 2),
        "total_s":      round(total / 1000, 3),
        "under_60s":    total < 60_000,
        "n_foods_in":   len(foods_df),
        "n_foods_safe": len(safe_df),
    }
