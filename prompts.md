# prompts.md — Key AI Prompts Used
BAX-423 Final Project · NutriAI · Spring 2026

---

## Prompt 1 — Architecture Design
**Prompt:** "Design a Python pipeline for a clinical diet planning app that integrates a Bloom filter for allergen screening and FAISS for meal candidate retrieval. The pipeline must handle 4 medical conditions (IBS, GERD, T2D, Hypertension) and run end-to-end in under 60 seconds."

**Used for:** Establishing the 6-stage pipeline architecture and choosing which BAX-423 techniques to apply at which stage.

**How modified:** Added the RL bandit adaptive layer (not in initial design), switched FAISS from L2 to IndexFlatIP after L2-normalisation for cosine similarity semantics.

---

## Prompt 2 — Bloom Filter Implementation
**Prompt:** "Implement a Bloom filter in pure Python (no external libraries) with configurable false-positive rate, multiple SHA-256 hash functions, and a byte array bit vector. Include a benchmark comparing it to a Python set for allergen keyword lookup."

**Used for:** `pipeline/bloom_filter.py` — `BloomFilter` and `AllergenBloomChecker` classes.

**How modified:** Added the two-stage strategy (Bloom pre-screen → exact keyword confirm) to eliminate false positives while preserving zero false negatives. Critical for safety-critical allergen detection.

---

## Prompt 3 — FAISS Nutritional Embeddings
**Prompt:** "How should I represent food items as vectors for FAISS similarity search in a diet planning context? I want semantically meaningful embeddings without using a large language model."

**Used for:** `pipeline/faiss_engine.py` — using normalised nutritional profiles (13 nutrients) as the embedding space.

**How modified:** Added L2 normalisation before indexing so inner product equals cosine similarity. Added a fallback to numpy brute-force when faiss-cpu is not installed.

---

## Prompt 4 — RL Bandit for Adaptive Learning
**Prompt:** "Implement epsilon-greedy and Thompson Sampling bandits in Python for learning meal category preferences from 1-5 star ratings. Show a comparison learning curve."

**Used for:** `pipeline/bandit.py` — `EpsilonGreedyBandit`, `ThompsonBandit`, `simulate_learning_curve`.

**How modified:** Added epsilon decay (multiplicative) to epsilon-greedy. Normalised 1-5 ratings to [0,1] reward scale. Added Beta distribution Thompson Sampling with binary success/failure (rating ≥ 4 = success).

---

## Prompt 5 — Clinical Rule Encoding
**Prompt:** "What are the specific dietary rules for IBS (low-FODMAP), GERD, Type 2 Diabetes (GI-based), and Hypertension (DASH diet) that should be encoded in a food filtering system? Cite sources."

**Used for:** `config.py` — `HIGH_FODMAP_INGREDIENTS`, `GERD_TRIGGERS`, `GI_DATABASE`, `DASH_SODIUM_MAX`.

**How modified:** Cross-referenced Monash University FODMAP list, American College of Gastroenterology GERD guidelines, University of Sydney GI database, and NHLBI DASH guidelines. Added clinical source citations as comments.

---

## Prompt 6 — NIH RDA Table
**Prompt:** "Provide the NIH Dietary Reference Intakes (RDA) for calories, protein, carbs, fat, fiber, iron, calcium, vitamin B12, vitamin D, zinc, sodium, potassium, and magnesium by age group and sex."

**Used for:** `config.py` — `RDA_TABLE` dictionary.

**How modified:** Structured as a Python dict keyed by (sex, age_min, age_max) for O(1) lookups. Added 4 age brackets per sex covering 19–99.

---

## Prompt 7 — Streamlit UI Design
**Prompt:** "Design a Streamlit app with sidebar profile input and 6 tabs: 7-day plan view, nutrition dashboard, exclusion log, adaptive learning, benchmarks, and persona tests. Include plotly charts for macro bars and micronutrient radar."

**Used for:** `app.py` structure and CSS styling.

**How modified:** Added session state management for plan persistence across tab switches. Added streaming per-day expand/collapse UI. Added real-time benchmark button.

---

## Prompt 8 — Diversity Scoring
**Prompt:** "Implement a diversity score for a meal plan that measures how varied the meals are across 7 days. Use Jaccard similarity between meal food sets."

**Used for:** `pipeline/diversity_engine.py` — `diversity_score()` method.

**How modified:** Extended to also track category-level diversity (not just food-level) and report a `category_spread` dict.

---

## Prompt 9 — PDF Export
**Prompt:** "Generate a formatted PDF from a 7-day meal plan dictionary using fpdf2. Include a title, user profile summary, a table of meals per day with nutrient values, and an exclusion log."

**Used for:** `utils/export.py` — `plan_to_pdf()`.

**How modified:** Added graceful fallback to plain text bytes if fpdf2 is not installed. Added Streamlit download button integration.

---

*Note: All AI-generated code was reviewed, tested, and modified. The author understands every design decision and can walk through the codebase in the live demo.*
