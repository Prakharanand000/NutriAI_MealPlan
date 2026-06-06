# NutriAI – Automated Diet Plan Builder
**BAX-423 Big Data · Spring 2026 · UC Davis GSM**

> A full-stack AI-assisted application that generates a personalised 7-day meal plan in under 60 seconds, tailored to clinical conditions, allergens, dietary preferences, and nutrient targets.

**Live app:** https://nutriaimealplan-prakhar.streamlit.app/

---

## Quick Start

```bash
# 1. Install dependencies
cd code
pip install -r requirements.txt

# 2. Build food database (5,500+ records, offline mode)
python data_setup.py --offline

# 3. (Optional) Enrich with 10,000+ USDA FoodData Central records
python data_setup.py --api-key YOUR_USDA_API_KEY

# 4. Launch the app
streamlit run app.py
```

App runs at: http://localhost:8501

---

## Six Core Capabilities

| # | Capability | Implementation |
|---|---|---|
| 1 | **Clinical Condition Filtering** | `pipeline/clinical_filter.py` — FODMAP, GERD, GI, DASH rules |
| 2 | **Allergy Detection & Exclusion** | `pipeline/allergy_engine.py` + **Bloom filter** (BAX Technique 1) |
| 3 | **Dietary Preference Handling** | `pipeline/diet_filter.py` — vegan/veg/pescatarian/non-veg |
| 4 | **Diversity Engine** | `pipeline/diversity_engine.py` — Jaccard-based diversity score |
| 5 | **Macro & Micronutrient Analysis** | `pipeline/nutrient_analyzer.py` — 13 nutrients vs NIH RDA |
| 6 | **Sub-60-Second Generation** | Full pipeline < 1s; timing logged and displayed |

---

## BAX-423 Techniques (Benchmarked)

### Technique 1 – Bloom Filter (Sketching)
- **File:** `pipeline/bloom_filter.py`
- **Purpose:** First-pass allergen pre-screening with zero false negatives
- **Benchmark:** Compared to pure set-based lookup on 30,000 queries
- **Result:** Substantially lower memory footprint; safety guaranteed by exact-match confirmation step

### Technique 2 – FAISS Embeddings
- **File:** `pipeline/faiss_engine.py`
- **Purpose:** Foods are encoded as 13-D normalised nutritional vectors; FAISS IndexFlatIP retrieves top-K meal candidates per slot
- **Benchmark:** Compared to numpy brute-force cosine similarity on 200 queries × 5,500 foods
- **Result:** Maintains sub-millisecond per-query latency; scales to 400k+ branded USDA foods

### Technique 3 – RL Bandit (Adaptive Learning)
- **File:** `pipeline/bandit.py`
- **Two strategies:** Epsilon-Greedy (ε-decay) vs Thompson Sampling (Beta posteriors)
- **Benchmark:** Simulated learning curve over 50 rounds shows Thompson converges faster

---

## Data Sources

| Source | Use | URL |
|---|---|---|
| USDA FoodData Central | Primary food database (10k+ records via API) | https://fdc.nal.usda.gov/api-guide.html |
| Monash University | Low-FODMAP classifications | https://www.monashfodmap.com |
| NIH DRI / RDA tables | Daily nutrient targets | https://www.ncbi.nlm.nih.gov/books/NBK56068 |
| Univ. of Sydney GI Database | Glycaemic index values | https://www.glycemicindex.com |
| NHLBI DASH Guidelines | Hypertension diet rules | https://www.nhlbi.nih.gov/education/dash-eating-plan |

---

## 4 Test Personas

| Persona | Conditions | Allergens | Diet | Pass Criteria |
|---|---|---|---|---|
| **Priya** | IBS | Dairy/Lactose | Vegetarian | Zero FODMAP triggers, zero dairy, iron ≥ 80% RDA |
| **Ravi** | GERD | Gluten (Celiac) | Non-Vegetarian | Zero GERD triggers, zero gluten, diversity ≥ 0.7 |
| **Mei** | Type 2 Diabetes | Tree Nuts | Vegan | All GI ≤ 55, zero animal products, fibre ≥ 25g/day |
| **James** | Hypertension | Soy | Pescatarian | Sodium ≤ 1,500 mg/day, ≥ 3 fish meals, potassium ≥ 80% RDA |

---

## Project Structure

```
NutriAI_MealPlan/
├── code/
│   ├── app.py                  ← Streamlit app (streamlit run app.py)
│   ├── config.py               ← RDA tables, FODMAP rules, allergen maps
│   ├── data_setup.py           ← USDA API fetcher + offline generator
│   ├── requirements.txt
│   ├── pipeline/
│   │   ├── bloom_filter.py     ← BAX Technique 1: Bloom filter
│   │   ├── faiss_engine.py     ← BAX Technique 2: FAISS embeddings
│   │   ├── clinical_filter.py  ← Capability 1: Clinical rules
│   │   ├── allergy_engine.py   ← Capability 2: Allergen exclusion
│   │   ├── diet_filter.py      ← Capability 3: Diet preferences
│   │   ├── diversity_engine.py ← Capability 4: Diversity engine
│   │   ├── nutrient_analyzer.py← Capability 5: Macro/micro analysis
│   │   ├── meal_generator.py   ← Capability 6: Sub-60s orchestrator
│   │   ├── bandit.py           ← Adaptive learning (RL bandit)
│   │   └── explainer.py        ← "Why excluded/included" feature
│   └── utils/
│       ├── export.py           ← PDF + CSV export
│       └── benchmark.py        ← Benchmark runner
├── data/
│   ├── foods_snapshot.csv      ← 5,500 item offline snapshot
│   └── foods.db                ← SQLite database
├── .streamlit/config.toml
├── README.md
└── prompts.md
```

---

## Deployment (Streamlit Cloud)

1. Push this repository to GitHub
2. Go to https://share.streamlit.io → New App → select your repo
3. Set **Main file path** to `code/app.py`
4. Click Deploy

The app will auto-install `requirements.txt` and run `data_setup.py --offline` on first launch (via Streamlit's init mechanism).

---

## Submission

ZIP the project as: `LastName_FirstName_BAX423_Final.zip`

Contents:
- `code/`     — all source code
- `data/`     — foods_snapshot.csv + foods.db
- `brief.pdf` — 4-page technical brief
- `prompts.md`— key AI prompts used

Plus the live hosted app URL.

---

*Built with Streamlit · FAISS · Bloom Filter · RL Bandit · USDA FoodData Central*
*BAX-423 Big Data · Dr. Rahul Makhijani · UC Davis GSM · Spring 2026*
