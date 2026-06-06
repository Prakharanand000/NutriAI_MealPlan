# Key AI Prompts

These are the main prompts I used while building NutriAI, with a quick note on what I did with each output and what I changed. I used Claude and ChatGPT as coding assistants, but I went through every file myself, tested it, and can walk through any of it in the demo.

## 1. Pipeline architecture
> "Design a Python pipeline for a clinical diet planning app that uses a Bloom filter for allergen screening and FAISS for meal candidate retrieval. It has to handle IBS, GERD, type 2 diabetes and hypertension, and finish in under 60 seconds."

I used this to settle on the six-stage layout. I didn't keep it as-is though: the bandit layer wasn't in the original sketch, and I later swapped FAISS from plain L2 to inner-product on normalised vectors so the similarity actually means cosine.

## 2. Bloom filter
> "Write a Bloom filter in plain Python, no libraries, with a tunable false-positive rate and a few hash functions. Add a quick benchmark against a normal set for keyword lookups."

This became `pipeline/bloom_filter.py`. The generated version had false positives, which is unacceptable for allergens, so I added a second exact-match pass after the Bloom screen to get the false-negative rate to zero.

## 3. Food vectors for FAISS
> "How do I turn food items into vectors for FAISS similarity search in a diet app, without using an LLM for embeddings?"

I went with the 13 nutrient values as the vector. I added the L2 normalisation step myself so inner product equals cosine, and wrote a numpy fallback for when faiss-cpu isn't installed (it isn't always available on the cloud build).

## 4. Adaptive bandit
> "Implement epsilon-greedy and Thompson Sampling bandits in Python to learn meal-category preferences from 1 to 5 star ratings, and plot a comparison curve."

This is `pipeline/bandit.py`. I added epsilon decay, mapped the 1-5 ratings down to a 0-1 reward, and treated a rating of 4 or 5 as a success for the Beta posteriors in Thompson Sampling.

## 5. Clinical rules
> "What are the actual dietary rules for IBS (low-FODMAP), GERD, type 2 diabetes (GI based) and hypertension (DASH)? Give me sources."

I used the answer as a starting point and then checked everything against Monash (FODMAP), the ACG GERD guidance, the Sydney GI database and NHLBI DASH before putting it in `config.py`. The source links are in the comments.

## 6. RDA table
> "Give me the NIH RDA values for calories, protein, carbs, fat, fiber, iron, calcium, B12, vitamin D, zinc, sodium, potassium and magnesium, split by age and sex."

I reshaped this into the `RDA_TABLE` dict keyed by sex and age range so lookups are O(1), and split it into four adult age brackets.

## 7. Streamlit UI
> "Lay out a Streamlit app with a sidebar for the profile and tabs for the 7-day plan, nutrition dashboard, exclusion log, adaptive learning, benchmarks and persona tests."

Used for the overall `app.py` layout. I added the session-state handling so the plan survives tab switches, and wired up the live benchmark button.

## 8. Diversity score
> "Write a diversity score for a 7-day meal plan based on how much the meals vary. Use Jaccard similarity between the food sets."

This is the `diversity_score()` in `pipeline/diversity_engine.py`. I extended it to also look at category spread, not just individual foods, since two different chicken dishes still felt repetitive.

## 9. PDF export
> "Generate a PDF from a 7-day meal plan dict with fpdf2: title, profile summary, a per-day table with nutrients, and the exclusion log."

Used for `plan_to_pdf()` in `utils/export.py`. I added a plain-text fallback for when fpdf2 isn't installed and hooked it into the Streamlit download button.
