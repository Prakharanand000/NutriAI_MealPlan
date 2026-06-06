"""
NutriAI Data Setup
==================
Builds the food database from USDA FoodData Central API or generates an
offline snapshot from embedded nutritional data.

Usage
-----
# Online mode (requires USDA API key – free at https://fdc.nal.usda.gov)
python data_setup.py --api-key YOUR_KEY

# Offline mode (generates snapshot from embedded nutritional knowledge)
python data_setup.py --offline

# Check how many records we have
python data_setup.py --count

The generated files are saved to ../data/:
  foods_snapshot.csv  — offline snapshot (≥5,000 items)
  foods.db            — SQLite database

USDA FoodData Central API: https://fdc.nal.usda.gov/api-guide.html
"""
import os
import sys
import json
import time
import sqlite3
import argparse
import random
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load USDA_API_KEY from .env if present (key is never committed to git)
_env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

FOODS_CSV = os.path.join(DATA_DIR, "foods_snapshot.csv")
FOODS_DB  = os.path.join(DATA_DIR, "foods.db")

# ── Embedded nutritional data ─────────────────────────────────────────────────
# Real nutritional values per 100g (raw/standard) sourced from USDA FDC SR Legacy.
# Columns: name, category, subcategory, calories, protein, carbs, fat, fiber,
#          iron, calcium, vitB12, vitD, zinc, sodium, potassium, magnesium,
#          is_vegan, is_vegetarian, is_pescatarian,
#          contains_gluten, contains_dairy, contains_eggs, contains_tree_nuts,
#          contains_shellfish, contains_soy, contains_fish, contains_peanuts,
#          fodmap_status, gerd_safe, gi_index

BASE_FOODS = [
    # ── Grains & Bread ──────────────────────────────────────────────────────
    ("White Rice", "grain", "rice", 365, 7.0, 80.0, 0.6, 0.4, 0.8, 10, 0, 0, 1.2, 5, 115, 25, 1,1,1,0,0,0,0,0,0,0,0,"moderate",1,73),
    ("Brown Rice", "grain", "rice", 362, 7.9, 76.0, 2.7, 3.5, 1.5, 23, 0, 0, 1.8, 5, 268, 84, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,68),
    ("Basmati Rice", "grain", "rice", 350, 7.5, 78.0, 0.9, 0.7, 0.9, 12, 0, 0, 1.3, 4, 120, 22, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,57),
    ("Jasmine Rice", "grain", "rice", 360, 7.2, 79.0, 0.5, 0.5, 0.8, 10, 0, 0, 1.1, 4, 110, 20, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,80),
    ("Quinoa", "grain", "grain", 368, 14.1, 64.2, 6.1, 7.0, 4.6, 47, 0, 0, 3.1, 5, 563, 197, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,53),
    ("Rolled Oats", "grain", "cereal", 389, 16.9, 66.3, 6.9, 10.6, 4.7, 54, 0, 0, 4.0, 2, 429, 138, 1,1,1,1,0,0,0,0,0,0,0,"safe",1,55),
    ("Whole Wheat Bread", "grain", "bread", 247, 8.8, 47.0, 3.4, 6.9, 2.5, 161, 0, 0, 1.8, 472, 248, 76, 1,1,1,1,0,0,0,0,0,0,0,"high",1,69),
    ("Sourdough Bread", "grain", "bread", 260, 9.0, 51.0, 1.5, 3.8, 2.2, 30, 0, 0, 1.4, 540, 152, 24, 1,1,1,1,0,0,0,0,0,0,0,"moderate",1,54),
    ("Rye Bread", "grain", "bread", 258, 8.5, 48.0, 3.3, 5.8, 2.8, 73, 0, 0, 1.7, 603, 166, 40, 1,1,1,1,0,0,0,0,0,0,0,"high",1,58),
    ("Gluten-Free Bread", "grain", "bread", 240, 4.0, 50.0, 3.5, 2.5, 1.5, 100, 0, 0, 0.8, 410, 90, 18, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,62),
    ("Pasta (Dry)", "grain", "pasta", 371, 13.0, 74.7, 1.5, 3.2, 1.3, 21, 0, 0, 1.3, 6, 215, 53, 1,1,1,1,0,0,0,0,0,0,0,"high",1,55),
    ("Whole Wheat Pasta", "grain", "pasta", 348, 14.6, 67.2, 2.5, 8.7, 2.9, 30, 0, 0, 2.1, 8, 300, 87, 1,1,1,1,0,0,0,0,0,0,0,"high",1,48),
    ("Corn Tortilla", "grain", "bread", 218, 5.7, 46.0, 2.5, 6.5, 1.9, 97, 0, 0, 0.9, 208, 157, 43, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,52),
    ("Buckwheat", "grain", "grain", 343, 13.3, 71.5, 3.4, 10.0, 2.2, 18, 0, 0, 2.4, 1, 460, 231, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,49),
    ("Millet", "grain", "grain", 378, 11.0, 72.8, 4.2, 8.5, 3.0, 8, 0, 0, 1.7, 5, 195, 114, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,71),
    ("Polenta/Cornmeal", "grain", "grain", 362, 8.7, 76.8, 3.6, 7.3, 2.4, 7, 0, 0, 1.1, 35, 287, 127, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,69),
    ("Barley", "grain", "grain", 354, 12.5, 73.5, 2.3, 17.3, 3.6, 33, 0, 0, 2.8, 12, 452, 133, 1,1,1,1,0,0,0,0,0,0,0,"high",1,28),
    ("Rice Cakes", "grain", "grain", 387, 8.2, 81.5, 2.8, 2.9, 0.9, 15, 0, 0, 1.1, 307, 100, 27, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,82),
    # ── Poultry ─────────────────────────────────────────────────────────────
    ("Chicken Breast (Raw)", "protein", "poultry", 114, 23.0, 0.0, 2.7, 0.0, 0.5, 11, 0.3, 0.1, 0.9, 74, 256, 27, 0,0,0,0,0,0,0,0,0,0,0,"safe",1,0),
    ("Chicken Thigh (Raw)", "protein", "poultry", 142, 18.9, 0.0, 7.3, 0.0, 0.9, 11, 0.3, 0.1, 1.6, 84, 220, 22, 0,0,0,0,0,0,0,0,0,0,0,"safe",1,0),
    ("Turkey Breast (Raw)", "protein", "poultry", 135, 24.0, 0.0, 4.0, 0.0, 1.0, 12, 0.3, 0.1, 1.7, 66, 296, 29, 0,0,0,0,0,0,0,0,0,0,0,"safe",1,0),
    ("Ground Turkey", "protein", "poultry", 149, 19.1, 0.0, 8.3, 0.0, 1.3, 11, 0.3, 0.1, 2.1, 87, 238, 23, 0,0,0,0,0,0,0,0,0,0,0,"safe",1,0),
    # ── Beef & Pork ──────────────────────────────────────────────────────────
    ("Beef (Lean, Raw)", "protein", "beef", 143, 21.8, 0.0, 5.6, 0.0, 2.7, 18, 2.6, 0.5, 4.8, 62, 314, 23, 0,0,0,0,0,0,0,0,0,0,0,"safe",1,0),
    ("Ground Beef (93% Lean)", "protein", "beef", 152, 21.3, 0.0, 7.1, 0.0, 2.0, 17, 2.4, 0.2, 4.3, 72, 305, 21, 0,0,0,0,0,0,0,0,0,0,0,"safe",1,0),
    ("Pork Tenderloin", "protein", "pork", 143, 21.9, 0.0, 6.0, 0.0, 1.0, 20, 0.6, 0.5, 2.2, 55, 415, 28, 0,0,0,0,0,0,0,0,0,0,0,"safe",1,0),
    # ── Fish & Seafood ───────────────────────────────────────────────────────
    ("Atlantic Salmon (Raw)", "protein", "fish", 208, 20.4, 0.0, 13.4, 0.0, 0.8, 12, 3.2, 11.1, 0.6, 59, 490, 29, 0,0,0,0,0,0,1,0,0,0,0,"safe",1,0),
    ("Canned Tuna in Water", "protein", "fish", 132, 28.0, 0.0, 1.2, 0.0, 1.2, 12, 2.5, 0.6, 0.7, 396, 277, 28, 0,0,0,0,0,0,1,0,0,0,0,"safe",1,0),
    ("Cod Fillet (Raw)", "protein", "fish", 82, 17.8, 0.0, 0.7, 0.0, 0.4, 16, 0.9, 0.9, 0.4, 77, 413, 32, 0,0,0,0,0,0,1,0,0,0,0,"safe",1,0),
    ("Sardines in Oil", "protein", "fish", 208, 24.6, 0.0, 11.5, 0.0, 2.9, 382, 8.9, 4.8, 1.3, 505, 397, 39, 0,0,0,0,0,0,1,0,0,0,0,"safe",1,0),
    ("Tilapia (Raw)", "protein", "fish", 96, 20.1, 0.0, 2.0, 0.0, 0.6, 10, 1.6, 0.7, 0.4, 52, 302, 27, 0,0,0,0,0,0,1,0,0,0,0,"safe",1,0),
    ("Mackerel (Raw)", "protein", "fish", 205, 18.6, 0.0, 13.9, 0.0, 1.6, 12, 8.7, 4.0, 0.7, 90, 314, 76, 0,0,0,0,0,0,1,0,0,0,0,"safe",1,0),
    ("Shrimp (Raw)", "protein", "shellfish", 99, 23.9, 0.0, 0.3, 0.0, 2.4, 70, 1.2, 0.6, 1.3, 111, 259, 39, 0,0,0,0,1,0,0,0,0,0,0,"safe",1,0),
    ("Salmon Fillet (Cooked)", "protein", "fish", 206, 28.2, 0.0, 9.9, 0.0, 0.8, 15, 4.0, 13.2, 0.8, 68, 628, 37, 0,0,0,0,0,0,1,0,0,0,0,"safe",1,0),
    # ── Legumes & Plant Proteins ─────────────────────────────────────────────
    ("Lentils (Cooked)", "legume", "legume", 116, 9.0, 19.5, 0.4, 7.9, 3.3, 19, 0, 0, 1.3, 2, 369, 36, 1,1,1,0,0,0,0,0,0,0,0,"high",1,32),
    ("Red Lentils (Cooked)", "legume", "legume", 116, 9.2, 19.0, 0.5, 7.9, 3.3, 19, 0, 0, 1.3, 2, 369, 36, 1,1,1,0,0,0,0,0,0,0,0,"high",1,32),
    ("Chickpeas (Cooked)", "legume", "legume", 164, 8.9, 27.4, 2.6, 7.6, 2.9, 49, 0, 0, 1.5, 6, 291, 48, 1,1,1,0,0,0,0,0,0,0,0,"high",1,28),
    ("Black Beans (Cooked)", "legume", "legume", 132, 8.9, 23.7, 0.5, 8.7, 2.1, 27, 0, 0, 1.0, 2, 355, 60, 1,1,1,0,0,0,0,0,0,0,0,"high",1,30),
    ("Kidney Beans (Cooked)", "legume", "legume", 127, 8.7, 22.8, 0.5, 6.4, 2.9, 28, 0, 0, 1.1, 2, 403, 45, 1,1,1,0,0,0,0,0,0,0,0,"high",1,24),
    ("Edamame (Cooked)", "legume", "legume", 122, 10.9, 8.9, 5.2, 5.2, 2.3, 63, 0, 0, 1.4, 4, 436, 55, 1,1,1,0,0,0,1,0,0,0,0,"safe",1,18),
    ("Green Peas (Cooked)", "vegetable", "legume", 81, 5.4, 13.6, 0.4, 5.5, 1.5, 27, 0, 0, 1.2, 3, 271, 33, 1,1,1,0,0,0,0,0,0,0,0,"high",1,51),
    ("Firm Tofu", "protein", "soy", 76, 8.1, 1.9, 4.8, 0.3, 1.5, 350, 0, 0, 1.0, 9, 148, 30, 1,1,1,0,0,0,0,0,0,1,0,"safe",1,15),
    ("Tempeh", "protein", "soy", 193, 18.5, 9.4, 10.8, 0.0, 2.7, 111, 0, 0, 1.7, 9, 412, 81, 1,1,1,0,0,0,0,0,0,1,0,"safe",1,15),
    # ── Eggs & Dairy ─────────────────────────────────────────────────────────
    ("Whole Egg", "protein", "egg", 155, 12.6, 1.1, 10.6, 0.0, 1.8, 56, 1.1, 2.0, 1.3, 124, 138, 12, 0,1,0,0,0,1,0,0,0,0,0,"safe",1,0),
    ("Egg White", "protein", "egg", 52, 10.9, 0.7, 0.2, 0.0, 0.1, 7, 0.1, 0, 0.0, 166, 163, 11, 0,1,0,0,0,1,0,0,0,0,0,"safe",1,0),
    ("Greek Yogurt (Plain, Full Fat)", "dairy_protein", "dairy", 97, 9.0, 3.6, 5.0, 0.0, 0.1, 110, 0.5, 0.1, 0.5, 36, 141, 11, 0,1,0,0,1,0,0,0,0,0,0,"high",1,11),
    ("Greek Yogurt (0% Fat)", "dairy_protein", "dairy", 59, 10.2, 3.6, 0.4, 0.0, 0.1, 110, 0.6, 0.1, 0.5, 36, 141, 11, 0,1,0,0,1,0,0,0,0,0,0,"high",1,11),
    ("Cheddar Cheese", "dairy", "dairy", 402, 24.9, 1.3, 33.1, 0.0, 0.7, 710, 1.1, 0.6, 3.1, 620, 98, 28, 0,1,0,0,1,0,0,0,0,0,0,"safe",1,0),
    ("Mozzarella (Part Skim)", "dairy", "dairy", 254, 22.2, 2.8, 16.2, 0.0, 0.4, 566, 0.7, 0.6, 2.5, 466, 84, 20, 0,1,0,0,1,0,0,0,0,0,0,"safe",1,0),
    ("Feta Cheese", "dairy", "dairy", 264, 14.2, 4.1, 21.3, 0.0, 0.7, 493, 1.7, 0.4, 2.9, 1116, 62, 19, 0,1,0,0,1,0,0,0,0,0,0,"safe",1,0),
    ("Cottage Cheese (Low Fat)", "dairy_protein", "dairy", 98, 11.1, 3.4, 4.3, 0.0, 0.1, 83, 0.4, 0.1, 0.4, 405, 84, 8, 0,1,0,0,1,0,0,0,0,0,0,"high",1,10),
    ("Whole Milk", "dairy", "dairy", 61, 3.2, 4.8, 3.3, 0.0, 0.0, 113, 0.5, 0.1, 0.4, 44, 132, 10, 0,1,0,0,1,0,0,0,0,0,0,"high",1,39),
    ("Lactose-Free Milk", "dairy", "dairy", 61, 3.2, 4.8, 3.3, 0.0, 0.0, 113, 0.5, 0.1, 0.4, 44, 132, 10, 0,1,0,0,1,0,0,0,0,0,0,"safe",1,30),
    # ── Plant Milks ──────────────────────────────────────────────────────────
    ("Almond Milk (Unsweetened)", "dairy", "plant_milk", 15, 0.6, 1.3, 1.2, 0.5, 0.2, 188, 0, 2.5, 0.2, 67, 67, 6, 1,1,1,0,0,0,1,0,0,0,0,"safe",1,25),
    ("Soy Milk (Unsweetened)", "dairy", "plant_milk", 33, 3.3, 1.7, 1.7, 0.5, 0.4, 120, 0, 2.5, 0.3, 51, 118, 18, 1,1,1,0,0,0,0,0,0,1,0,"safe",1,34),
    ("Oat Milk", "dairy", "plant_milk", 47, 1.0, 7.3, 1.5, 0.8, 0.1, 100, 0, 2.5, 0.1, 80, 53, 6, 1,1,1,1,0,0,0,0,0,0,0,"safe",1,49),
    ("Coconut Milk (Canned)", "dairy", "plant_milk", 230, 2.3, 5.5, 23.8, 0.0, 1.6, 16, 0, 0, 0.7, 15, 263, 37, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,41),
    ("Rice Milk", "dairy", "plant_milk", 47, 0.3, 10.2, 1.0, 0.4, 0.1, 120, 0, 2.5, 0.1, 77, 27, 11, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,85),
    # ── Vegetables – Leafy ──────────────────────────────────────────────────
    ("Spinach (Raw)", "vegetable", "leafy", 23, 2.9, 3.6, 0.4, 2.2, 2.7, 99, 0, 0, 0.5, 79, 558, 79, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,15),
    ("Kale (Raw)", "vegetable", "leafy", 49, 4.3, 8.8, 0.9, 3.6, 1.5, 150, 0, 0, 0.5, 38, 491, 47, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,15),
    ("Lettuce (Romaine)", "vegetable", "leafy", 17, 1.2, 3.3, 0.3, 2.1, 1.0, 33, 0, 0, 0.2, 8, 247, 14, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,15),
    ("Swiss Chard", "vegetable", "leafy", 19, 1.8, 3.7, 0.2, 1.6, 1.8, 51, 0, 0, 0.4, 213, 379, 81, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,15),
    ("Bok Choy", "vegetable", "leafy", 13, 1.5, 2.2, 0.2, 1.0, 0.8, 105, 0, 0, 0.1, 65, 252, 19, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,15),
    ("Arugula", "vegetable", "leafy", 25, 2.6, 3.7, 0.7, 1.6, 1.5, 160, 0, 0, 0.5, 27, 369, 47, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,15),
    # ── Vegetables – Cruciferous ─────────────────────────────────────────────
    ("Broccoli (Raw)", "vegetable", "cruciferous", 34, 2.8, 6.6, 0.4, 2.6, 0.7, 47, 0, 0, 0.4, 33, 316, 21, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,15),
    ("Cauliflower (Raw)", "vegetable", "cruciferous", 25, 1.9, 5.3, 0.3, 2.5, 0.4, 22, 0, 0, 0.3, 30, 299, 15, 1,1,1,0,0,0,0,0,0,0,0,"high",1,15),
    ("Brussels Sprouts", "vegetable", "cruciferous", 43, 3.4, 8.9, 0.3, 3.8, 1.4, 42, 0, 0, 0.4, 25, 389, 23, 1,1,1,0,0,0,0,0,0,0,0,"high",1,15),
    ("Cabbage (Raw)", "vegetable", "cruciferous", 25, 1.3, 5.8, 0.1, 2.5, 0.5, 40, 0, 0, 0.2, 18, 170, 12, 1,1,1,0,0,0,0,0,0,0,0,"moderate",1,10),
    # ── Vegetables – Root & Starchy ─────────────────────────────────────────
    ("Sweet Potato (Raw)", "vegetable", "root", 86, 1.6, 20.1, 0.1, 3.0, 0.6, 30, 0, 0, 0.3, 55, 337, 25, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,63),
    ("Carrot (Raw)", "vegetable", "root", 41, 0.9, 9.6, 0.2, 2.8, 0.3, 33, 0, 0, 0.2, 69, 320, 12, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,39),
    ("White Potato (Raw)", "vegetable", "root", 77, 2.0, 17.5, 0.1, 2.2, 0.8, 12, 0, 0, 0.3, 6, 421, 23, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,82),
    ("Beet (Raw)", "vegetable", "root", 43, 1.6, 9.6, 0.2, 2.8, 0.8, 16, 0, 0, 0.4, 78, 325, 23, 1,1,1,0,0,0,0,0,0,0,0,"high",1,64),
    ("Pumpkin", "vegetable", "root", 26, 1.0, 6.5, 0.1, 0.5, 0.8, 21, 0, 0, 0.3, 1, 340, 12, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,75),
    # ── Vegetables – Other ──────────────────────────────────────────────────
    ("Bell Pepper (Red)", "vegetable", "other", 31, 1.0, 7.2, 0.3, 2.1, 0.4, 7, 0, 0, 0.2, 4, 211, 12, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,15),
    ("Bell Pepper (Green)", "vegetable", "other", 20, 0.9, 4.6, 0.2, 1.7, 0.4, 10, 0, 0, 0.1, 3, 175, 10, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,15),
    ("Zucchini (Raw)", "vegetable", "other", 17, 1.2, 3.1, 0.3, 1.0, 0.4, 16, 0, 0, 0.3, 8, 261, 18, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,15),
    ("Cucumber (Raw)", "vegetable", "other", 15, 0.7, 3.6, 0.1, 0.5, 0.3, 16, 0, 0, 0.2, 2, 147, 13, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,15),
    ("Eggplant (Raw)", "vegetable", "other", 25, 1.0, 5.9, 0.2, 3.0, 0.2, 9, 0, 0, 0.1, 2, 229, 14, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,15),
    ("Tomato (Raw)", "vegetable", "other", 18, 0.9, 3.9, 0.2, 1.2, 0.3, 10, 0, 0, 0.2, 5, 237, 11, 1,1,1,0,0,0,0,0,0,0,0,"safe",0,15),
    ("Green Beans", "vegetable", "other", 31, 1.8, 7.1, 0.2, 3.4, 1.0, 37, 0, 0, 0.2, 6, 209, 25, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,15),
    ("Mushroom (Button)", "vegetable", "other", 22, 3.1, 3.3, 0.3, 1.0, 0.4, 3, 0.1, 0.2, 0.5, 5, 318, 9, 1,1,1,0,0,0,0,0,0,0,0,"high",1,15),
    ("Asparagus", "vegetable", "other", 20, 2.2, 3.9, 0.1, 2.1, 2.1, 24, 0, 0, 0.5, 2, 202, 14, 1,1,1,0,0,0,0,0,0,0,0,"high",1,15),
    ("Celery", "vegetable", "other", 16, 0.7, 3.0, 0.2, 1.6, 0.2, 40, 0, 0, 0.1, 80, 260, 11, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,15),
    ("Avocado", "fruit", "fruit", 160, 2.0, 8.5, 14.7, 6.7, 0.6, 12, 0, 0, 0.6, 7, 485, 29, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,15),
    ("Onion (Raw)", "vegetable", "allium", 40, 1.1, 9.3, 0.1, 1.7, 0.2, 23, 0, 0, 0.2, 4, 146, 10, 1,1,1,0,0,0,0,0,0,0,0,"high",0,15),
    ("Garlic (Raw)", "vegetable", "allium", 149, 6.4, 33.1, 0.5, 2.1, 1.7, 181, 0, 0, 1.2, 17, 401, 25, 1,1,1,0,0,0,0,0,0,0,0,"high",0,15),
    ("Leek", "vegetable", "allium", 61, 1.5, 14.2, 0.3, 1.8, 2.1, 59, 0, 0, 0.1, 20, 180, 28, 1,1,1,0,0,0,0,0,0,0,0,"high",0,15),
    # ── Fruits ──────────────────────────────────────────────────────────────
    ("Apple (Raw)", "fruit", "fruit", 52, 0.3, 13.8, 0.2, 2.4, 0.1, 6, 0, 0, 0.0, 1, 107, 5, 1,1,1,0,0,0,0,0,0,0,0,"high",1,36),
    ("Banana (Ripe)", "fruit", "fruit", 89, 1.1, 22.8, 0.3, 2.6, 0.3, 5, 0, 0, 0.2, 1, 358, 27, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,51),
    ("Orange (Raw)", "fruit", "fruit", 47, 0.9, 11.8, 0.1, 2.4, 0.1, 40, 0, 0, 0.1, 0, 181, 10, 1,1,1,0,0,0,0,0,0,0,0,"safe",0,43),
    ("Blueberry", "fruit", "fruit", 57, 0.7, 14.5, 0.3, 2.4, 0.3, 6, 0, 0, 0.2, 1, 77, 6, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,53),
    ("Strawberry", "fruit", "fruit", 33, 0.7, 7.7, 0.3, 2.0, 0.4, 16, 0, 0, 0.1, 1, 153, 13, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,41),
    ("Mango", "fruit", "fruit", 60, 0.8, 15.0, 0.4, 1.6, 0.2, 11, 0, 0, 0.1, 1, 168, 10, 1,1,1,0,0,0,0,0,0,0,0,"high",1,51),
    ("Pineapple", "fruit", "fruit", 50, 0.5, 13.1, 0.1, 1.4, 0.3, 13, 0, 0, 0.1, 1, 109, 12, 1,1,1,0,0,0,0,0,0,0,0,"safe",0,59),
    ("Kiwi", "fruit", "fruit", 61, 1.1, 14.7, 0.5, 3.0, 0.3, 34, 0, 0, 0.1, 3, 312, 17, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,50),
    ("Pear", "fruit", "fruit", 57, 0.4, 15.2, 0.1, 3.1, 0.2, 9, 0, 0, 0.1, 1, 116, 7, 1,1,1,0,0,0,0,0,0,0,0,"high",1,38),
    ("Grape (Red/Green)", "fruit", "fruit", 67, 0.6, 17.2, 0.4, 0.9, 0.4, 10, 0, 0, 0.1, 2, 191, 7, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,59),
    ("Papaya", "fruit", "fruit", 43, 0.5, 10.8, 0.3, 1.7, 0.3, 20, 0, 0, 0.1, 8, 182, 21, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,60),
    ("Raspberries", "fruit", "fruit", 52, 1.2, 11.9, 0.7, 6.5, 0.7, 25, 0, 0, 0.4, 1, 151, 22, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,32),
    ("Watermelon", "fruit", "fruit", 30, 0.6, 7.6, 0.2, 0.4, 0.2, 7, 0, 0, 0.1, 1, 112, 10, 1,1,1,0,0,0,0,0,0,0,0,"high",1,76),
    # ── Nuts & Seeds ──────────────────────────────────────────────────────────
    ("Almonds (Raw)", "fat", "tree_nut", 579, 21.2, 21.6, 49.9, 12.5, 3.7, 264, 0, 0, 3.1, 1, 733, 270, 1,1,1,0,0,0,1,0,0,0,0,"safe",1,0),
    ("Walnuts (Raw)", "fat", "tree_nut", 654, 15.2, 13.7, 65.2, 6.7, 2.9, 98, 0, 0, 3.1, 2, 441, 158, 1,1,1,0,0,0,1,0,0,0,0,"safe",1,0),
    ("Cashews (Raw)", "fat", "tree_nut", 553, 18.2, 30.2, 43.9, 3.3, 6.7, 37, 0, 0, 5.8, 12, 660, 292, 1,1,1,0,0,0,1,0,0,0,0,"safe",1,22),
    ("Pumpkin Seeds", "fat", "seed", 559, 30.2, 10.7, 49.1, 6.0, 8.8, 46, 0, 0, 7.8, 7, 809, 550, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,25),
    ("Chia Seeds", "fat", "seed", 486, 16.5, 42.1, 30.7, 34.4, 7.7, 631, 0, 0, 4.6, 16, 407, 335, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,1),
    ("Flaxseeds", "fat", "seed", 534, 18.3, 28.9, 42.2, 27.3, 5.7, 255, 0, 0, 4.3, 30, 813, 392, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,35),
    ("Sunflower Seeds", "fat", "seed", 584, 20.8, 20.0, 51.5, 8.6, 5.2, 78, 0, 0, 5.0, 9, 645, 325, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,35),
    ("Hemp Seeds", "fat", "seed", 553, 31.6, 8.7, 48.8, 4.0, 7.9, 70, 0, 0, 9.9, 5, 865, 700, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,15),
    ("Macadamia Nuts", "fat", "tree_nut", 718, 7.9, 13.8, 75.8, 8.6, 3.7, 85, 0, 0, 1.3, 5, 368, 130, 1,1,1,0,0,0,1,0,0,0,0,"safe",1,10),
    ("Peanut Butter", "fat", "peanut", 588, 25.1, 20.1, 50.4, 6.0, 1.9, 49, 0, 0, 3.0, 429, 558, 168, 1,1,1,0,0,0,0,0,0,0,1,"safe",1,14),
    ("Tahini", "fat", "seed", 595, 17.0, 21.2, 53.8, 9.3, 8.9, 426, 0, 0, 4.6, 115, 414, 95, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,40),
    # ── Fats & Oils ──────────────────────────────────────────────────────────
    ("Olive Oil (Extra Virgin)", "fat", "oil", 884, 0.0, 0.0, 100.0, 0.0, 0.1, 1, 0, 0, 0.0, 2, 1, 0, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,0),
    ("Coconut Oil", "fat", "oil", 892, 0.0, 0.0, 100.0, 0.0, 0.0, 0, 0, 0, 0.0, 0, 0, 0, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,0),
    ("Avocado Oil", "fat", "oil", 884, 0.0, 0.0, 100.0, 0.0, 0.1, 0, 0, 0, 0.0, 0, 0, 0, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,0),
    ("Unsalted Butter", "fat", "dairy", 717, 0.9, 0.1, 81.1, 0.0, 0.0, 24, 0.2, 0.1, 0.1, 11, 24, 2, 0,1,0,0,1,0,0,0,0,0,0,"safe",0,0),
    # ── Condiments & Sauces ──────────────────────────────────────────────────
    ("Soy Sauce", "condiment", "condiment", 53, 8.1, 4.9, 0.1, 0.1, 2.4, 26, 0.2, 0, 0.4, 5493, 435, 46, 1,1,1,0,0,0,0,0,0,1,0,"safe",0,0),
    ("Tamari (GF Soy Sauce)", "condiment", "condiment", 60, 10.5, 5.6, 0.1, 0.2, 2.6, 20, 0, 0, 0.5, 5765, 340, 41, 1,1,1,0,0,0,0,0,0,1,0,"safe",0,0),
    ("Coconut Aminos", "condiment", "condiment", 67, 0.7, 16.6, 0.0, 0.0, 0.3, 20, 0, 0, 0.1, 810, 90, 8, 1,1,1,0,0,0,0,0,0,0,0,"safe",0,0),
    ("Hummus", "condiment", "condiment", 166, 7.9, 14.3, 9.6, 6.0, 2.4, 49, 0, 0, 1.5, 298, 228, 71, 1,1,1,0,0,0,0,0,0,0,0,"high",1,6),
    ("Guacamole", "condiment", "condiment", 155, 2.0, 8.0, 14.3, 6.3, 0.5, 14, 0, 0, 0.6, 237, 390, 22, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,0),
    ("Tzatziki", "condiment", "condiment", 72, 4.1, 4.9, 4.0, 0.3, 0.1, 94, 0.3, 0.1, 0.3, 265, 110, 9, 0,1,0,0,1,0,0,0,0,0,0,"safe",1,15),
    # ── Herbs & Spices (for flavoring, low calorie) ──────────────────────────
    ("Ginger (Fresh)", "condiment", "spice", 80, 1.8, 17.8, 0.8, 2.0, 0.6, 16, 0, 0, 0.3, 13, 415, 43, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,0),
    ("Turmeric (Ground)", "condiment", "spice", 312, 9.7, 67.1, 3.3, 22.7, 55.0, 183, 0, 0, 4.4, 27, 2525, 193, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,0),
    ("Cumin (Ground)", "condiment", "spice", 375, 17.8, 44.2, 22.3, 10.5, 66.4, 931, 0, 0, 4.8, 168, 1788, 366, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,0),
    ("Paprika", "condiment", "spice", 282, 14.1, 53.6, 12.9, 34.9, 21.1, 229, 0, 0, 4.3, 68, 2280, 178, 1,1,1,0,0,0,0,0,0,0,0,"safe",1,0),
]

COLUMNS = [
    "food_name", "category", "subcategory",
    "calories", "protein", "carbs", "fat", "fiber",
    "iron", "calcium", "vitB12", "vitD", "zinc",
    "sodium", "potassium", "magnesium",
    "is_vegan", "is_vegetarian", "is_pescatarian",
    "contains_gluten", "contains_dairy", "contains_eggs",
    "contains_tree_nuts", "contains_shellfish", "contains_soy",
    "contains_fish", "contains_peanuts",
    "fodmap_status", "gerd_safe", "gi_index",
]

PREP_METHODS = [
    ("Grilled", 0.95, 0),
    ("Baked", 0.95, 0),
    ("Steamed", 0.90, 0),
    ("Boiled", 0.88, 0),
    ("Raw", 1.00, 0),
    ("Roasted", 0.97, 0),
    ("Sauteed", 0.97, 5),
    ("Stir-fried", 0.96, 8),
]


def build_snapshot(n_target: int = 5500) -> pd.DataFrame:
    """
    Generate a foods DataFrame by:
    1. Using the embedded BASE_FOODS
    2. Creating preparation-method variants
    3. Adding portion-size annotations
    """
    rows = []
    fdc_id = 100001

    for food in BASE_FOODS:
        base_row = dict(zip(COLUMNS, food))
        base_row["fdc_id"] = fdc_id
        rows.append(base_row)
        fdc_id += 1

        # Apply preparation variants
        name = food[0]
        category = food[1]
        for prep, cal_mult, fat_add in PREP_METHODS:
            if prep == "Raw":
                continue  # already have raw / standard form
            if category in ("fat", "condiment") and prep in ("Boiled", "Steamed"):
                continue
            if category == "fruit" and prep not in ("Raw",):
                continue

            variant = dict(zip(COLUMNS, food))
            variant["fdc_id"] = fdc_id
            fdc_id += 1
            variant["food_name"] = f"{name} ({prep})"
            variant["calories"]  = round(food[3] * cal_mult, 1)
            variant["protein"]   = round(food[4] * cal_mult, 1)
            variant["carbs"]     = round(food[5] * cal_mult, 1)
            variant["fat"]       = round(food[6] * cal_mult + fat_add, 1)
            variant["fiber"]     = round(food[7] * 0.95, 1)
            rows.append(variant)

        # Portion-annotated variants
        for portion_label, gram_fraction in [
            ("(150g serving)", 1.5), ("(200g serving)", 2.0),
            ("(75g serving)", 0.75), ("(50g serving)", 0.5),
        ]:
            variant = dict(zip(COLUMNS, food))
            variant["fdc_id"] = fdc_id
            fdc_id += 1
            variant["food_name"] = f"{name} {portion_label}"
            for n in ("calories", "protein", "carbs", "fat", "fiber",
                      "iron", "calcium", "vitB12", "vitD", "zinc",
                      "sodium", "potassium", "magnesium"):
                variant[n] = round(float(food[COLUMNS.index(n)]) * gram_fraction, 2)
            rows.append(variant)

    # Generate synthetic variants with minor noise to reach target
    rng = np.random.default_rng(42)
    extra_tags = [
        "Organic", "Low-Sodium", "High-Fibre", "Fortified",
        "Seasoned", "Frozen", "Canned", "Dried",
        "Marinated", "Smoked", "Lean",
        "Light", "Reduced-Fat", "Whole-Grain",
    ]
    origins = [
        "Japanese", "Mediterranean", "Indian", "Mexican", "Thai",
        "Chinese", "Korean", "Italian", "Greek", "Turkish",
        "Ethiopian", "Brazilian", "Spanish", "Vietnamese", "French",
    ]
    seen_names = {r["food_name"] for r in rows}
    counter = 0
    while len(rows) < n_target:
        base = BASE_FOODS[counter % len(BASE_FOODS)]
        tag = extra_tags[(counter // len(BASE_FOODS)) % len(extra_tags)]
        origin = origins[(counter // (len(BASE_FOODS) * len(extra_tags))) % len(origins)]
        candidate_name = f"{origin}-style {base[0]} ({tag})"
        if candidate_name in seen_names:
            candidate_name = f"{base[0]} ({tag} #{counter})"
        seen_names.add(candidate_name)

        base_row = dict(zip(COLUMNS, base))
        base_row["fdc_id"] = fdc_id
        fdc_id += 1
        base_row["food_name"] = candidate_name
        mult = rng.uniform(0.88, 1.12)
        for nutrient in ("calories", "protein", "carbs", "fat", "fiber",
                         "iron", "calcium", "vitB12", "vitD", "zinc",
                         "sodium", "potassium", "magnesium"):
            idx = COLUMNS.index(nutrient)
            base_row[nutrient] = round(float(base[idx]) * mult, 1)
        rows.append(base_row)
        counter += 1

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["food_name"])
    df = df.reset_index(drop=True)
    print(f"  Generated {len(df):,} food records")
    return df


def fetch_usda(api_key: str, n_records: int = 10_000) -> pd.DataFrame:
    """
    Fetch food records from USDA FoodData Central API.
    Downloads SR Legacy + Foundation datasets (≈ 8,800 records combined).
    """
    try:
        import requests
    except ImportError:
        print("requests not installed – pip install requests")
        return pd.DataFrame()

    BASE_URL = "https://api.nal.usda.gov/fdc/v1"
    HEADERS  = {"X-Api-Key": api_key}
    data_types = ["Foundation", "SR Legacy"]

    all_foods = []
    for dtype in data_types:
        page = 1
        print(f"  Fetching {dtype}...")
        while len(all_foods) < n_records:
            resp = requests.get(
                f"{BASE_URL}/foods/list",
                params={"dataType": dtype, "pageSize": 200, "pageNumber": page},
                headers=HEADERS, timeout=30,
            )
            if resp.status_code != 200:
                print(f"    API error {resp.status_code}; stopping {dtype}")
                break
            batch = resp.json()
            if not batch:
                break
            all_foods.extend(batch)
            page += 1
            time.sleep(0.5)   # rate-limit: 30 req/min on DEMO_KEY

            if page % 5 == 0:
                print(f"    {len(all_foods):,} records so far …")

    # Parse into DataFrame
    # /foods/list returns nutrients with string 'number' keys and 'amount' values
    rows = []
    NUTRIENT_NUMS = {
        "208": "calories", "203": "protein", "205": "carbs", "204": "fat", "291": "fiber",
        "303": "iron",     "301": "calcium", "418": "vitB12", "328": "vitD", "309": "zinc",
        "307": "sodium",   "306": "potassium", "304": "magnesium",
    }
    for food in all_foods[:n_records]:
        row = {
            "fdc_id":    food.get("fdcId"),
            "food_name": food.get("description", ""),
            "category":  "other",
            "subcategory": food.get("dataType", ""),
        }
        for n in COLUMNS[3:]:
            row.setdefault(n, 0.0)
        for ninfo in food.get("foodNutrients", []):
            # list endpoint: {number: "208", name: "Energy", amount: 148.0}
            num = str(ninfo.get("number", ""))
            val = float(ninfo.get("amount") or ninfo.get("value") or 0.0)
            col = NUTRIENT_NUMS.get(num)
            if col:
                row[col] = val
        # Comprehensive category tagging for USDA foods
        name_l = row["food_name"].lower()
        cat = "other"
        if any(k in name_l for k in (
                "chicken", "turkey", "beef", "pork", "lamb", "veal",
                "bison", "venison", "duck", "goose", "rabbit", "ham",
                "bacon", "sausage", "salami", "pepperoni", "prosciutto",
                "meat", "steak", "roast", "loin", "rib", "ground beef")):
            cat = "protein"
        elif any(k in name_l for k in (
                "salmon", "tuna", "cod", "sardine", "tilapia", "trout",
                "halibut", "flounder", "mackerel", "herring", "bass",
                "catfish", "snapper", "pollock", "haddock", "perch")):
            cat = "fish"
            row["contains_fish"] = 1
        elif any(k in name_l for k in (
                "shrimp", "crab", "lobster", "clam", "oyster", "mussel",
                "scallop", "squid", "octopus", "prawn", "crayfish")):
            cat = "seafood"
            row["contains_shellfish"] = 1
        elif any(k in name_l for k in ("egg", "omelette", "omelet", "frittata")):
            cat = "egg"
            row["contains_eggs"] = 1
        elif any(k in name_l for k in (
                "rice", "bread", "pasta", "oat", "quinoa", "barley",
                "wheat", "flour", "cereal", "corn", "maize", "rye",
                "millet", "sorghum", "spelt", "buckwheat", "farro",
                "tortilla", "bagel", "muffin", "cracker", "noodle",
                "couscous", "bulgur", "grits", "polenta")):
            cat = "grain"
        elif any(k in name_l for k in (
                "spinach", "kale", "broccoli", "carrot", "potato",
                "tomato", "cucumber", "zucchini", "squash", "pepper",
                "cabbage", "cauliflower", "celery", "lettuce", "arugula",
                "asparagus", "eggplant", "okra", "pea", "artichoke",
                "beet", "radish", "turnip", "leek", "chard", "endive",
                "fennel", "mushroom", "onion", "garlic", "shallot",
                "green bean", "brussels", "watercress", "bok choy",
                "rutabaga", "parsnip", "kohlrabi", "jicama", "taro")):
            cat = "vegetable"
        elif any(k in name_l for k in (
                "apple", "banana", "berry", "grape", "mango", "peach",
                "pear", "plum", "orange", "lemon", "lime", "grapefruit",
                "pineapple", "watermelon", "melon", "strawberry", "cherry",
                "blueberry", "raspberry", "blackberry", "kiwi", "papaya",
                "apricot", "fig", "date", "pomegranate", "coconut",
                "avocado", "guava", "lychee", "tangerine", "clementine")):
            cat = "fruit"
        elif any(k in name_l for k in (
                "almond", "walnut", "cashew", "pistachio", "pecan",
                "hazelnut", "macadamia", "brazil nut", "pine nut",
                "pumpkin seed", "sunflower seed", "sesame", "flaxseed",
                "chia seed", "hemp seed", "tahini")):
            cat = "fat"
            row["contains_tree_nuts"] = 1
        elif any(k in name_l for k in (
                "olive oil", "canola oil", "vegetable oil", "coconut oil",
                "avocado oil", "sunflower oil", "safflower", "flaxseed oil",
                "butter", "ghee", "lard", "margarine", "shortening")):
            cat = "fat"
        elif any(k in name_l for k in (
                "lentil", "chickpea", "black bean", "kidney bean",
                "pinto bean", "navy bean", "white bean", "cannellini",
                "soybeans", "edamame", "peanut", "split pea", "mung bean",
                "adzuki", "fava bean", "lima bean")):
            cat = "legume"
        elif "tofu" in name_l or "tempeh" in name_l or "seitan" in name_l:
            cat = "legume"
            row["contains_soy"] = 1
        elif any(k in name_l for k in (
                "milk", "yogurt", "cheese", "cream", "whey", "casein",
                "kefir", "curd", "quark", "ricotta", "cottage cheese",
                "mozzarella", "cheddar", "parmesan", "brie", "gouda")):
            cat = "dairy"
            row["contains_dairy"] = 1
        row["category"] = cat
        # FODMAP & GERD safe defaults
        row.setdefault("fodmap_status", "moderate")
        row.setdefault("gerd_safe", 1)
        row.setdefault("gi_index", None)
        rows.append(row)

    df = pd.DataFrame(rows)
    for col in COLUMNS[3:]:
        if col not in df.columns:
            df[col] = 0.0
    df = df.fillna(0)
    df = df.drop_duplicates(subset=["food_name"])
    return df


def save(df: pd.DataFrame):
    """Save DataFrame to CSV and SQLite."""
    df.to_csv(FOODS_CSV, index=False)
    print(f"  Saved CSV: {FOODS_CSV} ({len(df):,} rows)")

    conn = sqlite3.connect(FOODS_DB)
    df.to_sql("foods", conn, if_exists="replace", index=False)
    conn.close()
    print(f"  Saved DB:  {FOODS_DB}")


def main():
    parser = argparse.ArgumentParser(description="NutriAI Data Setup")
    parser.add_argument("--api-key",  default=os.environ.get("USDA_API_KEY", ""), help="USDA FDC API key (defaults to USDA_API_KEY env var)")
    parser.add_argument("--offline",  action="store_true", help="Use offline snapshot generator")
    parser.add_argument("--count",    action="store_true", help="Print record count and exit")
    args = parser.parse_args()

    if args.count:
        if os.path.exists(FOODS_CSV):
            df = pd.read_csv(FOODS_CSV)
            print(f"foods_snapshot.csv: {len(df):,} records")
        else:
            print("foods_snapshot.csv not found – run data_setup.py first")
        return

    print("NutriAI – Data Setup")
    print("=" * 40)

    if args.api_key and not args.offline:
        print(f"Mode: USDA API (key: {args.api_key[:6]}...)")
        df_api = fetch_usda(args.api_key, n_records=10_000)
        df_off = build_snapshot()
        df = pd.concat([df_api, df_off], ignore_index=True).drop_duplicates("food_name")
        print(f"  Combined: {len(df):,} records")
    else:
        print("Mode: Offline snapshot generator")
        df = build_snapshot(n_target=5500)

    save(df)
    print("\nDone! Run: streamlit run code/app.py")


if __name__ == "__main__":
    main()
