"""
NutriAI – centralized configuration.
All clinical rules, RDA tables, allergen maps, FODMAP lists, and
GI thresholds live here so the pipeline modules stay thin.
"""
import os

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FOODS_CSV = os.path.join(DATA_DIR, "foods_snapshot.csv")
FOODS_DB  = os.path.join(DATA_DIR, "foods.db")

# ── NIH RDA targets (per day) by (sex, age_min, age_max) ─────────────
# Source: NIH Dietary Reference Intakes https://www.ncbi.nlm.nih.gov/books/NBK56068
RDA_TABLE = {
    ("M", 19, 30): dict(calories=2500, protein=56,  carbs=325, fat=78,  fiber=38,
                        iron=8,  calcium=1000, vitB12=2.4, vitD=15, zinc=11,
                        sodium=2300, potassium=3400, magnesium=400),
    ("M", 31, 50): dict(calories=2400, protein=56,  carbs=310, fat=75,  fiber=38,
                        iron=8,  calcium=1000, vitB12=2.4, vitD=15, zinc=11,
                        sodium=2300, potassium=3400, magnesium=420),
    ("M", 51, 70): dict(calories=2200, protein=56,  carbs=285, fat=69,  fiber=30,
                        iron=8,  calcium=1200, vitB12=2.4, vitD=20, zinc=11,
                        sodium=2300, potassium=3400, magnesium=420),
    ("M", 71, 99): dict(calories=2000, protein=56,  carbs=260, fat=62,  fiber=30,
                        iron=8,  calcium=1200, vitB12=2.4, vitD=20, zinc=11,
                        sodium=2300, potassium=3400, magnesium=420),
    ("F", 19, 30): dict(calories=2000, protein=46,  carbs=260, fat=62,  fiber=25,
                        iron=18, calcium=1000, vitB12=2.4, vitD=15, zinc=8,
                        sodium=2300, potassium=2600, magnesium=310),
    ("F", 31, 50): dict(calories=1800, protein=46,  carbs=235, fat=58,  fiber=25,
                        iron=18, calcium=1000, vitB12=2.4, vitD=15, zinc=8,
                        sodium=2300, potassium=2600, magnesium=320),
    ("F", 51, 70): dict(calories=1600, protein=46,  carbs=210, fat=55,  fiber=21,
                        iron=8,  calcium=1200, vitB12=2.4, vitD=20, zinc=8,
                        sodium=2300, potassium=2600, magnesium=320),
    ("F", 71, 99): dict(calories=1600, protein=46,  carbs=210, fat=55,  fiber=21,
                        iron=8,  calcium=1200, vitB12=2.4, vitD=20, zinc=8,
                        sodium=2300, potassium=2600, magnesium=320),
}

NUTRIENT_UNITS = {
    "calories": "kcal", "protein": "g", "carbs": "g", "fat": "g", "fiber": "g",
    "iron": "mg", "calcium": "mg", "vitB12": "µg", "vitD": "µg", "zinc": "mg",
    "sodium": "mg", "potassium": "mg", "magnesium": "mg",
}

NUTRIENT_DISPLAY = {
    "calories": "Calories", "protein": "Protein", "carbs": "Carbohydrates",
    "fat": "Total Fat", "fiber": "Dietary Fiber",
    "iron": "Iron", "calcium": "Calcium", "vitB12": "Vitamin B12",
    "vitD": "Vitamin D", "zinc": "Zinc",
    "sodium": "Sodium", "potassium": "Potassium", "magnesium": "Magnesium",
}

RDA_THRESHOLD = 0.80   # flag days where any key nutrient < 80% RDA

# ── Clinical Conditions ───────────────────────────────────────────────
# High-FODMAP ingredients to exclude for IBS.
# Source: Monash University FODMAP list (https://www.monashfodmap.com)
HIGH_FODMAP_INGREDIENTS = {
    "garlic", "onion", "leek", "shallot", "scallion", "spring onion",
    "wheat", "rye", "barley", "wheat flour", "semolina", "couscous",
    "bulgur", "spelt", "kamut",
    "apple", "apricot", "cherry", "mango", "peach", "pear", "plum",
    "watermelon", "lychee", "blackberry", "dried fruit",
    "lentils", "chickpeas", "black beans", "kidney beans", "baked beans",
    "butter beans", "navy beans", "split peas",
    "milk", "soft cheese", "cottage cheese", "ricotta", "ice cream",
    "cream cheese", "sour cream", "custard",
    "cauliflower", "mushroom", "asparagus", "artichoke",
    "celery", "pea", "green pea", "savoy cabbage", "beetroot",
    "honey", "high-fructose corn syrup", "agave",
}

# Foods that are confirmed low-FODMAP (safe for IBS)
SAFE_FODMAP_FOODS = {
    "rice", "white rice", "brown rice", "basmati rice", "jasmine rice",
    "potato", "sweet potato", "yam", "taro",
    "quinoa", "millet", "buckwheat", "cornmeal", "polenta",
    "oats", "gluten-free oats", "gluten-free bread",
    "banana", "blueberry", "strawberry", "grape", "kiwi",
    "orange", "mandarin", "raspberry", "cantaloupe", "pineapple",
    "papaya", "passion fruit", "lime",
    "carrot", "bell pepper", "capsicum", "zucchini", "squash",
    "cucumber", "spinach", "kale", "lettuce", "bok choy", "cabbage",
    "tomato", "eggplant", "pumpkin", "ginger", "chives",
    "broccoli", "green beans", "bean sprouts", "bamboo shoots",
    "chicken", "beef", "lamb", "pork", "turkey", "duck",
    "salmon", "tuna", "cod", "shrimp", "prawns", "tilapia",
    "eggs", "egg white", "egg yolk",
    "firm tofu", "tempeh",
    "hard cheese", "cheddar", "parmesan", "feta",
    "lactose-free milk", "almond milk", "oat milk", "rice milk",
    "coconut milk", "hemp milk",
    "almonds", "macadamia", "pumpkin seeds", "sunflower seeds",
    "chia seeds", "flaxseeds", "sesame seeds",
    "olive oil", "coconut oil", "avocado oil", "butter",
    "maple syrup", "brown sugar", "white sugar",
}

# GERD trigger foods to exclude
# Source: American College of Gastroenterology guidelines
GERD_TRIGGERS = {
    "orange", "orange juice", "grapefruit", "grapefruit juice",
    "lemon juice", "lime juice", "citrus",
    "tomato", "tomato sauce", "ketchup", "marinara", "salsa",
    "coffee", "espresso", "caffeine", "energy drink",
    "chocolate", "cocoa", "dark chocolate", "milk chocolate",
    "fried", "deep-fried", "french fries", "potato chips", "fried chicken",
    "spicy", "hot sauce", "chili", "jalapeño", "sriracha", "curry powder",
    "pepper flakes", "cayenne",
    "alcohol", "beer", "wine", "spirits",
    "peppermint", "spearmint", "mint",
    "onion", "garlic", "leek",
    "butter", "cream sauce", "heavy cream", "full-fat cheese",
}

# Glycemic Index thresholds
GI_LOW    = 55   # safe for T2D
GI_MEDIUM = 70
GI_HIGH   = 71   # avoid for T2D

# GI values for common foods (source: University of Sydney GI database)
GI_DATABASE = {
    "white bread": 75, "whole wheat bread": 69, "sourdough bread": 54,
    "white rice": 73, "brown rice": 68, "basmati rice": 57, "jasmine rice": 80,
    "pasta": 55, "whole wheat pasta": 48, "rice noodles": 61,
    "oats": 55, "instant oatmeal": 79, "muesli": 57,
    "quinoa": 53, "barley": 28, "buckwheat": 49, "millet": 71,
    "cornflakes": 81, "bran flakes": 74,
    "potato": 82, "sweet potato": 63, "yam": 54,
    "lentils": 32, "chickpeas": 28, "black beans": 30, "kidney beans": 24,
    "soybeans": 16, "tofu": 15, "tempeh": 15,
    "apple": 36, "banana": 51, "orange": 43, "grapes": 59,
    "mango": 51, "pineapple": 59, "watermelon": 76, "cherry": 22,
    "strawberry": 41, "blueberry": 53, "peach": 42, "pear": 38,
    "milk": 39, "yogurt": 41, "ice cream": 62,
    "sugar": 65, "honey": 61, "maple syrup": 54,
    "rice cakes": 82, "corn tortilla": 52, "rye bread": 58,
}

# DASH diet limits for hypertension
DASH_SODIUM_MAX = 1500   # mg/day
DASH_POTASSIUM_MIN_PCT = 0.8   # 80% RDA minimum

# ── Allergen Definitions ──────────────────────────────────────────────
# Maps canonical allergen name → ingredient keywords
ALLERGEN_KEYWORDS = {
    "gluten": {
        "wheat", "barley", "rye", "spelt", "kamut", "triticale",
        "wheat flour", "white flour", "whole wheat", "wheat bran", "wheat germ",
        "semolina", "couscous", "bulgur", "farro", "durum",
        "bread", "pasta", "noodle", "cracker", "crouton", "breadcrumb",
        "cereal", "granola", "muesli", "soy sauce", "malt", "beer",
        "oats", "oatmeal",  # unless certified GF
    },
    "dairy": {
        "milk", "whole milk", "skim milk", "cream", "heavy cream", "half and half",
        "butter", "ghee", "cheese", "cheddar", "mozzarella", "parmesan",
        "feta", "brie", "gouda", "swiss cheese", "ricotta", "cottage cheese",
        "cream cheese", "sour cream", "yogurt", "greek yogurt", "kefir",
        "whey", "casein", "lactose", "ice cream", "gelato", "custard",
    },
    "eggs": {
        "egg", "eggs", "egg white", "egg yolk", "egg powder",
        "mayonnaise", "meringue", "hollandaise", "aioli",
    },
    "tree_nuts": {
        "almond", "almonds", "almond butter", "almond flour", "almond milk",
        "cashew", "cashews", "cashew butter",
        "walnut", "walnuts", "pecan", "pecans",
        "pistachio", "pistachios", "hazelnut", "hazelnuts",
        "brazil nut", "brazil nuts", "macadamia", "pine nut", "pine nuts",
        "chestnut", "chestnuts",
    },
    "shellfish": {
        "shrimp", "prawn", "crab", "lobster", "oyster",
        "clam", "scallop", "mussel", "crayfish", "crawfish",
        "barnacle", "abalone",
    },
    "soy": {
        "soy", "soybean", "soybeans", "soy milk", "soy sauce",
        "tofu", "tempeh", "miso", "edamame", "tamari",
        "soy protein", "textured vegetable protein", "tvp",
    },
    "peanuts": {
        "peanut", "peanuts", "peanut butter", "peanut oil",
        "groundnut", "groundnuts", "satay",
    },
    "fish": {
        "salmon", "tuna", "cod", "halibut", "tilapia", "trout",
        "bass", "anchovy", "anchovies", "herring", "sardine", "sardines",
        "mackerel", "flounder", "snapper", "catfish", "grouper",
        "mahi mahi", "swordfish", "pollock", "haddock",
    },
}

# Cross-contamination risk pairs: if user avoids X, warn about Y
CROSS_CONTAMINATION = {
    "gluten": ["oats", "corn", "buckwheat"],
    "tree_nuts": ["peanuts", "seeds"],
    "shellfish": ["fish"],
    "soy": ["miso", "hoisin sauce"],
}

# ── Diet Type Rules ───────────────────────────────────────────────────
DIET_EXCLUSIONS = {
    "vegan": {"meat", "fish", "seafood", "shellfish", "poultry", "pork",
              "beef", "lamb", "chicken", "turkey",
              "milk", "cheese", "butter", "yogurt", "cream", "egg",
              "honey", "gelatin", "whey", "casein"},
    "vegetarian": {"meat", "fish", "seafood", "shellfish", "poultry", "pork",
                   "beef", "lamb", "chicken", "turkey", "duck", "veal",
                   "anchovy", "anchovies"},
    "pescatarian": {"meat", "poultry", "pork", "beef", "lamb",
                    "chicken", "turkey", "duck", "veal"},
    "non_vegetarian": set(),  # no exclusions
}

# ── FAISS Embedding Config ────────────────────────────────────────────
# Nutrient vector components used for FAISS indexing
FAISS_NUTRIENTS = [
    "calories", "protein", "carbs", "fat", "fiber",
    "iron", "calcium", "vitB12", "vitD", "zinc",
    "sodium", "potassium", "magnesium",
]

# Normalization factors (approximate max per 100g)
FAISS_NORM = {
    "calories": 900, "protein": 90, "carbs": 100, "fat": 100, "fiber": 50,
    "iron": 30, "calcium": 2000, "vitB12": 100, "vitD": 50, "zinc": 30,
    "sodium": 3000, "potassium": 5000, "magnesium": 600,
}

# ── Bandit Config ─────────────────────────────────────────────────────
BANDIT_EPSILON = 0.15   # exploration rate for epsilon-greedy
BANDIT_DECAY   = 0.995  # epsilon decay per step

# ── Meal Structure ────────────────────────────────────────────────────
MEAL_TYPES = ["breakfast", "lunch", "dinner"]
MEAL_CALORIE_SPLIT = {"breakfast": 0.30, "lunch": 0.35, "dinner": 0.35}

# Standard serving sizes per food category (grams)
SERVING_SIZES = {
    "grain":       {"breakfast": 80, "lunch": 90, "dinner": 90},
    "protein":     {"breakfast": 80, "lunch": 120, "dinner": 150},
    "vegetable":   {"breakfast": 80, "lunch": 150, "dinner": 150},
    "fruit":       {"breakfast": 100, "lunch": 80, "dinner": 0},
    "dairy":       {"breakfast": 150, "lunch": 80, "dinner": 0},
    "fat":         {"breakfast": 10, "lunch": 15, "dinner": 15},
    "legume":      {"breakfast": 0, "lunch": 120, "dinner": 120},
}
