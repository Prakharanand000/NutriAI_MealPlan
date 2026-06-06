"""
BAX-423 Technique 1: Bloom Filter for allergen pre-screening.

A Bloom filter is a probabilistic data structure that provides O(1) membership
testing with no false negatives and a configurable false-positive rate.
For allergen checking at scale (400 k+ USDA branded foods) it allows the
pipeline to instantly discard safe foods before the expensive allergen-lookup.

Reference: Burton Bloom (1970), "Space/Time Trade-offs in Hash Coding".
"""
import hashlib
import math
import time
from typing import Iterable


class BloomFilter:
    """
    Space-efficient probabilistic set membership structure.

    Parameters
    ----------
    n_items : int
        Expected number of items to insert.
    false_positive_rate : float
        Target false-positive probability (e.g. 0.01 = 1 %).
    """

    def __init__(self, n_items: int = 10_000, false_positive_rate: float = 0.01):
        self.fpr = false_positive_rate
        self.n_items = n_items
        # Optimal bit-array size and hash count
        self.size = max(1, int(
            -n_items * math.log(false_positive_rate) / (math.log(2) ** 2)
        ))
        self.hash_count = max(1, int(self.size / n_items * math.log(2)))
        self._bits = bytearray(math.ceil(self.size / 8))
        self._count = 0

    # ── Internals ─────────────────────────────────────────────────────
    def _get_bit_positions(self, item: str):
        """Return `hash_count` bit positions for *item*."""
        base = item.lower().strip()
        positions = []
        for seed in range(self.hash_count):
            digest = hashlib.sha256(f"{base}_{seed}".encode()).hexdigest()
            pos = int(digest, 16) % self.size
            positions.append(pos)
        return positions

    def _set_bit(self, pos: int):
        byte_idx, bit_idx = divmod(pos, 8)
        self._bits[byte_idx] |= 1 << bit_idx

    def _check_bit(self, pos: int) -> bool:
        byte_idx, bit_idx = divmod(pos, 8)
        return bool(self._bits[byte_idx] & (1 << bit_idx))

    # ── Public API ────────────────────────────────────────────────────
    def add(self, item: str):
        for pos in self._get_bit_positions(item):
            self._set_bit(pos)
        self._count += 1

    def add_all(self, items: Iterable[str]):
        for item in items:
            self.add(item)

    def check(self, item: str) -> bool:
        """
        Return True if *item* is POSSIBLY in the set (may be a false positive).
        Return False only if *item* is DEFINITELY NOT in the set.
        """
        return all(self._check_bit(pos) for pos in self._get_bit_positions(item))

    @property
    def memory_bytes(self) -> int:
        return len(self._bits)

    @property
    def fill_ratio(self) -> float:
        return sum(bin(b).count("1") for b in self._bits) / (self.size or 1)


class AllergenBloomChecker:
    """
    Wraps one BloomFilter per allergen category.
    Used as the first-pass allergen screen in the NutriAI pipeline.
    """

    def __init__(self, allergen_keywords: dict, false_positive_rate: float = 0.005):
        self._filters: dict[str, BloomFilter] = {}
        self._keyword_sets: dict[str, set] = {}
        for allergen, keywords in allergen_keywords.items():
            bf = BloomFilter(n_items=max(len(keywords) * 3, 500),
                             false_positive_rate=false_positive_rate)
            kw_lower = {k.lower() for k in keywords}
            bf.add_all(kw_lower)
            # also index individual words in multi-word keywords
            for kw in kw_lower:
                for word in kw.split():
                    bf.add(word)
            self._filters[allergen] = bf
            self._keyword_sets[allergen] = kw_lower

    def screen(self, food_name: str, allergens: list[str]) -> tuple[bool, list[str]]:
        """
        First-pass screen via Bloom filter, then confirm with exact set lookup.

        Returns (is_allergen, matched_allergens)
        """
        name_lower = food_name.lower()
        tokens = set(name_lower.split())
        tokens.add(name_lower)

        matched = []
        for allergen in allergens:
            if allergen not in self._filters:
                continue
            bf = self._filters[allergen]
            # Bloom pre-screen (fast, no false negatives)
            bloom_hit = any(bf.check(tok) for tok in tokens)
            if bloom_hit:
                # Confirm with exact keyword set (eliminates false positives)
                kws = self._keyword_sets[allergen]
                confirmed = any(kw in name_lower for kw in kws)
                if confirmed:
                    matched.append(allergen)
        return bool(matched), matched

    def stats(self) -> dict:
        return {
            allergen: {
                "size_kb": round(bf.memory_bytes / 1024, 2),
                "hash_count": bf.hash_count,
                "fill_ratio": round(bf.fill_ratio, 3),
            }
            for allergen, bf in self._filters.items()
        }


def benchmark_bloom_vs_set(
    allergen_keywords: dict,
    n_queries: int = 50_000,
) -> dict:
    """
    Compare Bloom-filter-first strategy vs pure set-based lookup.
    Returns timing and memory stats for both approaches.
    """
    import random
    import string

    # Build structures
    all_keywords = set()
    for kws in allergen_keywords.values():
        all_keywords.update(k.lower() for k in kws)

    checker = AllergenBloomChecker(allergen_keywords)
    allergen_list = list(allergen_keywords.keys())

    # Generate realistic query mix: 10 % real allergen names, 90 % random food names
    real_queries = list(all_keywords)[:max(1, len(all_keywords))]
    random_words = [
        "".join(random.choices(string.ascii_lowercase, k=random.randint(4, 12)))
        for _ in range(n_queries)
    ]
    queries = []
    for i, q in enumerate(random_words):
        if i % 10 == 0 and real_queries:
            queries.append(random.choice(real_queries))
        else:
            queries.append(q)

    # ── Bloom-first approach ──────────────────────────────────────────
    t0 = time.perf_counter()
    for q in queries:
        checker.screen(q, allergen_list)
    bloom_time = time.perf_counter() - t0

    # ── Pure set-based approach ────────────────────────────────────────
    def set_check(name: str, allergens: list[str]) -> list[str]:
        name_l = name.lower()
        matched = []
        for a in allergens:
            kws = allergen_keywords.get(a, set())
            if any(kw in name_l for kw in kws):
                matched.append(a)
        return matched

    t0 = time.perf_counter()
    for q in queries:
        set_check(q, allergen_list)
    set_time = time.perf_counter() - t0

    bloom_mem  = sum(f.memory_bytes for f in checker._filters.values())
    set_mem    = sum(
        sum(len(k) for k in kws) for kws in allergen_keywords.values()
    )

    return {
        "n_queries": n_queries,
        "bloom_ms": round(bloom_time * 1000, 2),
        "set_ms":   round(set_time * 1000, 2),
        "speedup_x": round(set_time / bloom_time, 2) if bloom_time else 0,
        "bloom_memory_kb": round(bloom_mem / 1024, 2),
        "set_memory_kb":   round(set_mem / 1024, 2),
        "memory_saving_pct": round(
            (1 - bloom_mem / max(set_mem, 1)) * 100, 1
        ),
    }
