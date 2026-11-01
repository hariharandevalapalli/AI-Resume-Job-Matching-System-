"""Standard information-retrieval ranking metrics.

All functions take `ranked_grades`: the relevance grades of the candidates in the
order the system ranked them (best first).
"""

from __future__ import annotations

import math

from scipy.stats import spearmanr


def dcg_at_k(ranked_grades: list[float], k: int) -> float:
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ranked_grades[:k]))


def ndcg_at_k(ranked_grades: list[float], k: int) -> float:
    ideal = dcg_at_k(sorted(ranked_grades, reverse=True), k)
    return dcg_at_k(ranked_grades, k) / ideal if ideal > 0 else 0.0


def precision_at_k(ranked_grades: list[float], k: int, threshold: float) -> float:
    if k <= 0:
        return 0.0
    top = ranked_grades[:k]
    return sum(1 for g in top if g >= threshold) / k


def average_precision(ranked_grades: list[float], threshold: float) -> float:
    relevant_total = sum(1 for g in ranked_grades if g >= threshold)
    if relevant_total == 0:
        return 0.0
    hits = 0
    score = 0.0
    for i, grade in enumerate(ranked_grades, start=1):
        if grade >= threshold:
            hits += 1
            score += hits / i
    return score / relevant_total


def reciprocal_rank(ranked_grades: list[float], threshold: float) -> float:
    for i, grade in enumerate(ranked_grades, start=1):
        if grade >= threshold:
            return 1.0 / i
    return 0.0


def spearman(scores: list[float], grades: list[float]) -> float:
    """Rank correlation between model scores and human grades (0 if undefined)."""
    if len(scores) < 2 or len(set(scores)) < 2 or len(set(grades)) < 2:
        return 0.0
    value = spearmanr(scores, grades).statistic
    return 0.0 if value is None or math.isnan(value) else float(value)
