from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


def mcnemar_test(y_true, y_pred_a, y_pred_b) -> Dict[str, float]:
    """McNemar's test for paired binary classifiers."""
    y_true = np.asarray(y_true).astype(int)
    y_pred_a = np.asarray(y_pred_a).astype(int)
    y_pred_b = np.asarray(y_pred_b).astype(int)

    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true

    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0}

    stat = (abs(b - c) - 1) ** 2 / n
    p_value = math.erfc(math.sqrt(stat / 2.0))
    return {"b": b, "c": c, "statistic": float(stat), "p_value": float(p_value)}


def _rank_row(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values)
    ranks = np.zeros_like(values, dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and np.isclose(values[order[j]], values[order[j + 1]]):
            j += 1
        rank = (i + 1 + j + 1) / 2.0
        ranks[order[i : j + 1]] = rank
        i = j + 1
    return ranks


def _rank_matrix(scores: np.ndarray) -> np.ndarray:
    return np.vstack([_rank_row(row) for row in scores])


def _gammaincc(a: float, x: float) -> float:
    if x < 0 or a <= 0:
        return float("nan")
    if x == 0:
        return 1.0
    eps = 1e-12
    max_iter = 200
    fp_min = 1e-300
    if x < a + 1.0:
        ap = a
        summ = 1.0 / a
        delta = summ
        for _ in range(max_iter):
            ap += 1.0
            delta *= x / ap
            summ += delta
            if abs(delta) < abs(summ) * eps:
                break
        log_term = -x + a * math.log(x) - math.lgamma(a)
        p = summ * math.exp(log_term)
        return max(0.0, 1.0 - p)

    b = x + 1.0 - a
    c = 1.0 / fp_min
    d = 1.0 / b
    h = d
    for i in range(1, max_iter + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fp_min:
            d = fp_min
        c = b + an / c
        if abs(c) < fp_min:
            c = fp_min
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    log_term = -x + a * math.log(x) - math.lgamma(a)
    return max(0.0, math.exp(log_term) * h)


def _chi2_sf(stat: float, df: int) -> float:
    return _gammaincc(df / 2.0, stat / 2.0)


def friedman_test(scores_matrix: np.ndarray) -> Dict[str, float | np.ndarray]:
    scores = np.asarray(scores_matrix, dtype=float)
    if scores.ndim != 2:
        raise ValueError("scores_matrix must be 2D (n_datasets, n_models)")
    n_datasets, n_models = scores.shape
    if n_datasets < 2 or n_models < 2:
        raise ValueError("scores_matrix must have at least 2 datasets and 2 models")

    ranks = _rank_matrix(scores)
    avg_ranks = ranks.mean(axis=0)
    stat = (12.0 * n_datasets) / (n_models * (n_models + 1)) * np.sum(avg_ranks ** 2) - 3 * n_datasets * (n_models + 1)
    stat = max(0.0, float(stat))
    p_value = _chi2_sf(stat, n_models - 1)
    return {"statistic": float(stat), "p_value": float(p_value), "avg_ranks": avg_ranks}


def nemenyi_posthoc(scores_matrix: np.ndarray) -> Dict[str, object]:
    scores = np.asarray(scores_matrix, dtype=float)
    if scores.ndim != 2:
        raise ValueError("scores_matrix must be 2D (n_datasets, n_models)")
    n_datasets, n_models = scores.shape
    if n_datasets < 2 or n_models < 2:
        raise ValueError("scores_matrix must have at least 2 datasets and 2 models")

    ranks = _rank_matrix(scores)
    avg_ranks = ranks.mean(axis=0)

    q_alpha_table = {
        2: 1.960,
        3: 2.343,
        4: 2.569,
        5: 2.728,
        6: 2.850,
        7: 2.949,
        8: 3.031,
        9: 3.102,
        10: 3.164,
    }
    q_alpha = q_alpha_table.get(n_models, q_alpha_table[10])
    cd = q_alpha * math.sqrt(n_models * (n_models + 1) / (6.0 * n_datasets))

    sig = np.zeros((n_models, n_models), dtype=bool)
    for i in range(n_models):
        for j in range(i + 1, n_models):
            sig_val = abs(avg_ranks[i] - avg_ranks[j]) > cd
            sig[i, j] = sig_val
            sig[j, i] = sig_val

    return {
        "avg_ranks": avg_ranks,
        "critical_difference": float(cd),
        "significance": sig,
        "q_alpha": float(q_alpha),
    }
