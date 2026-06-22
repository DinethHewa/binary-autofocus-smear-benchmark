from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np


def bootstrap_metric(
    y_true,
    y_prob,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores.append(metric_fn(y_true[idx], y_prob[idx]))
    scores = np.asarray(scores)
    alpha = (1.0 - ci) / 2.0
    low = np.quantile(scores, alpha)
    high = np.quantile(scores, 1.0 - alpha)
    return {"mean": float(scores.mean()), "ci_low": float(low), "ci_high": float(high)}


def bootstrap_difference(
    y_true,
    y_prob_a,
    y_prob_b,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob_a = np.asarray(y_prob_a)
    y_prob_b = np.asarray(y_prob_b)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        score_a = metric_fn(y_true[idx], y_prob_a[idx])
        score_b = metric_fn(y_true[idx], y_prob_b[idx])
        diffs.append(score_a - score_b)
    diffs = np.asarray(diffs)
    alpha = (1.0 - ci) / 2.0
    low = np.quantile(diffs, alpha)
    high = np.quantile(diffs, 1.0 - alpha)
    return {"mean": float(diffs.mean()), "ci_low": float(low), "ci_high": float(high)}


def paired_bootstrap_test(
    y_true,
    y_prob_a,
    y_prob_b,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n: int = 10000,
    seed: int = 42,
    ci: float = 0.95,
) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob_a = np.asarray(y_prob_a)
    y_prob_b = np.asarray(y_prob_b)
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)

    deltas = []
    for _ in range(n):
        idx = rng.integers(0, n_samples, size=n_samples)
        score_a = metric_fn(y_true[idx], y_prob_a[idx])
        score_b = metric_fn(y_true[idx], y_prob_b[idx])
        deltas.append(score_a - score_b)
    deltas = np.asarray(deltas)

    alpha = (1.0 - ci) / 2.0
    ci_low = float(np.quantile(deltas, alpha))
    ci_high = float(np.quantile(deltas, 1.0 - alpha))
    mean_delta = float(np.mean(deltas))
    p_lower = float(np.mean(deltas <= 0.0))
    p_upper = float(np.mean(deltas >= 0.0))
    p_value = float(min(1.0, 2.0 * min(p_lower, p_upper)))

    return {
        "mean_delta": mean_delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
    }
