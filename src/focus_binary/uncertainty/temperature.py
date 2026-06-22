from __future__ import annotations

from typing import Tuple

import numpy as np


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    eps = 1e-6
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _nll(y_true: np.ndarray, probs: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    probs = np.asarray(probs, dtype=float)
    eps = 1e-6
    probs = np.clip(probs, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(probs) + (1.0 - y_true) * np.log(1.0 - probs)))


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    logits = _logit(probs)
    scaled = logits / max(float(temperature), 1e-6)
    return _sigmoid(scaled)


def fit_temperature(
    y_true: np.ndarray,
    probs: np.ndarray,
    grid: Tuple[float, float, int] = (0.05, 5.0, 60),
) -> float:
    """Fit temperature on validation probabilities by minimizing NLL."""
    y_true = np.asarray(y_true, dtype=float)
    probs = np.asarray(probs, dtype=float)
    t_min, t_max, steps = grid
    temps = np.logspace(np.log10(t_min), np.log10(t_max), int(steps))

    best_t = 1.0
    best_nll = float("inf")
    for t in temps:
        scaled = apply_temperature(probs, float(t))
        nll = _nll(y_true, scaled)
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)

    return float(best_t)
