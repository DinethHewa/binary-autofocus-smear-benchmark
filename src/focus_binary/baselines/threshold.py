from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from focus_binary.classical_ml.models import compute_focus_vectors
from focus_binary.eval.metrics import compute_metrics


def select_threshold(y_true: np.ndarray, scores: np.ndarray, metric: str = "f1", num_thresholds: int = 200) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(scores) == 0:
        return 0.5

    quantiles = np.linspace(0.0, 1.0, num_thresholds)
    thresholds = np.unique(np.quantile(scores, quantiles))

    best_t = thresholds[0]
    best_score = float("-inf")
    for t in thresholds:
        metrics = compute_metrics(y_true, scores, threshold=float(t))
        value = metrics.get(metric)
        if value is None:
            value = metrics.get("f1")
        if value is not None and value > best_score:
            best_score = value
            best_t = float(t)
    return float(best_t)


def compute_split_scores(
    df,
    input_size: int,
    measures: List[str],
    batch_size: int = 64,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    vectors = compute_focus_vectors(
        df["image_path"].astype(str).tolist(),
        input_size=input_size,
        enabled_measures=measures,
        batch_size=batch_size,
    )
    scores = {name: vectors[:, idx] for idx, name in enumerate(measures)}
    return vectors, scores


def build_composite_scores(
    train_vectors: np.ndarray,
    split_vectors: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mean = train_vectors.mean(axis=0)
    std = train_vectors.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    z = (split_vectors - mean) / std
    composite = z.mean(axis=1)
    weights = 1.0 / std
    return composite, weights
