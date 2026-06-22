from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from focus_binary.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    bin_acc = np.zeros(n_bins, dtype=float)
    bin_conf = np.zeros(n_bins, dtype=float)
    bin_count = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = bin_ids == i
        if not np.any(mask):
            continue
        bin_count[i] = int(mask.sum())
        bin_acc[i] = y_true[mask].mean()
        bin_conf[i] = y_prob[mask].mean()
    return bin_acc, bin_conf, bin_count


def reliability_bins(
    y_true,
    y_prob,
    n_bins: int = 15,
) -> Dict[str, list]:
    """Return per-bin accuracy/confidence/counts for reliability plots."""

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    bin_acc = np.full(n_bins, np.nan, dtype=float)
    bin_conf = np.full(n_bins, np.nan, dtype=float)
    bin_count = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = bin_ids == i
        if not np.any(mask):
            continue
        bin_count[i] = int(mask.sum())
        bin_acc[i] = float(y_true[mask].mean())
        bin_conf[i] = float(y_prob[mask].mean())

    return {
        "bin_edges": bins.tolist(),
        "bin_acc": bin_acc.tolist(),
        "bin_conf": bin_conf.tolist(),
        "bin_count": bin_count.tolist(),
    }


def compute_ece(y_true, y_prob, n_bins: int = 15) -> float:
    bins = reliability_bins(y_true, y_prob, n_bins=n_bins)
    acc = np.asarray(bins["bin_acc"], dtype=float)
    conf = np.asarray(bins["bin_conf"], dtype=float)
    count = np.asarray(bins["bin_count"], dtype=float)
    total = count.sum()
    if total == 0:
        return 0.0
    weights = count / total
    valid = ~np.isnan(acc) & ~np.isnan(conf)
    if not np.any(valid):
        return 0.0
    return float(np.sum(weights[valid] * np.abs(acc[valid] - conf[valid])))


def compute_brier(y_true, y_prob) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    return compute_ece(y_true, y_prob, n_bins=n_bins)


def temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray:
    return logits / max(temperature, 1e-6)


def fit_temperature(y_true, y_logit, max_iter: int = 50, lr: float = 0.01) -> float:
    """Fit temperature by minimizing NLL on logits."""
    y_true = np.asarray(y_true).astype(float).reshape(-1, 1)
    y_logit = np.asarray(y_logit).astype(float).reshape(-1, 1)

    if tf is None:
        logger.warning("TensorFlow not installed; returning temperature=1.0")
        return 1.0

    temperature = tf.Variable(1.0, dtype=tf.float32)
    y_true_tf = tf.constant(y_true, dtype=tf.float32)
    logits_tf = tf.constant(y_logit, dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    for _ in range(max_iter):
        with tf.GradientTape() as tape:
            scaled = logits_tf / temperature
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true_tf, logits=scaled))
        grads = tape.gradient(loss, [temperature])
        optimizer.apply_gradients(zip(grads, [temperature]))
        temperature.assign(tf.maximum(temperature, 1e-3))

    return float(temperature.numpy())


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _to_logits(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    probs = np.clip(probs, eps, 1.0 - eps)
    return np.log(probs / (1.0 - probs))


def temperature_scaling(
    y_true_val,
    logits_or_probs_val,
) -> Tuple[float, np.ndarray]:
    """Fit temperature on validation data and return calibrated probabilities."""

    y_true = np.asarray(y_true_val).astype(float).reshape(-1)
    values = np.asarray(logits_or_probs_val).astype(float).reshape(-1)

    if values.size == 0:
        return 1.0, values

    if values.min() >= 0.0 and values.max() <= 1.0:
        logits = _to_logits(values)
    else:
        logits = values

    if tf is not None:
        temperature = fit_temperature(y_true, logits)
    else:
        logger.warning("TensorFlow not installed; using grid-search temperature scaling.")
        temps = np.linspace(0.5, 5.0, 50)
        best_temp = 1.0
        best_loss = float("inf")
        for temp in temps:
            scaled = logits / max(temp, 1e-6)
            probs = _sigmoid(scaled)
            probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
            loss = -np.mean(y_true * np.log(probs) + (1.0 - y_true) * np.log(1.0 - probs))
            if loss < best_loss:
                best_loss = loss
                best_temp = temp
        temperature = float(best_temp)

    calibrated_probs = _sigmoid(logits / max(temperature, 1e-6))
    return float(temperature), calibrated_probs


def choose_threshold(y_true_val, y_prob_val, metric: str = "f1") -> float:
    """Select a threshold on validation data for F1 or Youden's J."""

    y_true = np.asarray(y_true_val).astype(int)
    y_prob = np.asarray(y_prob_val).astype(float)
    if y_true.size == 0:
        return 0.5

    metric = metric.lower()
    thresholds = np.linspace(0.0, 1.0, 101)
    best_score = float("-inf")
    best_thresh = 0.5

    for thresh in thresholds:
        y_hat = (y_prob >= thresh).astype(int)
        tp = int(((y_hat == 1) & (y_true == 1)).sum())
        tn = int(((y_hat == 0) & (y_true == 0)).sum())
        fp = int(((y_hat == 1) & (y_true == 0)).sum())
        fn = int(((y_hat == 0) & (y_true == 1)).sum())

        if metric == "youdenj":
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            score = tpr - fpr
        else:
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            score = 2 * precision * recall / max(precision + recall, 1e-8)

        if score > best_score:
            best_score = score
            best_thresh = thresh

    return float(best_thresh)
