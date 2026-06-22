from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None

try:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
except Exception:  # pragma: no cover
    accuracy_score = None
    f1_score = None
    roc_auc_score = None


def _require_tf() -> None:
    if tf is None:
        raise ImportError("TensorFlow is required for feature importance.")


def _as_probabilities(preds: np.ndarray) -> np.ndarray:
    preds = np.asarray(preds)
    if preds.ndim > 1:
        if preds.shape[-1] == 1:
            preds = preds.reshape(-1)
        elif preds.shape[-1] >= 2:
            preds = preds[..., 1]
    return preds.reshape(-1).astype(float)


def _predict(model, inputs) -> np.ndarray:
    _require_tf()
    preds = model.predict(inputs, verbose=0)
    return _as_probabilities(preds)


def _split_inputs(inputs) -> Tuple[object, np.ndarray]:
    if isinstance(inputs, (tuple, list)) and len(inputs) == 2:
        images, features = inputs
        return (np.asarray(images), np.asarray(features)), np.asarray(features)
    return np.asarray(inputs), np.asarray(inputs)


def _score_metric(y_true: np.ndarray, y_prob: np.ndarray, metric: str) -> float:
    metric = metric.lower()
    if metric == "auc":
        if roc_auc_score is None:
            raise ImportError("scikit-learn is required for AUC metric.")
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))
    if metric == "f1":
        if f1_score is None:
            raise ImportError("scikit-learn is required for F1 metric.")
        y_pred = (y_prob >= 0.5).astype(int)
        return float(f1_score(y_true, y_pred, zero_division=0))
    if metric in {"acc", "accuracy"}:
        if accuracy_score is None:
            raise ImportError("scikit-learn is required for accuracy metric.")
        y_pred = (y_prob >= 0.5).astype(int)
        return float(accuracy_score(y_true, y_pred))
    raise ValueError(f"Unsupported metric: {metric}")


def permutation_importance(
    model,
    X_val,
    y_val: Iterable[int],
    metric: str = "auc",
    n_repeats: int = 10,
    seed: int = 0,
) -> Dict[str, object]:
    """Permutation importance for feature vectors (supports hybrid input tuples)."""
    y_true = np.asarray(y_val).astype(int)
    inputs, features = _split_inputs(X_val)
    baseline_prob = _predict(model, inputs)
    baseline = _score_metric(y_true, baseline_prob, metric)

    rng = np.random.default_rng(seed)
    importances: List[Dict[str, float]] = []
    if features.ndim != 2:
        raise ValueError("Features must be a 2D array for permutation importance.")

    for idx in range(features.shape[1]):
        scores = []
        for _ in range(n_repeats):
            permuted = np.array(features, copy=True)
            permuted[:, idx] = rng.permutation(permuted[:, idx])
            perm_inputs = (inputs[0], permuted) if isinstance(inputs, tuple) else permuted
            y_prob = _predict(model, perm_inputs)
            scores.append(_score_metric(y_true, y_prob, metric))
        deltas = [baseline - s for s in scores if not np.isnan(s)]
        mean = float(np.mean(deltas)) if deltas else float("nan")
        std = float(np.std(deltas)) if deltas else float("nan")
        importances.append({"feature_index": idx, "mean": mean, "std": std})

    importances.sort(key=lambda x: np.nan_to_num(x["mean"], nan=-np.inf), reverse=True)
    return {"metric": metric, "baseline": float(baseline), "importances": importances}


def gradient_sensitivity(model, X_batch) -> np.ndarray:
    """Mean absolute gradient of predictions with respect to focus features."""
    _require_tf()

    inputs, features = _split_inputs(X_batch)
    features_tf = tf.convert_to_tensor(features, dtype=tf.float32)

    if isinstance(inputs, tuple):
        images_tf = tf.convert_to_tensor(inputs[0], dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(features_tf)
            preds = model([images_tf, features_tf], training=False)
            preds = tf.reshape(preds, [-1])
            loss = tf.reduce_mean(preds)
    else:
        with tf.GradientTape() as tape:
            tape.watch(features_tf)
            preds = model(features_tf, training=False)
            preds = tf.reshape(preds, [-1])
            loss = tf.reduce_mean(preds)

    grads = tape.gradient(loss, features_tf)
    if grads is None:
        return np.zeros((features_tf.shape[-1],), dtype=np.float32)
    importance = tf.reduce_mean(tf.abs(grads), axis=0)
    return importance.numpy()
