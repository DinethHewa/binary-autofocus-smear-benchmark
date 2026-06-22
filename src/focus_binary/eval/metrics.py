from __future__ import annotations

import numpy as np

try:  # Optional dependency
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    roc_auc_score = None


def _as_probabilities(y_pred) -> np.ndarray:
    preds = np.asarray(y_pred).astype(float)
    if preds.ndim == 0:
        preds = preds.reshape(1)
    if preds.ndim > 1:
        if preds.shape[-1] == 1:
            preds = preds.reshape(-1)
        elif preds.shape[-1] >= 2:
            preds = preds[..., 1]
    return preds.reshape(-1)


def _roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    if roc_auc_score is not None:
        return float(roc_auc_score(y_true, y_prob))

    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    pos = (y_sorted == 1).astype(float)
    neg = (y_sorted == 0).astype(float)
    tps = np.cumsum(pos)
    fps = np.cumsum(neg)
    total_pos = tps[-1] if tps.size else 0.0
    total_neg = fps[-1] if fps.size else 0.0
    if total_pos == 0 or total_neg == 0:
        return float("nan")
    tpr = tps / total_pos
    fpr = fps / total_neg
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    return float(np.trapz(tpr, fpr))


def compute_metrics(y_true, y_pred, threshold: float = 0.5) -> dict:
    """Compute binary classification metrics from true labels and predicted probabilities."""

    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = _as_probabilities(y_pred)
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_pred must have the same length")

    y_hat = (y_prob >= threshold).astype(int)

    tp = int(((y_hat == 1) & (y_true == 1)).sum())
    tn = int(((y_hat == 0) & (y_true == 0)).sum())
    fp = int(((y_hat == 1) & (y_true == 0)).sum())
    fn = int(((y_hat == 0) & (y_true == 1)).sum())

    total = max(len(y_true), 1)
    accuracy = (tp + tn) / total
    precision = tp / max((tp + fp), 1)
    recall = tp / max((tp + fn), 1)
    f1 = 2 * precision * recall / max((precision + recall), 1e-8)
    auc = _roc_auc(y_true, y_prob)
    confusion = [[tn, fp], [fn, tp]]

    return {
        "accuracy": accuracy,
        "binary_accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": confusion,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tn": tn,
    }


def compute_basic_metrics(y_true, y_pred) -> dict:
    """Backward-compatible alias with threshold=0.5."""
    return compute_metrics(y_true, y_pred, threshold=0.5)
