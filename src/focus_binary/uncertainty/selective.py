from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

from focus_binary.eval.metrics import compute_metrics


def _as_probabilities(y_prob: np.ndarray) -> np.ndarray:
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)
    return y_prob


def selective_summary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    delta: float | None = None,
    tau: float | None = None,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = _as_probabilities(y_prob)
    if len(y_true) == 0:
        return {"coverage": 0.0, "auc": float("nan"), "f1": float("nan"), "risk": float("nan")}

    confidence = np.maximum(y_prob, 1.0 - y_prob)
    mask = np.ones_like(y_prob, dtype=bool)
    if delta is not None:
        mask &= np.abs(y_prob - 0.5) >= float(delta)
    if tau is not None:
        mask &= confidence >= float(tau)

    if mask.sum() == 0:
        return {"coverage": 0.0, "auc": float("nan"), "f1": float("nan"), "risk": float("nan")}

    metrics = compute_metrics(y_true[mask], y_prob[mask], threshold=0.5)
    accuracy = metrics.get("accuracy", float("nan"))
    risk = 1.0 - float(accuracy) if accuracy == accuracy else float("nan")
    return {
        "coverage": float(mask.mean()),
        "auc": float(metrics.get("auc", float("nan"))),
        "f1": float(metrics.get("f1", float("nan"))),
        "risk": risk,
    }


def risk_coverage_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    mode: str = "tau",
    thresholds: Iterable[float] | None = None,
) -> List[Dict[str, float]]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = _as_probabilities(y_prob)
    if thresholds is None:
        if mode == "delta":
            thresholds = np.linspace(0.0, 0.49, 20)
        else:
            thresholds = np.linspace(0.5, 0.99, 20)

    rows: List[Dict[str, float]] = []
    for t in thresholds:
        if mode == "delta":
            summary = selective_summary(y_true, y_prob, delta=float(t), tau=None)
        else:
            summary = selective_summary(y_true, y_prob, delta=None, tau=float(t))
        rows.append(
            {
                "threshold": float(t),
                "coverage": summary["coverage"],
                "risk": summary["risk"],
                "auc": summary["auc"],
                "f1": summary["f1"],
            }
        )
    return rows
