from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover
    IsotonicRegression = None
    LogisticRegression = None


EPS = 1e-6


@dataclass
class GateCalibrator:
    method: str
    fitted_method: str
    status: str
    params: dict[str, Any] = field(default_factory=dict)
    model: Any = None


def _as_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _as_int_array(values) -> np.ndarray:
    return np.asarray(values, dtype=int).reshape(-1)


def _valid_binary_arrays(y_true, y_prob) -> tuple[np.ndarray, np.ndarray]:
    labels = _as_int_array(y_true)
    probs = clip_prob(y_prob)
    n = min(labels.size, probs.size)
    labels = labels[:n]
    probs = probs[:n]
    mask = np.isfinite(labels) & np.isfinite(probs)
    return labels[mask], probs[mask]


def clip_prob(y_prob, eps: float = EPS) -> np.ndarray:
    return np.clip(_as_float_array(y_prob), eps, 1.0 - eps)


def logit(y_prob, eps: float = EPS) -> np.ndarray:
    probs = clip_prob(y_prob, eps=eps)
    return np.log(probs / (1.0 - probs))


def sigmoid(y_logit) -> np.ndarray:
    logits = _as_float_array(y_logit)
    out = np.empty_like(logits, dtype=float)
    positive = logits >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_x = np.exp(logits[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return np.clip(out, EPS, 1.0 - EPS)


def _bin_table_from_ids(y_true: np.ndarray, y_prob: np.ndarray, bin_ids: np.ndarray, n_bins: int, edges: np.ndarray | None = None) -> pd.DataFrame:
    total = max(int(y_true.size), 1)
    rows: list[dict[str, float | int]] = []
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        count = int(mask.sum())
        if edges is not None:
            lower = float(edges[bin_id])
            upper = float(edges[bin_id + 1])
        elif count:
            lower = float(np.min(y_prob[mask]))
            upper = float(np.max(y_prob[mask]))
        else:
            lower = float("nan")
            upper = float("nan")
        if count:
            mean_prob = float(np.mean(y_prob[mask]))
            observed_rate = float(np.mean(y_true[mask]))
            abs_error = abs(observed_rate - mean_prob)
        else:
            mean_prob = float("nan")
            observed_rate = float("nan")
            abs_error = float("nan")
        rows.append(
            {
                "bin_id": bin_id,
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "weight": count / total,
                "mean_prob": mean_prob,
                "observed_rate": observed_rate,
                "abs_error": abs_error,
            }
        )
    return pd.DataFrame(rows)


def make_reliability_table(y_true, y_prob, n_bins: int = 15) -> pd.DataFrame:
    labels, probs = _valid_binary_arrays(y_true, y_prob)
    n_bins = max(int(n_bins), 1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    if labels.size == 0:
        return _bin_table_from_ids(labels, probs, np.asarray([], dtype=int), n_bins, edges)
    bin_ids = np.digitize(probs, edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    return _bin_table_from_ids(labels, probs, bin_ids, n_bins, edges)


def ece_fixed_width(y_true, y_prob, n_bins: int = 15) -> float:
    table = make_reliability_table(y_true, y_prob, n_bins=n_bins)
    valid = table["count"].to_numpy(dtype=int) > 0
    if not np.any(valid):
        return 0.0
    return float(np.nansum(table.loc[valid, "weight"] * table.loc[valid, "abs_error"]))


def adaptive_ece(y_true, y_prob, n_bins: int = 15) -> float:
    labels, probs = _valid_binary_arrays(y_true, y_prob)
    if labels.size == 0:
        return 0.0
    n_bins = max(1, min(int(n_bins), labels.size))
    order = np.argsort(probs)
    groups = [group for group in np.array_split(order, n_bins) if group.size]
    total = labels.size
    value = 0.0
    for group in groups:
        mean_prob = float(np.mean(probs[group]))
        observed_rate = float(np.mean(labels[group]))
        value += (group.size / total) * abs(observed_rate - mean_prob)
    return float(value)


def classwise_ece(y_true, y_prob, n_bins: int = 15) -> dict[str, float]:
    labels, probs = _valid_binary_arrays(y_true, y_prob)
    if labels.size == 0:
        return {"class_0_ece": 0.0, "class_1_ece": 0.0, "classwise_ece": 0.0}
    class_1 = ece_fixed_width(labels, probs, n_bins=n_bins)
    class_0 = ece_fixed_width(1 - labels, 1.0 - probs, n_bins=n_bins)
    return {
        "class_0_ece": float(class_0),
        "class_1_ece": float(class_1),
        "classwise_ece": float(0.5 * (class_0 + class_1)),
    }


def maximum_calibration_error(y_true, y_prob, n_bins: int = 15) -> float:
    table = make_reliability_table(y_true, y_prob, n_bins=n_bins)
    valid = table["count"].to_numpy(dtype=int) > 0
    if not np.any(valid):
        return 0.0
    return float(np.nanmax(table.loc[valid, "abs_error"].to_numpy(dtype=float)))


def brier_score(y_true, y_prob) -> float:
    labels, probs = _valid_binary_arrays(y_true, y_prob)
    if labels.size == 0:
        return float("nan")
    return float(np.mean((probs - labels) ** 2))


def negative_log_likelihood(y_true, y_prob) -> float:
    labels, probs = _valid_binary_arrays(y_true, y_prob)
    if labels.size == 0:
        return float("nan")
    probs = clip_prob(probs)
    return float(-np.mean(labels * np.log(probs) + (1 - labels) * np.log(1.0 - probs)))


def calibration_slope_intercept(y_true, y_prob) -> dict[str, float]:
    labels, probs = _valid_binary_arrays(y_true, y_prob)
    if labels.size == 0:
        return {"calibration_intercept": float("nan"), "calibration_slope": float("nan")}
    base_rate = float(np.mean(labels))
    if np.unique(labels).size < 2:
        return {
            "calibration_intercept": float(logit([base_rate])[0]),
            "calibration_slope": 0.0,
        }
    logits = logit(probs).reshape(-1, 1)
    if float(np.std(logits)) == 0.0 or LogisticRegression is None:
        return {
            "calibration_intercept": float(logit([base_rate])[0]),
            "calibration_slope": 0.0,
        }
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(logits, labels)
    return {
        "calibration_intercept": float(model.intercept_[0]),
        "calibration_slope": float(model.coef_[0, 0]),
    }


def brier_decomposition(y_true, y_prob, n_bins: int = 15) -> dict[str, float]:
    labels, probs = _valid_binary_arrays(y_true, y_prob)
    if labels.size == 0:
        return {"reliability": float("nan"), "resolution": float("nan"), "uncertainty": float("nan")}
    table = make_reliability_table(labels, probs, n_bins=n_bins)
    valid = table["count"].to_numpy(dtype=int) > 0
    base_rate = float(np.mean(labels))
    uncertainty = base_rate * (1.0 - base_rate)
    if not np.any(valid):
        return {"reliability": 0.0, "resolution": 0.0, "uncertainty": float(uncertainty)}
    weights = table.loc[valid, "weight"].to_numpy(dtype=float)
    mean_probs = table.loc[valid, "mean_prob"].to_numpy(dtype=float)
    observed = table.loc[valid, "observed_rate"].to_numpy(dtype=float)
    reliability = float(np.sum(weights * (mean_probs - observed) ** 2))
    resolution = float(np.sum(weights * (observed - base_rate) ** 2))
    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": float(uncertainty),
    }


def evaluate_calibration(y_true, y_prob, n_bins: int = 15) -> dict[str, float]:
    metrics = {
        "ece": ece_fixed_width(y_true, y_prob, n_bins=n_bins),
        "adaptive_ece": adaptive_ece(y_true, y_prob, n_bins=n_bins),
        "mce": maximum_calibration_error(y_true, y_prob, n_bins=n_bins),
        "brier": brier_score(y_true, y_prob),
        "nll": negative_log_likelihood(y_true, y_prob),
    }
    metrics.update(classwise_ece(y_true, y_prob, n_bins=n_bins))
    metrics.update(calibration_slope_intercept(y_true, y_prob))
    brier_parts = brier_decomposition(y_true, y_prob, n_bins=n_bins)
    metrics.update({f"brier_{key}": value for key, value in brier_parts.items()})
    return {key: float(value) for key, value in metrics.items()}


def _constant_calibrator(y_true, method: str, status: str) -> GateCalibrator:
    labels = _as_int_array(y_true)
    rate = float(np.mean(labels)) if labels.size else 0.5
    return GateCalibrator(
        method=method,
        fitted_method="constant",
        status=status,
        params={"constant_probability": float(clip_prob([rate])[0])},
    )


def _fit_temperature_grid(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    logits = logit(y_prob)
    temps = np.concatenate(
        [
            np.linspace(0.05, 1.0, 80),
            np.linspace(1.025, 10.0, 160),
        ]
    )
    best_temp = 1.0
    best_loss = float("inf")
    for temp in temps:
        probs = sigmoid(logits / max(float(temp), EPS))
        loss = negative_log_likelihood(y_true, probs)
        if np.isfinite(loss) and loss < best_loss:
            best_loss = loss
            best_temp = float(temp)
    return best_temp


def fit_calibrator(y_true_val, y_prob_val, method: str) -> GateCalibrator:
    labels, probs = _valid_binary_arrays(y_true_val, y_prob_val)
    method_name = method.lower().strip().replace("-", "_").replace(" ", "_")
    aliases = {
        "identity": "none",
        "uncalibrated": "none",
        "temperature": "temperature_scaling",
        "temp": "temperature_scaling",
        "platt": "platt_scaling",
        "logistic": "platt_scaling",
        "logistic_calibration": "platt_scaling",
        "isotonic": "isotonic_regression",
        "beta": "beta_calibration",
    }
    method_name = aliases.get(method_name, method_name)

    if method_name == "none":
        return GateCalibrator(method=method, fitted_method="none", status="fitted")

    if labels.size == 0:
        return _constant_calibrator(labels, method, "empty_validation_fallback")

    if np.unique(labels).size < 2:
        return _constant_calibrator(labels, method, "single_class_validation_fallback")

    if method_name == "temperature_scaling":
        temperature = _fit_temperature_grid(labels, probs)
        return GateCalibrator(
            method=method,
            fitted_method="temperature_scaling",
            status="fitted",
            params={"temperature": temperature},
        )

    if method_name == "platt_scaling":
        if LogisticRegression is None:
            return GateCalibrator(method=method, fitted_method="none", status="skipped_sklearn_unavailable")
        model = LogisticRegression(solver="lbfgs", max_iter=1000)
        model.fit(logit(probs).reshape(-1, 1), labels)
        return GateCalibrator(method=method, fitted_method="platt_scaling", status="fitted", model=model)

    if method_name == "isotonic_regression":
        if IsotonicRegression is None:
            return GateCalibrator(method=method, fitted_method="none", status="skipped_sklearn_unavailable")
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(probs, labels)
        return GateCalibrator(method=method, fitted_method="isotonic_regression", status="fitted", model=model)

    if method_name == "beta_calibration":
        if LogisticRegression is None:
            return GateCalibrator(method=method, fitted_method="none", status="skipped_sklearn_unavailable")
        features = np.column_stack([np.log(clip_prob(probs)), np.log(clip_prob(1.0 - probs))])
        model = LogisticRegression(solver="lbfgs", max_iter=1000)
        model.fit(features, labels)
        return GateCalibrator(method=method, fitted_method="beta_calibration", status="fitted", model=model)

    raise ValueError(f"Unknown calibration method: {method}")


def apply_calibrator(y_prob, calibrator: GateCalibrator) -> np.ndarray:
    probs = clip_prob(y_prob)
    fitted = calibrator.fitted_method
    if fitted == "none":
        return probs
    if fitted == "constant":
        return np.full_like(probs, float(calibrator.params.get("constant_probability", 0.5)), dtype=float)
    if fitted == "temperature_scaling":
        temperature = float(calibrator.params.get("temperature", 1.0))
        return sigmoid(logit(probs) / max(temperature, EPS))
    if fitted == "platt_scaling":
        if calibrator.model is None:
            return probs
        return clip_prob(calibrator.model.predict_proba(logit(probs).reshape(-1, 1))[:, 1])
    if fitted == "isotonic_regression":
        if calibrator.model is None:
            return probs
        return clip_prob(calibrator.model.predict(probs))
    if fitted == "beta_calibration":
        if calibrator.model is None:
            return probs
        features = np.column_stack([np.log(clip_prob(probs)), np.log(clip_prob(1.0 - probs))])
        return clip_prob(calibrator.model.predict_proba(features)[:, 1])
    return probs


__all__ = [
    "GateCalibrator",
    "adaptive_ece",
    "apply_calibrator",
    "brier_decomposition",
    "brier_score",
    "calibration_slope_intercept",
    "classwise_ece",
    "clip_prob",
    "ece_fixed_width",
    "evaluate_calibration",
    "fit_calibrator",
    "logit",
    "make_reliability_table",
    "maximum_calibration_error",
    "negative_log_likelihood",
    "sigmoid",
]
