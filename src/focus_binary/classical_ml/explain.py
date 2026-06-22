from __future__ import annotations

from typing import Dict, List

import numpy as np

try:
    from sklearn.inspection import permutation_importance
except Exception:  # pragma: no cover
    permutation_importance = None


def _unwrap_estimator(model):
    if hasattr(model, "named_steps"):
        model = model.named_steps.get("clf", model)
    return model


def _coef_importance(model, feature_names: List[str]) -> List[Dict[str, float]]:
    if not hasattr(model, "coef_"):
        return []
    coef = np.asarray(model.coef_).reshape(-1)
    rows = []
    for name, value in zip(feature_names, coef):
        rows.append({"feature": name, "importance": float(value)})
    rows.sort(key=lambda x: abs(x["importance"]), reverse=True)
    return rows


def _tree_importance(model, feature_names: List[str]) -> List[Dict[str, float]]:
    if not hasattr(model, "feature_importances_"):
        return []
    importances = np.asarray(model.feature_importances_).reshape(-1)
    rows = []
    for name, value in zip(feature_names, importances):
        rows.append({"feature": name, "importance": float(value)})
    rows.sort(key=lambda x: x["importance"], reverse=True)
    return rows


def _calibrated_base(model):
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        return model.calibrated_classifiers_[0].estimator
    return model


def permutation_importance_scores(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 10,
    seed: int = 0,
    scoring: str = "roc_auc",
) -> List[Dict[str, float]]:
    if permutation_importance is None:
        return []
    result = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=n_repeats,
        random_state=seed,
        scoring=scoring,
    )
    means = result.importances_mean
    stds = result.importances_std
    rows = []
    for name, mean, std in zip(feature_names, means, stds):
        rows.append({"feature": name, "mean": float(mean), "std": float(std)})
    rows.sort(key=lambda x: x["mean"], reverse=True)
    return rows


def explain_classical_model(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    seed: int = 0,
) -> Dict[str, object]:
    base = _unwrap_estimator(model)
    base = _calibrated_base(base)
    coef_rows = _coef_importance(base, feature_names)
    tree_rows = _tree_importance(base, feature_names)
    perm_rows = permutation_importance_scores(model, X_val, y_val, feature_names, seed=seed)

    ranked = perm_rows or tree_rows or coef_rows
    top_features = [row["feature"] for row in ranked[:3]]

    return {
        "coefficients": coef_rows,
        "tree_importances": tree_rows,
        "permutation_importance": perm_rows,
        "top_features": top_features,
    }
