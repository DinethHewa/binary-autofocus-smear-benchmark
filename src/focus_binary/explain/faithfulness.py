from __future__ import annotations

from typing import Any, Iterable, List

import numpy as np

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None

try:
    from PIL import Image, ImageFilter
except Exception:  # pragma: no cover
    Image = None
    ImageFilter = None


DEFAULT_STEPS = [0.05 * i for i in range(1, 11)]
DEFAULT_AUC_FRACTIONS = [0.0, 0.05, 0.15, 0.30, 0.50, 0.75, 1.0]


def _require_tf() -> None:
    if tf is None:
        raise ImportError("TensorFlow is required for faithfulness tests.")


def probability_from_output(preds: Any) -> tuple[float, bool, str]:
    """Convert sigmoid/logit or two-class softmax/logits output to P(class=1)."""
    arr = np.asarray(preds.numpy() if hasattr(preds, "numpy") else preds, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=np.nan, posinf=np.nan, neginf=np.nan)
    if not np.isfinite(arr).any():
        return float("nan"), False, "model_output_nonfinite"
    warning = ""
    if arr.ndim == 0:
        val = float(arr)
        if val < 0.0 or val > 1.0:
            val = 1.0 / (1.0 + np.exp(-np.clip(val, -60.0, 60.0)))
            warning = "scalar_logit_sigmoid_applied"
        return float(np.clip(val, 0.0, 1.0)), True, warning
    if arr.ndim == 1 or arr.shape[-1] == 1:
        val = float(arr.reshape(-1)[0])
        if val < 0.0 or val > 1.0:
            val = 1.0 / (1.0 + np.exp(-np.clip(val, -60.0, 60.0)))
            warning = "logit_sigmoid_applied"
        return float(np.clip(val, 0.0, 1.0)), True, warning
    flat = arr.reshape((-1, arr.shape[-1]))
    if flat.shape[-1] >= 2:
        row = flat[0]
        row_sum = float(np.sum(row))
        if np.nanmin(row) >= 0.0 and np.nanmax(row) <= 1.0 and abs(row_sum - 1.0) <= 1e-3:
            return float(np.clip(row[1], 0.0, 1.0)), True, warning
        shifted = row - np.nanmax(row)
        exp = np.exp(np.clip(shifted, -60.0, 60.0))
        denom = float(np.sum(exp))
        if denom <= 0.0 or not np.isfinite(denom):
            return float("nan"), False, "softmax_denominator_invalid"
        warning = "softmax_applied_to_logits"
        return float(np.clip(exp[1] / denom, 0.0, 1.0)), True, warning
    return float("nan"), False, "unsupported_model_output_shape"


def normalize_saliency_with_status(saliency: np.ndarray) -> tuple[np.ndarray, bool, str]:
    sal = np.asarray(saliency, dtype=np.float32)
    if sal.ndim == 3:
        sal = np.mean(np.abs(sal), axis=-1)
    warning_parts: list[str] = []
    if not np.all(np.isfinite(sal)):
        warning_parts.append("heatmap_nonfinite_replaced")
    sal = np.nan_to_num(sal, nan=0.0, posinf=0.0, neginf=0.0)
    if sal.size == 0:
        return np.zeros((1, 1), dtype=np.float32), False, "heatmap_empty"
    min_val = float(np.min(sal))
    max_val = float(np.max(sal))
    if not np.isfinite(min_val) or not np.isfinite(max_val):
        return np.zeros_like(sal, dtype=np.float32), False, "heatmap_range_nonfinite"
    if max_val <= min_val:
        warning_parts.append("heatmap_constant_zero_map")
        return np.zeros_like(sal, dtype=np.float32), False, ";".join(warning_parts)
    sal = (sal - min_val) / (max_val - min_val)
    return np.clip(sal, 0.0, 1.0).astype(np.float32), len(warning_parts) == 0, ";".join(warning_parts)


def _normalize_saliency(saliency: np.ndarray) -> np.ndarray:
    sal, _, _ = normalize_saliency_with_status(saliency)
    return sal


def normalize_fractions(fractions: Iterable[float] | None = None) -> np.ndarray:
    raw = DEFAULT_AUC_FRACTIONS if fractions is None else list(fractions)
    values = np.asarray(raw, dtype=np.float64)
    values = values[np.isfinite(values)]
    values = np.clip(values, 0.0, 1.0)
    values = np.unique(values)
    if values.size == 0:
        values = np.asarray([0.0, 1.0], dtype=np.float64)
    if values[0] > 0.0:
        values = np.insert(values, 0, 0.0)
    if values[-1] < 1.0:
        values = np.append(values, 1.0)
    return np.sort(values)


def probability_curve_auc(fractions: Iterable[float], probabilities: Iterable[float]) -> tuple[float, bool, str]:
    x = np.asarray(list(fractions), dtype=np.float64)
    y = np.asarray(list(probabilities), dtype=np.float64)
    if x.size != y.size or x.size < 2:
        return float("nan"), False, "curve_length_invalid"
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return float("nan"), False, "curve_nonfinite"
    if np.any(np.diff(x) < -1e-12) or float(np.min(x)) < 0.0 or float(np.max(x)) > 1.0:
        return float("nan"), False, "curve_fraction_not_monotonic_0_1"
    if float(np.min(y)) < -1e-8 or float(np.max(y)) > 1.0 + 1e-8:
        return float("nan"), False, "curve_probability_outside_0_1"
    auc = float(np.trapezoid(np.clip(y, 0.0, 1.0), x))
    valid = np.isfinite(auc) and 0.0 <= auc <= 1.0
    return auc, bool(valid), "" if valid else "auc_probability_outside_0_1"


def faithfulness_metrics_from_curves(
    fractions: Iterable[float],
    baseline_probability: float,
    deletion_probabilities: Iterable[float],
    insertion_probabilities: Iterable[float],
) -> dict[str, float | bool | str]:
    x = normalize_fractions(fractions)
    deletion = np.asarray(list(deletion_probabilities), dtype=np.float64)
    insertion = np.asarray(list(insertion_probabilities), dtype=np.float64)
    base = float(baseline_probability)
    warnings: list[str] = []
    probability_valid = bool(np.isfinite(base) and 0.0 <= base <= 1.0)
    if deletion.size != x.size or insertion.size != x.size:
        warnings.append("curve_length_mismatch")
        probability_valid = False
    else:
        if not np.all(np.isfinite(deletion)) or not np.all(np.isfinite(insertion)):
            warnings.append("curve_probability_nonfinite")
            probability_valid = False
        if np.any(deletion < -1e-8) or np.any(deletion > 1.0 + 1e-8) or np.any(insertion < -1e-8) or np.any(insertion > 1.0 + 1e-8):
            warnings.append("curve_probability_outside_0_1")
            probability_valid = False
    if probability_valid:
        deletion = np.clip(deletion, 0.0, 1.0)
        insertion = np.clip(insertion, 0.0, 1.0)
    deletion_auc_prob, deletion_auc_valid, deletion_auc_warning = probability_curve_auc(x, deletion)
    insertion_auc_prob, insertion_auc_valid, insertion_auc_warning = probability_curve_auc(x, insertion)
    if deletion_auc_warning:
        warnings.append(f"deletion_{deletion_auc_warning}")
    if insertion_auc_warning:
        warnings.append(f"insertion_{insertion_auc_warning}")
    if deletion.size == x.size and np.all(np.isfinite(deletion)) and np.isfinite(base):
        deletion_drop_auc = float(np.trapezoid(np.clip(base - deletion, -1.0, 1.0), x))
    else:
        deletion_drop_auc = float("nan")
    xai_metric_valid = bool(probability_valid and deletion_auc_valid and insertion_auc_valid)
    return {
        "deletion_auc": deletion_auc_prob,
        "insertion_auc": insertion_auc_prob,
        "deletion_auc_prob": deletion_auc_prob,
        "insertion_auc_prob": insertion_auc_prob,
        "deletion_drop_auc": deletion_drop_auc,
        "probability_valid": probability_valid,
        "xai_metric_valid": xai_metric_valid,
        "xai_metric_warning": ";".join(dict.fromkeys(w for w in warnings if w)),
    }


def saliency_to_mask(saliency_map: np.ndarray, top_frac: float) -> np.ndarray:
    sal = _normalize_saliency(saliency_map)
    if top_frac <= 0:
        return np.zeros_like(sal, dtype=bool)
    if top_frac >= 1:
        return np.ones_like(sal, dtype=bool)
    if float(np.max(sal)) <= 0.0:
        return np.zeros_like(sal, dtype=bool)
    flat = sal.reshape(-1)
    k = int(np.ceil(len(flat) * float(top_frac)))
    if k <= 0:
        return np.zeros_like(sal, dtype=bool)
    threshold = np.partition(flat, -k)[-k]
    return sal >= threshold


def _baseline_image(image: np.ndarray, mode: str = "mean") -> np.ndarray:
    img = np.asarray(image, dtype=np.float32)
    if mode == "mean":
        mean_val = img.mean(axis=(0, 1), keepdims=True)
        return np.ones_like(img) * mean_val
    if mode == "blur":
        if Image is None or ImageFilter is None:
            mean_val = img.mean(axis=(0, 1), keepdims=True)
            return np.ones_like(img) * mean_val
        img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        pil = Image.fromarray(img_uint8)
        blurred = pil.filter(ImageFilter.GaussianBlur(radius=8))
        return np.asarray(blurred, dtype=np.float32) / 255.0
    raise ValueError("mode must be 'mean' or 'blur'")


def apply_mask(image: np.ndarray, mask: np.ndarray, mode: str = "mean") -> np.ndarray:
    img = np.asarray(image, dtype=np.float32)
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.ndim == 2 and img.ndim == 3:
        mask_arr = mask_arr[..., None]
    baseline = _baseline_image(img, mode=mode)
    out = np.array(img, copy=True)
    out[mask_arr] = baseline[mask_arr]
    return out


def _predict_prob(model, image: np.ndarray, extra_inputs: Iterable[np.ndarray] | None = None) -> float:
    _require_tf()
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 3:
        img = np.expand_dims(img, axis=0)
    if extra_inputs:
        inputs: List[np.ndarray] = [img]
        for extra in extra_inputs:
            extra_arr = np.asarray(extra, dtype=np.float32)
            if extra_arr.ndim == 1:
                extra_arr = np.expand_dims(extra_arr, axis=0)
            inputs.append(extra_arr)
        preds = model.predict(inputs, verbose=0)
    else:
        preds = model.predict(img, verbose=0)
    prob, valid, _ = probability_from_output(preds)
    if not valid:
        return float("nan")
    return prob


def deletion_curve(
    model,
    image: np.ndarray,
    saliency_map: np.ndarray,
    steps: Iterable[float] | None = None,
    mode: str = "mean",
    extra_inputs: Iterable[np.ndarray] | None = None,
) -> List[float]:
    if steps is None:
        steps = DEFAULT_STEPS
    probs: List[float] = []
    for frac in steps:
        mask = saliency_to_mask(saliency_map, float(frac))
        masked = apply_mask(image, mask, mode=mode)
        probs.append(_predict_prob(model, masked, extra_inputs=extra_inputs))
    return probs


def insertion_curve(
    model,
    image: np.ndarray,
    saliency_map: np.ndarray,
    steps: Iterable[float] | None = None,
    mode: str = "mean",
    extra_inputs: Iterable[np.ndarray] | None = None,
) -> List[float]:
    if steps is None:
        steps = DEFAULT_STEPS
    baseline = _baseline_image(image, mode=mode)
    probs: List[float] = []
    for frac in steps:
        mask = saliency_to_mask(saliency_map, float(frac))
        mask_arr = mask
        if mask_arr.ndim == 2 and image.ndim == 3:
            mask_arr = mask_arr[..., None]
        current = np.array(baseline, copy=True)
        current[mask_arr] = image[mask_arr]
        probs.append(_predict_prob(model, current, extra_inputs=extra_inputs))
    return probs


def curve_auc_drop(deletion_probs: Iterable[float], base_prob: float) -> float:
    probs = np.asarray(list(deletion_probs), dtype=np.float32)
    if probs.size == 0:
        return float("nan")
    xs = normalize_fractions(np.linspace(0.0, 1.0, num=len(probs)))
    if xs.size != probs.size:
        xs = np.linspace(0.0, 1.0, num=len(probs))
    drop = np.clip(float(base_prob) - probs, -1.0, 1.0)
    return float(np.trapezoid(drop, xs))


def curve_auc_gain(insertion_probs: Iterable[float], base_prob: float) -> float:
    probs = np.asarray(list(insertion_probs), dtype=np.float32)
    if probs.size == 0:
        return float("nan")
    xs = normalize_fractions(np.linspace(0.0, 1.0, num=len(probs)))
    if xs.size != probs.size:
        xs = np.linspace(0.0, 1.0, num=len(probs))
    return float(np.trapezoid(np.clip(probs, 0.0, 1.0), xs))


def deletion_auc(
    model,
    image: np.ndarray,
    saliency: np.ndarray,
    fractions: Iterable[float] | None = None,
    mode: str = "mean",
    extra_inputs: Iterable[np.ndarray] | None = None,
) -> float:
    if fractions is None:
        fractions = DEFAULT_STEPS
    base_prob = _predict_prob(model, image, extra_inputs=extra_inputs)
    probs = deletion_curve(
        model=model,
        image=image,
        saliency_map=saliency,
        steps=fractions,
        mode=mode,
        extra_inputs=extra_inputs,
    )
    return curve_auc_drop(probs, base_prob)


def insertion_auc(
    model,
    image: np.ndarray,
    saliency: np.ndarray,
    fractions: Iterable[float] | None = None,
    mode: str = "mean",
    extra_inputs: Iterable[np.ndarray] | None = None,
) -> float:
    if fractions is None:
        fractions = DEFAULT_STEPS
    base_prob = _predict_prob(model, image, extra_inputs=extra_inputs)
    probs = insertion_curve(
        model=model,
        image=image,
        saliency_map=saliency,
        steps=fractions,
        mode=mode,
        extra_inputs=extra_inputs,
    )
    return curve_auc_gain(probs, base_prob)
