from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from focus_binary.eval.metrics import compute_metrics
from focus_binary.explain.faithfulness import (
    DEFAULT_STEPS,
    curve_auc_drop,
    curve_auc_gain,
    deletion_curve,
    insertion_curve,
)
from focus_binary.explain.feature_importance import gradient_sensitivity, permutation_importance
from focus_binary.explain.gradcam import gradcam_heatmap, overlay_heatmap
from focus_binary.explain.vit_rollout import extract_attention_matrices, attention_rollout, upscale_to_image
from focus_binary.features.vectorize import compute_focus_vector
from focus_binary.classical_ml.models import predict_probabilities
from focus_binary.utils.io import save_json
from focus_binary.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None

try:
    from PIL import Image, ImageFilter
except Exception:  # pragma: no cover
    Image = None
    ImageFilter = None


@dataclass(frozen=True)
class ExplainConfig:
    method: str = "gradcam"
    layer_name: Optional[str] = None
    use_cls_token: bool = False
    normalize: bool = True


@dataclass
class ExplainResult:
    image_path: str
    method: str
    heatmap: np.ndarray
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "method": self.method,
            "metadata": self.metadata,
        }


def _require_tf():
    if tf is None:
        raise ImportError("TensorFlow is required for explainability.")


def _infer_input_size(model, fallback: Sequence[int] = (224, 224)) -> Sequence[int]:
    shape = model.input_shape
    if isinstance(shape, (list, tuple)) and shape and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    if shape is None:
        return fallback
    if len(shape) >= 3 and shape[1] is not None and shape[2] is not None:
        return (int(shape[1]), int(shape[2]))
    return fallback


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _grad_vector_focus(model, image: Optional[tf.Tensor], focus_vec: tf.Tensor) -> np.ndarray:
    focus_vec = tf.convert_to_tensor(focus_vec, dtype=tf.float32)
    if focus_vec.ndim == 1:
        focus_vec = tf.expand_dims(focus_vec, axis=0)

    with tf.GradientTape() as tape:
        tape.watch(focus_vec)
        if image is None:
            preds = model(focus_vec, training=False)
        else:
            image = tf.convert_to_tensor(image, dtype=tf.float32)
            if image.ndim == 3:
                image = tf.expand_dims(image, axis=0)
            preds = model([image, focus_vec], training=False)
        preds = tf.reshape(preds, [-1])
        loss = tf.reduce_mean(preds)
    grads = tape.gradient(loss, focus_vec)
    if grads is None:
        return np.zeros((focus_vec.shape[-1],), dtype=np.float32)
    return tf.reduce_mean(tf.abs(grads), axis=0).numpy()


def _load_image(path: str, target_size: Sequence[int]) -> tf.Tensor:
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, target_size)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def _save_image(path: Path, image: np.ndarray) -> None:
    if Image is None:
        raise ImportError("Pillow is required for saving images.")
    img = np.asarray(image)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def _heavy_blur_np(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32)
    if Image is None or ImageFilter is None:
        mean_val = img.mean(axis=(0, 1), keepdims=True)
        return np.ones_like(img) * mean_val
    img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img_uint8)
    blurred = pil.filter(ImageFilter.GaussianBlur(radius=8))
    return np.asarray(blurred, dtype=np.float32) / 255.0


def _explain_single(
    model,
    image_tensor: tf.Tensor,
    model_family: str,
    input_size: Sequence[int],
    layer_name: Optional[str] = None,
    extra_inputs: Optional[Sequence[tf.Tensor]] = None,
) -> np.ndarray:
    if model_family in {"vit", "hybrid_vit"}:
        attn_mats = extract_attention_matrices(model, image_tensor)
        rollout = attention_rollout(attn_mats)
        heatmap = upscale_to_image(rollout[0], input_size)
    else:
        heatmap = gradcam_heatmap(
            model,
            image_tensor,
            layer_name=layer_name,
            extra_inputs=extra_inputs,
        )
        heatmap = tf.image.resize(heatmap[..., None], input_size, method="bilinear")
        heatmap = tf.squeeze(heatmap, axis=-1).numpy()
    heatmap = np.clip(heatmap, 0.0, 1.0)
    return heatmap


def _apply_transforms(image: tf.Tensor) -> Dict[str, tf.Tensor]:
    bright = tf.clip_by_value(tf.image.adjust_brightness(image, delta=0.05), 0.0, 1.0)
    contrast = tf.clip_by_value(tf.image.adjust_contrast(image, contrast_factor=0.9), 0.0, 1.0)
    shifted = tf.roll(image, shift=[2, 2], axis=[0, 1])
    return {
        "brightness": bright,
        "contrast": contrast,
        "translate": shifted,
    }


def _heatmap_similarity(base: np.ndarray, other: np.ndarray) -> float:
    _require_tf()
    base_tf = tf.convert_to_tensor(base[..., None], dtype=tf.float32)
    other_tf = tf.convert_to_tensor(other[..., None], dtype=tf.float32)
    score = tf.image.ssim(base_tf, other_tf, max_val=1.0)
    return float(score.numpy())


def _faithfulness_metrics(
    model,
    source_df: pd.DataFrame,
    model_family: str,
    input_size: Sequence[int],
    preprocess_fn: Optional[Any],
    enabled_measures: Sequence[str],
    per_dataset_samples: int,
    seed: int,
    layer_name: Optional[str],
    feature_rank: Optional[List[int]] = None,
    predict_fn: Optional[Any] = None,
) -> Dict[str, object]:
    image_families = {
        "cnn",
        "cnn_attention",
        "transfer",
        "vit",
        "hybrid_vit",
        "cnn_focus_hybrid",
        "convnext",
        "swin",
    }
    vector_families = {"focus_dnn", "classical_ml", "threshold_baselines"}

    if "dataset" in source_df.columns:
        datasets = sorted(source_df["dataset"].dropna().unique())
    else:
        datasets = ["all"]

    summary_rows: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []

    steps = DEFAULT_STEPS

    if model_family in vector_families:
        for dataset_name in datasets:
            subset = source_df if dataset_name == "all" else source_df[source_df["dataset"].astype(str) == str(dataset_name)]
            if subset.empty:
                continue
            sample = subset.sample(n=min(per_dataset_samples, len(subset)), random_state=seed)
            focus_vecs: List[np.ndarray] = []
            labels: List[int] = []
            paths: List[str] = []

            for _, row in sample.iterrows():
                image_path = str(row["image_path"])
                image = _load_image(image_path, input_size)
                focus_vec = compute_focus_vector(image, list(enabled_measures)).numpy()
                focus_vecs.append(focus_vec)
                labels.append(int(row.get("label", 0)))
                paths.append(image_path)

            focus_arr = np.stack(focus_vecs, axis=0) if focus_vecs else np.zeros((0, len(enabled_measures)))
            labels_arr = np.asarray(labels, dtype=int)

            if feature_rank is None:
                if model_family == "focus_dnn":
                    perm = permutation_importance(
                        model=model,
                        X_val=focus_arr,
                        y_val=labels_arr,
                        metric="auc",
                        n_repeats=5,
                        seed=seed,
                    )
                    feature_rank = [int(entry["feature_index"]) for entry in perm.get("importances", [])]
                else:
                    corrs = []
                    for idx in range(focus_arr.shape[1]):
                        col = focus_arr[:, idx]
                        if np.std(col) == 0 or len(np.unique(labels_arr)) < 2:
                            corrs.append(0.0)
                        else:
                            corrs.append(float(np.corrcoef(col, labels_arr)[0, 1]))
                    feature_rank = list(np.argsort(-np.abs(corrs)))

            if predict_fn is None:
                if model_family == "focus_dnn":
                    def _pred_fn(x):
                        preds = model.predict(x, verbose=0)
                        preds = np.asarray(preds).reshape(-1)
                        return preds.astype(float)
                else:
                    def _pred_fn(x):
                        return predict_probabilities(model, x)
                predict_fn = _pred_fn

            deletion_scores: List[float] = []
            for idx, vec in enumerate(focus_arr):
                base_prob = float(predict_fn(vec[None, :])[0])
                del_probs: List[float] = []
                for frac in steps:
                    k = int(np.ceil(float(frac) * len(feature_rank)))
                    ablated = vec.copy()
                    if k > 0:
                        ablated[feature_rank[:k]] = 0.0
                    del_probs.append(float(predict_fn(ablated[None, :])[0]))
                del_auc = curve_auc_drop(del_probs, base_prob)
                deletion_scores.append(del_auc)
                sample_rows.append(
                    {
                        "family": model_family,
                        "dataset": str(dataset_name),
                        "image_path": paths[idx],
                        "deletion_auc_feature": del_auc,
                    }
                )

            summary_rows.append(
                {
                    "family": model_family,
                    "dataset": str(dataset_name),
                    "deletion_auc_feature": float(np.mean(deletion_scores)) if deletion_scores else float("nan"),
                    "insertion_auc": float("nan"),
                    "stability_ssim": float("nan"),
                    "n_samples": int(len(deletion_scores)),
                }
            )

        return {"summary_rows": summary_rows, "sample_rows": sample_rows}

    if model_family not in image_families:
        for dataset_name in datasets:
            summary_rows.append(
                {
                    "family": model_family,
                    "dataset": str(dataset_name),
                    "deletion_auc": float("nan"),
                    "insertion_auc": float("nan"),
                    "stability_ssim": float("nan"),
                    "n_samples": 0,
                }
            )
        return {"summary_rows": summary_rows, "sample_rows": sample_rows}

    for dataset_name in datasets:
        subset = source_df if dataset_name == "all" else source_df[source_df["dataset"].astype(str) == str(dataset_name)]
        if subset.empty:
            continue
        sample = subset.sample(n=min(per_dataset_samples, len(subset)), random_state=seed)
        deletion_scores: List[float] = []
        insertion_scores: List[float] = []
        stability_scores: List[float] = []

        for _, row in sample.iterrows():
            image_path = str(row["image_path"])
            image = _load_image(image_path, input_size)
            model_input = image
            if preprocess_fn is not None:
                model_input = preprocess_fn(model_input * 255.0)

            extra_inputs = None
            if model_family == "cnn_focus_hybrid":
                focus_vec = compute_focus_vector(image, list(enabled_measures)).numpy()
                extra_inputs = [focus_vec]

            heatmap = _explain_single(
                model,
                model_input,
                model_family=model_family,
                input_size=input_size,
                layer_name=layer_name,
                extra_inputs=extra_inputs,
            )
            model_input_np = model_input.numpy() if hasattr(model_input, "numpy") else np.asarray(model_input)
            if extra_inputs:
                focus_batch = np.expand_dims(np.asarray(extra_inputs[0], dtype=np.float32), axis=0)
                base_prob = float(
                    model.predict([model_input_np[None, ...], focus_batch], verbose=0).reshape(-1)[0]
                )
            else:
                base_prob = float(model.predict(model_input_np[None, ...], verbose=0).reshape(-1)[0])

            del_probs = deletion_curve(
                model,
                image=model_input_np,
                saliency_map=heatmap,
                steps=steps,
                mode="mean",
                extra_inputs=extra_inputs,
            )
            ins_probs = insertion_curve(
                model,
                image=model_input_np,
                saliency_map=heatmap,
                steps=steps,
                mode="mean",
                extra_inputs=extra_inputs,
            )
            del_auc = curve_auc_drop(del_probs, base_prob)
            ins_auc = curve_auc_gain(ins_probs, base_prob)
            deletion_scores.append(del_auc)
            insertion_scores.append(ins_auc)

            transforms = _apply_transforms(image)
            sim_scores = []
            for _, transformed in transforms.items():
                transformed_input = transformed
                if preprocess_fn is not None:
                    transformed_input = preprocess_fn(transformed_input * 255.0)
                transformed_heatmap = _explain_single(
                    model,
                    transformed_input,
                    model_family=model_family,
                    input_size=input_size,
                    layer_name=layer_name,
                    extra_inputs=extra_inputs,
                )
                sim_scores.append(_heatmap_similarity(heatmap, transformed_heatmap))
            stability = float(np.mean(sim_scores))
            stability_scores.append(stability)

            sample_rows.append(
                {
                    "family": model_family,
                    "dataset": str(dataset_name),
                    "image_path": image_path,
                    "deletion_auc": del_auc,
                    "insertion_auc": ins_auc,
                    "stability_ssim": stability,
                }
            )

        summary_rows.append(
            {
                "family": model_family,
                "dataset": str(dataset_name),
                "deletion_auc": float(np.mean(deletion_scores)) if deletion_scores else float("nan"),
                "insertion_auc": float(np.mean(insertion_scores)) if insertion_scores else float("nan"),
                "stability_ssim": float(np.mean(stability_scores)) if stability_scores else float("nan"),
                "n_samples": int(len(deletion_scores)),
            }
        )

    return {"summary_rows": summary_rows, "sample_rows": sample_rows}


def _run_focus_explainability(
    model,
    df: pd.DataFrame,
    model_family: str,
    out_dir: Path,
    input_size: Sequence[int],
    enabled_measures: Sequence[str],
    preprocess_fn: Optional[Any] = None,
    layer_name: Optional[str] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    stability_scores: List[float] = []
    images: List[np.ndarray] = []
    focus_vectors: List[np.ndarray] = []
    preds_all: List[float] = []

    for idx, row in df.iterrows():
        image_path = str(row["image_path"])
        label = int(row.get("label", 0))
        image = _load_image(image_path, input_size)
        focus_vec = compute_focus_vector(image, list(enabled_measures))

        model_input = image
        if preprocess_fn is not None:
            model_input = preprocess_fn(model_input * 255.0)

        if model_family == "focus_dnn":
            pred = float(model(tf.expand_dims(focus_vec, axis=0), training=False).numpy().reshape(-1)[0])
            heatmap = None
        else:
            pred = float(
                model(
                    [tf.expand_dims(model_input, axis=0), tf.expand_dims(focus_vec, axis=0)],
                    training=False,
                )
                .numpy()
                .reshape(-1)[0]
            )
            heatmap = _explain_single(
                model,
                model_input,
                model_family=model_family,
                input_size=input_size,
                layer_name=layer_name,
                extra_inputs=[focus_vec],
            )

        preds_all.append(pred)
        images.append(image.numpy())
        focus_vectors.append(focus_vec.numpy())

        sample_prefix = f"sample_{idx:03d}"
        orig_path = out_dir / f"{sample_prefix}_orig.png"
        _save_image(orig_path, image.numpy())

        heatmap_path = ""
        overlay_path = ""
        if heatmap is not None:
            overlay = overlay_heatmap(image.numpy(), heatmap, alpha=0.4)
            heatmap_path = out_dir / f"{sample_prefix}_heatmap.png"
            overlay_path = out_dir / f"{sample_prefix}_overlay.png"
            _save_image(heatmap_path, heatmap)
            _save_image(overlay_path, overlay)

        transforms = _apply_transforms(image)
        sim_scores = []
        if model_family == "focus_dnn":
            base_grad = _grad_vector_focus(model, None, focus_vec)
            for transformed in transforms.values():
                t_vec = compute_focus_vector(transformed, list(enabled_measures))
                grad_vec = _grad_vector_focus(model, None, t_vec)
                sim_scores.append(_cosine_similarity(base_grad, grad_vec))
        else:
            for transformed in transforms.values():
                t_vec = compute_focus_vector(transformed, list(enabled_measures))
                transformed_input = transformed
                if preprocess_fn is not None:
                    transformed_input = preprocess_fn(transformed_input * 255.0)
                transformed_heatmap = _explain_single(
                    model,
                    transformed_input,
                    model_family=model_family,
                    input_size=input_size,
                    layer_name=layer_name,
                    extra_inputs=[t_vec],
                )
                sim_scores.append(_heatmap_similarity(heatmap, transformed_heatmap))

        stability_score = float(np.mean(sim_scores)) if sim_scores else float("nan")
        stability_scores.append(stability_score)

        records.append(
            {
                "image_path": image_path,
                "label": label,
                "pred": pred,
                "dataset": str(row.get("dataset", "")),
                "orig_image": str(orig_path),
                "heatmap_image": str(heatmap_path) if heatmap_path else "",
                "overlay_image": str(overlay_path) if overlay_path else "",
                "stability": stability_score,
            }
        )

    images_arr = np.stack(images) if images else np.zeros((0, *input_size, 3), dtype=np.float32)
    focus_arr = np.stack(focus_vectors) if focus_vectors else np.zeros((0, len(enabled_measures)), dtype=np.float32)
    labels = df["label"].astype(int).to_numpy()

    if model_family == "focus_dnn":
        perm = permutation_importance(
            model=model,
            X_val=focus_arr,
            y_val=labels,
            metric="auc",
            n_repeats=10,
            seed=seed,
        )
        grad = gradient_sensitivity(model, focus_arr)
    else:
        perm = permutation_importance(
            model=model,
            X_val=(images_arr, focus_arr),
            y_val=labels,
            metric="auc",
            n_repeats=10,
            seed=seed,
        )
        grad = gradient_sensitivity(model, (images_arr, focus_arr))

    perm_lookup = {entry["feature_index"]: entry for entry in perm.get("importances", [])}
    feature_rows: List[Dict[str, object]] = []
    for idx, name in enumerate(enabled_measures):
        entry = perm_lookup.get(idx, {"mean": float("nan"), "std": float("nan")})
        grad_val = float(grad[idx]) if idx < len(grad) else float("nan")
        feature_rows.append(
            {
                "feature": name,
                "perm_mean": float(entry.get("mean", float("nan"))),
                "perm_std": float(entry.get("std", float("nan"))),
                "grad_sensitivity": grad_val,
            }
        )

    feature_rows_sorted = sorted(
        feature_rows,
        key=lambda r: np.nan_to_num(r["perm_mean"], nan=-np.inf),
        reverse=True,
    )
    top_features = [row["feature"] for row in feature_rows_sorted[:3]]
    feature_rank = [enabled_measures.index(row["feature"]) for row in feature_rows_sorted if row["feature"] in enabled_measures]

    feature_csv = out_dir / "feature_importance.csv"
    pd.DataFrame(feature_rows).to_csv(feature_csv, index=False)

    ablation: Dict[str, Any] | None = None
    ablation_path: str | None = None
    if model_family == "cnn_focus_hybrid" and len(labels) > 0:
        base_images = images_arr
        if preprocess_fn is not None:
            base_images = preprocess_fn(base_images * 255.0).numpy()
        focus_zero = np.zeros_like(focus_arr)
        blurred = np.stack([_heavy_blur_np(img) for img in images_arr], axis=0)
        blurred_images = blurred
        if preprocess_fn is not None:
            blurred_images = preprocess_fn(blurred_images * 255.0).numpy()

        base_probs = model([base_images, focus_arr], training=False).numpy().reshape(-1)
        focus_probs = model([base_images, focus_zero], training=False).numpy().reshape(-1)
        blur_probs = model([blurred_images, focus_arr], training=False).numpy().reshape(-1)

        def _delta_metrics(mask: np.ndarray | None = None) -> Dict[str, float]:
            if mask is None:
                mask = np.ones(len(labels), dtype=bool)
            base_metrics = compute_metrics(labels[mask], base_probs[mask], threshold=0.5)
            focus_metrics = compute_metrics(labels[mask], focus_probs[mask], threshold=0.5)
            blur_metrics = compute_metrics(labels[mask], blur_probs[mask], threshold=0.5)
            return {
                "focus_zero_delta_auc": float(base_metrics["auc"] - focus_metrics["auc"]),
                "focus_zero_delta_f1": float(base_metrics["f1"] - focus_metrics["f1"]),
                "focus_zero_delta_prob": float(np.mean(base_probs[mask] - focus_probs[mask])),
                "blur_delta_auc": float(base_metrics["auc"] - blur_metrics["auc"]),
                "blur_delta_f1": float(base_metrics["f1"] - blur_metrics["f1"]),
                "blur_delta_prob": float(np.mean(base_probs[mask] - blur_probs[mask])),
            }

        ablation = {"all": _delta_metrics()}
        if "dataset" in df.columns:
            for dataset_name in sorted(df["dataset"].dropna().unique()):
                mask = df["dataset"].astype(str).to_numpy() == str(dataset_name)
                ablation[str(dataset_name)] = _delta_metrics(mask)

        ablation_rows = []
        for dataset_name, metrics in ablation.items():
            row = {"dataset": str(dataset_name)}
            row.update(metrics)
            ablation_rows.append(row)
        ablation_path = str(out_dir / "hybrid_branch_ablation.csv")
        pd.DataFrame(ablation_rows).to_csv(ablation_path, index=False)

    stability_mean = float(np.mean(stability_scores)) if stability_scores else float("nan")
    stability_std = float(np.std(stability_scores)) if stability_scores else float("nan")

    records_path = out_dir / "explanations.csv"
    pd.DataFrame(records).to_csv(records_path, index=False)

    summary = {
        "n_samples": int(len(records)),
        "stability_mean": stability_mean,
        "stability_std": stability_std,
        "records": records,
        "records_csv": str(records_path),
        "feature_importance_csv": str(feature_csv),
        "feature_importance": perm,
        "top_features": top_features,
        "feature_rank": feature_rank,
        "ablation": ablation,
        "hybrid_ablation_csv": ablation_path,
    }
    return summary


def run_explainability(
    model,
    sample_batch,
    model_family: str,
    out_dir: str | Path,
    n_samples: int = 32,
    seed: int = 0,
    preprocess_fn: Optional[Any] = None,
    layer_name: Optional[str] = None,
    enabled_measures: Optional[Sequence[str]] = None,
    faithfulness_samples_per_dataset: int = 50,
    run_faithfulness: bool = True,
) -> Dict[str, Any]:
    """Run explainability and save per-sample artifacts."""
    _require_tf()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(sample_batch, pd.DataFrame):
        sample_batch = pd.DataFrame(sample_batch)
    if sample_batch.empty:
        raise ValueError("sample_batch is empty.")

    if enabled_measures is None:
        enabled_measures = ["lapvar", "tenengrad", "brenner", "sml"]

    source_df = sample_batch
    df = source_df.sample(n=min(n_samples, len(source_df)), random_state=seed).reset_index(drop=True)
    input_size = _infer_input_size(model)

    def _stat(values: List[float]) -> Tuple[float, float]:
        arr = np.asarray([v for v in values if v == v], dtype=float)
        if arr.size == 0:
            return float("nan"), float("nan")
        return float(arr.mean()), float(arr.std())

    if model_family in {"focus_dnn", "cnn_focus_hybrid"}:
        summary = _run_focus_explainability(
            model=model,
            df=df,
            model_family=model_family,
            out_dir=out_dir,
            input_size=input_size,
            enabled_measures=enabled_measures,
            preprocess_fn=preprocess_fn,
            layer_name=layer_name,
            seed=seed,
        )
        metrics_path = None
        samples_path = None
        metrics_rows = []
        if run_faithfulness:
            faithfulness = _faithfulness_metrics(
                model=model,
                source_df=source_df,
                model_family=model_family,
                input_size=input_size,
                preprocess_fn=preprocess_fn,
                enabled_measures=enabled_measures,
                per_dataset_samples=faithfulness_samples_per_dataset,
                seed=seed,
                layer_name=layer_name,
                feature_rank=summary.get("feature_rank"),
            )
            metrics_rows = faithfulness.get("summary_rows", [])
            metrics_path = out_dir / "explainability_metrics.csv"
            pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
            samples = faithfulness.get("sample_rows", [])
            samples_path = out_dir / "faithfulness_samples.csv"
            pd.DataFrame(samples).to_csv(samples_path, index=False)

            del_vals = [row.get("deletion_auc") for row in samples if row.get("deletion_auc") is not None]
            ins_vals = [row.get("insertion_auc") for row in samples if row.get("insertion_auc") is not None]
            feat_vals = [row.get("deletion_auc_feature") for row in samples if row.get("deletion_auc_feature") is not None]
            del_mean, del_std = _stat([v for v in del_vals if v == v])
            ins_mean, ins_std = _stat([v for v in ins_vals if v == v])
            feat_mean, feat_std = _stat([v for v in feat_vals if v == v])
            summary["deletion_auc_mean"] = del_mean
            summary["deletion_auc_std"] = del_std
            summary["insertion_auc_mean"] = ins_mean
            summary["insertion_auc_std"] = ins_std
            summary["deletion_auc_feature_mean"] = feat_mean
            summary["deletion_auc_feature_std"] = feat_std

        summary["explainability_metrics_csv"] = str(metrics_path) if metrics_path else None
        summary["faithfulness_samples_csv"] = str(samples_path) if samples_path else None
        logger.info("explainability run complete", extra={"samples": len(df), "out_dir": str(out_dir)})
        return summary

    records = []
    stability_scores: List[float] = []

    for idx, row in df.iterrows():
        image_path = str(row["image_path"])
        label = int(row.get("label", 0))
        image = _load_image(image_path, input_size)
        model_input = image
        if preprocess_fn is not None:
            model_input = preprocess_fn(model_input * 255.0)

        pred = float(model(tf.expand_dims(model_input, axis=0), training=False).numpy().reshape(-1)[0])
        heatmap = _explain_single(
            model,
            model_input,
            model_family=model_family,
            input_size=input_size,
            layer_name=layer_name,
        )
        overlay = overlay_heatmap(image.numpy(), heatmap, alpha=0.4)

        sample_prefix = f"sample_{idx:03d}"
        orig_path = out_dir / f"{sample_prefix}_orig.png"
        heatmap_path = out_dir / f"{sample_prefix}_heatmap.png"
        overlay_path = out_dir / f"{sample_prefix}_overlay.png"

        _save_image(orig_path, image.numpy())
        _save_image(heatmap_path, heatmap)
        _save_image(overlay_path, overlay)

        transforms = _apply_transforms(image)
        sim_scores = []
        for _, transformed in transforms.items():
            transformed_input = transformed
            if preprocess_fn is not None:
                transformed_input = preprocess_fn(transformed_input * 255.0)
            transformed_heatmap = _explain_single(
                model,
                transformed_input,
                model_family=model_family,
                input_size=input_size,
                layer_name=layer_name,
            )
            sim_scores.append(_heatmap_similarity(heatmap, transformed_heatmap))
        stability_score = float(np.mean(sim_scores))
        stability_scores.append(stability_score)

        records.append(
            {
                "image_path": image_path,
                "label": label,
                "pred": pred,
                "dataset": str(row.get("dataset", "")),
                "orig_image": str(orig_path),
                "heatmap_image": str(heatmap_path),
                "overlay_image": str(overlay_path),
                "stability": stability_score,
            }
        )

    stability_mean = float(np.mean(stability_scores)) if stability_scores else float("nan")
    stability_std = float(np.std(stability_scores)) if stability_scores else float("nan")

    records_path = out_dir / "explanations.csv"
    pd.DataFrame(records).to_csv(records_path, index=False)

    metrics_path = None
    samples_path = None
    metrics_rows = []
    if run_faithfulness:
        faithfulness = _faithfulness_metrics(
            model=model,
            source_df=source_df,
            model_family=model_family,
            input_size=input_size,
            preprocess_fn=preprocess_fn,
            enabled_measures=enabled_measures,
            per_dataset_samples=faithfulness_samples_per_dataset,
            seed=seed,
            layer_name=layer_name,
        )
        metrics_rows = faithfulness.get("summary_rows", [])
        metrics_path = out_dir / "explainability_metrics.csv"
        pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
        samples = faithfulness.get("sample_rows", [])
        samples_path = out_dir / "faithfulness_samples.csv"
        pd.DataFrame(samples).to_csv(samples_path, index=False)
        del_vals = [row.get("deletion_auc") for row in samples if row.get("deletion_auc") is not None]
        ins_vals = [row.get("insertion_auc") for row in samples if row.get("insertion_auc") is not None]
        del_mean, del_std = _stat([v for v in del_vals if v == v])
        ins_mean, ins_std = _stat([v for v in ins_vals if v == v])
        summary_deletion = (del_mean, del_std)
        summary_insertion = (ins_mean, ins_std)

    summary = {
        "n_samples": int(len(records)),
        "stability_mean": stability_mean,
        "stability_std": stability_std,
        "deletion_auc_mean": summary_deletion[0] if run_faithfulness else None,
        "deletion_auc_std": summary_deletion[1] if run_faithfulness else None,
        "insertion_auc_mean": summary_insertion[0] if run_faithfulness else None,
        "insertion_auc_std": summary_insertion[1] if run_faithfulness else None,
        "records": records,
        "records_csv": str(records_path),
        "explainability_metrics_csv": str(metrics_path) if metrics_path else None,
        "faithfulness_samples_csv": str(samples_path) if samples_path else None,
    }
    logger.info("explainability run complete", extra={"samples": len(records), "out_dir": str(out_dir)})
    return summary
