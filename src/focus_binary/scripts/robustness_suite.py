from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from focus_binary import paths
from focus_binary.calib.calibration import choose_threshold, compute_brier, compute_ece
from focus_binary.data.tfdata import build_datasets
from focus_binary.data.tfdata_features import build_feature_datasets
from focus_binary.eval.metrics import compute_metrics
from focus_binary.models.transfer import get_preprocess
from focus_binary.robust.perturb import (
    apply_perturbation,
    feature_dropout,
    feature_gaussian_noise,
)
from focus_binary.utils.io import load_yaml, save_json
from focus_binary.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run robustness perturbation suite.")
    parser.add_argument("--model-path", required=True, help="Path to model (.keras)")
    parser.add_argument("--manifest", required=True, help="Manifest CSV with splits")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--split", default="test", help="Split to evaluate")
    parser.add_argument("--grid", default="light", help="Perturbation grid preset")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--input-size", type=int, default=224, help="Square input size")
    parser.add_argument("--backbone", default=None, help="Transfer backbone name for preprocessing")
    parser.add_argument("--save-preds", action="store_true", help="Save per-perturbation predictions CSVs")
    return parser.parse_args(argv)


def _load_model(model_path: Path):
    if tf is None:
        raise ImportError("TensorFlow is required for robustness evaluation.")
    from focus_binary.models.vit import _CLSToken, _PositionalEmbedding  # type: ignore
    from focus_binary.models.swin_tiny import WindowPartition, WindowReverse  # type: ignore
    from focus_binary.models.convnext import ConvNeXtPreprocess  # type: ignore

    custom_objects = {}
    if _CLSToken is not None:
        custom_objects["_CLSToken"] = _CLSToken
    if _PositionalEmbedding is not None:
        custom_objects["_PositionalEmbedding"] = _PositionalEmbedding
    if WindowPartition is not None:
        custom_objects["WindowPartition"] = WindowPartition
    if WindowReverse is not None:
        custom_objects["WindowReverse"] = WindowReverse
    if ConvNeXtPreprocess is not None:
        custom_objects["ConvNeXtPreprocess"] = ConvNeXtPreprocess

    if custom_objects:
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return tf.keras.models.load_model(model_path)


def _as_probabilities(preds: np.ndarray) -> np.ndarray:
    preds = np.asarray(preds)
    if preds.ndim > 1:
        if preds.shape[-1] == 1:
            preds = preds.reshape(-1)
        elif preds.shape[-1] >= 2:
            preds = preds[..., 1]
    return preds.reshape(-1).astype(float)


def _predict_probs(model, ds) -> np.ndarray:
    preds = model.predict(ds, verbose=0)
    return _as_probabilities(preds)


def _apply_preprocess(ds, preprocess_fn):
    def _map(inputs, label):
        if isinstance(inputs, (tuple, list)):
            img = inputs[0]
            rest = inputs[1:]
            img = preprocess_fn(img * 255.0)
            if len(rest) == 1:
                return (img, rest[0]), label
            return (img, *rest), label
        return preprocess_fn(inputs * 255.0), label

    return ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)


def _grid_preset(name: str) -> List[Tuple[str, str, Dict[str, float]]]:
    if name != "light":
        raise ValueError("Only 'light' grid preset is supported right now.")

    items: List[Tuple[str, str, Dict[str, float]]] = [("clean", "none", {})]
    for stddev in (0.01, 0.03, 0.05):
        items.append(("gaussian_noise", f"{stddev:.2f}", {"stddev": stddev}))
    for quality in (95, 80, 60):
        items.append(("jpeg_compression", str(quality), {"quality": quality}))
    for delta in (0.05, 0.1):
        items.append(("brightness", f"{delta:.2f}", {"delta": delta}))
    for factor in (0.8, 0.6):
        items.append(("contrast", f"{factor:.2f}", {"factor": factor}))
    for sigma in (0.5, 1.0):
        items.append(("slight_blur", f"{sigma:.1f}", {"sigma": sigma}))
    return items


def _feature_grid() -> List[Tuple[str, str, Dict[str, float]]]:
    items: List[Tuple[str, str, Dict[str, float]]] = []
    for stddev in (0.01, 0.03, 0.05):
        items.append((f"feat_noise_{stddev:.2f}", "feature_gaussian_noise", {"stddev": stddev}))
    for p in (0.1, 0.2):
        items.append((f"feat_dropout_{p:.1f}", "feature_dropout", {"p": p}))
    return items


def _apply_feature_perturb(x, kind: str, kwargs: Dict[str, float]):
    if kind == "feature_gaussian_noise":
        return feature_gaussian_noise(x, **kwargs)
    if kind == "feature_dropout":
        return feature_dropout(x, **kwargs)
    raise ValueError(f"Unknown feature perturbation: {kind}")


def _metrics_row(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    dataset_name: str,
    perturb: str,
    level: str,
) -> Dict[str, object]:
    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    return {
        "perturb": perturb,
        "level": level,
        "dataset": dataset_name,
        "auc": float(metrics.get("auc", float("nan"))),
        "f1": float(metrics.get("f1", float("nan"))),
        "ece": compute_ece(y_true, y_prob),
        "brier": compute_brier(y_true, y_prob),
    }


def run_robustness_suite(
    *,
    model,
    manifest_path: str | Path,
    out_dir: str | Path,
    split: str = "test",
    grid: str = "light",
    batch_size: int = 16,
    input_size: int = 224,
    backbone: str | None = None,
    save_predictions: bool = False,
) -> Path:
    if tf is None:
        raise ImportError("TensorFlow is required for robustness evaluation.")

    manifest_path = Path(manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = paths.PROJECT_ROOT / manifest_path

    out_dir = Path(out_dir)
    if not out_dir.is_absolute():
        out_dir = paths.PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    split_df = df[df["split"] == split].reset_index(drop=True)
    if split_df.empty:
        raise ValueError(f"No rows for split '{split}' in manifest.")

    preprocess_fn = get_preprocess(backbone) if backbone else None
    cfg = {}
    try:
        cfg = load_yaml(paths.CONFIG_DIR / "default.yaml")
    except FileNotFoundError:
        cfg = {}
    enabled_measures = cfg.get("enabled_focus_measures", ["lapvar", "tenengrad", "brenner", "sml"])
    focus_from_augmented = bool(cfg.get("focus_vector_from_augmented", False))

    model_inputs = model.inputs if isinstance(model.inputs, (list, tuple)) else [model.input]
    expects_focus = len(model_inputs) == 2 or (len(model_inputs) == 1 and len(model_inputs[0].shape) == 2)
    expects_image = len(model_inputs) == 2 or (len(model_inputs) == 1 and len(model_inputs[0].shape) == 4)

    threshold = 0.5
    if "val" in df["split"].unique():
        val_df = df[df["split"] == "val"].reset_index(drop=True)
        if not val_df.empty:
            if expects_focus:
                val_ds = build_feature_datasets(
                    manifest_csv=manifest_path,
                    split="val",
                    batch_size=batch_size,
                    input_size=input_size,
                    image_mode="rgb",
                    enabled_measures=enabled_measures,
                    augment_images=False,
                    shuffle=False,
                    seed=42,
                    compute_from_augmented=focus_from_augmented,
                )
                if len(model_inputs) == 1 and len(model_inputs[0].shape) == 2:
                    val_ds = val_ds.map(lambda x, y: (x[1], y), num_parallel_calls=tf.data.AUTOTUNE)
            else:
                val_ds = build_datasets(
                    manifest_csv=manifest_path,
                    split="val",
                    batch_size=batch_size,
                    input_size=input_size,
                    image_mode="rgb",
                    augment=False,
                    shuffle=False,
                    seed=42,
                    force_rgb=True,
                )
            if preprocess_fn is not None and expects_image and not expects_focus:
                val_ds = _apply_preprocess(val_ds, preprocess_fn)
            y_prob_val = _predict_probs(model, val_ds)
            threshold = choose_threshold(val_df["label"].to_numpy(), y_prob_val, metric="f1")

    rows: List[Dict[str, object]] = []
    items = _grid_preset(grid)
    feature_items = _feature_grid() if expects_focus else []

    def _build_dataset():
        if expects_focus:
            ds = build_feature_datasets(
                manifest_csv=manifest_path,
                split=split,
                batch_size=batch_size,
                input_size=input_size,
                image_mode="rgb",
                enabled_measures=enabled_measures,
                augment_images=False,
                shuffle=False,
                seed=42,
                compute_from_augmented=focus_from_augmented,
            )
            if len(model_inputs) == 1 and len(model_inputs[0].shape) == 2:
                ds = ds.map(lambda x, y: (x[1], y), num_parallel_calls=tf.data.AUTOTUNE)
            return ds
        return build_datasets(
            manifest_csv=manifest_path,
            split=split,
            batch_size=batch_size,
            input_size=input_size,
            image_mode="rgb",
            augment=False,
            shuffle=False,
            seed=42,
            force_rgb=True,
        )

    def _apply_feature(ds, kind: str, kwargs: Dict[str, float]):
        def _map(inputs, label):
            if isinstance(inputs, (tuple, list)):
                img, feat = inputs
                return (img, _apply_feature_perturb(feat, kind, kwargs)), label
            return _apply_feature_perturb(inputs, kind, kwargs), label

        return ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)

    for perturb, level, kwargs in items:
        if not expects_image and perturb != "clean":
            continue
        ds = _build_dataset()
        if expects_image and perturb != "clean":
            ds = apply_perturbation(ds, perturb, **kwargs)
        if preprocess_fn is not None and expects_image and not expects_focus:
            ds = _apply_preprocess(ds, preprocess_fn)

        y_prob = _predict_probs(model, ds)
        y_true = split_df["label"].to_numpy().astype(int)

        rows.append(_metrics_row(y_true, y_prob, threshold, "all", perturb, level))

        for dataset_name in sorted(split_df["dataset"].unique()):
            mask = split_df["dataset"] == dataset_name
            rows.append(_metrics_row(y_true[mask], y_prob[mask], threshold, dataset_name, perturb, level))

        if save_predictions:
            preds_path = out_dir / f"predictions_{perturb}_{level}.csv"
            pd.DataFrame(
                {
                    "image_path": split_df["image_path"].astype(str),
                    "y_true": y_true,
                    "y_prob": y_prob,
                    "dataset": split_df["dataset"].astype(str),
                    "perturb": perturb,
                    "level": level,
                }
            ).to_csv(preds_path, index=False)

    if feature_items:
        for label, kind, kwargs in feature_items:
            ds = _build_dataset()
            if expects_focus:
                ds = _apply_feature(ds, kind, kwargs)
            y_prob = _predict_probs(model, ds)
            y_true = split_df["label"].to_numpy().astype(int)

            rows.append(_metrics_row(y_true, y_prob, threshold, "all", label, label))
            for dataset_name in sorted(split_df["dataset"].unique()):
                mask = split_df["dataset"] == dataset_name
                rows.append(_metrics_row(y_true[mask], y_prob[mask], threshold, dataset_name, label, label))

            if save_predictions:
                preds_path = out_dir / f"predictions_{label}.csv"
                pd.DataFrame(
                    {
                        "image_path": split_df["image_path"].astype(str),
                        "y_true": y_true,
                        "y_prob": y_prob,
                        "dataset": split_df["dataset"].astype(str),
                        "perturb": label,
                        "level": label,
                    }
                ).to_csv(preds_path, index=False)

        if expects_image and expects_focus and len(model_inputs) == 2:
            for perturb, level, kwargs in items:
                if perturb == "clean":
                    continue
                for label, kind, fkwargs in feature_items:
                    ds = _build_dataset()
                    ds = apply_perturbation(ds, perturb, **kwargs)
                    ds = _apply_feature(ds, kind, fkwargs)
                    y_prob = _predict_probs(model, ds)
                    y_true = split_df["label"].to_numpy().astype(int)
                    combo_name = f"{perturb}+{label}"
                    rows.append(_metrics_row(y_true, y_prob, threshold, "all", combo_name, level))
                    for dataset_name in sorted(split_df["dataset"].unique()):
                        mask = split_df["dataset"] == dataset_name
                        rows.append(
                            _metrics_row(y_true[mask], y_prob[mask], threshold, dataset_name, combo_name, level)
                        )

    results_df = pd.DataFrame(rows)
    curves_path = out_dir / "robustness_curves.csv"
    results_df.to_csv(curves_path, index=False)

    summary_path = out_dir / "robustness_summary.json"
    save_json({"threshold": threshold, "grid": grid, "results_csv": str(curves_path)}, summary_path)
    logger.info("robustness suite complete", extra={"summary": str(summary_path)})
    return curves_path


def main(argv: List[str] | None = None) -> Path:
    args = parse_args(argv)
    if tf is None:
        raise ImportError("TensorFlow is required for robustness evaluation.")

    model = _load_model(Path(args.model_path))
    return run_robustness_suite(
        model=model,
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        split=args.split,
        grid=args.grid,
        batch_size=args.batch_size,
        input_size=args.input_size,
        backbone=args.backbone,
        save_predictions=args.save_preds,
    )


if __name__ == "__main__":
    main()
