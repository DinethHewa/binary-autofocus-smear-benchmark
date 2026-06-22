from __future__ import annotations

import argparse
import inspect
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from focus_binary import paths
from focus_binary.calib.calibration import compute_brier, compute_ece, expected_calibration_error
from focus_binary.data.balance import compute_class_weights, report_and_check_imbalance
from focus_binary.data.tfdata import build_datasets
from focus_binary.data.tfdata_features import build_feature_datasets
from focus_binary.eval.metrics import compute_metrics
from focus_binary.models.cnn_attention import build_cnn_attention
from focus_binary.models.cnn_baseline import build_cnn_baseline
from focus_binary.models.hybrid import build_hybrid_vit
from focus_binary.models.convnext import build_convnext
from focus_binary.models.focus_dnn import build_focus_dnn
from focus_binary.models.cnn_focus_hybrid import build_cnn_focus_hybrid
from focus_binary.models.transfer import build_transfer_model, get_preprocess, train_with_finetune_schedule
from focus_binary.models.vit import _CLSToken, _PositionalEmbedding, build_vit
from focus_binary.models.swin_tiny import WindowPartition, WindowReverse, build_swin_tiny
from focus_binary.robust.leakage import assert_no_leakage_manifest
from focus_binary.robust.seeds import set_global_determinism
from focus_binary.scripts.robustness_suite import run_robustness_suite
from focus_binary.utils.io import load_json, save_json, load_yaml
from focus_binary.utils.logging import get_logger
from focus_binary.uncertainty.temperature import apply_temperature, fit_temperature
from focus_binary.uncertainty.selective import risk_coverage_curve, selective_summary

logger = get_logger(__name__)

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None


if tf is not None:
    class _EpochProgressCallback(tf.keras.callbacks.Callback):
        def __init__(self, label: str, total_epochs: int):
            super().__init__()
            self.label = label
            self.total_epochs = max(1, int(total_epochs))
            self.started = 0.0

        def on_epoch_begin(self, epoch, logs=None):
            self.started = time.perf_counter()
            print(f"{self.label}: epoch {epoch + 1}/{self.total_epochs} start", flush=True)

        def on_epoch_end(self, epoch, logs=None):
            elapsed = time.perf_counter() - self.started if self.started else 0.0
            logs = logs or {}
            metric_parts = []
            for key in ("loss", "auc", "val_loss", "val_auc"):
                if key in logs:
                    try:
                        metric_parts.append(f"{key}={float(logs[key]):.4f}")
                    except Exception:
                        metric_parts.append(f"{key}={logs[key]}")
            suffix = " " + " ".join(metric_parts) if metric_parts else ""
            print(
                f"{self.label}: epoch {epoch + 1}/{self.total_epochs} done in {elapsed:.1f}s{suffix}",
                flush=True,
            )
else:
    _EpochProgressCallback = None


def _progress_callback(label: str, total_epochs: int):
    if _EpochProgressCallback is None:
        return []
    return [_EpochProgressCallback(label, total_epochs)]


def _filter_builder_kwargs(builder, hparams: Dict[str, object]) -> Dict[str, object]:
    """Drop Keras Tuner bookkeeping keys before calling builders with **kwargs."""
    signature = inspect.signature(builder)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return {k: v for k, v in hparams.items() if not str(k).startswith("tuner/")}
    allowed = set(signature.parameters)
    return {k: v for k, v in hparams.items() if k in allowed and not str(k).startswith("tuner/")}


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-seed evaluation workflow.")
    parser.add_argument(
        "--family",
        required=True,
        choices=[
            "cnn",
            "cnn_attention",
            "transfer",
            "vit",
            "hybrid_vit",
            "focus_dnn",
            "cnn_focus_hybrid",
            "convnext",
            "swin",
        ],
    )
    parser.add_argument("--manifest", required=True, help="Manifest CSV with splits")
    parser.add_argument("--runs-dir", required=True, help="Root directory containing best_model/hparams per family")
    parser.add_argument("--out-dir", required=True, help="Output directory root")
    parser.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated seed list")
    parser.add_argument(
        "--retrain",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retrain from scratch per seed (recommended).",
    )
    parser.add_argument("--phase1-epochs", type=int, default=3, help="Transfer phase1 epochs")
    parser.add_argument("--phase2-epochs", type=int, default=5, help="Transfer phase2 epochs")
    parser.add_argument("--phase2-lr-mult", type=float, default=0.1, help="Transfer phase2 LR multiplier")
    parser.add_argument(
        "--run-robustness",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run robustness suite after evaluation.",
    )
    parser.add_argument(
        "--robustness-per-seed",
        action="store_true",
        help="Run robustness for each seed instead of once.",
    )
    parser.add_argument("--robustness-grid", default="light", help="Perturbation grid preset")
    parser.add_argument("--robustness-save-preds", action="store_true", help="Save robustness predictions")
    parser.add_argument(
        "--leakage-check",
        choices=["full", "stack_sha1", "stack", "none"],
        default="full",
        help=(
            "Leakage audit mode before training. 'full' checks stack, sha1, and phash; "
            "'stack_sha1' checks stack and exact image hashes; 'stack' checks split groups only."
        ),
    )
    return parser.parse_args(argv)


def _load_model(model_path: Path):
    if tf is None:
        raise ImportError("TensorFlow is required for evaluation.")
    custom_objects = {}
    if _CLSToken is not None:
        custom_objects["_CLSToken"] = _CLSToken
    if _PositionalEmbedding is not None:
        custom_objects["_PositionalEmbedding"] = _PositionalEmbedding
    if WindowPartition is not None:
        custom_objects["WindowPartition"] = WindowPartition
    if WindowReverse is not None:
        custom_objects["WindowReverse"] = WindowReverse
    try:
        from focus_binary.models.convnext import ConvNeXtPreprocess  # type: ignore
    except Exception:
        ConvNeXtPreprocess = None
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


def _predict_with_latency(model, ds, n_samples: int) -> Tuple[np.ndarray, float]:
    start = time.perf_counter()
    preds = model.predict(ds, verbose=0)
    elapsed = time.perf_counter() - start
    y_prob = _as_probabilities(preds)
    latency_ms = (elapsed / max(n_samples, 1)) * 1000.0
    return y_prob, latency_ms


def _split_transfer_epochs(best_epoch: int, phase1_epochs: int) -> Tuple[int, int]:
    if best_epoch <= 0:
        return max(1, phase1_epochs), 0
    if best_epoch <= phase1_epochs:
        return best_epoch, 0
    return phase1_epochs, best_epoch - phase1_epochs


def _get_val_auc(history) -> List[float]:
    if history is None:
        return []
    if isinstance(history, dict):
        values = history.get("val_auc")
    else:
        values = history.history.get("val_auc")
    if values:
        return [float(v) for v in values]
    if isinstance(history, dict):
        return []
    for key, vals in history.history.items():
        if key.startswith("val_") and vals:
            return [float(v) for v in vals]
    return []


def _build_model(family: str, hparams: Dict[str, object], input_size: int):
    input_shape = (input_size, input_size, 3)
    if family == "cnn":
        return build_cnn_baseline(input_shape=input_shape, **_filter_builder_kwargs(build_cnn_baseline, hparams)), None, False
    if family == "cnn_attention":
        attention_type = str(hparams.get("attention_type", "se"))
        return build_cnn_attention(hparams, input_shape=input_shape, attention_type=attention_type), None, False
    if family == "vit":
        return build_vit(input_shape=input_shape, **_filter_builder_kwargs(build_vit, hparams)), None, False
    if family == "hybrid_vit":
        return build_hybrid_vit(input_shape=input_shape, **_filter_builder_kwargs(build_hybrid_vit, hparams)), None, False
    if family == "focus_dnn":
        return build_focus_dnn(hparams, input_dim=int(hparams.get("focus_dim", 0) or 1)), None, False
    if family == "cnn_focus_hybrid":
        focus_dim = int(hparams.get("focus_dim", 0) or 1)
        return build_cnn_focus_hybrid(hparams, input_shape=input_shape, focus_dim=focus_dim), None, False
    if family == "transfer":
        spec = build_transfer_model(
            backbone=str(hparams.get("backbone", "MobileNetV2")),
            input_size=input_size,
            pooling=str(hparams.get("pooling", "avg")),
            head_units=int(hparams.get("head_units", 0)),
            dropout=float(hparams.get("dropout", 0.0)),
            base_trainable=int(hparams.get("base_trainable_blocks", 0)),
            lr=float(hparams.get("lr", 1e-3)),
            label_smoothing=float(hparams.get("label_smoothing", 0.0)),
            weights=str(hparams.get("weights", "imagenet")),
        )
        return spec["model"], spec.get("preprocess_input"), bool(spec.get("preprocess_in_model", False)), spec
    if family == "convnext":
        return build_convnext(input_shape=input_shape, **_filter_builder_kwargs(build_convnext, hparams)), None, False
    if family == "swin":
        return build_swin_tiny(input_shape=input_shape, **_filter_builder_kwargs(build_swin_tiny, hparams)), None, False
    raise ValueError(f"Unsupported family: {family}")


def _apply_preprocess(ds, preprocess_fn):
    def _map(img, label):
        return preprocess_fn(img * 255.0), label

    return ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)


def _metrics_with_calibration(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    brier = float(np.mean((y_prob - y_true) ** 2)) if len(y_true) else float("nan")
    return {
        "auc": float(metrics.get("auc", float("nan"))),
        "f1": float(metrics.get("f1", float("nan"))),
        "acc": float(metrics.get("accuracy", float("nan"))),
        "precision": float(metrics.get("precision", float("nan"))),
        "recall": float(metrics.get("recall", float("nan"))),
        "fp": float(metrics.get("fp", 0.0)),
        "fn": float(metrics.get("fn", 0.0)),
        "ece": float(ece),
        "brier": float(brier),
    }


def _bootstrap_ci(values: np.ndarray, n_resamples: int = 2000, seed: int = 42) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_resamples, len(values)), replace=True)
    means = samples.mean(axis=1)
    ci_low, ci_high = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def _summarize_metrics(metrics_df: pd.DataFrame) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    metrics_cols = [
        "auc",
        "f1",
        "acc",
        "precision",
        "recall",
        "fp",
        "fn",
        "ece",
        "brier",
        "params_count",
        "latency_ms",
    ]
    for (dataset, split), group in metrics_df.groupby(["dataset", "split"]):
        dataset_key = f"{dataset}:{split}"
        summary[dataset_key] = {}
        for metric in metrics_cols:
            summary[dataset_key][metric] = _bootstrap_ci(group[metric].to_numpy())
    return summary


def _train_phase_a(
    family: str,
    hparams: Dict[str, object],
    input_size: int,
    train_ds,
    val_ds,
    class_weight: Dict[int, float] | None,
    epochs: int,
    phase1_epochs: int,
    phase2_epochs: int,
    phase2_lr_mult: float,
) -> Tuple[object, int, Dict[str, object]]:
    if family == "transfer":
        model, preprocess_fn, preprocess_in_model, spec = _build_model(family, hparams, input_size)
        if preprocess_fn is not None and not preprocess_in_model:
            train_ds = _apply_preprocess(train_ds, preprocess_fn)
            val_ds = _apply_preprocess(val_ds, preprocess_fn)
        histories = train_with_finetune_schedule(
            model=model,
            backbone_model=spec["backbone_model"],
            train_ds=train_ds,
            val_ds=val_ds,
            phase1_epochs=phase1_epochs,
            phase2_epochs=phase2_epochs,
            phase1_lr=float(hparams.get("lr", 1e-3)),
            phase2_lr=float(hparams.get("lr", 1e-3)) * phase2_lr_mult,
            base_trainable_blocks=int(hparams.get("base_trainable_blocks", 0)),
            backbone_name=spec["backbone_name"],
            label_smoothing=float(hparams.get("label_smoothing", 0.0)),
            callbacks=_progress_callback(f"{family} phase-a", phase1_epochs + phase2_epochs),
            class_weight=class_weight,
        )
        val_scores = _get_val_auc(histories.get("phase1")) + _get_val_auc(histories.get("phase2"))
        best_epoch = int(np.argmax(val_scores) + 1) if val_scores else phase1_epochs + phase2_epochs
        return model, best_epoch, spec

    model, preprocess_fn, preprocess_in_model = _build_model(family, hparams, input_size)
    if preprocess_fn is not None and not preprocess_in_model:
        train_ds = _apply_preprocess(train_ds, preprocess_fn)
        val_ds = _apply_preprocess(val_ds, preprocess_fn)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(epochs),
        verbose=0,
        callbacks=_progress_callback(f"{family} phase-a", int(epochs)),
        class_weight=class_weight,
    )
    val_scores = _get_val_auc(history)
    best_epoch = int(np.argmax(val_scores) + 1) if val_scores else int(epochs)
    return model, best_epoch, {}


def _train_final(
    family: str,
    hparams: Dict[str, object],
    input_size: int,
    trainval_ds,
    class_weight: Dict[int, float] | None,
    best_epoch: int,
    phase1_epochs: int,
    phase2_epochs: int,
    phase2_lr_mult: float,
) -> object:
    if family == "transfer":
        model, preprocess_fn, preprocess_in_model, spec = _build_model(family, hparams, input_size)
        if preprocess_fn is not None and not preprocess_in_model:
            trainval_ds = _apply_preprocess(trainval_ds, preprocess_fn)
        p1, p2 = _split_transfer_epochs(best_epoch, phase1_epochs)
        train_with_finetune_schedule(
            model=model,
            backbone_model=spec["backbone_model"],
            train_ds=trainval_ds,
            val_ds=None,
            phase1_epochs=p1,
            phase2_epochs=p2,
            phase1_lr=float(hparams.get("lr", 1e-3)),
            phase2_lr=float(hparams.get("lr", 1e-3)) * phase2_lr_mult,
            base_trainable_blocks=int(hparams.get("base_trainable_blocks", 0)),
            backbone_name=spec["backbone_name"],
            label_smoothing=float(hparams.get("label_smoothing", 0.0)),
            callbacks=_progress_callback(f"{family} final", max(1, p1 + p2)),
            class_weight=class_weight,
        )
        return model

    model, preprocess_fn, preprocess_in_model = _build_model(family, hparams, input_size)
    if preprocess_fn is not None and not preprocess_in_model:
        trainval_ds = _apply_preprocess(trainval_ds, preprocess_fn)
    model.fit(
        trainval_ds,
        epochs=best_epoch,
        verbose=0,
        callbacks=_progress_callback(f"{family} final", int(best_epoch)),
        class_weight=class_weight,
    )
    return model


def _prepare_trainval_manifest(df: pd.DataFrame, dest: Path) -> Path:
    trainval = df[df["split"].isin(["train", "val"])].copy()
    trainval["split"] = "train"
    dest.parent.mkdir(parents=True, exist_ok=True)
    trainval.to_csv(dest, index=False)
    return dest


def main(argv: List[str] | None = None) -> Path:
    args = parse_args(argv)
    if tf is None:
        raise ImportError("TensorFlow is required for multi-seed evaluation.")

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = paths.PROJECT_ROOT / manifest_path

    print(f"Running leakage check ({args.leakage_check}) on {manifest_path}", flush=True)
    if args.leakage_check == "full":
        assert_no_leakage_manifest(manifest_path)
    elif args.leakage_check == "stack_sha1":
        assert_no_leakage_manifest(manifest_path, check_sha1=True, check_phash=False)
    elif args.leakage_check == "stack":
        assert_no_leakage_manifest(manifest_path, check_sha1=False, check_phash=False)
    else:
        print("Leakage check skipped by --leakage-check none", flush=True)

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = paths.PROJECT_ROOT / runs_dir

    family_dir = runs_dir / args.family
    best_model_path = family_dir / "best_model.keras"
    best_hparams_path = family_dir / "best_hparams.json"
    summary_path = family_dir / "summary.json"

    if not best_hparams_path.exists():
        raise FileNotFoundError(f"Missing best_hparams.json at {best_hparams_path}")

    hparams = load_json(best_hparams_path)
    summary = load_json(summary_path) if summary_path.exists() else {}

    input_size = int(summary.get("input_size", 224))
    batch_size = int(summary.get("batch_size", 16))
    train_epochs = int(summary.get("epochs", 10))

    cfg = {}
    try:
        cfg = load_yaml(paths.CONFIG_DIR / "default.yaml")
    except FileNotFoundError:
        cfg = {}
    enabled_measures = cfg.get("enabled_focus_measures", ["lapvar", "tenengrad", "brenner", "sml"])
    focus_from_augmented = bool(cfg.get("focus_vector_from_augmented", False))
    if args.family in {"focus_dnn", "cnn_focus_hybrid"}:
        hparams["focus_dim"] = len(enabled_measures)

    df = pd.read_csv(manifest_path)
    _, extreme = report_and_check_imbalance(df)
    class_weight = compute_class_weights(df, split="train") if extreme else None

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = paths.PROJECT_ROOT / out_dir
    family_out = out_dir / args.family
    family_out.mkdir(parents=True, exist_ok=True)

    metrics_rows: List[Dict[str, object]] = []
    all_predictions: List[pd.DataFrame] = []
    calibration_rows: List[Dict[str, object]] = []
    selective_rows: List[Dict[str, object]] = []

    last_model = None
    backbone_name = str(hparams.get("backbone", "MobileNetV2")) if args.family == "transfer" else None

    for seed in seeds:
        print(f"Starting multiseed job: family={args.family} seed={seed}", flush=True)
        set_global_determinism(seed)
        seed_dir = family_out / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        if args.family in {"focus_dnn", "cnn_focus_hybrid"}:
            train_ds = build_feature_datasets(
                manifest_csv=manifest_path,
                split="train",
                batch_size=batch_size,
                input_size=input_size,
                image_mode="rgb",
                enabled_measures=enabled_measures,
                augment_images=False,
                shuffle=True,
                seed=seed,
                compute_from_augmented=focus_from_augmented,
            )
            val_ds = build_feature_datasets(
                manifest_csv=manifest_path,
                split="val",
                batch_size=batch_size,
                input_size=input_size,
                image_mode="rgb",
                enabled_measures=enabled_measures,
                augment_images=False,
                shuffle=False,
                seed=seed,
                compute_from_augmented=focus_from_augmented,
            )
            test_ds = build_feature_datasets(
                manifest_csv=manifest_path,
                split="test",
                batch_size=batch_size,
                input_size=input_size,
                image_mode="rgb",
                enabled_measures=enabled_measures,
                augment_images=False,
                shuffle=False,
                seed=seed,
                compute_from_augmented=focus_from_augmented,
            )
            if args.family == "focus_dnn":
                train_ds = train_ds.map(lambda x, y: (x[1], y), num_parallel_calls=tf.data.AUTOTUNE)
                val_ds = val_ds.map(lambda x, y: (x[1], y), num_parallel_calls=tf.data.AUTOTUNE)
                test_ds = test_ds.map(lambda x, y: (x[1], y), num_parallel_calls=tf.data.AUTOTUNE)
        else:
            train_ds = build_datasets(
                manifest_csv=manifest_path,
                split="train",
                batch_size=batch_size,
                input_size=input_size,
                image_mode="rgb",
                augment=False,
                shuffle=True,
                seed=seed,
                force_rgb=True,
            )
            val_ds = build_datasets(
                manifest_csv=manifest_path,
                split="val",
                batch_size=batch_size,
                input_size=input_size,
                image_mode="rgb",
                augment=False,
                shuffle=False,
                seed=seed,
                force_rgb=True,
            )
            test_ds = build_datasets(
                manifest_csv=manifest_path,
                split="test",
                batch_size=batch_size,
                input_size=input_size,
                image_mode="rgb",
                augment=False,
                shuffle=False,
                seed=seed,
                force_rgb=True,
            )

        seed_train_start = time.perf_counter()
        if args.retrain:
            _, best_epoch, _ = _train_phase_a(
                family=args.family,
                hparams=hparams,
                input_size=input_size,
                train_ds=train_ds,
                val_ds=val_ds,
                class_weight=class_weight,
                epochs=train_epochs,
                phase1_epochs=args.phase1_epochs,
                phase2_epochs=args.phase2_epochs,
                phase2_lr_mult=args.phase2_lr_mult,
            )

            trainval_manifest = _prepare_trainval_manifest(df, seed_dir / "trainval_manifest.csv")
            if args.family in {"focus_dnn", "cnn_focus_hybrid"}:
                trainval_ds = build_feature_datasets(
                    manifest_csv=trainval_manifest,
                    split="train",
                    batch_size=batch_size,
                    input_size=input_size,
                    image_mode="rgb",
                    enabled_measures=enabled_measures,
                    augment_images=False,
                    shuffle=True,
                    seed=seed,
                    compute_from_augmented=focus_from_augmented,
                )
                if args.family == "focus_dnn":
                    trainval_ds = trainval_ds.map(lambda x, y: (x[1], y), num_parallel_calls=tf.data.AUTOTUNE)
            else:
                trainval_ds = build_datasets(
                    manifest_csv=trainval_manifest,
                    split="train",
                    batch_size=batch_size,
                    input_size=input_size,
                    image_mode="rgb",
                    augment=False,
                    shuffle=True,
                    seed=seed,
                    force_rgb=True,
                )

            model = _train_final(
                family=args.family,
                hparams=hparams,
                input_size=input_size,
                trainval_ds=trainval_ds,
                class_weight=class_weight,
                best_epoch=best_epoch,
                phase1_epochs=args.phase1_epochs,
                phase2_epochs=args.phase2_epochs,
                phase2_lr_mult=args.phase2_lr_mult,
            )
        else:
            if not best_model_path.exists():
                raise FileNotFoundError(f"Missing best model at {best_model_path}")
            model = _load_model(best_model_path)
        training_walltime_s = time.perf_counter() - seed_train_start if args.retrain else 0.0

        preprocess_fn = None
        if args.family == "transfer":
            preprocess_fn = get_preprocess(backbone_name)

        if preprocess_fn is not None:
            test_ds = _apply_preprocess(test_ds, preprocess_fn)

        test_df = df[df["split"] == "test"].reset_index(drop=True)
        val_df = df[df["split"] == "val"].reset_index(drop=True)
        y_true = test_df["label"].to_numpy().astype(int)
        y_prob, latency_ms = _predict_with_latency(model, test_ds, n_samples=len(test_df))

        val_probs = None
        temperature = 1.0
        if not val_df.empty:
            val_ds_pred = val_ds
            if preprocess_fn is not None:
                val_ds_pred = _apply_preprocess(val_ds_pred, preprocess_fn)
            val_preds = model.predict(val_ds_pred, verbose=0)
            val_probs = _as_probabilities(val_preds)
            if len(val_probs) == len(val_df):
                temperature = fit_temperature(val_df["label"].to_numpy().astype(int), val_probs)

        y_prob_cal = apply_temperature(y_prob, temperature) if temperature != 1.0 else y_prob

        preds_df = pd.DataFrame(
            {
                "seed": seed,
                "image_path": test_df["image_path"].astype(str),
                "y_true": y_true,
                "y_prob": y_prob,
                "dataset": test_df["dataset"].astype(str),
            }
        )
        preds_path = seed_dir / "predictions.csv"
        preds_df.to_csv(preds_path, index=False)
        all_predictions.append(preds_df)

        params_count = int(model.count_params())
        overall_metrics = _metrics_with_calibration(y_true, y_prob)
        overall_row = {
            "seed": seed,
            "dataset": "all",
            "split": "test",
            **overall_metrics,
            "params_count": params_count,
            "latency_ms": float(latency_ms),
        }
        metrics_rows.append(overall_row)

        if val_probs is not None and len(val_probs) == len(val_df):
            val_true = val_df["label"].to_numpy().astype(int)
            val_probs_cal = apply_temperature(val_probs, temperature) if temperature != 1.0 else val_probs
            calibration_rows.append(
                {
                    "seed": seed,
                    "split": "val",
                    "ece_before": compute_ece(val_true, val_probs),
                    "brier_before": compute_brier(val_true, val_probs),
                    "ece_after": compute_ece(val_true, val_probs_cal),
                    "brier_after": compute_brier(val_true, val_probs_cal),
                    "temperature": float(temperature),
                }
            )
            calibration_rows.append(
                {
                    "seed": seed,
                    "split": "test",
                    "ece_before": compute_ece(y_true, y_prob),
                    "brier_before": compute_brier(y_true, y_prob),
                    "ece_after": compute_ece(y_true, y_prob_cal),
                    "brier_after": compute_brier(y_true, y_prob_cal),
                    "temperature": float(temperature),
                }
            )

            for mode in ("tau", "delta"):
                curve = risk_coverage_curve(y_true, y_prob_cal, mode=mode)
                for row in curve:
                    selective_rows.append(
                        {
                            "seed": seed,
                            "split": "test",
                            "mode": mode,
                            "threshold": row["threshold"],
                            "coverage": row["coverage"],
                            "risk": row["risk"],
                            "auc": row["auc"],
                            "f1": row["f1"],
                            "temperature": float(temperature),
                        }
                    )
            summary_tau = selective_summary(y_true, y_prob_cal, tau=0.9)
            summary_delta = selective_summary(y_true, y_prob_cal, delta=0.1)
            selective_rows.append(
                {
                    "seed": seed,
                    "split": "test",
                    "mode": "tau_summary",
                    "threshold": 0.9,
                    "coverage": summary_tau["coverage"],
                    "risk": summary_tau["risk"],
                    "auc": summary_tau["auc"],
                    "f1": summary_tau["f1"],
                    "temperature": float(temperature),
                }
            )
            selective_rows.append(
                {
                    "seed": seed,
                    "split": "test",
                    "mode": "delta_summary",
                    "threshold": 0.1,
                    "coverage": summary_delta["coverage"],
                    "risk": summary_delta["risk"],
                    "auc": summary_delta["auc"],
                    "f1": summary_delta["f1"],
                    "temperature": float(temperature),
                }
            )

        per_dataset_metrics: Dict[str, Dict[str, float]] = {}
        for dataset_name in sorted(test_df["dataset"].unique()):
            mask = test_df["dataset"] == dataset_name
            metrics = _metrics_with_calibration(y_true[mask], y_prob[mask])
            per_dataset_metrics[dataset_name] = metrics
            metrics_rows.append(
                {
                    "seed": seed,
                    "dataset": dataset_name,
                    "split": "test",
                    **metrics,
                    "params_count": params_count,
                    "latency_ms": float(latency_ms),
                }
            )

        seed_metrics_path = seed_dir / "metrics.json"
        save_json(
            {
                "seed": seed,
                "params_count": params_count,
                "latency_ms": float(latency_ms),
                "training_walltime_s": float(training_walltime_s),
                "overall": overall_metrics,
                "per_dataset": per_dataset_metrics,
            },
            seed_metrics_path,
        )
        logger.info(
            "seed complete",
            extra={"seed": seed, "metrics": overall_metrics, "training_walltime_s": round(training_walltime_s, 2)},
        )
        last_model = model

        if args.run_robustness and args.robustness_per_seed:
            robustness_out = seed_dir / "robustness"
            run_robustness_suite(
                model=model,
                manifest_path=manifest_path,
                out_dir=robustness_out,
                split="test",
                grid=args.robustness_grid,
                batch_size=batch_size,
                input_size=input_size,
                backbone=backbone_name,
                save_predictions=args.robustness_save_preds,
            )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = family_out / "multiseed_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    preds_all = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    preds_all_path = family_out / "predictions_all.csv"
    preds_all.to_csv(preds_all_path, index=False)

    if calibration_rows:
        calibration_df = pd.DataFrame(calibration_rows)
        calibration_df.to_csv(family_out / "calibration_before_after.csv", index=False)
    if selective_rows:
        selective_df = pd.DataFrame(selective_rows)
        selective_df.to_csv(family_out / "selective_metrics.csv", index=False)

    summary = {
        "family": args.family,
        "seeds": seeds,
        "metrics": _summarize_metrics(metrics_df),
    }
    summary_path = family_out / "summary.json"
    save_json(summary, summary_path)

    if args.run_robustness and not args.robustness_per_seed:
        if last_model is None:
            last_model = _load_model(best_model_path)
        robustness_out = family_out / "robustness"
        run_robustness_suite(
            model=last_model,
            manifest_path=manifest_path,
            out_dir=robustness_out,
            split="test",
            grid=args.robustness_grid,
            batch_size=batch_size,
            input_size=input_size,
            backbone=backbone_name,
            save_predictions=args.robustness_save_preds,
        )

    logger.info("multi-seed evaluation complete", extra={"out_dir": str(family_out)})
    return family_out


if __name__ == "__main__":
    main()
