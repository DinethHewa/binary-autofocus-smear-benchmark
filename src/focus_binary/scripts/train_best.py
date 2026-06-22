from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from focus_binary import paths
from focus_binary.data.balance import compute_class_weights, report_and_check_imbalance
from focus_binary.data.splits import assert_no_leak
from focus_binary.data.tfdata import build_datasets
from focus_binary.data.tfdata_features import build_feature_datasets
from focus_binary.eval.evaluate import evaluate_model
from focus_binary.models.cnn_baseline import build_cnn_baseline
from focus_binary.models.cnn_attention import build_cnn_attention
from focus_binary.models.focus_dnn import build_focus_dnn
from focus_binary.models.cnn_focus_hybrid import build_cnn_focus_hybrid
from focus_binary.models.hybrid import build_hybrid_vit
from focus_binary.models.convnext import build_convnext
from focus_binary.models.transfer import build_transfer_model, train_with_finetune_schedule
from focus_binary.models.vit import build_vit
from focus_binary.models.swin_tiny import build_swin_tiny
from focus_binary.robust.leakage import assert_no_leakage_manifest
from focus_binary.utils.io import load_json, save_json, save_model, load_yaml
from focus_binary.utils.logging import get_logger
from focus_binary.utils.seed import set_global_seed
from focus_binary.utils.efficiency import count_params, hardware_string, measure_latency

logger = get_logger(__name__)

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train final model from best hyperparameters.")
    parser.add_argument("--family", required=True, help="Model family to train")
    parser.add_argument("--best-hparams", required=True, help="Path to best_hparams.json")
    parser.add_argument("--manifest", required=True, help="Manifest CSV with splits")
    parser.add_argument("--output-dir", default=str(paths.ARTIFACT_DIR / "final"), help="Where to write final model")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs for final training (non-transfer)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--input-size", type=int, default=224, help="Square input size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--augment", action="store_true", help="Enable lightweight augmentation")
    parser.add_argument("--phase1-epochs", type=int, default=3, help="Transfer phase 1 epochs")
    parser.add_argument("--phase2-epochs", type=int, default=5, help="Transfer phase 2 epochs")
    parser.add_argument("--phase1-lr", type=float, default=1e-3, help="Transfer phase 1 LR")
    parser.add_argument("--phase2-lr-mult", type=float, default=0.1, help="Transfer phase 2 LR multiplier")
    return parser.parse_args(argv)


def _write_trainval_manifest(df: pd.DataFrame, dest: Path) -> Path:
    trainval = df[df["split"].isin(["train", "val"])].copy()
    trainval["split"] = "train"
    dest.parent.mkdir(parents=True, exist_ok=True)
    trainval.to_csv(dest, index=False)
    return dest


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    if tf is None:
        raise ImportError("TensorFlow is required for training.")

    set_global_seed(args.seed)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = paths.PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    hparams = load_json(Path(args.best_hparams))
    input_size = int(hparams.get("input_size", args.input_size))
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = paths.PROJECT_ROOT / manifest_path

    df = pd.read_csv(manifest_path)
    assert_no_leakage_manifest(manifest_path)
    assert_no_leak(df, group_col="stack_id", split_col="split")
    _, extreme = report_and_check_imbalance(df)

    trainval_manifest = _write_trainval_manifest(df, output_dir / "trainval_manifest.csv")
    test_df = df[df["split"] == "test"].copy()

    cfg = {}
    try:
        cfg = load_yaml(paths.CONFIG_DIR / "default.yaml")
    except FileNotFoundError:
        cfg = {}
    enabled_measures = cfg.get("enabled_focus_measures", ["lapvar", "tenengrad", "brenner", "sml"])
    focus_from_augmented = bool(cfg.get("focus_vector_from_augmented", False))

    if args.family in {"focus_dnn", "cnn_focus_hybrid"}:
        train_ds = build_feature_datasets(
            manifest_csv=trainval_manifest,
            split="train",
            batch_size=args.batch_size,
            input_size=input_size,
            image_mode="rgb",
            enabled_measures=enabled_measures,
            augment_images=args.augment,
            shuffle=True,
            seed=args.seed,
            compute_from_augmented=focus_from_augmented,
        )
        test_ds = build_feature_datasets(
            manifest_csv=manifest_path,
            split="test",
            batch_size=args.batch_size,
            input_size=input_size,
            image_mode="rgb",
            enabled_measures=enabled_measures,
            augment_images=False,
            shuffle=False,
            seed=args.seed,
            compute_from_augmented=focus_from_augmented,
        )
        if args.family == "focus_dnn":
            train_ds = train_ds.map(lambda x, y: (x[1], y), num_parallel_calls=tf.data.AUTOTUNE)
            test_ds = test_ds.map(lambda x, y: (x[1], y), num_parallel_calls=tf.data.AUTOTUNE)
        focus_dim = len(enabled_measures)
    else:
        train_ds = build_datasets(
            manifest_csv=trainval_manifest,
            split="train",
            batch_size=args.batch_size,
            input_size=input_size,
            image_mode="rgb",
            augment=args.augment,
            shuffle=True,
            seed=args.seed,
            force_rgb=True,
        )
        test_ds = build_datasets(
            manifest_csv=manifest_path,
            split="test",
            batch_size=args.batch_size,
            input_size=input_size,
            image_mode="rgb",
            augment=False,
            shuffle=False,
            seed=args.seed,
            force_rgb=True,
        )
        focus_dim = 0

    class_weight = None
    if extreme:
        trainval_df = pd.read_csv(trainval_manifest)
        class_weight = compute_class_weights(trainval_df, split="train")

    model = None
    preprocess_fn = None
    preprocess_in_model = False

    train_start = time.perf_counter()

    if args.family == "cnn":
        model = build_cnn_baseline(input_shape=(input_size, input_size, 3), **hparams)
    elif args.family == "cnn_attention":
        attention_type = hparams.get("attention_type", "se")
        model = build_cnn_attention(hparams, input_shape=(input_size, input_size, 3), attention_type=attention_type)
    elif args.family == "vit":
        model = build_vit(input_shape=(input_size, input_size, 3), **hparams)
    elif args.family == "hybrid_vit":
        model = build_hybrid_vit(input_shape=(input_size, input_size, 3), **hparams)
    elif args.family == "focus_dnn":
        model = build_focus_dnn(hparams, input_dim=focus_dim)
    elif args.family == "cnn_focus_hybrid":
        model = build_cnn_focus_hybrid(hparams, input_shape=(input_size, input_size, 3), focus_dim=focus_dim)
    elif args.family == "transfer":
        spec = build_transfer_model(
            backbone=hparams.get("backbone", "MobileNetV2"),
            input_size=input_size,
            pooling=hparams.get("pooling", "avg"),
            head_units=hparams.get("head_units", 0),
            dropout=hparams.get("dropout", 0.0),
            base_trainable=hparams.get("base_trainable_blocks", 0),
            lr=hparams.get("lr", args.phase1_lr),
            label_smoothing=hparams.get("label_smoothing", 0.0),
            weights=hparams.get("weights", "imagenet"),
        )
        model = spec["model"]
        preprocess_fn = spec.get("preprocess_input")
        preprocess_in_model = bool(spec.get("preprocess_in_model", False))

        if preprocess_fn is not None and not preprocess_in_model:
            def _apply_preprocess(img, label):
                return preprocess_fn(img * 255.0), label

            train_ds = train_ds.map(_apply_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            test_ds = test_ds.map(_apply_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

        train_with_finetune_schedule(
            model=model,
            backbone_model=spec["backbone_model"],
            train_ds=train_ds,
            val_ds=None,
            phase1_epochs=args.phase1_epochs,
            phase2_epochs=args.phase2_epochs,
            phase1_lr=args.phase1_lr,
            phase2_lr=args.phase1_lr * args.phase2_lr_mult,
            base_trainable_blocks=hparams.get("base_trainable_blocks", 0),
            backbone_name=spec["backbone_name"],
            label_smoothing=hparams.get("label_smoothing", 0.0),
            callbacks=None,
            class_weight=class_weight,
        )
    elif args.family == "convnext":
        model = build_convnext(input_shape=(input_size, input_size, 3), **hparams)
    elif args.family == "swin":
        model = build_swin_tiny(input_shape=(input_size, input_size, 3), **hparams)
    else:
        raise ValueError(f"Unsupported family: {args.family}")

    if args.family != "transfer":
        model.fit(train_ds, epochs=args.epochs, verbose=1, class_weight=class_weight)
    training_walltime_s = time.perf_counter() - train_start

    final_model_path = output_dir / "final_model.keras"
    save_model(model, final_model_path)

    preds_path = output_dir / "final_predictions.csv"
    eval_result = evaluate_model(model, (test_ds, test_df), threshold=0.5, preds_path=preds_path)
    summary_path = output_dir / "final_summary.json"
    params_count = count_params(model)
    latency_mean = None
    latency_p95 = None
    try:
        latency_mean, latency_p95 = measure_latency(model, input_size=input_size, batch_size=1)
    except Exception as exc:
        logger.warning("latency measurement failed", extra={"error": str(exc)})

    save_json(
        {
            "family": args.family,
            "input_size": input_size,
            "best_hparams": hparams,
            "final_model": str(final_model_path),
            "metrics": eval_result.get("overall", {}),
            "predictions": str(preds_path),
            "params_count": params_count,
            "latency_ms_mean": latency_mean,
            "latency_ms_p95": latency_p95,
            "tuning_walltime_s": 0.0,
            "training_walltime_s": float(training_walltime_s),
            "hardware": hardware_string(),
        },
        summary_path,
    )

    logger.info(
        "Final training complete",
        extra={
            "model": str(final_model_path),
            "summary": str(summary_path),
            "training_walltime_s": round(training_walltime_s, 2),
        },
    )
    return final_model_path


if __name__ == "__main__":
    main()
