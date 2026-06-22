from __future__ import annotations

import argparse
import inspect
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from focus_binary import paths
from focus_binary.calib.calibration import choose_threshold, compute_brier, compute_ece
from focus_binary.classical_ml.models import build_classical_models, compute_focus_vectors, predict_probabilities
from focus_binary.data.balance import compute_class_weights, report_and_check_imbalance
from focus_binary.data.splits import assert_no_leak
from focus_binary.data.tfdata import build_datasets
from focus_binary.data.tfdata_features import build_feature_datasets
from focus_binary.eval.metrics import compute_metrics
from focus_binary.models.cnn_attention import build_cnn_attention
from focus_binary.models.cnn_baseline import build_cnn_baseline
from focus_binary.models.cnn_focus_hybrid import build_cnn_focus_hybrid
from focus_binary.models.focus_dnn import build_focus_dnn
from focus_binary.models.hybrid import build_hybrid_vit
from focus_binary.models.transfer import build_transfer_model, train_with_finetune_schedule
from focus_binary.models.vit import build_vit
from focus_binary.robust.leakage import assert_no_leakage_manifest
from focus_binary.robust.seeds import set_global_determinism
from focus_binary.utils.io import load_json, load_yaml, save_json
from focus_binary.utils.logging import get_logger
from focus_binary.baselines.threshold import build_composite_scores, select_threshold

logger = get_logger(__name__)

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None

try:
    from sklearn.utils.class_weight import compute_sample_weight
except Exception:  # pragma: no cover
    compute_sample_weight = None


DEFAULT_FAMILIES = [
    "cnn",
    "cnn_attention",
    "transfer",
    "vit",
    "hybrid_vit",
    "focus_dnn",
    "cnn_focus_hybrid",
    "classical_ml",
    "threshold_baselines",
]


def _filter_builder_kwargs(builder, hparams: Dict[str, object]) -> Dict[str, object]:
    """Drop Keras Tuner bookkeeping keys before calling builders with **kwargs."""
    signature = inspect.signature(builder)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return {k: v for k, v in hparams.items() if not str(k).startswith("tuner/")}
    allowed = set(signature.parameters)
    return {k: v for k, v in hparams.items() if k in allowed and not str(k).startswith("tuner/")}


def _str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    val = value.strip().lower()
    if val in {"true", "1", "yes", "y"}:
        return True
    if val in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {value}")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leave-one-dataset-out evaluation.")
    parser.add_argument(
        "--families",
        default=",".join(DEFAULT_FAMILIES),
        help="Comma-separated list of families",
    )
    parser.add_argument("--family", default=None, help="(Deprecated) Single family")
    parser.add_argument("--manifest", required=True, help="Manifest CSV with splits")
    parser.add_argument("--runs-dir", default="./runs", help="Runs directory with best_hparams.json")
    parser.add_argument("--out-dir", default="./reports/lodo", help="Output root for LODO")
    parser.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated seed list")
    parser.add_argument("--heldout", default="all", help="Heldout dataset name or 'all'")
    parser.add_argument("--use-best-hparams", type=_str2bool, default=True)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--early-stop", type=_str2bool, default=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    return parser.parse_args(argv)


def _parse_families(args: argparse.Namespace) -> List[str]:
    if args.family:
        return [args.family.strip()]
    return [f.strip() for f in args.families.split(",") if f.strip()]


def _resolve_path(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = paths.PROJECT_ROOT / p
    return p


def _apply_preprocess(ds, preprocess_fn):
    def _map(img, label):
        return preprocess_fn(img * 255.0), label

    return ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)


def _predict_probs(model, ds) -> np.ndarray:
    preds = model.predict(ds, verbose=0)
    preds = np.asarray(preds)
    if preds.ndim > 1:
        if preds.shape[-1] == 1:
            preds = preds.reshape(-1)
        elif preds.shape[-1] >= 2:
            preds = preds[..., 1]
    return preds.reshape(-1).astype(float)


def _resolve_input_size(hparams: Dict[str, object], args: argparse.Namespace, cfg: Dict[str, object]) -> int:
    if args.input_size:
        return int(args.input_size)
    if "input_size" in hparams:
        return int(hparams["input_size"])
    model_cfg = cfg.get("model") or {}
    if "input_size" in model_cfg:
        return int(model_cfg["input_size"])
    img_size = cfg.get("img_size")
    if isinstance(img_size, (list, tuple)) and len(img_size) >= 2:
        return int(img_size[0])
    return 224


def _prepare_lodo_manifest(df: pd.DataFrame, heldout: str) -> pd.DataFrame:
    train_df = df[(df["dataset"] != heldout) & (df["split"] == "train")].copy()
    val_df = df[(df["dataset"] != heldout) & (df["split"] == "val")].copy()
    test_df = df[(df["dataset"] == heldout) & (df["split"] == "test")].copy()

    if test_df.empty:
        test_df = df[df["dataset"] == heldout].copy()
        test_df["split"] = "test"

    if val_df.empty:
        logger.warning("val split empty for LODO; using train as validation", extra={"heldout": heldout})
        val_df = train_df.copy()
        val_df["split"] = "val"

    lodo_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return lodo_df


def _metrics_row(
    family: str,
    heldout: str,
    seed: int,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, object]:
    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    metrics.update(
        {
            "ece": compute_ece(y_true, y_prob),
            "brier": compute_brier(y_true, y_prob),
        }
    )
    return {
        "family": family,
        "heldout_dataset": heldout,
        "seed": seed,
        "auc": float(metrics.get("auc", float("nan"))),
        "f1": float(metrics.get("f1", float("nan"))),
        "ece": float(metrics.get("ece", float("nan"))),
        "brier": float(metrics.get("brier", float("nan"))),
        "threshold": float(threshold),
        "n_test": int(len(y_true)),
    }


def _build_model(
    family: str,
    hparams: Dict[str, object],
    input_size: int,
    focus_dim: int,
):
    input_shape = (input_size, input_size, 3)
    if family == "cnn":
        return build_cnn_baseline(input_shape=input_shape, **_filter_builder_kwargs(build_cnn_baseline, hparams)), None, False, None
    if family == "cnn_attention":
        attention_type = str(hparams.get("attention_type", "se"))
        return build_cnn_attention(hparams, input_shape=input_shape, attention_type=attention_type), None, False, None
    if family == "vit":
        return build_vit(input_shape=input_shape, **_filter_builder_kwargs(build_vit, hparams)), None, False, None
    if family == "hybrid_vit":
        return build_hybrid_vit(input_shape=input_shape, **_filter_builder_kwargs(build_hybrid_vit, hparams)), None, False, None
    if family == "focus_dnn":
        return build_focus_dnn(hparams, input_dim=focus_dim), None, False, None
    if family == "cnn_focus_hybrid":
        return build_cnn_focus_hybrid(hparams, input_shape=input_shape, focus_dim=focus_dim), None, False, None
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
    raise ValueError(f"Unsupported family: {family}")


def _train_deep_model(
    family: str,
    model,
    train_ds,
    val_ds,
    hparams: Dict[str, object],
    max_epochs: int,
    early_stop: bool,
    class_weight: Optional[Dict[int, float]],
    transfer_spec: Optional[Dict[str, object]],
):
    callbacks = []
    if early_stop:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=8,
                restore_best_weights=True,
            )
        )

    if family == "transfer":
        phase1_epochs = min(3, max_epochs)
        phase2_epochs = max(0, max_epochs - phase1_epochs)
        train_with_finetune_schedule(
            model=model,
            backbone_model=transfer_spec["backbone_model"],
            train_ds=train_ds,
            val_ds=val_ds,
            phase1_epochs=phase1_epochs,
            phase2_epochs=phase2_epochs,
            phase1_lr=float(hparams.get("lr", 1e-3)),
            phase2_lr=float(hparams.get("lr", 1e-3)) * 0.1,
            base_trainable_blocks=int(hparams.get("base_trainable_blocks", 0)),
            backbone_name=transfer_spec["backbone_name"],
            label_smoothing=float(hparams.get("label_smoothing", 0.0)),
            callbacks=callbacks,
            class_weight=class_weight,
        )
        return

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max_epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=0,
    )


def _fit_classical_model_with_weights(model, X, y, sample_weight) -> None:
    if sample_weight is None:
        model.fit(X, y)
        return
    try:
        model.fit(X, y, sample_weight=sample_weight)
        return
    except (TypeError, ValueError):
        pass

    step_name = None
    named_steps = getattr(model, "named_steps", None)
    if isinstance(named_steps, dict) and "clf" in named_steps:
        step_name = "clf"
    elif hasattr(model, "steps") and getattr(model, "steps"):
        step_name = model.steps[-1][0]

    if step_name is not None:
        try:
            model.fit(X, y, **{f"{step_name}__sample_weight": sample_weight})
            return
        except (TypeError, ValueError):
            pass

    model.fit(X, y)


def main(argv: List[str] | None = None) -> Path:
    args = parse_args(argv)
    families = _parse_families(args)

    manifest_path = _resolve_path(args.manifest)
    runs_dir = _resolve_path(args.runs_dir)
    out_root = _resolve_path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = {}
    try:
        cfg = load_yaml(paths.CONFIG_DIR / "default.yaml")
    except FileNotFoundError:
        cfg = {}

    enabled_measures = cfg.get("enabled_focus_measures", ["lapvar", "tenengrad", "brenner", "sml"])
    focus_from_augmented = bool(cfg.get("focus_vector_from_augmented", False))

    df = pd.read_csv(manifest_path)
    datasets = sorted(df["dataset"].dropna().unique())
    if not datasets:
        raise ValueError("No datasets found in manifest.")

    if args.heldout == "all":
        heldout_list = datasets
    else:
        heldout_list = [d.strip() for d in args.heldout.split(",") if d.strip()]

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    batch_size = int(args.batch_size or cfg.get("batch_size", 16))

    summary_rows: List[Dict[str, object]] = []

    for heldout in heldout_list:
        heldout_dir = out_root / f"heldout_{heldout}"
        heldout_dir.mkdir(parents=True, exist_ok=True)

        lodo_df = _prepare_lodo_manifest(df, heldout)
        manifest_out = heldout_dir / "lodo_manifest.csv"
        lodo_df.to_csv(manifest_out, index=False)

        assert_no_leak(lodo_df, group_col="stack_id", split_col="split")
        assert_no_leakage_manifest(manifest_out, check_sha1=False, check_phash=False)

        train_df = lodo_df[lodo_df["split"] == "train"].reset_index(drop=True)
        val_df = lodo_df[lodo_df["split"] == "val"].reset_index(drop=True)
        test_df = lodo_df[lodo_df["split"] == "test"].reset_index(drop=True)

        if train_df.empty or test_df.empty:
            raise ValueError(f"LODO splits empty for heldout {heldout}")

        _, extreme = report_and_check_imbalance(lodo_df)
        class_weight = compute_class_weights(lodo_df) if extreme else None

        for family in families:
            family_dir = out_root / family / f"heldout_{heldout}"
            family_dir.mkdir(parents=True, exist_ok=True)

            if family in {"classical_ml", "threshold_baselines"}:
                continue

            hparams: Dict[str, object] = {}
            if args.use_best_hparams:
                hparams_path = runs_dir / family / "best_hparams.json"
                if not hparams_path.exists():
                    raise FileNotFoundError(f"Missing best_hparams for {family}: {hparams_path}")
                hparams = load_json(hparams_path)

            input_size = _resolve_input_size(hparams, args, cfg)
            focus_dim = len(enabled_measures)

            for seed in seeds:
                set_global_determinism(seed)

                if family in {"focus_dnn", "cnn_focus_hybrid"}:
                    train_ds = build_feature_datasets(
                        manifest_csv=manifest_out,
                        split="train",
                        batch_size=batch_size,
                        input_size=input_size,
                        image_mode="rgb",
                        enabled_measures=enabled_measures,
                        augment_images=True,
                        shuffle=True,
                        seed=seed,
                        compute_from_augmented=focus_from_augmented,
                    )
                    val_ds = build_feature_datasets(
                        manifest_csv=manifest_out,
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
                        manifest_csv=manifest_out,
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
                    if family == "focus_dnn":
                        train_ds = train_ds.map(lambda x, y: (x[1], y), num_parallel_calls=tf.data.AUTOTUNE)
                        val_ds = val_ds.map(lambda x, y: (x[1], y), num_parallel_calls=tf.data.AUTOTUNE)
                        test_ds = test_ds.map(lambda x, y: (x[1], y), num_parallel_calls=tf.data.AUTOTUNE)
                else:
                    train_ds = build_datasets(
                        manifest_csv=manifest_out,
                        split="train",
                        batch_size=batch_size,
                        input_size=input_size,
                        image_mode="rgb",
                        augment=True,
                        shuffle=True,
                        seed=seed,
                        force_rgb=True,
                    )
                    val_ds = build_datasets(
                        manifest_csv=manifest_out,
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
                        manifest_csv=manifest_out,
                        split="test",
                        batch_size=batch_size,
                        input_size=input_size,
                        image_mode="rgb",
                        augment=False,
                        shuffle=False,
                        seed=seed,
                        force_rgb=True,
                    )

                model, preprocess_fn, preprocess_in_model, transfer_spec = _build_model(
                    family, hparams, input_size, focus_dim
                )

                if preprocess_fn is not None and not preprocess_in_model:
                    train_ds = _apply_preprocess(train_ds, preprocess_fn)
                    val_ds = _apply_preprocess(val_ds, preprocess_fn)
                    test_ds = _apply_preprocess(test_ds, preprocess_fn)

                start = time.perf_counter()
                _train_deep_model(
                    family,
                    model,
                    train_ds,
                    val_ds,
                    hparams,
                    max_epochs=args.max_epochs,
                    early_stop=args.early_stop,
                    class_weight=class_weight,
                    transfer_spec=transfer_spec,
                )
                train_time_s = time.perf_counter() - start

                y_prob_val = _predict_probs(model, val_ds)
                y_prob_test = _predict_probs(model, test_ds)
                threshold = choose_threshold(val_df["label"].to_numpy(), y_prob_val, metric="f1")

                y_true = test_df["label"].to_numpy().astype(int)
                metrics_row = _metrics_row(family, heldout, seed, y_true, y_prob_test, threshold)
                summary_rows.append(metrics_row)

                y_pred = (y_prob_test >= threshold).astype(int)
                preds_df = pd.DataFrame(
                    {
                        "dataset": test_df["dataset"].astype(str),
                        "image_path": test_df["image_path"].astype(str),
                        "y_true": y_true,
                        "y_prob": y_prob_test,
                        "y_pred": y_pred,
                        "seed": seed,
                        "heldout_dataset": heldout,
                    }
                )

                seed_dir = family_dir / f"seed_{seed}"
                seed_dir.mkdir(parents=True, exist_ok=True)
                preds_df.to_csv(seed_dir / "predictions.csv", index=False)

                metrics_payload = dict(metrics_row)
                metrics_payload.update(
                    {
                        "train_time_s": float(train_time_s),
                        "input_size": input_size,
                    }
                )
                save_json(metrics_payload, seed_dir / "metrics.json")

            family_summary = pd.DataFrame(
                [row for row in summary_rows if row["family"] == family]
            )
            family_summary.to_csv(out_root / family / "lodo_summary.csv", index=False)

        # classical_ml and threshold_baselines
        for seed in seeds:
            set_global_determinism(seed)

            if "classical_ml" in families:
                hparams = {}
                input_size = _resolve_input_size(hparams, args, cfg)
                all_vectors = compute_focus_vectors(
                    lodo_df["image_path"].astype(str).tolist(),
                    input_size=input_size,
                    enabled_measures=enabled_measures,
                    batch_size=64,
                    manifest_path=manifest_out,
                )
                train_idx = lodo_df.index[lodo_df["split"] == "train"].to_numpy()
                val_idx = lodo_df.index[lodo_df["split"] == "val"].to_numpy()
                test_idx = lodo_df.index[lodo_df["split"] == "test"].to_numpy()

                X_train = all_vectors[train_idx]
                X_val = all_vectors[val_idx]
                X_test = all_vectors[test_idx]

                y_train = lodo_df.loc[train_idx, "label"].to_numpy().astype(int)
                y_val = lodo_df.loc[val_idx, "label"].to_numpy().astype(int)
                y_test = lodo_df.loc[test_idx, "label"].to_numpy().astype(int)

                sample_weight = None
                if compute_sample_weight is not None:
                    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

                models = build_classical_models(seed=seed)
                best_name = None
                best_val_auc = float("-inf")
                best_model = None

                for name, model in models.items():
                    print(f"LODO classical_ml heldout={heldout} seed={seed}: fitting {name}", flush=True)
                    _fit_classical_model_with_weights(model, X_train, y_train, sample_weight)
                    y_prob_val = predict_probabilities(model, X_val)
                    val_auc = compute_metrics(y_val, y_prob_val).get("auc", float("nan"))
                    if val_auc > best_val_auc:
                        best_val_auc = val_auc
                        best_name = name
                        best_model = model

                if best_model is None:
                    raise RuntimeError("No classical ML model trained.")

                y_prob_val = predict_probabilities(best_model, X_val)
                y_prob_test = predict_probabilities(best_model, X_test)
                threshold = choose_threshold(y_val, y_prob_val, metric="f1")

                metrics_row = _metrics_row("classical_ml", heldout, seed, y_test, y_prob_test, threshold)
                summary_rows.append(metrics_row)

                y_pred = (y_prob_test >= threshold).astype(int)
                preds_df = pd.DataFrame(
                    {
                        "dataset": lodo_df.loc[test_idx, "dataset"].astype(str),
                        "image_path": lodo_df.loc[test_idx, "image_path"].astype(str),
                        "y_true": y_test,
                        "y_prob": y_prob_test,
                        "y_pred": y_pred,
                        "seed": seed,
                        "heldout_dataset": heldout,
                    }
                )

                seed_dir = out_root / "classical_ml" / f"heldout_{heldout}" / f"seed_{seed}"
                seed_dir.mkdir(parents=True, exist_ok=True)
                preds_df.to_csv(seed_dir / "predictions.csv", index=False)
                save_json(
                    {
                        **metrics_row,
                        "best_model": best_name,
                        "input_size": input_size,
                    },
                    seed_dir / "metrics.json",
                )

            if "threshold_baselines" in families:
                hparams = {}
                input_size = _resolve_input_size(hparams, args, cfg)
                all_vectors = compute_focus_vectors(
                    lodo_df["image_path"].astype(str).tolist(),
                    input_size=input_size,
                    enabled_measures=enabled_measures,
                    batch_size=64,
                    manifest_path=manifest_out,
                )

                train_idx = lodo_df.index[lodo_df["split"] == "train"].to_numpy()
                val_idx = lodo_df.index[lodo_df["split"] == "val"].to_numpy()
                test_idx = lodo_df.index[lodo_df["split"] == "test"].to_numpy()

                X_train = all_vectors[train_idx]
                X_val = all_vectors[val_idx]
                X_test = all_vectors[test_idx]

                y_val = lodo_df.loc[val_idx, "label"].to_numpy().astype(int)
                y_test = lodo_df.loc[test_idx, "label"].to_numpy().astype(int)

                scores: Dict[str, np.ndarray] = {}
                for idx, name in enumerate(enabled_measures):
                    scores[name] = X_val[:, idx]
                composite_val, _ = build_composite_scores(X_train, X_val)
                composite_test, _ = build_composite_scores(X_train, X_test)
                scores["composite"] = composite_val

                best_name = None
                best_auc = float("-inf")
                best_threshold = 0.5
                for name, vals in scores.items():
                    threshold = select_threshold(y_val, vals, metric="f1")
                    auc = compute_metrics(y_val, vals, threshold=threshold).get("auc", float("nan"))
                    if auc > best_auc:
                        best_auc = auc
                        best_name = name
                        best_threshold = threshold

                if best_name is None:
                    raise RuntimeError("No threshold baseline computed.")

                if best_name == "composite":
                    y_prob_test = composite_test
                else:
                    y_prob_test = X_test[:, enabled_measures.index(best_name)]

                metrics_row = _metrics_row("threshold_baselines", heldout, seed, y_test, y_prob_test, best_threshold)
                summary_rows.append(metrics_row)

                y_pred = (y_prob_test >= best_threshold).astype(int)
                preds_df = pd.DataFrame(
                    {
                        "dataset": lodo_df.loc[test_idx, "dataset"].astype(str),
                        "image_path": lodo_df.loc[test_idx, "image_path"].astype(str),
                        "y_true": y_test,
                        "y_prob": y_prob_test,
                        "y_pred": y_pred,
                        "seed": seed,
                        "heldout_dataset": heldout,
                    }
                )

                seed_dir = out_root / "threshold_baselines" / f"heldout_{heldout}" / f"seed_{seed}"
                seed_dir.mkdir(parents=True, exist_ok=True)
                preds_df.to_csv(seed_dir / "predictions.csv", index=False)
                save_json(
                    {
                        **metrics_row,
                        "best_measure": best_name,
                        "input_size": input_size,
                    },
                    seed_dir / "metrics.json",
                )

    summary_df = pd.DataFrame(summary_rows)
    summary_all_path = out_root / "lodo_summary_all.csv"
    summary_df.to_csv(summary_all_path, index=False)

    for family in families:
        fam_rows = summary_df[summary_df["family"] == family]
        fam_dir = out_root / family
        fam_dir.mkdir(parents=True, exist_ok=True)
        fam_rows.to_csv(fam_dir / "lodo_summary.csv", index=False)

    logger.info("LODO evaluation complete", extra={"summary": str(summary_all_path)})
    return summary_all_path


if __name__ == "__main__":
    main()
