from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from focus_binary import paths
from focus_binary.calib.calibration import choose_threshold, compute_brier, compute_ece
from focus_binary.classical_ml.explain import explain_classical_model
from focus_binary.classical_ml.models import build_classical_models, compute_focus_vectors, predict_probabilities
from focus_binary.eval.metrics import compute_metrics
from focus_binary.utils.io import load_yaml, save_json
from focus_binary.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from sklearn.utils.class_weight import compute_sample_weight
except Exception:  # pragma: no cover
    compute_sample_weight = None


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classical ML baselines on focus vectors.")
    parser.add_argument("--manifest", required=True, help="Manifest CSV with splits")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--input-size", type=int, default=224, help="Resize images to this size")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for vector extraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--measures", default=None, help="Comma-separated focus measures (optional)")
    parser.add_argument("--use-tensorflow-loader", action="store_true", help="Use TensorFlow loader instead of PIL")
    return parser.parse_args(argv)


def _params_count(model) -> int | None:
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        intercept = np.asarray(getattr(model, "intercept_", np.array([])))
        return int(coef.size + intercept.size)
    if hasattr(model, "feature_importances_"):
        return int(np.asarray(model.feature_importances_).size)
    return None


def main(argv: List[str] | None = None) -> Path:
    args = parse_args(argv)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = paths.PROJECT_ROOT / manifest_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = paths.PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    if "split" not in df.columns:
        raise KeyError("manifest must include 'split' column")

    cfg = {}
    try:
        cfg = load_yaml(paths.CONFIG_DIR / "default.yaml")
    except FileNotFoundError:
        cfg = {}
    measures = (
        [m.strip() for m in args.measures.split(",") if m.strip()]
        if args.measures
        else cfg.get("enabled_focus_measures", ["lapvar", "tenengrad", "brenner", "sml"])
    )

    train_idx = df.index[df["split"] == "train"].to_numpy()
    val_idx = df.index[df["split"] == "val"].to_numpy()
    test_idx = df.index[df["split"] == "test"].to_numpy()
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError("train/val/test splits must be non-empty for classical ML baselines.")

    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df = df.loc[val_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)

    start = time.perf_counter()
    all_vectors = compute_focus_vectors(
        df["image_path"].astype(str).tolist(),
        input_size=args.input_size,
        enabled_measures=measures,
        batch_size=args.batch_size,
        use_tensorflow_loader=args.use_tensorflow_loader,
        manifest_path=manifest_path,
    )
    X_train = all_vectors[train_idx]
    X_val = all_vectors[val_idx]
    X_test = all_vectors[test_idx]
    feature_time_s = time.perf_counter() - start

    y_train = train_df["label"].astype(int).to_numpy()
    y_val = val_df["label"].astype(int).to_numpy()
    y_test = test_df["label"].astype(int).to_numpy()

    sample_weight = None
    if compute_sample_weight is not None:
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    models = build_classical_models(seed=args.seed)

    metrics_rows: List[Dict[str, object]] = []
    predictions_rows: List[Dict[str, object]] = []
    explainability: Dict[str, object] = {
        "feature_names": measures,
        "models": {},
    }
    summary_models: Dict[str, object] = {}

    for name, model in models.items():
        logger.info("training classical model", extra={"model": name})
        _fit_model_with_weights(model, X_train, y_train, sample_weight)

        y_prob_val = predict_probabilities(model, X_val)
        y_prob_test = predict_probabilities(model, X_test)

        threshold = choose_threshold(y_val, y_prob_val, metric="f1")

        val_metrics = compute_metrics(y_val, y_prob_val, threshold=threshold)
        val_metrics.update({"ece": compute_ece(y_val, y_prob_val), "brier": compute_brier(y_val, y_prob_val)})

        test_metrics = compute_metrics(y_test, y_prob_test, threshold=threshold)
        test_metrics.update({"ece": compute_ece(y_test, y_prob_test), "brier": compute_brier(y_test, y_prob_test)})

        metrics_rows.append(
            {
                "model": name,
                "split": "val",
                "dataset": "all",
                "threshold": float(threshold),
                **val_metrics,
            }
        )
        metrics_rows.append(
            {
                "model": name,
                "split": "test",
                "dataset": "all",
                "threshold": float(threshold),
                **test_metrics,
            }
        )

        for dataset_name in sorted(test_df["dataset"].unique()):
            mask = test_df["dataset"] == dataset_name
            metrics = compute_metrics(y_test[mask], y_prob_test[mask], threshold=threshold)
            metrics.update(
                {
                    "ece": compute_ece(y_test[mask], y_prob_test[mask]),
                    "brier": compute_brier(y_test[mask], y_prob_test[mask]),
                }
            )
            metrics_rows.append(
                {
                    "model": name,
                    "split": "test",
                    "dataset": dataset_name,
                    "threshold": float(threshold),
                    **metrics,
                }
            )

        for split_name, df_split, y_prob in [
            ("val", val_df, y_prob_val),
            ("test", test_df, y_prob_test),
        ]:
            predictions_rows.extend(
                [
                    {
                        "model": name,
                        "split": split_name,
                        "dataset": df_split.loc[idx, "dataset"],
                        "image_path": df_split.loc[idx, "image_path"],
                        "y_true": int(df_split.loc[idx, "label"]),
                        "y_prob": float(y_prob[i]),
                    }
                    for i, idx in enumerate(df_split.index)
                ]
            )

        explainability["models"][name] = explain_classical_model(
            model=model,
            X_val=X_val,
            y_val=y_val,
            feature_names=measures,
            seed=args.seed,
        )
        summary_models[name] = {
            "params_count": _params_count(model),
            "threshold": float(threshold),
        }

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = out_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    predictions_df = pd.DataFrame(predictions_rows)
    predictions_path = out_dir / "predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    explainability_path = out_dir / "explainability.json"
    save_json(explainability, explainability_path)

    summary_path = out_dir / "summary.json"
    save_json(
        {
            "family": "classical_ml",
            "input_size": args.input_size,
            "feature_names": measures,
            "models": summary_models,
            "feature_extraction_time_s": float(feature_time_s),
        },
        summary_path,
    )

    logger.info(
        "classical ML run complete",
        extra={
            "metrics": str(metrics_path),
            "predictions": str(predictions_path),
            "explainability": str(explainability_path),
        },
    )
    return metrics_path


def _fit_model_with_weights(model, X, y, sample_weight) -> None:
    if sample_weight is None:
        model.fit(X, y)
        return
    try:
        model.fit(X, y, sample_weight=sample_weight)
        return
    except (TypeError, ValueError):
        pass

    step_name = None
    if hasattr(model, "named_steps") and isinstance(getattr(model, "named_steps"), dict):
        if "clf" in model.named_steps:
            step_name = "clf"
    if step_name is None and hasattr(model, "steps") and model.steps:
        step_name = model.steps[-1][0]
    if step_name is not None:
        try:
            model.fit(X, y, **{f"{step_name}__sample_weight": sample_weight})
            return
        except (TypeError, ValueError):
            pass
    model.fit(X, y)


if __name__ == "__main__":
    raise SystemExit(main())
