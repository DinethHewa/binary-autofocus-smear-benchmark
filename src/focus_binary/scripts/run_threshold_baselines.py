from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from focus_binary import paths
from focus_binary.baselines.threshold import build_composite_scores, compute_split_scores, select_threshold
from focus_binary.calib.calibration import compute_brier, compute_ece
from focus_binary.eval.metrics import compute_metrics
from focus_binary.utils.io import load_yaml, save_json
from focus_binary.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run threshold baselines on focus measures.")
    parser.add_argument("--manifest", required=True, help="Manifest CSV with splits")
    parser.add_argument("--out-dir", default=None, help="Output directory (default runs/threshold_baselines/<timestamp>)")
    parser.add_argument("--input-size", type=int, default=224, help="Resize input size")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for vector extraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--metric", default="f1", help="Threshold selection metric")
    parser.add_argument("--measures", default=None, help="Comma-separated focus measures (optional override)")
    return parser.parse_args(argv)


def _select_output_dir(arg_dir: str | None) -> Path:
    if arg_dir:
        return Path(arg_dir)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return paths.PROJECT_ROOT / "runs" / "threshold_baselines" / stamp


def _metrics_row(
    model_name: str,
    split: str,
    dataset: str,
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
    metrics.update({"model": model_name, "split": split, "dataset": dataset, "threshold": threshold})
    return metrics


def main(argv: List[str] | None = None) -> Path:
    args = parse_args(argv)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = paths.PROJECT_ROOT / manifest_path

    out_dir = _select_output_dir(args.out_dir)
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

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("train/val/test splits must be non-empty for threshold baselines.")

    train_vectors, train_scores = compute_split_scores(
        train_df,
        input_size=args.input_size,
        measures=measures,
        batch_size=args.batch_size,
    )
    val_vectors, val_scores = compute_split_scores(
        val_df,
        input_size=args.input_size,
        measures=measures,
        batch_size=args.batch_size,
    )
    test_vectors, test_scores = compute_split_scores(
        test_df,
        input_size=args.input_size,
        measures=measures,
        batch_size=args.batch_size,
    )

    metrics_rows: List[Dict[str, object]] = []
    calibration_rows: List[Dict[str, object]] = []
    predictions_rows: List[Dict[str, object]] = []
    explainability: Dict[str, object] = {"feature_names": measures, "models": {}}

    y_train = train_df["label"].astype(int).to_numpy()
    y_val = val_df["label"].astype(int).to_numpy()
    y_test = test_df["label"].astype(int).to_numpy()

    for measure in measures:
        val_score = val_scores[measure]
        test_score = test_scores[measure]
        threshold = select_threshold(y_val, val_score, metric=args.metric)

        metrics_rows.append(_metrics_row(measure, "val", "all", y_val, val_score, threshold))
        metrics_rows.append(_metrics_row(measure, "test", "all", y_test, test_score, threshold))

        for dataset_name in sorted(test_df["dataset"].unique()):
            mask = test_df["dataset"] == dataset_name
            metrics_rows.append(_metrics_row(measure, "test", dataset_name, y_test[mask], test_score[mask], threshold))

        for split_name, split_df, score in [
            ("val", val_df, val_score),
            ("test", test_df, test_score),
        ]:
            predictions_rows.extend(
                [
                    {
                        "model": measure,
                        "split": split_name,
                        "dataset": split_df.loc[idx, "dataset"],
                        "image_path": split_df.loc[idx, "image_path"],
                        "y_true": int(split_df.loc[idx, "label"]),
                        "y_prob": float(score[i]),
                    }
                    for i, idx in enumerate(split_df.index)
                ]
            )

        calibration_rows.append(
            {
                "model": measure,
                "split": "val",
                "dataset": "all",
                "ece": compute_ece(y_val, val_score),
                "brier": compute_brier(y_val, val_score),
            }
        )
        calibration_rows.append(
            {
                "model": measure,
                "split": "test",
                "dataset": "all",
                "ece": compute_ece(y_test, test_score),
                "brier": compute_brier(y_test, test_score),
            }
        )

        explainability["models"][measure] = {
            "feature_weights": [{"feature": measure, "weight": 1.0}],
            "top_features": [measure],
        }

    composite_val, weights = build_composite_scores(train_vectors, val_vectors)
    composite_test, _ = build_composite_scores(train_vectors, test_vectors)
    threshold = select_threshold(y_val, composite_val, metric=args.metric)
    model_name = "composite"

    metrics_rows.append(_metrics_row(model_name, "val", "all", y_val, composite_val, threshold))
    metrics_rows.append(_metrics_row(model_name, "test", "all", y_test, composite_test, threshold))

    for dataset_name in sorted(test_df["dataset"].unique()):
        mask = test_df["dataset"] == dataset_name
        metrics_rows.append(_metrics_row(model_name, "test", dataset_name, y_test[mask], composite_test[mask], threshold))

    for split_name, split_df, score in [
        ("val", val_df, composite_val),
        ("test", test_df, composite_test),
    ]:
        predictions_rows.extend(
            [
                {
                    "model": model_name,
                    "split": split_name,
                    "dataset": split_df.loc[idx, "dataset"],
                    "image_path": split_df.loc[idx, "image_path"],
                    "y_true": int(split_df.loc[idx, "label"]),
                    "y_prob": float(score[i]),
                }
                for i, idx in enumerate(split_df.index)
            ]
        )

    calibration_rows.append(
        {
            "model": model_name,
            "split": "val",
            "dataset": "all",
            "ece": compute_ece(y_val, composite_val),
            "brier": compute_brier(y_val, composite_val),
        }
    )
    calibration_rows.append(
        {
            "model": model_name,
            "split": "test",
            "dataset": "all",
            "ece": compute_ece(y_test, composite_test),
            "brier": compute_brier(y_test, composite_test),
        }
    )

    weight_rows = [
        {"feature": name, "weight": float(abs(weights[idx]))}
        for idx, name in enumerate(measures)
    ]
    weight_rows_sorted = sorted(weight_rows, key=lambda r: r["weight"], reverse=True)
    explainability["models"][model_name] = {
        "feature_weights": weight_rows_sorted,
        "top_features": [row["feature"] for row in weight_rows_sorted[:3]],
    }

    metrics_path = out_dir / "metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)

    preds_path = out_dir / "predictions.csv"
    pd.DataFrame(predictions_rows).to_csv(preds_path, index=False)

    calib_path = out_dir / "calibration.csv"
    pd.DataFrame(calibration_rows).to_csv(calib_path, index=False)

    explain_path = out_dir / "explainability.json"
    save_json(explainability, explain_path)

    summary_path = out_dir / "summary.json"
    save_json(
        {
            "family": "threshold_baselines",
            "input_size": args.input_size,
            "feature_names": measures,
            "models": list(explainability["models"].keys()),
            "output_dir": str(out_dir),
        },
        summary_path,
    )

    logger.info(
        "threshold baselines complete",
        extra={
            "metrics": str(metrics_path),
            "predictions": str(preds_path),
            "calibration": str(calib_path),
        },
    )
    return out_dir


if __name__ == "__main__":
    raise SystemExit(main())
