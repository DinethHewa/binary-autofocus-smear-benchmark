from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from focus_binary import paths
from focus_binary.data.tfdata import build_datasets
from focus_binary.data.tfdata_features import build_feature_datasets
from focus_binary.eval.evaluate import evaluate_model
from focus_binary.eval.metrics import compute_metrics
from focus_binary.models.transfer import get_preprocess
from focus_binary.utils.io import load_json, load_model, load_yaml
from focus_binary.utils.efficiency import hardware_string, measure_latency
from focus_binary.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare best models across families.")
    parser.add_argument("--manifest", required=False, help="Manifest CSV with splits")
    parser.add_argument("--runs-dir", required=True, help="Root directory containing family runs")
    parser.add_argument("--out-dir", required=False, help="Output directory for reports")
    parser.add_argument("--mode", default="best", choices=["best", "q1"], help="Comparison mode")
    parser.add_argument("--include-lodo", action="store_true", help="Append LODO summary rows to per_dataset_metrics.csv")
    return parser.parse_args(argv)


def _find_family_runs(runs_dir: Path) -> List[Path]:
    return [p for p in runs_dir.iterdir() if p.is_dir() and (p / "best_model.keras").exists()]


def _find_q1_runs(runs_dir: Path) -> List[Path]:
    return [p for p in runs_dir.iterdir() if p.is_dir() and (p / "multiseed_metrics.csv").exists()]


def _load_summary(family_dir: Path) -> Dict[str, Any]:
    summary_path = family_dir / "summary.json"
    if summary_path.exists():
        return load_json(summary_path)
    return {}


def _load_best_hparams(family_dir: Path) -> Dict[str, Any]:
    hparams_path = family_dir / "best_hparams.json"
    if hparams_path.exists():
        return load_json(hparams_path)
    return {}


def _resolve_input_size(summary: Dict[str, Any], hparams: Dict[str, Any]) -> int:
    return int(summary.get("input_size") or hparams.get("input_size") or 224)


def _load_model_with_custom_objects(model_path: Path):
    if tf is None:
        raise ImportError("TensorFlow is required for model evaluation.")
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

    return load_model(model_path) if not custom_objects else tf.keras.models.load_model(model_path, custom_objects=custom_objects)


def _evaluate_family(
    family: str,
    model,
    test_df: pd.DataFrame,
    manifest_csv: Path,
    input_size: int,
    preds_path: Path,
    hparams: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = {}
    try:
        cfg = load_yaml(paths.CONFIG_DIR / "default.yaml")
    except FileNotFoundError:
        cfg = {}
    enabled_measures = cfg.get("enabled_focus_measures", ["lapvar", "tenengrad", "brenner", "sml"])
    focus_from_augmented = bool(cfg.get("focus_vector_from_augmented", False))

    if family in {"focus_dnn", "cnn_focus_hybrid"}:
        test_ds = build_feature_datasets(
            manifest_csv=manifest_csv,
            split="test",
            batch_size=16,
            input_size=input_size,
            image_mode="rgb",
            enabled_measures=enabled_measures,
            augment_images=False,
            shuffle=False,
            seed=42,
            compute_from_augmented=focus_from_augmented,
        )
        if family == "focus_dnn" and tf is not None:
            test_ds = test_ds.map(lambda x, y: (x[1], y), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        test_ds = build_datasets(
            manifest_csv=manifest_csv,
            split="test",
            batch_size=16,
            input_size=input_size,
            image_mode="rgb",
            augment=False,
            shuffle=False,
            seed=42,
            force_rgb=True,
        )

    if family == "transfer":
        backbone = hparams.get("backbone", "MobileNetV2")
        preprocess_fn = get_preprocess(backbone)

        def _apply_preprocess(img, label):
            return preprocess_fn(img * 255.0), label

        test_ds = test_ds.map(_apply_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    result = evaluate_model(model, (test_ds, test_df), threshold=0.5, preds_path=preds_path)
    return result


def _write_leaderboard_md(df: pd.DataFrame, path: Path) -> None:
    def _fmt(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "NA"
        try:
            return f"{float(value):.4f}"
        except Exception:
            return str(value)

    lines = [
        "# Leaderboard",
        "",
        "Ranking: pooled test AUC desc, F1 desc, latency_ms_mean asc, params_count asc.",
        "",
        "| rank | family | model_name | auc | f1 | acc | params_count | latency_ms_mean | latency_ms_p95 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['rank']} | {row['family']} | {row['model_name']} | "
            f"{_fmt(row['auc'])} | {_fmt(row['f1'])} | {_fmt(row['acc'])} | {row['params_count']} | "
            f"{_fmt(row.get('latency_ms_mean'))} | {_fmt(row.get('latency_ms_p95'))} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _format_mean_std(mean_val: float, std_val: float) -> str:
    if mean_val is None or std_val is None:
        return "NA"
    try:
        mean_val = float(mean_val)
        std_val = 0.0 if np.isnan(float(std_val)) else float(std_val)
        return f"{mean_val:.4f}±{std_val:.4f}"
    except Exception:
        return "NA"


def _load_multiseed_metrics(family_dir: Path) -> pd.DataFrame:
    metrics_path = family_dir / "multiseed_metrics.csv"
    if not metrics_path.exists():
        return pd.DataFrame()
    return pd.read_csv(metrics_path)


def _load_classical_outputs(runs_dir: Path) -> Dict[str, Any] | None:
    classical_dir = runs_dir / "classical_ml"
    metrics_path = classical_dir / "metrics.csv"
    if not metrics_path.exists():
        return None
    metrics_df = pd.read_csv(metrics_path)
    test_rows = metrics_df[(metrics_df["split"] == "test") & (metrics_df["dataset"] == "all")]
    if test_rows.empty:
        return None
    best_row = test_rows.sort_values(by=["auc", "f1"], ascending=[False, False]).iloc[0]
    model_name = str(best_row["model"])
    summary_path = classical_dir / "summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}
    params_count = None
    models = summary.get("models", {})
    if isinstance(models, dict) and model_name in models:
        params_count = models[model_name].get("params_count")

    preds_path = classical_dir / "predictions.csv"
    preds_df = pd.read_csv(preds_path) if preds_path.exists() else pd.DataFrame()
    preds_df = preds_df[(preds_df.get("model") == model_name) & (preds_df.get("split") == "test")]

    per_dataset = metrics_df[
        (metrics_df["split"] == "test") & (metrics_df["dataset"] != "all") & (metrics_df["model"] == model_name)
    ]

    return {
        "model_name": model_name,
        "best_row": best_row.to_dict(),
        "params_count": params_count,
        "summary": summary,
        "preds_df": preds_df,
        "per_dataset": per_dataset,
        "classical_dir": classical_dir,
    }


def _latest_subdir(parent: Path) -> Path | None:
    if not parent.exists():
        return None
    dirs = sorted([d for d in parent.iterdir() if d.is_dir()])
    return dirs[-1] if dirs else None


def _load_threshold_outputs(runs_dir: Path) -> Dict[str, Any] | None:
    base_dir = runs_dir / "threshold_baselines"
    latest = _latest_subdir(base_dir)
    if latest is None:
        return None
    metrics_path = latest / "metrics.csv"
    if not metrics_path.exists():
        return None
    metrics_df = pd.read_csv(metrics_path)
    val_rows = metrics_df[(metrics_df["split"] == "val") & (metrics_df["dataset"] == "all")]
    if val_rows.empty:
        return None
    best_val = val_rows.sort_values(by=["auc", "f1"], ascending=[False, False]).iloc[0]
    model_name = str(best_val["model"])
    test_rows = metrics_df[
        (metrics_df["split"] == "test") & (metrics_df["dataset"] == "all") & (metrics_df["model"] == model_name)
    ]
    if test_rows.empty:
        return None
    best_test = test_rows.iloc[0]

    preds_path = latest / "predictions.csv"
    preds_df = pd.read_csv(preds_path) if preds_path.exists() else pd.DataFrame()
    preds_df = preds_df[(preds_df.get("model") == model_name) & (preds_df.get("split") == "test")]

    per_dataset = metrics_df[
        (metrics_df["split"] == "test") & (metrics_df["dataset"] != "all") & (metrics_df["model"] == model_name)
    ]

    summary_path = latest / "summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}

    return {
        "model_name": model_name,
        "best_row": best_test.to_dict(),
        "summary": summary,
        "preds_df": preds_df,
        "per_dataset": per_dataset,
        "threshold_dir": latest,
    }


def _robustness_summary(family_dir: Path) -> Dict[str, Any]:
    robustness_dir = family_dir / "robustness"
    curves_path = robustness_dir / "robustness_curves.csv"
    if not curves_path.exists():
        return {"robustness_path": "", "baseline_auc": None, "worst_auc": None, "delta_auc": None, "baseline_f1": None, "worst_f1": None, "delta_f1": None, "worst_perturb": "", "worst_level": ""}

    df = pd.read_csv(curves_path)
    df = df[df["dataset"] == "all"]
    baseline = df[df["perturb"] == "clean"]
    if baseline.empty:
        return {"robustness_path": str(robustness_dir), "baseline_auc": None, "worst_auc": None, "delta_auc": None, "baseline_f1": None, "worst_f1": None, "delta_f1": None, "worst_perturb": "", "worst_level": ""}

    baseline_auc = float(baseline["auc"].mean())
    baseline_f1 = float(baseline["f1"].mean())
    perturbed = df[df["perturb"] != "clean"]
    if perturbed.empty:
        return {
            "robustness_path": str(robustness_dir),
            "baseline_auc": baseline_auc,
            "worst_auc": baseline_auc,
            "delta_auc": 0.0,
            "baseline_f1": baseline_f1,
            "worst_f1": baseline_f1,
            "delta_f1": 0.0,
            "worst_perturb": "",
            "worst_level": "",
        }

    idx = perturbed["auc"].idxmin()
    worst_row = perturbed.loc[idx]
    worst_auc = float(worst_row["auc"])
    worst_f1 = float(worst_row["f1"])
    return {
        "robustness_path": str(robustness_dir),
        "baseline_auc": baseline_auc,
        "worst_auc": worst_auc,
        "delta_auc": worst_auc - baseline_auc,
        "baseline_f1": baseline_f1,
        "worst_f1": worst_f1,
        "delta_f1": worst_f1 - baseline_f1,
        "worst_perturb": str(worst_row.get("perturb", "")),
        "worst_level": str(worst_row.get("level", "")),
    }


def _explainability_path(family_dir: Path) -> str:
    summary_path = family_dir / "explainability_summary.json"
    if summary_path.exists():
        return str(summary_path)
    explain_dir = family_dir / "explain_samples"
    if explain_dir.exists():
        return str(explain_dir)
    return ""


def _write_q1_markdown(df: pd.DataFrame, path: Path) -> None:
    def _fmt(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "NA"
        return str(value)

    lines = [
        "# Q1 Leaderboard",
        "",
        "Metrics are mean±std across seeds (pooled test).",
        "",
        "| rank | family | auc | f1 | ece | params_count | latency_ms_mean | latency_ms_p95 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['rank']} | {row['family']} | {_fmt(row['auc'])} | {_fmt(row['f1'])} | {_fmt(row['ece'])} | "
            f"{row.get('params_count', '')} | {row.get('latency_ms_mean', '')} | {row.get('latency_ms_p95', '')} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = paths.PROJECT_ROOT / runs_dir
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is None:
        out_dir = paths.PROJECT_ROOT / ("reports/final_q1" if args.mode == "q1" else "reports/final")
    if not out_dir.is_absolute():
        out_dir = paths.PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "q1":
        leaderboard_rows: List[Dict[str, Any]] = []
        per_dataset_rows: List[Dict[str, Any]] = []
        robustness_rows: List[Dict[str, Any]] = []
        calibration_rows: List[Dict[str, Any]] = []

        for family_dir in _find_q1_runs(runs_dir):
            family = family_dir.name
            metrics_df = _load_multiseed_metrics(family_dir)
            if metrics_df.empty:
                logger.warning("Missing multiseed_metrics.csv", extra={"family": family})
                continue
            summary_path = family_dir / "summary.json"
            summary = load_json(summary_path) if summary_path.exists() else {}

            pooled = metrics_df[(metrics_df["dataset"] == "all") & (metrics_df["split"] == "test")]
            if pooled.empty:
                logger.warning("Missing pooled rows in multiseed_metrics.csv", extra={"family": family})
                continue

            auc_mean, auc_std = pooled["auc"].mean(), pooled["auc"].std(ddof=1)
            f1_mean, f1_std = pooled["f1"].mean(), pooled["f1"].std(ddof=1)
            ece_mean, ece_std = pooled["ece"].mean(), pooled["ece"].std(ddof=1)
            brier_mean, brier_std = pooled["brier"].mean(), pooled["brier"].std(ddof=1)

            params_count = int(pooled["params_count"].iloc[0]) if "params_count" in pooled.columns else summary.get("params_count")
            latency_mean = float(pooled["latency_ms"].mean()) if "latency_ms" in pooled.columns else None
            latency_p95 = float(pooled["latency_ms"].quantile(0.95)) if "latency_ms" in pooled.columns else None

            robustness = _robustness_summary(family_dir)
            explain_path = _explainability_path(family_dir)

            leaderboard_rows.append(
                {
                    "family": family,
                    "auc": _format_mean_std(auc_mean, auc_std),
                    "f1": _format_mean_std(f1_mean, f1_std),
                    "ece": _format_mean_std(ece_mean, ece_std),
                    "auc_mean": float(auc_mean),
                    "f1_mean": float(f1_mean),
                    "params_count": params_count,
                    "latency_ms_mean": latency_mean,
                    "latency_ms_p95": latency_p95,
                    "explainability_path": explain_path,
                    "robustness_path": robustness.get("robustness_path", ""),
                }
            )

            calibration_rows.append(
                {
                    "family": family,
                    "ece_mean": float(ece_mean),
                    "ece_std": float(ece_std) if not np.isnan(ece_std) else None,
                    "brier_mean": float(brier_mean),
                    "brier_std": float(brier_std) if not np.isnan(brier_std) else None,
                }
            )

            robustness_rows.append({"family": family, **robustness})

            per_dataset = metrics_df[(metrics_df["dataset"] != "all") & (metrics_df["split"] == "test")]
            if not per_dataset.empty:
                for dataset_name, group in per_dataset.groupby("dataset"):
                    per_dataset_rows.append(
                        {
                            "family": family,
                            "dataset": dataset_name,
                            "auc_mean": float(group["auc"].mean()),
                            "auc_std": float(group["auc"].std(ddof=1)),
                        }
                    )

        if not leaderboard_rows:
            raise RuntimeError("No multiseed results found in runs directory.")

        leaderboard_df = pd.DataFrame(leaderboard_rows)
        leaderboard_df = leaderboard_df.sort_values(
            by=["auc_mean", "f1_mean", "latency_ms_mean", "params_count"],
            ascending=[False, False, True, True],
            na_position="last",
        ).reset_index(drop=True)
        leaderboard_df.insert(0, "rank", range(1, len(leaderboard_df) + 1))

        leaderboard_csv = out_dir / "leaderboard.csv"
        leaderboard_df.to_csv(leaderboard_csv, index=False)

        leaderboard_md = out_dir / "leaderboard.md"
        _write_q1_markdown(leaderboard_df, leaderboard_md)

        per_dataset_path = out_dir / "per_dataset.csv"
        pd.DataFrame(per_dataset_rows).to_csv(per_dataset_path, index=False)

        robustness_path = out_dir / "robustness_summary.csv"
        pd.DataFrame(robustness_rows).to_csv(robustness_path, index=False)

        calibration_path = out_dir / "calibration_summary.csv"
        pd.DataFrame(calibration_rows).to_csv(calibration_path, index=False)

        logger.info(
            "q1 comparison complete",
            extra={
                "leaderboard": str(leaderboard_csv),
                "per_dataset": str(per_dataset_path),
                "robustness": str(robustness_path),
                "calibration": str(calibration_path),
            },
        )
        return leaderboard_csv

    if not args.manifest:
        raise ValueError("--manifest is required in best mode.")

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = paths.PROJECT_ROOT / manifest_path

    df = pd.read_csv(manifest_path)
    test_df = df[df["split"] == "test"].copy()

    leaderboard_rows: List[Dict[str, Any]] = []
    per_dataset_rows: List[Dict[str, Any]] = []
    confusion: Dict[str, Any] = {}
    predictions_all = []

    for family_dir in _find_family_runs(runs_dir):
        family = family_dir.name
        model_path = family_dir / "best_model.keras"
        if not model_path.exists():
            logger.warning("Skipping family without best_model.keras", extra={"family": family})
            continue

        hparams = _load_best_hparams(family_dir)
        summary = _load_summary(family_dir)
        input_size = _resolve_input_size(summary, hparams)

        model = _load_model_with_custom_objects(model_path)
        params_count = int(model.count_params())
        latency_mean = summary.get("latency_ms_mean")
        latency_p95 = summary.get("latency_ms_p95")
        if latency_mean is None or latency_p95 is None:
            try:
                latency_mean, latency_p95 = measure_latency(model, input_size=input_size, batch_size=1)
            except Exception as exc:
                logger.warning("latency measurement failed", extra={"family": family, "error": str(exc)})

        preds_path = out_dir / f"predictions_{family}.csv"
        result = _evaluate_family(family, model, test_df, manifest_path, input_size, preds_path, hparams)
        preds_df = pd.read_csv(preds_path)
        preds_df["family"] = family
        predictions_all.append(preds_df)

        pooled_metrics = compute_metrics(preds_df["y_true"].to_numpy(), preds_df["y_prob"].to_numpy(), threshold=0.5)
        model_name = family
        if family == "transfer" and hparams.get("backbone"):
            model_name = f"{family}:{hparams['backbone']}"

        leaderboard_rows.append(
            {
                "family": family,
                "model_name": model_name,
                "params_count": params_count,
                "input_size": input_size,
                "auc": pooled_metrics["auc"],
                "f1": pooled_metrics["f1"],
                "acc": pooled_metrics["accuracy"],
                "precision": pooled_metrics["precision"],
                "recall": pooled_metrics["recall"],
                "fp": pooled_metrics["fp"],
                "fn": pooled_metrics["fn"],
                "latency_ms_mean": latency_mean,
                "latency_ms_p95": latency_p95,
                "tuning_walltime_s": summary.get("tuning_walltime_s"),
                "training_walltime_s": summary.get("training_walltime_s"),
                "hardware": summary.get("hardware", hardware_string()),
            }
        )

        confusion.setdefault(family, {})
        confusion[family]["pooled"] = pooled_metrics["confusion_matrix"]

        for dataset_name, group in preds_df.groupby("dataset"):
            dataset_metrics = compute_metrics(group["y_true"].to_numpy(), group["y_prob"].to_numpy(), threshold=0.5)
            per_dataset_rows.append(
                {
                    "family": family,
                    "dataset": dataset_name,
                    "eval_type": "standard",
                    "seed": None,
                    "auc": dataset_metrics["auc"],
                    "f1": dataset_metrics["f1"],
                    "acc": dataset_metrics["accuracy"],
                    "precision": dataset_metrics["precision"],
                    "recall": dataset_metrics["recall"],
                    "fp": dataset_metrics["fp"],
                    "fn": dataset_metrics["fn"],
                }
            )
            confusion[family][dataset_name] = dataset_metrics["confusion_matrix"]

        logger.info(
            "evaluated family",
            extra={"family": family, "auc": pooled_metrics["auc"], "f1": pooled_metrics["f1"]},
        )

    classical_out = _load_classical_outputs(runs_dir)
    if classical_out is not None:
        best_row = classical_out["best_row"]
        model_name = classical_out["model_name"]
        preds_df = classical_out["preds_df"]
        summary = classical_out["summary"]
        params_count = classical_out.get("params_count")
        input_size = int(summary.get("input_size", 224))

        if not preds_df.empty:
            preds_df = preds_df.copy()
            preds_df["family"] = "classical_ml"
            predictions_all.append(preds_df)
            pooled_metrics = compute_metrics(preds_df["y_true"].to_numpy(), preds_df["y_prob"].to_numpy(), threshold=0.5)
        else:
            pooled_metrics = {
                "auc": best_row.get("auc"),
                "f1": best_row.get("f1"),
                "accuracy": best_row.get("accuracy"),
                "precision": best_row.get("precision"),
                "recall": best_row.get("recall"),
                "fp": best_row.get("fp"),
                "fn": best_row.get("fn"),
                "confusion_matrix": None,
            }

        leaderboard_rows.append(
            {
                "family": "classical_ml",
                "model_name": f"classical_ml:{model_name}",
                "params_count": params_count,
                "input_size": input_size,
                "auc": pooled_metrics.get("auc"),
                "f1": pooled_metrics.get("f1"),
                "acc": pooled_metrics.get("accuracy"),
                "precision": pooled_metrics.get("precision"),
                "recall": pooled_metrics.get("recall"),
                "fp": pooled_metrics.get("fp"),
                "fn": pooled_metrics.get("fn"),
                "latency_ms_mean": None,
                "latency_ms_p95": None,
                "tuning_walltime_s": None,
                "training_walltime_s": None,
                "hardware": summary.get("hardware", hardware_string()),
            }
        )

        confusion.setdefault("classical_ml", {})
        if pooled_metrics.get("confusion_matrix") is not None:
            confusion["classical_ml"]["pooled"] = pooled_metrics["confusion_matrix"]

        per_dataset = classical_out["per_dataset"]
        if not per_dataset.empty:
            for _, row in per_dataset.iterrows():
                per_dataset_rows.append(
                    {
                        "family": "classical_ml",
                        "dataset": row["dataset"],
                        "eval_type": "standard",
                        "seed": None,
                        "auc": row.get("auc"),
                        "f1": row.get("f1"),
                        "acc": row.get("accuracy"),
                        "precision": row.get("precision"),
                        "recall": row.get("recall"),
                        "fp": row.get("fp"),
                        "fn": row.get("fn"),
                    }
                )
        elif not preds_df.empty:
            for dataset_name, group in preds_df.groupby("dataset"):
                dataset_metrics = compute_metrics(
                    group["y_true"].to_numpy(),
                    group["y_prob"].to_numpy(),
                    threshold=0.5,
                )
                per_dataset_rows.append(
                    {
                        "family": "classical_ml",
                        "dataset": dataset_name,
                        "eval_type": "standard",
                        "seed": None,
                        "auc": dataset_metrics["auc"],
                        "f1": dataset_metrics["f1"],
                        "acc": dataset_metrics["accuracy"],
                        "precision": dataset_metrics["precision"],
                        "recall": dataset_metrics["recall"],
                        "fp": dataset_metrics["fp"],
                        "fn": dataset_metrics["fn"],
                    }
                )
                confusion["classical_ml"][dataset_name] = dataset_metrics["confusion_matrix"]

    threshold_out = _load_threshold_outputs(runs_dir)
    if threshold_out is not None:
        best_row = threshold_out["best_row"]
        model_name = threshold_out["model_name"]
        preds_df = threshold_out["preds_df"]
        summary = threshold_out["summary"]
        input_size = int(summary.get("input_size", 224))

        if not preds_df.empty:
            preds_df = preds_df.copy()
            preds_df["family"] = "threshold_baselines"
            predictions_all.append(preds_df)
            pooled_metrics = compute_metrics(
                preds_df["y_true"].to_numpy(),
                preds_df["y_prob"].to_numpy(),
                threshold=0.5,
            )
        else:
            pooled_metrics = {
                "auc": best_row.get("auc"),
                "f1": best_row.get("f1"),
                "accuracy": best_row.get("accuracy"),
                "precision": best_row.get("precision"),
                "recall": best_row.get("recall"),
                "fp": best_row.get("fp"),
                "fn": best_row.get("fn"),
                "confusion_matrix": None,
            }

        leaderboard_rows.append(
            {
                "family": "threshold_baselines",
                "model_name": f"threshold_baselines:{model_name}",
                "params_count": None,
                "input_size": input_size,
                "auc": pooled_metrics.get("auc"),
                "f1": pooled_metrics.get("f1"),
                "acc": pooled_metrics.get("accuracy"),
                "precision": pooled_metrics.get("precision"),
                "recall": pooled_metrics.get("recall"),
                "fp": pooled_metrics.get("fp"),
                "fn": pooled_metrics.get("fn"),
                "latency_ms_mean": None,
                "latency_ms_p95": None,
                "tuning_walltime_s": None,
                "training_walltime_s": None,
                "hardware": summary.get("hardware", hardware_string()),
            }
        )

        confusion.setdefault("threshold_baselines", {})
        if pooled_metrics.get("confusion_matrix") is not None:
            confusion["threshold_baselines"]["pooled"] = pooled_metrics["confusion_matrix"]

        per_dataset = threshold_out["per_dataset"]
        if not per_dataset.empty:
            for _, row in per_dataset.iterrows():
                per_dataset_rows.append(
                    {
                    "family": "threshold_baselines",
                    "dataset": row["dataset"],
                    "eval_type": "standard",
                    "seed": None,
                    "auc": row.get("auc"),
                    "f1": row.get("f1"),
                    "acc": row.get("accuracy"),
                    "precision": row.get("precision"),
                        "recall": row.get("recall"),
                        "fp": row.get("fp"),
                        "fn": row.get("fn"),
                    }
                )
        elif not preds_df.empty:
            for dataset_name, group in preds_df.groupby("dataset"):
                dataset_metrics = compute_metrics(
                    group["y_true"].to_numpy(),
                    group["y_prob"].to_numpy(),
                    threshold=0.5,
                )
                per_dataset_rows.append(
                    {
                    "family": "threshold_baselines",
                    "dataset": dataset_name,
                    "eval_type": "standard",
                    "seed": None,
                    "auc": dataset_metrics["auc"],
                    "f1": dataset_metrics["f1"],
                    "acc": dataset_metrics["accuracy"],
                    "precision": dataset_metrics["precision"],
                        "recall": dataset_metrics["recall"],
                        "fp": dataset_metrics["fp"],
                        "fn": dataset_metrics["fn"],
                    }
                )
                confusion["threshold_baselines"][dataset_name] = dataset_metrics["confusion_matrix"]

    if not leaderboard_rows:
        raise RuntimeError("No evaluated families found in runs directory.")

    leaderboard_df = pd.DataFrame(leaderboard_rows)
    leaderboard_df = leaderboard_df.sort_values(
        by=["auc", "f1", "latency_ms_mean", "params_count"],
        ascending=[False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    leaderboard_df.insert(0, "rank", range(1, len(leaderboard_df) + 1))

    leaderboard_csv = out_dir / "leaderboard.csv"
    leaderboard_df.to_csv(leaderboard_csv, index=False)
    leaderboard_md = out_dir / "leaderboard.md"
    _write_leaderboard_md(leaderboard_df, leaderboard_md)

    per_dataset_path = out_dir / "per_dataset_metrics.csv"
    per_dataset_df = pd.DataFrame(per_dataset_rows)
    if args.include_lodo:
        lodo_base = paths.PROJECT_ROOT / "reports" / "lodo"
        if not lodo_base.exists():
            lodo_base = runs_dir / "lodo"
        lodo_rows = []
        if lodo_base.exists():
            for family_dir in lodo_base.iterdir():
                if not family_dir.is_dir():
                    continue
                summary_path = family_dir / "lodo_summary.csv"
                if not summary_path.exists():
                    continue
                lodo_df = pd.read_csv(summary_path)
                for _, row in lodo_df.iterrows():
                    lodo_rows.append(
                        {
                            "family": family_dir.name,
                            "dataset": row.get("heldout_dataset"),
                            "eval_type": "lodo",
                            "auc": row.get("auc"),
                            "f1": row.get("f1"),
                            "acc": None,
                            "precision": None,
                            "recall": None,
                            "fp": None,
                            "fn": None,
                            "seed": row.get("seed"),
                        }
                    )
        if lodo_rows:
            lodo_df = pd.DataFrame(lodo_rows)
            per_dataset_df = pd.concat([per_dataset_df, lodo_df], ignore_index=True)

    per_dataset_df.to_csv(per_dataset_path, index=False)

    confusion_path = out_dir / "confusion_matrices.json"
    confusion_path.write_text(json.dumps(confusion, indent=2))

    predictions_path = out_dir / "predictions.csv"
    pd.concat(predictions_all, ignore_index=True).to_csv(predictions_path, index=False)

    logger.info(
        "comparison complete",
        extra={
            "leaderboard": str(leaderboard_csv),
            "per_dataset_metrics": str(per_dataset_path),
            "predictions": str(predictions_path),
        },
    )
    return leaderboard_csv


if __name__ == "__main__":
    main()
