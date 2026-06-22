from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .common import (
    PipelineContext,
    family_display_name,
    format_float,
    safe_div,
    safe_read_csv,
    save_image_grid,
    save_placeholder_figure,
    write_dataframe,
    write_text,
)
from .pillow_plots import (
    pillow_available,
    save_calibration_chart,
    save_multi_line_chart,
    save_two_panel_line_chart,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except Exception:  # pragma: no cover
    plt = None
    MATPLOTLIB_AVAILABLE = False


def _set_style() -> None:
    if not MATPLOTLIB_AVAILABLE:
        return
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )


def _load_representative_predictions(ctx: PipelineContext, summary_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    for _, row in summary_df.iterrows():
        family = str(row["family"])
        if family == "classical_ml":
            preds = safe_read_csv(ctx.project_root / "runs" / "classical_ml" / "predictions.csv", ctx)
            preds = preds[(preds["model"] == row["model_name"])].copy()
            preds["display_name"] = family_display_name(family, row["model_name"])
            outputs[family] = preds
        elif family == "threshold_baselines":
            preds = safe_read_csv(ctx.project_root / "runs" / "threshold_baselines" / "predictions.csv", ctx)
            preds = preds[(preds["model"] == row["model_name"])].copy()
            preds["display_name"] = family_display_name(family, row["model_name"])
            outputs[family] = preds
        else:
            pred_path = ctx.project_root / "reports" / "final" / f"predictions_{family}.csv"
            preds = safe_read_csv(pred_path, ctx)
            if preds.empty:
                continue
            preds["split"] = preds.get("split", "test")
            preds["model"] = row["model_name"]
            preds["display_name"] = row["display_name"]
            outputs[family] = preds
    return outputs


def _binary_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)
    bal = float(np.nanmean([recall, specificity]))
    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        mcc = float("nan")
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "mcc": mcc,
        "balanced_accuracy": bal,
    }


def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> tuple[float, pd.DataFrame]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    ece = 0.0
    for i in range(bins):
        left = edges[i]
        right = edges[i + 1]
        mask = (y_prob >= left) & (y_prob <= right) if i == bins - 1 else (y_prob >= left) & (y_prob < right)
        if not np.any(mask):
            rows.append({"bin_left": left, "bin_right": right, "count": 0, "mean_conf": np.nan, "empirical_acc": np.nan})
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        weight = float(np.mean(mask))
        ece += abs(conf - acc) * weight
        rows.append({"bin_left": left, "bin_right": right, "count": int(np.sum(mask)), "mean_conf": conf, "empirical_acc": acc})
    return ece, pd.DataFrame(rows)


def _priority2_feasibility(ctx: PipelineContext, prediction_map: dict[str, pd.DataFrame]) -> dict[str, str]:
    items = {
        "saved_models": "not_feasible_from_current_artifacts",
        "preprocessing_logic": "not_feasible_from_current_artifacts",
        "test_or_validation_images": "not_feasible_from_current_artifacts",
        "explainability_analysis": "not_feasible_from_current_artifacts",
        "threshold_analysis": "not_feasible_from_current_artifacts",
        "failure_case_gallery": "not_feasible_from_current_artifacts",
        "calibration_analysis": "not_feasible_from_current_artifacts",
    }
    model_paths = [
        ctx.project_root / "runs" / "cnn" / "best_model.keras",
        ctx.project_root / "runs" / "transfer" / "best_model.keras",
    ]
    if all(path.exists() for path in model_paths):
        items["saved_models"] = "feasible_now"
    if (ctx.project_root / "src" / "focus_binary" / "data" / "tfdata.py").exists():
        items["preprocessing_logic"] = "feasible_now"
    manifest_path = ctx.project_root / "data" / "manifest_with_splits.csv"
    manifest = safe_read_csv(manifest_path, ctx)
    if not manifest.empty and manifest["image_path"].astype(str).map(lambda p: Path(str(p)).exists()).mean() > 0.95:
        items["test_or_validation_images"] = "feasible_now"
    if prediction_map:
        items["threshold_analysis"] = "feasible_now"
        items["failure_case_gallery"] = "feasible_now"
        items["calibration_analysis"] = "feasible_now"
    if items["saved_models"] == "feasible_now" and items["preprocessing_logic"] == "feasible_now" and items["test_or_validation_images"] == "feasible_now":
        items["explainability_analysis"] = "feasible_now"
    elif items["saved_models"] == "feasible_now":
        items["explainability_analysis"] = "feasible_with_minor_fixes"
    return items


def _write_feasibility_report(ctx: PipelineContext, statuses: dict[str, str]) -> None:
    lines = [
        "# Priority 2 Feasibility Report",
        "",
        "Status legend:",
        "",
        "- `feasible_now`: all required saved artifacts are present and usable without retraining.",
        "- `feasible_with_minor_fixes`: the core artifacts exist but extra glue code or minor recovery is required.",
        "- `not_feasible_from_current_artifacts`: the required artifacts are absent or incomplete.",
        "",
        "## Item Status",
        "",
    ]
    for key, value in statuses.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "Important caveat:",
            "",
            "- Deep-family validation probabilities were not saved in the final report artifacts, so any threshold sweep for deep models is descriptive on the test set unless fresh inference is explicitly run.",
        ]
    )
    write_text(ctx.output_dir / "priority2_feasibility_report.md", "\n".join(lines))


def _threshold_and_curve_outputs(ctx: PipelineContext, prediction_map: dict[str, pd.DataFrame]) -> dict[str, Any]:
    threshold_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    plotted_models: list[tuple[str, np.ndarray, np.ndarray, float, float]] = []

    for family, df in prediction_map.items():
        if df.empty:
            continue
        display_name = str(df["display_name"].iloc[0])
        for split, split_df in df.groupby(df["split"].astype(str)):
            y_true = split_df["y_true"].astype(int).to_numpy()
            y_prob = split_df["y_prob"].astype(float).to_numpy()
            if np.min(y_prob) >= 0.0 and np.max(y_prob) <= 1.0:
                thresholds = np.linspace(0.05, 0.95, 19)
                score_type = "probability"
            else:
                thresholds = np.linspace(float(np.min(y_prob)), float(np.max(y_prob)), 19)
                score_type = "raw_score"
            for threshold in thresholds:
                threshold_rows.append(
                    {
                        "family": family,
                        "display_name": display_name,
                        "split": split,
                        "score_type": score_type,
                        "threshold": threshold,
                        **_binary_metrics_at_threshold(y_true, y_prob, threshold),
                    }
                )
            if len(np.unique(y_true)) >= 2:
                roc_auc = float(roc_auc_score(y_true, y_prob))
                ap = float(average_precision_score(y_true, y_prob))
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                curve_rows.append(
                    {
                        "family": family,
                        "display_name": display_name,
                        "split": split,
                        "roc_auc": roc_auc,
                        "average_precision": ap,
                        "n_samples": len(split_df),
                    }
                )
                if split == "test":
                    plotted_models.append((display_name, fpr, tpr, roc_auc, ap))

    threshold_df = pd.DataFrame(threshold_rows)
    curve_df = pd.DataFrame(curve_rows)
    write_dataframe(threshold_df, ctx.output_dir / "threshold_metrics.csv")
    write_dataframe(curve_df, ctx.output_dir / "roc_pr_summary.csv")

    if MATPLOTLIB_AVAILABLE:
        _set_style()
        fig, ax = plt.subplots(figsize=(7.2, 5.5))
        for display_name, fpr, tpr, roc_auc, _ in plotted_models:
            ax.plot(fpr, tpr, label=f"{display_name} (AUC={roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=0.8)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves From Saved Test Predictions")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(ctx.output_dir / "fig_roc_curves.png", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 5.5))
        for family, df in prediction_map.items():
            test_df = df[df["split"].astype(str) == "test"]
            if test_df.empty:
                continue
            y_true = test_df["y_true"].astype(int).to_numpy()
            y_prob = test_df["y_prob"].astype(float).to_numpy()
            if len(np.unique(y_true)) < 2:
                continue
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
            ax.plot(recall, precision, label=f"{test_df['display_name'].iloc[0]} (AP={ap:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves From Saved Test Predictions")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(ctx.output_dir / "fig_pr_curves.png", bbox_inches="tight")
        plt.close(fig)

        top_display = (
            threshold_df[threshold_df["split"] == "test"]
            .groupby("display_name")["balanced_accuracy"]
            .max()
            .sort_values(ascending=False)
            .head(4)
            .index.tolist()
        )
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharex=True)
        for display_name in top_display:
            subset = threshold_df[(threshold_df["split"] == "test") & (threshold_df["display_name"] == display_name)]
            axes[0].plot(subset["threshold"], subset["f1"], label=display_name)
            axes[1].plot(subset["threshold"], subset["mcc"], label=display_name)
        axes[0].set_title("F1 vs Threshold")
        axes[1].set_title("MCC vs Threshold")
        for axis in axes:
            axis.set_xlabel("Threshold")
            axis.legend(loc="best")
        fig.tight_layout()
        fig.savefig(ctx.output_dir / "fig_threshold_sweep.png", bbox_inches="tight")
        plt.close(fig)
    else:
        if pillow_available():
            roc_series = [
                {
                    "label": f"{display_name} (AUC={roc_auc:.3f})",
                    "x": np.asarray(fpr, dtype=float),
                    "y": np.asarray(tpr, dtype=float),
                }
                for display_name, fpr, tpr, roc_auc, _ in plotted_models
            ]
            if roc_series:
                save_multi_line_chart(
                    ctx.output_dir / "fig_roc_curves.png",
                    "ROC Curves From Saved Test Predictions",
                    roc_series,
                    "False Positive Rate",
                    "True Positive Rate",
                    xlim=(0.0, 1.0),
                    ylim=(0.0, 1.0),
                    diagonal=True,
                )
            else:
                save_placeholder_figure(ctx.output_dir / "fig_roc_curves.png", "ROC Figure Unavailable", "No usable saved ROC-curve inputs were recovered.")

            pr_series = []
            for family, df in prediction_map.items():
                test_df = df[df["split"].astype(str) == "test"]
                if test_df.empty:
                    continue
                y_true = test_df["y_true"].astype(int).to_numpy()
                y_prob = test_df["y_prob"].astype(float).to_numpy()
                if len(np.unique(y_true)) < 2:
                    continue
                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                ap = average_precision_score(y_true, y_prob)
                pr_series.append(
                    {
                        "label": f"{test_df['display_name'].iloc[0]} (AP={ap:.3f})",
                        "x": np.asarray(recall, dtype=float),
                        "y": np.asarray(precision, dtype=float),
                    }
                )
            if pr_series:
                save_multi_line_chart(
                    ctx.output_dir / "fig_pr_curves.png",
                    "Precision-Recall Curves From Saved Test Predictions",
                    pr_series,
                    "Recall",
                    "Precision",
                    xlim=(0.0, 1.0),
                    ylim=(0.0, 1.0),
                )
            else:
                save_placeholder_figure(ctx.output_dir / "fig_pr_curves.png", "PR Figure Unavailable", "No usable saved PR-curve inputs were recovered.")

            top_display = (
                threshold_df[threshold_df["split"] == "test"]
                .groupby("display_name")["balanced_accuracy"]
                .max()
                .sort_values(ascending=False)
                .head(4)
                .index.tolist()
            )
            left_series = []
            right_series = []
            for display_name in top_display:
                subset = threshold_df[(threshold_df["split"] == "test") & (threshold_df["display_name"] == display_name)].sort_values("threshold")
                left_series.append(
                    {
                        "label": display_name,
                        "x": subset["threshold"].astype(float).to_numpy(),
                        "y": subset["f1"].astype(float).to_numpy(),
                    }
                )
                right_series.append(
                    {
                        "label": display_name,
                        "x": subset["threshold"].astype(float).to_numpy(),
                        "y": subset["mcc"].astype(float).to_numpy(),
                    }
                )
            if left_series or right_series:
                save_two_panel_line_chart(
                    ctx.output_dir / "fig_threshold_sweep.png",
                    "Threshold Sweeps From Saved Test Predictions",
                    "F1 vs Threshold",
                    "MCC vs Threshold",
                    left_series,
                    right_series,
                    "Threshold",
                    "F1",
                    "MCC",
                    xlim=(0.0, 1.0),
                )
            else:
                save_placeholder_figure(ctx.output_dir / "fig_threshold_sweep.png", "Threshold Sweep Unavailable", "No usable saved threshold-sweep inputs were recovered.")
            ctx.log("Rendered ROC, PR, and threshold-sweep figures with the Pillow fallback renderer.")
        else:
            ctx.warn("Neither matplotlib nor Pillow is available; ROC, PR, and threshold-sweep figures are placeholders.")
            save_placeholder_figure(ctx.output_dir / "fig_roc_curves.png", "ROC Figure Skipped", "Neither matplotlib nor Pillow was available in the execution environment.")
            save_placeholder_figure(ctx.output_dir / "fig_pr_curves.png", "PR Figure Skipped", "Neither matplotlib nor Pillow was available in the execution environment.")
            save_placeholder_figure(ctx.output_dir / "fig_threshold_sweep.png", "Threshold Sweep Figure Skipped", "Neither matplotlib nor Pillow was available in the execution environment.")

    lines = [
        "# Threshold Analysis",
        "",
        "- ROC and PR summaries were computed from saved per-sample probabilities.",
        "- Saved validation probabilities existed for classical and threshold baselines; deep-family final reports only preserved test probabilities.",
        "- Any deep-model threshold sweep in this package is therefore descriptive and should not be presented as a validation-selected operating point.",
    ]
    write_text(ctx.output_dir / "threshold_analysis.md", "\n".join(lines))
    return {"threshold_df": threshold_df, "curve_df": curve_df}


def _calibration_outputs(ctx: PipelineContext, prediction_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    plot_rows: list[dict[str, Any]] = []
    if MATPLOTLIB_AVAILABLE:
        _set_style()
        fig, ax = plt.subplots(figsize=(7.2, 5.5))
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=0.8, label="Perfect calibration")
    for family, df in prediction_map.items():
        test_df = df[df["split"].astype(str) == "test"]
        if test_df.empty:
            continue
        y_true = test_df["y_true"].astype(int).to_numpy()
        y_prob = test_df["y_prob"].astype(float).to_numpy()
        if float(np.min(y_prob)) < 0.0 or float(np.max(y_prob)) > 1.0:
            ctx.warn(
                f"Skipping calibration for '{test_df['display_name'].iloc[0]}' because the saved score range was [{float(np.min(y_prob)):.3f}, {float(np.max(y_prob)):.3f}] rather than a probability range."
            )
            continue
        ece, bins_df = _compute_ece(y_true, y_prob, bins=10)
        brier = float(brier_score_loss(y_true, y_prob))
        rows.append(
            {
                "family": family,
                "display_name": test_df["display_name"].iloc[0],
                "ece": ece,
                "brier_score": brier,
            }
        )
        if MATPLOTLIB_AVAILABLE:
            valid_bins = bins_df.dropna(subset=["mean_conf", "empirical_acc"])
            ax.plot(valid_bins["mean_conf"], valid_bins["empirical_acc"], marker="o", label=f"{test_df['display_name'].iloc[0]} (ECE={ece:.3f})")
        else:
            valid_bins = bins_df.dropna(subset=["mean_conf", "empirical_acc"])
            plot_rows.append(
                {
                    "label": str(test_df["display_name"].iloc[0]),
                    "ece": ece,
                    "x": valid_bins["mean_conf"].astype(float).tolist(),
                    "y": valid_bins["empirical_acc"].astype(float).tolist(),
                }
            )
    if MATPLOTLIB_AVAILABLE:
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Empirical positive rate")
        ax.set_title("Reliability Diagram From Saved Test Predictions")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(ctx.output_dir / "fig_calibration.png", bbox_inches="tight")
        plt.close(fig)
    else:
        if pillow_available() and plot_rows:
            save_calibration_chart(
                ctx.output_dir / "fig_calibration.png",
                "Reliability Diagram From Saved Test Predictions",
                plot_rows,
            )
            ctx.log("Rendered fig_calibration.png with the Pillow fallback renderer.")
        elif pillow_available():
            save_placeholder_figure(ctx.output_dir / "fig_calibration.png", "Calibration Figure Unavailable", "No usable saved probability curves were recovered for calibration plotting.")
        else:
            ctx.warn("Neither matplotlib nor Pillow is available; fig_calibration.png is a placeholder.")
            save_placeholder_figure(ctx.output_dir / "fig_calibration.png", "Calibration Figure Skipped", "Neither matplotlib nor Pillow was available in the execution environment.")

    calibration_df = pd.DataFrame(rows).sort_values("ece")
    write_dataframe(calibration_df, ctx.output_dir / "calibration_summary.csv")
    lines = [
        "# Calibration Analysis",
        "",
        "- ECE and Brier score were computed from saved test-set probabilities.",
        "- The reliability diagram is descriptive only; no recalibration was fitted because retraining / refitting was out of scope.",
    ]
    write_text(ctx.output_dir / "calibration_analysis.md", "\n".join(lines))
    return calibration_df


def _image_array_to_rgb_float(image: Image.Image) -> np.ndarray:
    """Load PIL image data as RGB float without clipping 16-bit microscopy PNGs."""
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.ndim == 3 and array.shape[-1] >= 3:
        array = array[..., :3]
    else:
        raise ValueError(f"Unsupported image shape for '{image.mode}': {array.shape}")

    if np.issubdtype(array.dtype, np.integer):
        max_value = float(np.iinfo(array.dtype).max)
        return array.astype(np.float32) / max_value
    return np.clip(array.astype(np.float32), 0.0, 1.0)


def _contrast_normalize_for_display(image: np.ndarray) -> np.ndarray:
    """Create a readable RGB display image while preserving the model input path."""
    display = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(display, [1.0, 99.0])
    if high > low:
        display = (display - low) / (high - low)
    return np.clip(display, 0.0, 1.0)


def _load_image(path: str, target_size: tuple[int, int] | None = None, display_normalize: bool = False) -> np.ndarray:
    image = Image.open(path)
    if target_size is not None:
        image = image.resize((target_size[1], target_size[0]))
    array = _image_array_to_rgb_float(image)
    if display_normalize:
        array = _contrast_normalize_for_display(array)
    return array


def _failure_gallery(ctx: PipelineContext, prediction_map: dict[str, pd.DataFrame], best_model_name: str) -> pd.DataFrame:
    df = None
    for preds in prediction_map.values():
        if preds.empty:
            continue
        if str(preds["display_name"].iloc[0]) == best_model_name:
            df = preds[preds["split"].astype(str) == "test"].copy()
            break
    if df is None or df.empty:
        write_dataframe(pd.DataFrame(), ctx.output_dir / "failure_case_gallery.csv")
        write_text(ctx.output_dir / "failure_analysis.md", "# Failure Analysis\n\nNo saved prediction table was available for the selected top model.")
        return pd.DataFrame()

    df["pred_label"] = (df["y_prob"].astype(float) >= 0.5).astype(int)
    candidates: list[dict[str, Any]] = []
    subsets = {
        "true_positive": df[(df["y_true"] == 1) & (df["pred_label"] == 1)].sort_values("y_prob", ascending=False).head(1),
        "true_negative": df[(df["y_true"] == 0) & (df["pred_label"] == 0)].sort_values("y_prob", ascending=True).head(1),
        "high_confidence_false_positive": df[(df["y_true"] == 0) & (df["pred_label"] == 1)].sort_values("y_prob", ascending=False).head(1),
        "high_confidence_false_negative": df[(df["y_true"] == 1) & (df["pred_label"] == 0)].sort_values("y_prob", ascending=True).head(1),
        "borderline_correct_positive": df[(df["y_true"] == 1) & (df["pred_label"] == 1)].assign(distance=lambda x: (x["y_prob"] - 0.5).abs()).sort_values("distance").head(1),
        "borderline_correct_negative": df[(df["y_true"] == 0) & (df["pred_label"] == 0)].assign(distance=lambda x: (x["y_prob"] - 0.5).abs()).sort_values("distance").head(1),
    }
    for label, subset in subsets.items():
        if subset.empty:
            continue
        row = subset.iloc[0].to_dict()
        row["case_type"] = label
        candidates.append(row)

    out_df = pd.DataFrame(candidates)
    write_dataframe(out_df, ctx.output_dir / "failure_case_gallery.csv")

    temp_dir = ctx.output_dir / "_tmp_failure_gallery"
    temp_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    captions = []
    for row in out_df.itertuples():
        img = Image.fromarray((_load_image(row.image_path, target_size=(224, 224), display_normalize=True) * 255.0).astype(np.uint8))
        temp_path = temp_dir / f"{row.case_type}.png"
        img.save(temp_path)
        image_paths.append(temp_path)
        captions.append(f"{row.case_type}\ntrue={row.y_true} prob={row.y_prob:.3f}")
    save_image_grid(image_paths, captions, ctx.output_dir / "fig_failure_gallery.png", f"Failure / Borderline Gallery: {best_model_name}", cols=3)

    lines = [
        "# Failure Analysis",
        "",
        f"- Gallery constructed from saved test predictions for `{best_model_name}`.",
        "- The panel includes representative true positives, true negatives, high-confidence errors, and borderline correct predictions.",
        "- These examples are descriptive and should be discussed as case studies rather than as a substitute for full error-taxonomy analysis.",
    ]
    write_text(ctx.output_dir / "failure_analysis.md", "\n".join(lines))
    return out_df


def _load_model(path: Path):
    import tensorflow as tf
    from focus_binary.models.convnext import ConvNeXtPreprocess  # type: ignore
    from focus_binary.models.swin_tiny import WindowPartition, WindowReverse  # type: ignore
    from focus_binary.models.vit import _CLSToken, _PositionalEmbedding  # type: ignore

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
    return tf.keras.models.load_model(path, custom_objects=custom_objects or None)


def _gradcam_and_occlusion(
    ctx: PipelineContext,
    family: str,
    display_name: str,
    model_path: Path,
    samples: pd.DataFrame,
    preprocess_fn: Any = None,
) -> list[dict[str, Any]]:
    import tensorflow as tf
    from focus_binary.explain.gradcam import overlay_heatmap

    model = _load_model(model_path)
    out_dir = ctx.output_dir / "priority2_explainability" / family
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    def compute_gradcam(image_tensor: "tf.Tensor") -> np.ndarray:
        image_tensor = tf.convert_to_tensor(image_tensor, dtype=tf.float32)
        if image_tensor.ndim == 3:
            image_tensor = tf.expand_dims(image_tensor, axis=0)

        if family == "transfer":
            backbone = next((layer for layer in model.layers if isinstance(layer, tf.keras.Model)), None)
            if backbone is None:
                raise ValueError("Transfer backbone model was not found inside the saved model.")
            target_layer = None
            for layer in reversed(backbone.layers):
                if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                    target_layer = layer
                    break
            if target_layer is None:
                raise ValueError("No Conv2D layer found inside the transfer backbone.")
            intermediate_model = tf.keras.Model(
                backbone.input,
                [backbone.get_layer(target_layer.name).output, backbone.output],
            )
            head_input = tf.keras.Input(shape=backbone.output_shape[1:])
            x = head_input
            start_idx = model.layers.index(backbone) + 1
            for layer in model.layers[start_idx:]:
                x = layer(x)
            head_model = tf.keras.Model(head_input, x)
            with tf.GradientTape() as tape:
                conv_outputs, backbone_out = intermediate_model(image_tensor, training=False)
                tape.watch(conv_outputs)
                preds = head_model(backbone_out, training=False)
                target = preds[:, 0] if preds.shape[-1] == 1 else preds[:, tf.argmax(preds[0])]
            grads = tape.gradient(target, conv_outputs)
        else:
            target_layer = None
            for layer in reversed(model.layers):
                if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                    target_layer = layer
                    break
            if target_layer is None:
                raise ValueError("No Conv2D layer found for Grad-CAM.")
            grad_model = tf.keras.Model(inputs=model.inputs, outputs=[target_layer.output, model.output])
            with tf.GradientTape() as tape:
                conv_outputs, preds = grad_model(image_tensor, training=False)
                target = preds[:, 0] if preds.shape[-1] == 1 else preds[:, tf.argmax(preds[0])]
            grads = tape.gradient(target, conv_outputs)
        pooled = tf.reduce_mean(grads, axis=(1, 2))
        heatmap = tf.reduce_sum(conv_outputs * pooled[:, None, None, :], axis=-1)
        heatmap = tf.nn.relu(heatmap)
        max_val = tf.reduce_max(heatmap, axis=(1, 2), keepdims=True)
        heatmap = tf.where(max_val > 0, heatmap / max_val, heatmap)
        resized = tf.image.resize(heatmap[..., None], (224, 224), method="bilinear")
        return tf.squeeze(resized, axis=(0, -1)).numpy()

    def occlusion_map(img: np.ndarray, window: int = 28, stride: int = 16) -> np.ndarray:
        base_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
        if preprocess_fn is not None:
            base_input = preprocess_fn(base_tensor * 255.0)
        else:
            base_input = base_tensor
        base_pred = float(model(tf.expand_dims(base_input, axis=0), training=False).numpy().reshape(-1)[0])
        heat = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        counts = np.zeros_like(heat)
        for top in range(0, img.shape[0] - window + 1, stride):
            for left in range(0, img.shape[1] - window + 1, stride):
                occluded = img.copy()
                occluded[top : top + window, left : left + window, :] = occluded.mean(axis=(0, 1), keepdims=True)
                occ_tensor = tf.convert_to_tensor(occluded, dtype=tf.float32)
                if preprocess_fn is not None:
                    occ_input = preprocess_fn(occ_tensor * 255.0)
                else:
                    occ_input = occ_tensor
                occ_pred = float(model(tf.expand_dims(occ_input, axis=0), training=False).numpy().reshape(-1)[0])
                drop = max(base_pred - occ_pred, 0.0)
                heat[top : top + window, left : left + window] += drop
                counts[top : top + window, left : left + window] += 1.0
        counts[counts == 0] = 1.0
        heat = heat / counts
        if float(np.max(heat)) > 0:
            heat = heat / float(np.max(heat))
        return heat

    for _, row in samples.iterrows():
        image = _load_image(str(row["image_path"]), target_size=(224, 224))
        display_image = _load_image(str(row["image_path"]), target_size=(224, 224), display_normalize=True)
        tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        if preprocess_fn is not None:
            model_input = preprocess_fn(tensor * 255.0)
        else:
            model_input = tensor
        grad = compute_gradcam(model_input)
        occ = occlusion_map(image)
        if float(np.max(grad)) > 0:
            grad = grad / float(np.max(grad))
        if float(np.max(occ)) > 0:
            occ = occ / float(np.max(occ))
        grad_overlay = overlay_heatmap((display_image * 255.0).astype(np.uint8), grad, alpha=0.4)
        occ_overlay = overlay_heatmap((display_image * 255.0).astype(np.uint8), occ, alpha=0.4)

        border_mask = np.zeros_like(grad, dtype=bool)
        border = max(1, int(0.15 * grad.shape[0]))
        border_mask[:border, :] = True
        border_mask[-border:, :] = True
        border_mask[:, :border] = True
        border_mask[:, -border:] = True
        foreground_mask = np.mean(image, axis=-1) < 0.95
        grad_sum = float(np.sum(grad)) + 1e-8
        border_focus_share = float(np.sum(grad[border_mask]) / grad_sum)
        foreground_focus_share = float(np.sum(grad[foreground_mask]) / grad_sum)

        stem = f"{row['case_type']}_{Path(str(row['image_path'])).stem}"
        orig_path = out_dir / f"{stem}_orig.png"
        grad_path = out_dir / f"{stem}_gradcam.png"
        occ_path = out_dir / f"{stem}_occlusion.png"
        Image.fromarray((display_image * 255.0).astype(np.uint8)).save(orig_path)
        Image.fromarray(grad_overlay.astype(np.uint8)).save(grad_path)
        Image.fromarray(occ_overlay.astype(np.uint8)).save(occ_path)
        records.append(
            {
                "family": family,
                "display_name": display_name,
                "case_type": row["case_type"],
                "image_path": row["image_path"],
                "y_true": row["y_true"],
                "y_prob": row["y_prob"],
                "foreground_focus_share": foreground_focus_share,
                "border_focus_share": border_focus_share,
                "orig_image": str(orig_path),
                "gradcam_image": str(grad_path),
                "occlusion_image": str(occ_path),
            }
        )
    return records


def _plot_explanation_panels(ctx: PipelineContext, family: str, display_name: str, records: pd.DataFrame) -> None:
    ordered = []
    for case_type in ["true_positive", "true_negative", "high_confidence_false_positive", "high_confidence_false_negative"]:
        subset = records[records["case_type"] == case_type]
        if not subset.empty:
            ordered.append(subset.iloc[0])

    if not ordered:
        return

    grad_paths = [Path(row["gradcam_image"]) for row in ordered]
    grad_captions = [f"{row['case_type']}\nprob={row['y_prob']:.3f}" for row in ordered]
    save_image_grid(grad_paths, grad_captions, ctx.output_dir / f"fig_gradcam_panel_{family}.png", f"Grad-CAM: {display_name}", cols=len(ordered))

    occ_paths = [Path(row["occlusion_image"]) for row in ordered]
    occ_captions = [f"{row['case_type']}\nfg={row['foreground_focus_share']:.2f}" for row in ordered]
    save_image_grid(occ_paths, occ_captions, ctx.output_dir / f"fig_explanation_panel_{family}.png", f"Occlusion Sensitivity: {display_name}", cols=len(ordered))


def _explainability_outputs(ctx: PipelineContext, prediction_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    selected = []
    for family in ["cnn", "cnn_attention", "transfer"]:
        df = prediction_map.get(family)
        if df is None or df.empty:
            ctx.warn(f"Skipping explainability for '{family}' because saved prediction rows were unavailable.")
            continue
        test_df = df[df["split"].astype(str) == "test"].copy()
        test_df["pred_label"] = (test_df["y_prob"].astype(float) >= 0.5).astype(int)
        subsets = {
            "true_positive": test_df[(test_df["y_true"] == 1) & (test_df["pred_label"] == 1)].sort_values("y_prob", ascending=False).head(1),
            "true_negative": test_df[(test_df["y_true"] == 0) & (test_df["pred_label"] == 0)].sort_values("y_prob", ascending=True).head(1),
            "high_confidence_false_positive": test_df[(test_df["y_true"] == 0) & (test_df["pred_label"] == 1)].sort_values("y_prob", ascending=False).head(1),
            "high_confidence_false_negative": test_df[(test_df["y_true"] == 1) & (test_df["pred_label"] == 0)].sort_values("y_prob", ascending=True).head(1),
        }
        sample_rows = []
        for case_type, subset in subsets.items():
            if subset.empty:
                continue
            row = subset.iloc[0].to_dict()
            row["case_type"] = case_type
            sample_rows.append(row)
        if not sample_rows:
            continue
        display_name = str(test_df["display_name"].iloc[0])
        model_path = ctx.project_root / "runs" / family / "best_model.keras"
        preprocess_fn = None
        if family == "transfer":
            from focus_binary.models.transfer import get_preprocess
            import json

            summary_json = json.loads((ctx.project_root / "runs" / "transfer" / "summary.json").read_text(encoding="utf-8"))
            preprocess_fn = get_preprocess(summary_json.get("best_hparams", {}).get("backbone", "MobileNet"))
        try:
            records = _gradcam_and_occlusion(ctx, family, display_name, model_path, pd.DataFrame(sample_rows), preprocess_fn=preprocess_fn)
            selected.extend(records)
            _plot_explanation_panels(ctx, family, display_name, pd.DataFrame(records))
        except Exception as exc:  # pragma: no cover - defensive
            ctx.warn(f"Explainability generation failed for '{family}': {exc}")
    explain_df = pd.DataFrame(selected)
    write_dataframe(explain_df, ctx.output_dir / "explainability_samples.csv")
    if explain_df.empty:
        write_text(ctx.output_dir / "explainability_analysis.md", "# Explainability Analysis\n\nExplainability artifacts could not be generated from the current saved models and prediction files.")
        return explain_df

    recovered = ", ".join(f"`{family}`" for family in sorted(explain_df["family"].unique()))
    lines = [
        "# Explainability Analysis",
        "",
        f"- Grad-CAM and occlusion-sensitivity panels were successfully generated for: {recovered}.",
        "- The saved-artifact pipeline attempted both the top custom CNN and the best saved transfer model; any missing model reflects a recoverability issue that is documented in `warnings_and_limitations.md`.",
        "- Quantitative support fields were restricted to simple heatmap-mass heuristics (`foreground_focus_share`, `border_focus_share`) to avoid overclaiming from qualitative overlays.",
        "- Across the selected examples, heatmaps were interpreted conservatively: foreground-heavy attention is compatible with attention to smear material, but the current artifacts do not justify strong claims about specific subcellular structures such as nuclei.",
        "",
        "Observed heuristic summary:",
        "",
    ]
    for family, fam_df in explain_df.groupby("family"):
        lines.append(
            f"- `{family}`: median foreground focus share `{format_float(fam_df['foreground_focus_share'].median())}`, median border focus share `{format_float(fam_df['border_focus_share'].median())}`."
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Higher foreground-focus share suggests that the explanation mass concentrated on non-background image regions.",
            "- Elevated border-focus share in an error case should be discussed as a potential artifact-attention failure mode rather than as evidence of clinically meaningful localization.",
        ]
    )
    write_text(ctx.output_dir / "explainability_analysis.md", "\n".join(lines))
    return explain_df


def run_priority2(ctx: PipelineContext, priority1_outputs: dict[str, Any]) -> dict[str, Any]:
    summary_df: pd.DataFrame = priority1_outputs["summary"]
    prediction_map = _load_representative_predictions(ctx, summary_df)
    statuses = _priority2_feasibility(ctx, prediction_map)
    _write_feasibility_report(ctx, statuses)

    curve_outputs = {}
    calibration_df = pd.DataFrame()
    failure_df = pd.DataFrame()
    explain_df = pd.DataFrame()

    if statuses["threshold_analysis"] == "feasible_now":
        curve_outputs = _threshold_and_curve_outputs(ctx, prediction_map)
    if statuses["calibration_analysis"] == "feasible_now":
        calibration_df = _calibration_outputs(ctx, prediction_map)
    if statuses["failure_case_gallery"] == "feasible_now" and not summary_df.empty:
        best_model_name = str(summary_df.sort_values(["balanced_accuracy", "mcc"], ascending=[False, False]).iloc[0]["display_name"])
        failure_df = _failure_gallery(ctx, prediction_map, best_model_name)
    if statuses["explainability_analysis"] == "feasible_now":
        explain_df = _explainability_outputs(ctx, prediction_map)
        if explain_df.empty:
            statuses["explainability_analysis"] = "feasible_with_minor_fixes"
        elif explain_df["family"].nunique() < 2:
            statuses["explainability_analysis"] = "feasible_with_minor_fixes"

    _write_feasibility_report(ctx, statuses)

    ctx.log("Completed Priority 2 feasibility and inference-only post-processing.")
    return {
        "prediction_map": prediction_map,
        "statuses": statuses,
        "threshold": curve_outputs,
        "calibration": calibration_df,
        "failure": failure_df,
        "explainability": explain_df,
    }
