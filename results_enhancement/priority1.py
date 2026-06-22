from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import (
    PipelineContext,
    compute_confusion_metrics,
    family_display_name,
    format_float,
    parse_confusion_matrix,
    read_json,
    save_placeholder_figure,
    safe_read_csv,
    write_dataframe,
    write_text,
)
from .pillow_plots import (
    pillow_available,
    save_distribution_boxplot,
    save_dual_horizontal_bar_chart,
    save_heatmap_table,
    save_horizontal_bar_chart,
    save_line_grid_chart,
    save_panel_bar_chart,
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
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def _load_priority1_sources(ctx: PipelineContext) -> dict[str, Any]:
    return {
        "leaderboard": safe_read_csv(ctx.project_root / "reports" / "final" / "leaderboard.csv", ctx),
        "per_dataset": safe_read_csv(ctx.project_root / "reports" / "final" / "per_dataset_metrics.csv", ctx),
        "confusions": read_json(ctx.project_root / "reports" / "final" / "confusion_matrices.json", ctx),
        "classical_metrics": safe_read_csv(ctx.project_root / "runs" / "classical_ml" / "metrics.csv", ctx),
        "threshold_metrics": safe_read_csv(ctx.project_root / "runs" / "threshold_baselines" / "metrics.csv", ctx),
        "manifest": safe_read_csv(ctx.project_root / "data" / "manifest_with_splits.csv", ctx),
    }


def _deep_metric_rows(ctx: PipelineContext, leaderboard: pd.DataFrame, per_dataset: pd.DataFrame, confusions: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in leaderboard.iterrows():
        family = str(row["family"])
        model_name = str(row["model_name"])
        if family in {"classical_ml", "threshold_baselines"}:
            continue
        cm = parse_confusion_matrix(confusions.get(family, {}).get("pooled"))
        if cm is None:
            ctx.warn(f"Missing pooled confusion matrix for deep family '{family}'.")
            continue
        tn, fp, fn, tp = cm
        metrics = compute_confusion_metrics(tn, fp, fn, tp)
        rows.append(
            {
                "source_group": "deep_family",
                "family": family,
                "model_name": model_name,
                "display_name": family_display_name(family, model_name),
                "split": "test",
                "dataset": "all",
                "threshold": 0.5,
                "auc": row.get("auc"),
                "source_path": "reports/final/leaderboard.csv",
                **metrics,
            }
        )

    if per_dataset.empty:
        return rows

    for _, row in per_dataset.iterrows():
        family = str(row["family"])
        dataset = str(row["dataset"])
        cm = parse_confusion_matrix(confusions.get(family, {}).get(dataset))
        if cm is None:
            ctx.warn(f"Missing per-dataset confusion matrix for '{family}' on '{dataset}'.")
            continue
        tn, fp, fn, tp = cm
        metrics = compute_confusion_metrics(tn, fp, fn, tp)
        rows.append(
            {
                "source_group": "deep_family",
                "family": family,
                "model_name": family,
                "display_name": family,
                "split": "test",
                "dataset": dataset,
                "threshold": 0.5,
                "auc": row.get("auc"),
                "source_path": "reports/final/per_dataset_metrics.csv",
                **metrics,
            }
        )
    return rows


def _baseline_metric_rows(metrics_df: pd.DataFrame, family: str, source_path: str, ctx: PipelineContext) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if metrics_df.empty:
        ctx.warn(f"Metrics file for '{family}' is empty or unreadable.")
        return rows
    for _, row in metrics_df.iterrows():
        if {"tn", "fp", "fn", "tp"}.issubset(metrics_df.columns):
            tn = row.get("tn")
            fp = row.get("fp")
            fn = row.get("fn")
            tp = row.get("tp")
        else:
            parsed = parse_confusion_matrix(row.get("confusion_matrix"))
            if parsed is None:
                ctx.warn(f"Skipping '{family}' row without recoverable confusion terms: {row.to_dict()}")
                continue
            tn, fp, fn, tp = parsed
        metrics = compute_confusion_metrics(float(tn), float(fp), float(fn), float(tp))
        model = str(row.get("model", family))
        rows.append(
            {
                "source_group": family,
                "family": family,
                "model_name": model,
                "display_name": family_display_name(family, model),
                "split": str(row.get("split", "unknown")),
                "dataset": str(row.get("dataset", "all")),
                "threshold": row.get("threshold"),
                "auc": row.get("auc"),
                "ece": row.get("ece"),
                "brier": row.get("brier"),
                "source_path": source_path,
                **metrics,
            }
        )
    return rows


def _representative_baseline_rows(all_rows: pd.DataFrame, family: str, ctx: PipelineContext) -> pd.DataFrame:
    subset = all_rows[(all_rows["family"] == family)]
    if subset.empty:
        ctx.warn(f"No rows available to select representative model for '{family}'.")
        return subset
    val_all = subset[(subset["split"] == "val") & (subset["dataset"] == "all")]
    if not val_all.empty:
        best_val = val_all.sort_values(["auc", "f1"], ascending=[False, False]).iloc[0]
        match = subset[
            (subset["split"] == "test")
            & (subset["dataset"] == "all")
            & (subset["model_name"] == best_val["model_name"])
        ]
        if not match.empty:
            out = match.iloc[[0]].copy()
            out["selection_basis"] = "validation_selected"
            return out
        ctx.warn(f"Validation-selected representative for '{family}' had no corresponding test/all row; falling back to test selection.")
    test_all = subset[(subset["split"] == "test") & (subset["dataset"] == "all")]
    if not test_all.empty:
        out = test_all.sort_values(["auc", "f1"], ascending=[False, False]).iloc[[0]].copy()
        out["selection_basis"] = "test_selected_fallback"
        return out
    return subset.head(1).assign(selection_basis="first_available_fallback")


def _write_metric_methods(ctx: PipelineContext) -> None:
    lines = [
        "# Derived Metrics Methods",
        "",
        "All derived metrics were computed directly from saved confusion counts (`TN`, `FP`, `FN`, `TP`).",
        "",
        "- Accuracy = `(TP + TN) / (TP + TN + FP + FN)`",
        "- Precision = `TP / (TP + FP)`",
        "- Recall / Sensitivity = `TP / (TP + FN)`",
        "- Specificity = `TN / (TN + FP)`",
        "- F1-score = `2 * Precision * Recall / (Precision + Recall)`",
        "- Balanced accuracy = `(Recall + Specificity) / 2`",
        "- MCC = `(TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))`",
        "- False positive rate = `FP / (FP + TN)`",
        "- False negative rate = `FN / (FN + TP)`",
        "- Negative predictive value = `TN / (TN + FN)`",
        "- Geometric mean = `sqrt(Recall * Specificity)`",
        "",
        "Rules and safeguards:",
        "",
        "- AUC was copied only when it already existed in saved artifacts; it was never estimated from confusion counts.",
        "- `collapsed_positive_only` flags models whose predictions contained no negatives (`TN + FN = 0`).",
        "- `collapsed_negative_only` flags models whose predictions contained no positives (`TP + FP = 0`).",
        "- `specificity_zero_flag` marks models with zero specificity.",
        "- `balanced_metric_warning` is raised when the gap between raw accuracy and balanced accuracy was at least 0.05 or when collapse / specificity failure was detected.",
        "- Classical and threshold baseline representatives were validation-selected when a saved validation summary existed; otherwise a test-only fallback was reported explicitly.",
    ]
    write_text(ctx.output_dir / "derived_metrics_methods.md", "\n".join(lines))


def _plot_accuracy_ranking(ctx: PipelineContext, summary_df: pd.DataFrame) -> None:
    if not MATPLOTLIB_AVAILABLE:
        if pillow_available():
            df = summary_df.sort_values("accuracy", ascending=True)
            colors = ["#1f77b4" if family not in {"classical_ml", "threshold_baselines"} else "#7f7f7f" for family in df["family"]]
            save_horizontal_bar_chart(
                ctx.output_dir / "fig_accuracy_ranking.png",
                df["display_name"].tolist(),
                df["accuracy"].astype(float).tolist(),
                "Model Ranking By Raw Accuracy",
                "Accuracy",
                colors=colors,
                subtitle="Rendered with the Pillow fallback because matplotlib is unavailable in this environment.",
            )
            ctx.log("Rendered fig_accuracy_ranking.png with the Pillow fallback renderer.")
        else:
            ctx.warn("Neither matplotlib nor Pillow is available; fig_accuracy_ranking.png is a placeholder.")
            save_placeholder_figure(
                ctx.output_dir / "fig_accuracy_ranking.png",
                "Accuracy Ranking Figure Skipped",
                "Neither matplotlib nor Pillow was available, so the publication-style ranking plot could not be rendered.",
            )
        return
    _set_style()
    df = summary_df.sort_values("accuracy", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = ["#1f77b4" if family not in {"classical_ml", "threshold_baselines"} else "#7f7f7f" for family in df["family"]]
    ax.barh(df["display_name"], df["accuracy"], color=colors)
    ax.set_xlabel("Accuracy")
    ax.set_title("Model Ranking By Raw Accuracy")
    ax.set_xlim(0.0, min(1.0, max(0.6, df["accuracy"].max() + 0.03)))
    for i, value in enumerate(df["accuracy"]):
        ax.text(value + 0.005, i, format_float(value), va="center")
    fig.tight_layout()
    fig.savefig(ctx.output_dir / "fig_accuracy_ranking.png", bbox_inches="tight")
    plt.close(fig)


def _plot_balanced_ranking(ctx: PipelineContext, summary_df: pd.DataFrame) -> None:
    if not MATPLOTLIB_AVAILABLE:
        if pillow_available():
            df = summary_df.sort_values("balanced_accuracy", ascending=True)
            colors = ["#2a9d8f" if family not in {"classical_ml", "threshold_baselines"} else "#7f7f7f" for family in df["family"]]
            save_dual_horizontal_bar_chart(
                ctx.output_dir / "fig_balanced_ranking.png",
                df["display_name"].tolist(),
                df["balanced_accuracy"].astype(float).tolist(),
                df["mcc"].astype(float).tolist(),
                "Ranking By Imbalance-Aware Metrics",
                "Balanced Accuracy",
                "Matthews Correlation Coefficient",
                "Balanced Accuracy",
                "MCC",
                colors=colors,
            )
            ctx.log("Rendered fig_balanced_ranking.png with the Pillow fallback renderer.")
        else:
            ctx.warn("Neither matplotlib nor Pillow is available; fig_balanced_ranking.png is a placeholder.")
            save_placeholder_figure(
                ctx.output_dir / "fig_balanced_ranking.png",
                "Balanced Ranking Figure Skipped",
                "Neither matplotlib nor Pillow was available, so the publication-style ranking plot could not be rendered.",
            )
        return
    _set_style()
    df = summary_df.sort_values("balanced_accuracy", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    colors = ["#2a9d8f" if family not in {"classical_ml", "threshold_baselines"} else "#7f7f7f" for family in df["family"]]
    axes[0].barh(df["display_name"], df["balanced_accuracy"], color=colors)
    axes[0].set_xlabel("Balanced Accuracy")
    axes[0].set_title("Ranking By Balanced Accuracy")
    axes[1].barh(df["display_name"], df["mcc"], color=colors)
    axes[1].set_xlabel("MCC")
    axes[1].set_title("Ranking By MCC")
    for axis, metric in zip(axes, ["balanced_accuracy", "mcc"]):
        values = df[metric].fillna(0.0).tolist()
        for i, value in enumerate(values):
            axis.text(value + 0.01, i, format_float(value), va="center")
    fig.tight_layout()
    fig.savefig(ctx.output_dir / "fig_balanced_ranking.png", bbox_inches="tight")
    plt.close(fig)


def _build_ranking_analysis(ctx: PipelineContext, summary_df: pd.DataFrame) -> pd.DataFrame:
    ranking_df = summary_df.copy()
    metric_order = {
        "accuracy": False,
        "balanced_accuracy": False,
        "mcc": False,
        "specificity": False,
    }
    for metric, ascending in metric_order.items():
        ranking_df[f"rank_{metric}"] = ranking_df[metric].rank(method="min", ascending=ascending).astype(int)
    ranking_df["shift_balanced_vs_accuracy"] = ranking_df["rank_balanced_accuracy"] - ranking_df["rank_accuracy"]
    ranking_df["shift_mcc_vs_accuracy"] = ranking_df["rank_mcc"] - ranking_df["rank_accuracy"]
    ranking_df["shift_specificity_vs_accuracy"] = ranking_df["rank_specificity"] - ranking_df["rank_accuracy"]
    ranking_df = ranking_df.sort_values(["rank_accuracy", "rank_balanced_accuracy", "rank_mcc"])
    write_dataframe(ranking_df, ctx.output_dir / "ranking_comparison.csv")

    spearman_ab = ranking_df["rank_accuracy"].corr(ranking_df["rank_balanced_accuracy"], method="spearman")
    spearman_mcc = ranking_df["rank_accuracy"].corr(ranking_df["rank_mcc"], method="spearman")
    biggest_shift = ranking_df.iloc[ranking_df["shift_balanced_vs_accuracy"].abs().idxmax()]
    lines = [
        "# Ranking Shift Analysis",
        "",
        f"- Compared `{len(ranking_df)}` representative models/families.",
        f"- Spearman correlation between accuracy rank and balanced-accuracy rank: `{format_float(spearman_ab)}`.",
        f"- Spearman correlation between accuracy rank and MCC rank: `{format_float(spearman_mcc)}`.",
        f"- Largest balanced-vs-accuracy rank shift: `{biggest_shift['display_name']}` ({int(biggest_shift['shift_balanced_vs_accuracy']):+d} positions).",
        "",
        "Interpretation:",
        "",
        "- Raw accuracy remained high for multiple models because the negative class dominated the pooled benchmark.",
        "- Balanced accuracy and MCC penalized asymmetric error profiles more strongly, especially when recall and specificity were poorly matched.",
        "- Any row marked with `balanced_metric_warning` should be discussed with imbalance-aware metrics rather than accuracy alone.",
    ]
    write_text(ctx.output_dir / "ranking_shift_analysis.md", "\n".join(lines))

    _plot_accuracy_ranking(ctx, summary_df)
    _plot_balanced_ranking(ctx, summary_df)
    return ranking_df


def _normalize_trials(family: str, df: pd.DataFrame, summary: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    out["family"] = family
    out["tuner_type"] = summary.get("tuner_type")
    out["objective"] = summary.get("objective")
    out["batch_size"] = summary.get("batch_size")
    out["epochs_configured"] = summary.get("epochs")
    out["learning_rate_std"] = out["learning_rate"] if "learning_rate" in out.columns else out.get("lr")
    out["optimizer_std"] = out["optimizer"] if "optimizer" in out.columns else pd.NA
    if "dropout" in out.columns:
        out["dropout_std"] = out["dropout"]
    elif "fusion_dropout" in out.columns:
        out["dropout_std"] = out["fusion_dropout"]
    else:
        out["dropout_std"] = pd.NA
    if "dense_units" in out.columns:
        out["dense_units_std"] = out["dense_units"]
    elif "units" in out.columns:
        out["dense_units_std"] = out["units"]
    elif "fusion_units" in out.columns:
        out["dense_units_std"] = out["fusion_units"]
    elif "head_units" in out.columns:
        out["dense_units_std"] = out["head_units"]
    else:
        out["dense_units_std"] = pd.NA
    out["depth_std"] = (
        out["num_layers"]
        if "num_layers" in out.columns
        else out["num_blocks"]
        if "num_blocks" in out.columns
        else out["fusion_layers"]
        if "fusion_layers" in out.columns
        else pd.NA
    )
    out["backbone_std"] = (
        out["backbone_choice"]
        if "backbone_choice" in out.columns
        else out["backbone"]
        if "backbone" in out.columns
        else pd.NA
    )
    if "score" in out.columns:
        out["score_rank_within_family"] = out["score"].rank(method="min", ascending=False)
    return out


def _numeric_bin(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(pd.NA, index=series.index, dtype="object")
    unique = valid.nunique()
    if unique <= 6:
        return valid.round(6).astype(str).reindex(series.index)
    try:
        binned = pd.qcut(valid, q=min(4, unique), duplicates="drop")
        return binned.astype(str).reindex(series.index)
    except Exception:
        return pd.cut(valid, bins=min(4, unique), duplicates="drop").astype(str).reindex(series.index)


def _build_tuner_analysis(ctx: PipelineContext) -> dict[str, Any]:
    custom_families = ["cnn", "cnn_attention", "focus_dnn", "cnn_focus_hybrid"]
    frames: list[pd.DataFrame] = []
    summaries: dict[str, dict[str, Any]] = {}
    for family in custom_families:
        tuning_path = ctx.project_root / "runs" / family / "tuning_results.csv"
        summary_path = ctx.project_root / "runs" / family / "summary.json"
        if not tuning_path.exists():
            ctx.warn(f"Missing tuning_results.csv for family '{family}'.")
            continue
        df = safe_read_csv(tuning_path, ctx)
        if df.empty:
            ctx.warn(f"Tuning results for family '{family}' were empty.")
            continue
        summary = read_json(summary_path, ctx) if summary_path.exists() else {}
        summaries[family] = summary
        frames.append(_normalize_trials(family, df, summary))

    if not frames:
        empty = pd.DataFrame()
        for name in ["tuner_trials_cleaned.csv", "tuner_factor_effects.csv", "tuner_top_trials.csv"]:
            write_dataframe(empty, ctx.output_dir / name)
        write_text(ctx.output_dir / "tuner_analysis.md", "# Tuner Analysis\n\nNo usable tuner trial tables were found.")
        return {"cleaned": empty, "top_trials": empty}

    cleaned = pd.concat(frames, ignore_index=True, sort=False)
    cleaned = cleaned.sort_values(["family", "score"], ascending=[True, False]).reset_index(drop=True)
    write_dataframe(cleaned, ctx.output_dir / "tuner_trials_cleaned.csv")

    top_trials = cleaned.sort_values("score", ascending=False).head(10).copy()
    write_dataframe(top_trials, ctx.output_dir / "tuner_top_trials.csv")

    factor_rows: list[dict[str, Any]] = []
    factor_cols = [
        "optimizer_std",
        "learning_rate_std",
        "dropout_std",
        "dense_units_std",
        "depth_std",
        "activation",
        "filters_base",
        "kernel_size",
        "attention_type",
        "backbone_std",
    ]
    for family, fam_df in cleaned.groupby("family"):
        for factor in factor_cols:
            if factor not in fam_df.columns:
                continue
            raw = fam_df[factor]
            if raw.dropna().empty:
                continue
            display = _numeric_bin(raw) if factor in {"learning_rate_std", "dropout_std", "dense_units_std"} else raw.astype(str)
            temp = fam_df.assign(_factor_value=display)
            grouped = temp.dropna(subset=["_factor_value"]).groupby("_factor_value")
            for value, group in grouped:
                factor_rows.append(
                    {
                        "family": family,
                        "factor_name": factor,
                        "factor_value": value,
                        "n_trials": int(len(group)),
                        "mean_score": float(group["score"].mean()),
                        "median_score": float(group["score"].median()),
                        "std_score": float(group["score"].std(ddof=0)),
                        "best_score": float(group["score"].max()),
                    }
                )
    factor_effects = pd.DataFrame(factor_rows).sort_values(["family", "factor_name", "mean_score"], ascending=[True, True, False])
    write_dataframe(factor_effects, ctx.output_dir / "tuner_factor_effects.csv")

    _set_style()
    if MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        families = sorted(cleaned["family"].unique())
        data = [cleaned.loc[cleaned["family"] == family, "score"].dropna().to_numpy() for family in families]
        ax.boxplot(data, labels=families, patch_artist=True)
        ax.set_ylabel("Validation score")
        ax.set_title("Distribution Of Saved Tuner Trial Scores")
        fig.tight_layout()
        fig.savefig(ctx.output_dir / "fig_tuner_distribution.png", bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        plot_specs = [
            ("optimizer_std", "Optimizer"),
            ("dropout_std", "Dropout"),
            ("learning_rate_std", "Learning Rate"),
            ("dense_units_std", "Dense / Hidden Units"),
        ]
        for axis, (factor, title) in zip(axes.ravel(), plot_specs):
            rows = factor_effects[factor_effects["factor_name"] == factor]
            if rows.empty:
                axis.text(0.5, 0.5, "No saved data", ha="center", va="center")
                axis.set_axis_off()
                continue
            top_rows = rows.sort_values("mean_score", ascending=False).head(10).sort_values("mean_score")
            labels = [f"{row.family}:{row.factor_value}" for row in top_rows.itertuples()]
            axis.barh(labels, top_rows["mean_score"], color="#4c78a8")
            axis.set_title(title)
            axis.set_xlabel("Mean validation score")
        fig.tight_layout()
        fig.savefig(ctx.output_dir / "fig_hparam_effects.png", bbox_inches="tight")
        plt.close(fig)
    else:
        if pillow_available():
            families = sorted(cleaned["family"].unique())
            data = [cleaned.loc[cleaned["family"] == family, "score"].dropna().astype(float).tolist() for family in families]
            save_distribution_boxplot(
                ctx.output_dir / "fig_tuner_distribution.png",
                families,
                data,
                "Distribution Of Saved Tuner Trial Scores",
                "Validation score",
            )
            plot_specs = [
                ("optimizer_std", "Optimizer"),
                ("dropout_std", "Dropout"),
                ("learning_rate_std", "Learning Rate"),
                ("dense_units_std", "Dense / Hidden Units"),
            ]
            panels = []
            for factor, title in plot_specs:
                rows = factor_effects[factor_effects["factor_name"] == factor]
                top_rows = rows.sort_values("mean_score", ascending=False).head(8).sort_values("mean_score") if not rows.empty else pd.DataFrame()
                panels.append(
                    {
                        "title": title,
                        "labels": [f"{row.family}:{row.factor_value}" for row in top_rows.itertuples()],
                        "values": top_rows["mean_score"].astype(float).tolist() if not top_rows.empty else [],
                        "xlabel": "Mean validation score",
                    }
                )
            save_panel_bar_chart(
                ctx.output_dir / "fig_hparam_effects.png",
                panels,
                "Mean Saved Validation Score By Hyperparameter Category",
                cols=2,
            )
            ctx.log("Rendered tuner distribution and hyperparameter-effect figures with the Pillow fallback renderer.")
        else:
            ctx.warn("Neither matplotlib nor Pillow is available; tuner distribution and hyperparameter-effect figures are placeholders.")
            save_placeholder_figure(
                ctx.output_dir / "fig_tuner_distribution.png",
                "Tuner Distribution Figure Skipped",
                "Neither matplotlib nor Pillow was available in the execution environment.",
            )
            save_placeholder_figure(
                ctx.output_dir / "fig_hparam_effects.png",
                "Hyperparameter Effects Figure Skipped",
                "Neither matplotlib nor Pillow was available in the execution environment.",
            )

    top_lines = ["| family | trial_id | score |", "|---|---:|---:|"]
    for row in top_trials[["family", "trial_id", "score"]].itertuples(index=False):
        top_lines.append(f"| {row.family} | {row.trial_id} | {format_float(row.score)} |")

    analysis_lines = [
        "# Tuner Analysis",
        "",
        "This section uses only saved trial tables from custom CNN / DNN families (`cnn`, `cnn_attention`, `focus_dnn`, `cnn_focus_hybrid`).",
        "",
        "## Top 10 Trials",
        "",
        *top_lines,
        "",
        "## Stability Assessment",
        "",
    ]
    for family, fam_df in cleaned.groupby("family"):
        best = float(fam_df["score"].max())
        median = float(fam_df["score"].median())
        near_best = int((fam_df["score"] >= (best - 0.005)).sum())
        fragile = near_best <= 2 and (best - median) > 0.02
        analysis_lines.append(
            f"- `{family}`: best score `{format_float(best)}`, median `{format_float(median)}`, "
            f"`{near_best}` trials within 0.005 of the best score. "
            + ("This looks fragile/outlier-driven." if fragile else "This looks like a broader good-performing region rather than a single isolated outlier.")
        )
    analysis_lines.extend(
        [
            "",
            "## Factor-Level Summary",
            "",
            "- Mean and median score summaries by saved hyperparameter category are provided in `tuner_factor_effects.csv`.",
            "- Exact declared tuner bounds were not fully recoverable from the saved artifacts for every family, so this analysis reports observed completed-trial settings rather than claiming full search-space reconstruction.",
        ]
    )
    write_text(ctx.output_dir / "tuner_analysis.md", "\n".join(analysis_lines))
    return {"cleaned": cleaned, "top_trials": top_trials, "summaries": summaries}


def _find_history_path(family_dir: Path, trial_id: Any) -> Path | None:
    kt_dir = family_dir / "kt"
    if not kt_dir.exists():
        return None
    desired = str(int(float(trial_id)))
    for path in kt_dir.glob("trial_*/exec_0/history.csv"):
        token = path.parents[1].name.split("_", 1)[-1]
        try:
            if str(int(token)) == desired:
                return path
        except Exception:
            continue
    return None


def _best_epoch_info(history: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    history = history.copy()
    if "val_auc" in history.columns:
        idx_best = history["val_auc"].astype(float).idxmax()
        out["best_epoch_metric"] = "val_auc"
    elif "val_binary_accuracy" in history.columns:
        idx_best = history["val_binary_accuracy"].astype(float).idxmax()
        out["best_epoch_metric"] = "val_binary_accuracy"
    elif "val_loss" in history.columns:
        idx_best = history["val_loss"].astype(float).idxmin()
        out["best_epoch_metric"] = "val_loss"
    else:
        idx_best = history.index[-1]
        out["best_epoch_metric"] = "last_epoch_fallback"
    best_row = history.loc[idx_best]
    out["best_epoch"] = int(best_row["epoch"])
    out["epoch_of_min_val_loss"] = int(history["val_loss"].astype(float).idxmin()) if "val_loss" in history.columns else np.nan
    out["epoch_of_peak_val_accuracy"] = int(history["val_binary_accuracy"].astype(float).idxmax()) if "val_binary_accuracy" in history.columns else np.nan
    if {"binary_accuracy", "val_binary_accuracy"}.issubset(history.columns):
        out["generalization_gap_accuracy"] = float(best_row["binary_accuracy"]) - float(best_row["val_binary_accuracy"])
    else:
        out["generalization_gap_accuracy"] = np.nan
    if {"loss", "val_loss"}.issubset(history.columns):
        out["generalization_gap_loss"] = float(best_row["val_loss"]) - float(best_row["loss"])
    else:
        out["generalization_gap_loss"] = np.nan

    onset = np.nan
    if {"loss", "val_loss"}.issubset(history.columns) and len(history) >= 4:
        loss = history["loss"].astype(float).to_numpy()
        val_loss = history["val_loss"].astype(float).to_numpy()
        for i in range(2, len(history)):
            if val_loss[i] > val_loss[i - 1] > val_loss[i - 2] and loss[i] <= loss[i - 1]:
                onset = int(history.iloc[i]["epoch"])
                break
    out["overfitting_onset_epoch"] = onset
    return out


def _plot_training_grid(ctx: PipelineContext, histories: dict[str, pd.DataFrame], train_col: str, val_col: str, title: str, out_name: str) -> None:
    if not MATPLOTLIB_AVAILABLE:
        valid_items = [(family, df) for family, df in histories.items() if train_col in df.columns and val_col in df.columns]
        if not valid_items:
            return
        if pillow_available():
            panels = []
            for family, history in valid_items:
                panels.append(
                    {
                        "title": family,
                        "xlabel": "Epoch",
                        "ylabel": train_col.replace("_", " ").title(),
                        "series": [
                            {
                                "label": train_col,
                                "x": history["epoch"].astype(float).tolist(),
                                "y": history[train_col].astype(float).tolist(),
                                "color": "#1f77b4",
                            },
                            {
                                "label": val_col,
                                "x": history["epoch"].astype(float).tolist(),
                                "y": history[val_col].astype(float).tolist(),
                                "color": "#d62728",
                            },
                        ],
                    }
                )
            save_line_grid_chart(ctx.output_dir / out_name, title, panels, cols=2)
            ctx.log(f"Rendered {out_name} with the Pillow fallback renderer.")
        else:
            ctx.warn(f"Neither matplotlib nor Pillow is available; {out_name} is a placeholder.")
            save_placeholder_figure(
                ctx.output_dir / out_name,
                f"{title} Skipped",
                "Neither matplotlib nor Pillow was available, so the curve figure could not be rendered.",
            )
        return
    _set_style()
    valid_items = [(family, df) for family, df in histories.items() if train_col in df.columns and val_col in df.columns]
    if not valid_items:
        return
    n = len(valid_items)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3.2 * rows), squeeze=False)
    for axis in axes.ravel():
        axis.set_visible(False)
    for axis, (family, history) in zip(axes.ravel(), valid_items):
        axis.set_visible(True)
        axis.plot(history["epoch"], history[train_col], label=train_col, color="#1f77b4")
        axis.plot(history["epoch"], history[val_col], label=val_col, color="#d62728")
        axis.set_title(family)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(train_col.replace("_", " ").title())
        axis.legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(ctx.output_dir / out_name, bbox_inches="tight")
    plt.close(fig)


def _build_training_dynamics(ctx: PipelineContext) -> dict[str, Any]:
    families = ["cnn", "cnn_attention", "transfer", "vit", "hybrid_vit", "focus_dnn", "cnn_focus_hybrid"]
    rows: list[dict[str, Any]] = []
    histories: dict[str, pd.DataFrame] = {}
    for family in families:
        tuning_path = ctx.project_root / "runs" / family / "tuning_results.csv"
        if not tuning_path.exists():
            continue
        tuning_df = safe_read_csv(tuning_path, ctx)
        if tuning_df.empty or "score" not in tuning_df.columns:
            continue
        best_trial = tuning_df.sort_values("score", ascending=False).iloc[0]
        history_path = _find_history_path(ctx.project_root / "runs" / family, best_trial["trial_id"])
        if history_path is None:
            ctx.warn(f"Could not locate best-trial history.csv for family '{family}'.")
            continue
        history = safe_read_csv(history_path, ctx)
        if history.empty or "epoch" not in history.columns:
            ctx.warn(f"History file for family '{family}' was empty or missing the epoch column.")
            continue
        histories[family] = history
        best_info = _best_epoch_info(history)
        rows.append(
            {
                "family": family,
                "best_trial_id": best_trial["trial_id"],
                "best_trial_score": best_trial["score"],
                "history_path": str(history_path.relative_to(ctx.project_root)),
                **best_info,
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("best_trial_score", ascending=False)
    write_dataframe(summary_df, ctx.output_dir / "training_dynamics_summary.csv")
    _plot_training_grid(ctx, histories, "loss", "val_loss", "Training vs Validation Loss", "fig_training_curves_loss.png")
    _plot_training_grid(ctx, histories, "binary_accuracy", "val_binary_accuracy", "Training vs Validation Accuracy", "fig_training_curves_accuracy.png")
    _plot_training_grid(ctx, histories, "auc", "val_auc", "Training vs Validation AUC", "fig_training_curves_auc.png")

    lines = [
        "# Training Dynamics",
        "",
        "This analysis uses saved `history.csv` files from the best completed tuner trial for each family.",
        "",
        "- No TensorBoard event logs were available, so the reconstruction is limited to the saved CSV histories.",
        "- `best_epoch` follows the saved selection metric priority `val_auc -> val_binary_accuracy -> val_loss`.",
        "- `overfitting_onset_epoch` is a heuristic first epoch with two consecutive validation-loss increases while training loss still decreased.",
    ]
    write_text(ctx.output_dir / "training_dynamics.md", "\n".join(lines))
    return {"summary": summary_df, "histories": histories}


def _build_search_space_summary(ctx: PipelineContext) -> pd.DataFrame:
    tuning_config_path = ctx.project_root / "configs" / "tuning.yaml"
    import yaml

    cfg = yaml.safe_load(tuning_config_path.read_text(encoding="utf-8")) if tuning_config_path.exists() else {}
    family_meta = cfg.get("families", {}) if isinstance(cfg, dict) else {}

    rows: list[dict[str, Any]] = []
    tuned_families = ["cnn", "cnn_attention", "focus_dnn", "cnn_focus_hybrid", "transfer", "vit", "hybrid_vit"]
    for family in tuned_families:
        tuning_path = ctx.project_root / "runs" / family / "tuning_results.csv"
        if not tuning_path.exists():
            continue
        df = safe_read_csv(tuning_path, ctx)
        if df.empty:
            continue
        for column in df.columns:
            if column.startswith("tuner/") or column in {"trial_id", "score", "status", "tuner/trial_id"}:
                continue
            series = df[column].dropna()
            if series.empty:
                continue
            if pd.api.types.is_numeric_dtype(series):
                observed = f"{format_float(series.min(), 6)} .. {format_float(series.max(), 6)}"
                dtype = "numeric"
            else:
                values = sorted({str(value) for value in series.astype(str)})
                observed = ", ".join(values[:8])
                dtype = "categorical"
            rows.append(
                {
                    "family": family,
                    "hyperparameter": column,
                    "observed_type": dtype,
                    "n_unique_observed": int(series.nunique()),
                    "observed_values_or_range": observed,
                    "configured_max_trials": family_meta.get(family, {}).get("max_trials"),
                    "source": f"runs/{family}/tuning_results.csv",
                }
            )
    search_df = pd.DataFrame(rows).sort_values(["family", "hyperparameter"])
    write_dataframe(search_df, ctx.output_dir / "search_space_table.csv")

    _set_style()
    if MATPLOTLIB_AVAILABLE:
        families = sorted(search_df["family"].unique())
        hparams = sorted(search_df["hyperparameter"].unique())
        matrix = np.zeros((len(families), len(hparams)))
        for i, family in enumerate(families):
            present = set(search_df.loc[search_df["family"] == family, "hyperparameter"])
            for j, hparam in enumerate(hparams):
                matrix[i, j] = 1.0 if hparam in present else 0.0
        fig, ax = plt.subplots(figsize=(max(8, len(hparams) * 0.5), max(3.5, len(families) * 0.5)))
        ax.imshow(matrix, cmap="Greys", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(hparams)))
        ax.set_xticklabels(hparams, rotation=60, ha="right")
        ax.set_yticks(range(len(families)))
        ax.set_yticklabels(families)
        ax.set_title("Observed Search Dimensions By Model Family")
        fig.tight_layout()
        fig.savefig(ctx.output_dir / "fig_search_space_summary.png", bbox_inches="tight")
        plt.close(fig)
    else:
        if pillow_available():
            families = sorted(search_df["family"].unique())
            hparams = sorted(search_df["hyperparameter"].unique())
            matrix = np.zeros((len(hparams), len(families)))
            for j, family in enumerate(families):
                present = set(search_df.loc[search_df["family"] == family, "hyperparameter"])
                for i, hparam in enumerate(hparams):
                    matrix[i, j] = 1.0 if hparam in present else 0.0
            save_heatmap_table(
                ctx.output_dir / "fig_search_space_summary.png",
                "Observed Search Dimensions By Model Family",
                hparams,
                families,
                matrix,
            )
            ctx.log("Rendered fig_search_space_summary.png with the Pillow fallback renderer.")
        else:
            ctx.warn("Neither matplotlib nor Pillow is available; fig_search_space_summary.png is a placeholder.")
            save_placeholder_figure(
                ctx.output_dir / "fig_search_space_summary.png",
                "Search-Space Summary Figure Skipped",
                "Neither matplotlib nor Pillow was available in the execution environment.",
            )

    lines = [
        "# Search Space Summary",
        "",
        "This table reports the observed hyperparameter settings present in the saved completed-trial tables.",
        "",
        "- Exact declared lower/upper bounds were not fully recoverable from the saved tuner state for every family.",
        "- `search_space_table.csv` therefore reflects the empirically observed search space, which is the defensible subset of the original space supported by current artifacts.",
    ]
    write_text(ctx.output_dir / "search_space_summary.md", "\n".join(lines))
    return search_df


def _load_layer_count(model_path: Path, ctx: PipelineContext) -> float | None:
    try:
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
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects or None)
        return float(len(model.layers))
    except Exception as exc:  # pragma: no cover - defensive
        ctx.warn(f"Could not load model for layer-count extraction '{model_path.name}': {exc}")
        return None


def _build_efficiency_summary(ctx: PipelineContext, summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in summary_df.iterrows():
        family = str(row["family"])
        family_dir = ctx.project_root / "runs" / family
        summary_path = family_dir / "summary.json"
        summary = read_json(summary_path, ctx) if summary_path.exists() else {}
        model_path = family_dir / "best_model.keras"
        model_size_mb = (model_path.stat().st_size / (1024 * 1024)) if model_path.exists() else np.nan
        rows.append(
            {
                "family": family,
                "model_name": row["model_name"],
                "params_count": row.get("params_count"),
                "latency_ms_mean": row.get("latency_ms_mean"),
                "latency_ms_p95": row.get("latency_ms_p95"),
                "model_file_size_mb": model_size_mb,
                "layer_count": _load_layer_count(model_path, ctx) if model_path.exists() else np.nan,
                "hardware": row.get("hardware"),
                "input_size": row.get("input_size"),
                "tuning_walltime_s": row.get("tuning_walltime_s"),
                "feature_extraction_time_s": summary.get("feature_extraction_time_s"),
            }
        )
    efficiency_df = pd.DataFrame(rows).sort_values(["latency_ms_mean", "params_count"], ascending=[True, True])
    write_dataframe(efficiency_df, ctx.output_dir / "efficiency_summary.csv")
    lines = [
        "# Deployment Discussion",
        "",
        "- Saved summary artifacts show a clear trade-off between discrimination and deployability.",
        "- The smallest deep architectures (`cnn_attention`, `cnn_focus_hybrid`, `focus_dnn`) were materially lighter than transfer / transformer families, while the transfer and transformer models retained higher file sizes and/or latency.",
        "- Any deployment argument should therefore report both balanced discrimination and the saved CPU latency / file-size evidence rather than AUC alone.",
        "- No new latency benchmark was run for this package; only existing saved latency artifacts were used.",
    ]
    write_text(ctx.output_dir / "deployment_discussion.md", "\n".join(lines))
    return efficiency_df


def _class_distribution_text(manifest: pd.DataFrame) -> str:
    if manifest.empty or "split" not in manifest.columns or "label" not in manifest.columns:
        return "Saved manifest statistics were unavailable."
    test_df = manifest[manifest["split"] == "test"]
    if test_df.empty:
        return "Saved manifest statistics were unavailable for the test split."
    pos_ratio = test_df["label"].mean()
    return f"The pooled saved test split contains `{len(test_df)}` images with a positive prevalence of `{format_float(pos_ratio)}`."


def _write_manuscript_drafts(
    ctx: PipelineContext,
    summary_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    tuner_outputs: dict[str, Any],
    training_outputs: dict[str, Any],
    manifest: pd.DataFrame,
) -> None:
    best_row = summary_df.sort_values(["balanced_accuracy", "mcc"], ascending=[False, False]).iloc[0]
    biggest_shift = ranking_df.iloc[ranking_df["shift_balanced_vs_accuracy"].abs().idxmax()]
    results_lines = [
        "# Results Draft (Priority 1)",
        "",
        "## Comparative Benchmark Overview",
        "",
        _class_distribution_text(manifest),
        "",
        f"The strongest saved overall representative under imbalance-aware ranking was `{best_row['display_name']}` with balanced accuracy `{format_float(best_row['balanced_accuracy'])}`, MCC `{format_float(best_row['mcc'])}`, and AUC `{format_float(best_row['auc'])}`.",
        f"Raw accuracy alone overstated some models: the largest rank shift between accuracy and balanced accuracy was observed for `{biggest_shift['display_name']}` ({int(biggest_shift['shift_balanced_vs_accuracy']):+d} positions).",
        "",
        "## Imbalance-Aware Interpretation",
        "",
        "- Report balanced accuracy, specificity, MCC, and confusion-derived failure rates alongside accuracy.",
        "- Explicitly flag any rows marked `balanced_metric_warning`, `collapsed_positive_only`, or `collapsed_negative_only` in the derived tables.",
        "",
        "## Hyperparameter Search Insights",
        "",
        f"The custom-family tuner analysis recovered `{len(tuner_outputs['cleaned'])}` completed saved trials across CNN/DNN families.",
        "The top-trial table and factor-effect summaries should be interpreted as observed completed-trial evidence rather than a full declarative reconstruction of the search space.",
        "",
        "## Training Dynamics And Convergence",
        "",
        f"Best-trial histories were recovered for `{len(training_outputs['summary'])}` families. No TensorBoard event logs were available, so convergence claims should be limited to the saved CSV histories.",
        "",
        "## Deployment-Relevant Implications",
        "",
        "The efficiency summary shows that smaller custom models can remain competitive while materially reducing saved file size and/or CPU latency relative to heavier transfer and transformer families.",
    ]
    write_text(ctx.output_dir / "results_priority1_draft.md", "\n".join(results_lines))

    discussion_lines = [
        "# Discussion Draft (Priority 1)",
        "",
        "- Accuracy alone is inadequate for this benchmark because the negative class dominates the pooled evaluation set.",
        "- Models with similar accuracy can have materially different specificity, MCC, and balanced accuracy; this affects clinical and deployment trustworthiness.",
        "- Collapse-style failure modes should be discussed explicitly wherever specificity or recall approaches zero on any dataset slice.",
        "- Trust should favor models that remain strong under balanced metrics, show non-fragile tuner behavior, and do not require disproportionate deployment cost.",
        "- The saved artifact set supports strong comparative benchmarking, but not stronger claims about external generalization or threshold transfer without additional held-out evaluation.",
    ]
    write_text(ctx.output_dir / "discussion_priority1_draft.md", "\n".join(discussion_lines))

    figure_lines = [
        "# Figure Captions (Priority 1)",
        "",
        "- `fig_accuracy_ranking.png`: Representative model ranking by pooled test accuracy using saved benchmark artifacts.",
        "- `fig_balanced_ranking.png`: Representative model ranking by balanced accuracy and MCC, highlighting imbalance-aware ordering changes.",
        "- `fig_tuner_distribution.png`: Distribution of saved tuner trial scores for custom CNN/DNN families.",
        "- `fig_hparam_effects.png`: Mean saved validation score by selected hyperparameter groupings across custom CNN/DNN families.",
        "- `fig_training_curves_loss.png`: Saved train/validation loss curves for the best recovered trial in each family.",
        "- `fig_training_curves_accuracy.png`: Saved train/validation binary-accuracy curves for the best recovered trial in each family.",
        "- `fig_training_curves_auc.png`: Saved train/validation AUC curves for families where AUC histories were recorded.",
        "- `fig_search_space_summary.png`: Observed hyperparameter dimensions recovered from saved completed-trial tables.",
    ]
    write_text(ctx.output_dir / "figure_captions_priority1.md", "\n".join(figure_lines))

    table_lines = [
        "# Table Captions (Priority 1)",
        "",
        "- `derived_metrics_summary_table.csv`: Representative pooled test metrics with confusion-derived imbalance-aware endpoints for each family.",
        "- `ranking_comparison.csv`: Model rank shifts when moving from raw accuracy to balanced accuracy, MCC, and specificity.",
        "- `tuner_top_trials.csv`: Top 10 saved completed tuner trials across custom CNN/DNN families.",
        "- `training_dynamics_summary.csv`: Best-epoch and overfitting heuristics reconstructed from saved per-epoch histories.",
        "- `efficiency_summary.csv`: Deployment-relevant summary from saved parameter counts, file sizes, and existing latency artifacts.",
    ]
    write_text(ctx.output_dir / "table_captions_priority1.md", "\n".join(table_lines))


def run_priority1(ctx: PipelineContext) -> dict[str, Any]:
    sources = _load_priority1_sources(ctx)
    deep_rows = _deep_metric_rows(ctx, sources["leaderboard"], sources["per_dataset"], sources["confusions"])
    classical_rows = _baseline_metric_rows(
        sources["classical_metrics"], "classical_ml", "runs/classical_ml/metrics.csv", ctx
    )
    threshold_rows = _baseline_metric_rows(
        sources["threshold_metrics"], "threshold_baselines", "runs/threshold_baselines/metrics.csv", ctx
    )
    all_metrics = pd.DataFrame(deep_rows + classical_rows + threshold_rows)
    if all_metrics.empty:
        ctx.warn("No derived metrics could be built from saved artifacts.")
    write_dataframe(all_metrics, ctx.output_dir / "derived_metrics_all_models.csv")
    _write_metric_methods(ctx)

    deep_summary = all_metrics[(all_metrics["source_group"] == "deep_family") & (all_metrics["dataset"] == "all")]
    classical_summary = _representative_baseline_rows(all_metrics, "classical_ml", ctx)
    threshold_summary = _representative_baseline_rows(all_metrics, "threshold_baselines", ctx)
    summary_df = pd.concat([deep_summary, classical_summary, threshold_summary], ignore_index=True, sort=False)

    leaderboard = sources["leaderboard"].copy()
    merge_cols = [
        "family",
        "model_name",
        "display_name",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "balanced_accuracy",
        "mcc",
        "specificity",
        "false_positive_rate",
        "false_negative_rate",
        "negative_predictive_value",
        "geometric_mean",
        "collapsed_positive_only",
        "collapsed_negative_only",
        "specificity_zero_flag",
        "balanced_metric_warning",
        "tn",
        "fp",
        "fn",
        "tp",
        "n_total",
    ]
    summary_df = summary_df[merge_cols + ["split", "dataset", "threshold", "source_group"]]
    summary_df = summary_df.merge(
        leaderboard[["family", "model_name", "params_count", "input_size", "latency_ms_mean", "latency_ms_p95", "tuning_walltime_s", "training_walltime_s", "hardware"]],
        on=["family", "model_name"],
        how="left",
    )
    summary_df = summary_df.sort_values(
        ["balanced_accuracy", "mcc", "specificity", "accuracy"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    summary_df.insert(0, "rank_balanced_primary", np.arange(1, len(summary_df) + 1))
    write_dataframe(summary_df, ctx.output_dir / "derived_metrics_summary_table.csv")

    ranking_df = _build_ranking_analysis(ctx, summary_df)
    tuner_outputs = _build_tuner_analysis(ctx)
    training_outputs = _build_training_dynamics(ctx)
    search_space_df = _build_search_space_summary(ctx)
    efficiency_df = _build_efficiency_summary(ctx, summary_df)
    _write_manuscript_drafts(ctx, summary_df, ranking_df, tuner_outputs, training_outputs, sources["manifest"])
    ctx.log("Completed Priority 1 outputs from saved artifacts.")
    return {
        "all_metrics": all_metrics,
        "summary": summary_df,
        "ranking": ranking_df,
        "tuner": tuner_outputs,
        "training": training_outputs,
        "search_space": search_space_df,
        "efficiency": efficiency_df,
    }
