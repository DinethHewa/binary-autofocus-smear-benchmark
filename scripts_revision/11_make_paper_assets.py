from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from revision_utils import (  # noqa: E402
    append_missing,
    build_metadata,
    config_hash,
    display_model_name,
    ensure_dir,
    fresh_all,
    load_config,
    repo_path,
    safe_write_csv,
    safe_write_text,
    save_metadata_for_outputs,
    standardize_model_name,
)


SCRIPT_NAME = Path(__file__).name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-ready BSPC revision tables and figures.")
    parser.add_argument("--config", default="configs/bspc_revision.yaml")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _copy_or_empty(src: Path, dest: Path, columns: list[str]) -> None:
    if src.exists():
        df = pd.read_csv(src)
    else:
        df = pd.DataFrame(columns=columns)
    safe_write_csv(df, dest)


def _datasetwise_leaders(audit_path: Path, dest: Path) -> None:
    if not audit_path.exists():
        safe_write_csv(pd.DataFrame(columns=["dataset", "model_display_name", "balanced_accuracy", "AUC"]), dest)
        return
    df = pd.read_csv(audit_path)
    if df.empty:
        safe_write_csv(df, dest)
        return
    leaders = (
        df.sort_values(["dataset", "balanced_accuracy", "AUC", "F1"], ascending=[True, False, False, False])
        .groupby("dataset", as_index=False)
        .head(1)
    )
    safe_write_csv(leaders, dest)


def _efficiency_summary(config: dict, dest: Path) -> None:
    leaderboard = repo_path((config.get("paths") or {}).get("final_reports_dir", "reports/final")) / "leaderboard.csv"
    if not leaderboard.exists():
        safe_write_csv(pd.DataFrame(columns=["model_display_name", "params_count", "latency_ms_mean"]), dest)
        return
    df = pd.read_csv(leaderboard)
    if "family" in df.columns:
        df["model_display_name"] = df["family"].map(lambda x: display_model_name(str(x), config))
    cols = [c for c in ["family", "model_display_name", "params_count", "latency_ms_mean", "latency_ms_p95", "training_walltime_s", "hardware"] if c in df.columns]
    safe_write_csv(df[cols], dest)


def _plot_workflow(path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    labels = ["Artifact audit", "Validation probabilities", "Threshold selection", "CAOPS", "Statistics", "Paper assets"]
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.axis("off")
    x_positions = np.linspace(0.08, 0.92, len(labels))
    for x, label in zip(x_positions, labels):
        ax.text(x, 0.55, label, ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.35", fc="#f2f2f2", ec="#333333", lw=1))
    for x0, x1 in zip(x_positions[:-1], x_positions[1:]):
        ax.annotate("", xy=(x1 - 0.065, 0.55), xytext=(x0 + 0.065, 0.55), arrowprops=dict(arrowstyle="->", lw=1.4, color="#333333"))
    ax.set_title("Revision Analysis Workflow", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, path)


def _save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    try:
        fig.savefig(path.with_suffix(".pdf"))
    except Exception:
        pass


def _placeholder(path: Path, message: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    fig.tight_layout()
    _save_fig(fig, path)
    plt.close(fig)


def _metric_ranking_ci(stats_path: Path, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not stats_path.exists():
        _placeholder(path, "Bootstrap confidence intervals not available")
        return
    df = pd.read_csv(stats_path)
    work = df[df["metric"] == "balanced_accuracy"].copy()
    if work.empty:
        _placeholder(path, "Balanced accuracy confidence intervals not available")
        return
    work = work.sort_values("estimate", ascending=True)
    y = np.arange(len(work))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(work["estimate"], y, xerr=[work["estimate"] - work["ci_low"], work["ci_high"] - work["estimate"]], fmt="o", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(work["model_display_name"])
    ax.set_xlabel("Balanced accuracy")
    ax.set_title("Model Ranking with 95% Bootstrap CI")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    _save_fig(fig, path)
    plt.close(fig)


def _load_test_predictions(pred_dir: Path, config: dict) -> list[pd.DataFrame]:
    frames = []
    for model in [standardize_model_name(m) for m in config.get("model_families", [])]:
        p = pred_dir / f"test_predictions_{model}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if df.empty:
            continue
        df["model_display_name"] = display_model_name(model, config)
        frames.append(df)
    return frames


def _calibration_plot(frames: list[pd.DataFrame], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not frames:
        _placeholder(path, "Calibration predictions not available")
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Ideal")
    bins = np.linspace(0, 1, 11)
    for df in frames:
        y = df["true_label"].astype(int).to_numpy()
        p = df["probability_focused"].astype(float).to_numpy()
        xs, ys = [], []
        for i in range(10):
            mask = (p >= bins[i]) & ((p <= bins[i + 1]) if i == 9 else (p < bins[i + 1]))
            if mask.any():
                xs.append(float(p[mask].mean()))
                ys.append(float(y[mask].mean()))
        ax.plot(xs, ys, marker="o", linewidth=1.2, markersize=3, label=df["model_display_name"].iloc[0])
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed focused fraction")
    ax.set_title("Calibration Reliability")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    _save_fig(fig, path)
    plt.close(fig)


def _roc_pr_plots(frames: list[pd.DataFrame], roc_path: Path, pr_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, roc_curve
    except Exception:
        _placeholder(roc_path, "ROC curve dependencies not available")
        _placeholder(pr_path, "PR curve dependencies not available")
        return
    if not frames:
        _placeholder(roc_path, "Test predictions not available")
        _placeholder(pr_path, "Test predictions not available")
        return
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
    for df in frames:
        y = df["true_label"].astype(int).to_numpy()
        p = df["probability_focused"].astype(float).to_numpy()
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        label = df["model_display_name"].iloc[0]
        ax_roc.plot(fpr, tpr, linewidth=1.2, label=label)
        ax_pr.plot(rec, prec, linewidth=1.2, label=label)
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC Curves")
    ax_roc.grid(alpha=0.25)
    ax_roc.legend(fontsize=7, ncol=2)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curves")
    ax_pr.grid(alpha=0.25)
    ax_pr.legend(fontsize=7, ncol=2)
    fig_roc.tight_layout()
    fig_pr.tight_layout()
    _save_fig(fig_roc, roc_path)
    _save_fig(fig_pr, pr_path)
    plt.close(fig_roc)
    plt.close(fig_pr)


def _latex_blocks(table_dir: Path, fig_dir: Path, export_dir: Path) -> None:
    tables = sorted([p for p in table_dir.glob("table_*.csv")])
    figures = sorted([p for p in fig_dir.glob("fig_*.png")])
    table_lines = ["% Auto-generated table includes"]
    for p in tables:
        label = p.stem.replace("_", "-")
        table_lines.append(f"% {p.name}")
        table_lines.append(f"\\input{{{p.with_suffix('').as_posix()}}} % label: tab:{label}")
    figure_lines = ["% Auto-generated figure blocks"]
    for p in figures:
        label = p.stem.replace("_", "-")
        figure_lines.extend(
            [
                "\\begin{figure}[t]",
                "\\centering",
                f"\\includegraphics[width=0.95\\linewidth]{{{p.as_posix()}}}",
                f"\\caption{{{p.stem.replace('_', ' ').title()}.}}",
                f"\\label{{fig:{label}}}",
                "\\end{figure}",
                "",
            ]
        )
    safe_write_text("\n".join(table_lines) + "\n", export_dir / "table_blocks_latex.tex")
    safe_write_text("\n".join(figure_lines) + "\n", export_dir / "figure_blocks_latex.tex")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    cfg_hash = config_hash(config)
    table_dir = ensure_dir((config.get("subdirs") or {}).get("tables", "revision_outputs/tables"))
    fig_dir = ensure_dir((config.get("subdirs") or {}).get("figures", "revision_outputs/figures"))
    export_dir = ensure_dir((config.get("subdirs") or {}).get("paper_exports", "revision_outputs/paper_exports"))
    audit_dir = repo_path((config.get("subdirs") or {}).get("audit", "revision_outputs/audit"))
    arch_dir = repo_path((config.get("subdirs") or {}).get("architecture", "revision_outputs/architecture"))
    pred_dir = repo_path((config.get("subdirs") or {}).get("predictions", "revision_outputs/predictions"))
    stat_dir = repo_path((config.get("subdirs") or {}).get("statistics", "revision_outputs/statistics"))
    threshold_dir = repo_path((config.get("subdirs") or {}).get("thresholds", "revision_outputs/thresholds"))
    caops_dir = repo_path((config.get("subdirs") or {}).get("caops", "revision_outputs/caops"))
    loso_dir = repo_path((config.get("subdirs") or {}).get("loso", "revision_outputs/loso"))
    outputs = [
        table_dir / "table_label_provenance.csv",
        table_dir / "table_datasetwise_leading_models.csv",
        table_dir / "table_efficiency_summary.csv",
        fig_dir / "fig_revision_workflow.png",
        fig_dir / "fig_metric_ranking_ci.png",
        fig_dir / "fig_calibration_reliability.png",
        fig_dir / "fig_pr_curves.png",
        fig_dir / "fig_roc_curves.png",
        export_dir / "table_blocks_latex.tex",
        export_dir / "figure_blocks_latex.tex",
    ]
    input_files = [args.config, audit_dir / "label_provenance_table.csv", audit_dir / "anomaly_audit_table.csv"]
    if not args.force and fresh_all(outputs, input_files, cfg_hash):
        print("paper assets are fresh; skipping")
        return 0

    _copy_or_empty(audit_dir / "label_provenance_table.csv", table_dir / "table_label_provenance.csv", ["dataset", "label_provenance_category"])
    if not (table_dir / "table_architecture_summary.csv").exists():
        safe_write_csv(pd.DataFrame(columns=["model_code_name", "model_display_name", "attention_mechanism"]), table_dir / "table_architecture_summary.csv")
    _datasetwise_leaders(audit_dir / "anomaly_audit_table.csv", table_dir / "table_datasetwise_leading_models.csv")
    _copy_or_empty(threshold_dir / "test_metrics_by_selected_threshold.csv", table_dir / "table_threshold_selection.csv", ["model_code_name", "selection_rule"])
    _copy_or_empty(caops_dir / "caops_test_results.csv", table_dir / "table_caops_comparison.csv", ["model_code_name", "selection_rule"])
    _copy_or_empty(stat_dir / "paired_tests.csv", table_dir / "table_statistical_comparison.csv", ["model_a_code_name", "model_b_code_name"])
    _copy_or_empty(loso_dir / "metrics" / "loso_metrics.csv", table_dir / "table_loso_metrics.csv", ["model_code_name", "holdout_dataset", "status"])
    if not (table_dir / "table_multiseed_pooled_metrics.csv").exists():
        safe_write_csv(pd.DataFrame(columns=["model_code_name", "model_display_name", "n_seeds", "status"]), table_dir / "table_multiseed_pooled_metrics.csv")
    _efficiency_summary(config, table_dir / "table_efficiency_summary.csv")

    if (caops_dir / "caops_test_results.csv").exists():
        pass
    else:
        append_missing("Paper assets: CAOPS table missing", config)

    _plot_workflow(fig_dir / "fig_revision_workflow.png")
    _metric_ranking_ci(stat_dir / "bootstrap_ci_main_metrics.csv", fig_dir / "fig_metric_ranking_ci.png")
    frames = _load_test_predictions(pred_dir, config)
    _calibration_plot(frames, fig_dir / "fig_calibration_reliability.png")
    _roc_pr_plots(frames, fig_dir / "fig_roc_curves.png", fig_dir / "fig_pr_curves.png")
    if not (fig_dir / "fig_loso_heatmap_auc.png").exists():
        _placeholder(fig_dir / "fig_loso_heatmap_auc.png", "LOSO metrics not available")
    if not (fig_dir / "fig_loso_heatmap_balanced_accuracy.png").exists():
        _placeholder(fig_dir / "fig_loso_heatmap_balanced_accuracy.png", "LOSO metrics not available")

    _latex_blocks(table_dir, fig_dir, export_dir)
    generated = list(table_dir.glob("table_*.csv")) + list(fig_dir.glob("fig_*.png")) + [
        export_dir / "table_blocks_latex.tex",
        export_dir / "figure_blocks_latex.tex",
    ]
    metadata = build_metadata(script_name=SCRIPT_NAME, input_files=input_files, cfg_hash=cfg_hash, args=args)
    save_metadata_for_outputs(generated, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
