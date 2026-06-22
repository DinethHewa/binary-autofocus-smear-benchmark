from __future__ import annotations

from pathlib import Path

import pandas as pd

from results_enhancement.common import PipelineContext, copy_text, ensure_dir, format_float, write_text
from results_enhancement.inventory import build_inventory, extract_archives
from results_enhancement.priority1 import run_priority1
from results_enhancement.priority2 import run_priority2


def _main_paper_recommendations(priority1: dict) -> str:
    summary_df: pd.DataFrame = priority1["summary"]
    if summary_df.empty:
        strongest = "No representative benchmark row was recoverable."
    else:
        strongest_row = summary_df.sort_values(["balanced_accuracy", "mcc"], ascending=[False, False]).iloc[0]
        strongest = (
            f"`{strongest_row['display_name']}` should anchor the main paper because it combines "
            f"balanced accuracy `{format_float(strongest_row['balanced_accuracy'])}`, MCC `{format_float(strongest_row['mcc'])}`, "
            f"and AUC `{format_float(strongest_row['auc'])}`."
        )

    lines = [
        "# Main Paper Recommendations",
        "",
        "## Main-paper tables",
        "",
        "- `derived_metrics_summary_table.csv`",
        "- `ranking_comparison.csv`",
        "- `efficiency_summary.csv`",
        "",
        "## Main-paper figures",
        "",
        "- `fig_balanced_ranking.png`",
        "- `fig_accuracy_ranking.png`",
        "- `fig_training_curves_loss.png` or `fig_training_curves_accuracy.png`",
        "- `fig_calibration.png`",
        "",
        "## Supplementary tables",
        "",
        "- `derived_metrics_all_models.csv`",
        "- `tuner_trials_cleaned.csv`",
        "- `tuner_factor_effects.csv`",
        "- `training_dynamics_summary.csv`",
        "- `search_space_table.csv`",
        "- `threshold_metrics.csv`",
        "",
        "## Supplementary figures",
        "",
        "- `fig_tuner_distribution.png`",
        "- `fig_hparam_effects.png`",
        "- `fig_search_space_summary.png`",
        "- `fig_roc_curves.png`",
        "- `fig_pr_curves.png`",
        "- `fig_threshold_sweep.png`",
        "- `fig_failure_gallery.png`",
        "- `fig_gradcam_panel_cnn.png` / `fig_explanation_panel_cnn.png`",
        "- `fig_gradcam_panel_cnn_attention.png` / `fig_explanation_panel_cnn_attention.png`",
        "- `fig_gradcam_panel_transfer.png` / `fig_explanation_panel_transfer.png`",
        "",
        "## Suggested Results section structure",
        "",
        "1. Dataset composition and imbalance context",
        "2. Comparative benchmark with imbalance-aware metrics",
        "3. Hyperparameter search and model-selection evidence",
        "4. Training dynamics / convergence",
        "5. Deployment and calibration implications",
        "",
        "## Suggested Discussion section structure",
        "",
        "1. Why accuracy alone is insufficient for this benchmark",
        "2. Which models remain trustworthy under balanced metrics",
        "3. Search stability and overfitting risk",
        "4. Deployment trade-offs",
        "5. Limitations of the saved-artifact-only analysis",
        "",
        "Strongest recommended paper claim:",
        "",
        f"- {strongest}",
    ]
    return "\n".join(lines)


def _final_summary(inventory_rows: int, priority1: dict, priority2: dict) -> str:
    summary_df: pd.DataFrame = priority1["summary"]
    strongest_rows = summary_df.sort_values(["balanced_accuracy", "mcc"], ascending=[False, False]).head(3)
    strongest_points = [
        f"- `{row.display_name}`: balanced accuracy `{format_float(row.balanced_accuracy)}`, MCC `{format_float(row.mcc)}`, AUC `{format_float(row.auc)}`."
        for row in strongest_rows.itertuples()
    ]
    if not strongest_points:
        strongest_points = ["- No representative benchmark row could be recovered."]

    unavailable = []
    for key, value in priority2["statuses"].items():
        if value != "feasible_now":
            unavailable.append(f"- `{key}`: `{value}`")
    if not unavailable:
        unavailable = ["- All requested Priority 2 items were feasible from current artifacts."]

    lines = [
        "# Final Summary",
        "",
        "## Completed successfully",
        "",
        f"- Built `paper_ready_outputs/` and generated `{inventory_rows}` inventory rows.",
        "- Generated Priority 1 tables, figures, and manuscript-ready draft text strictly from saved artifacts.",
        "- Generated Priority 2 feasibility, threshold, calibration, failure-case, and explainability outputs without retraining.",
        "",
        "## Not possible or limited",
        "",
        *unavailable,
        "- No TensorBoard event logs or notebooks were present in the project workspace, so convergence analysis relies on saved CSV histories only.",
        "- Exact declared hyperparameter bounds were not fully recoverable for every family; the search-space report is based on observed completed trials plus config-level metadata.",
        "- `matplotlib` was unavailable in the offline execution environment; chart-oriented figures were rendered with a Pillow fallback, so styling may differ slightly from a native matplotlib export.",
        "",
        "## Artifacts used",
        "",
        "- `reports/final/*.csv` and `reports/final/*.json`",
        "- `runs/*/summary.json`, `best_hparams.json`, `tuning_results.csv`, `kt/*/history.csv`, and top-level `best_model.keras` files",
        "- `runs/classical_ml/*.csv` and `runs/threshold_baselines/*.csv`",
        "- `data/manifest_with_splits.csv` and the manifest-referenced image files",
        "",
        "## Strongest publishable findings",
        "",
        *strongest_points,
        "",
        "## Main paper vs supplementary material",
        "",
        "- Main paper: the representative imbalance-aware benchmark table, the balanced-ranking figure, one convergence figure, the efficiency summary, and calibration / failure-case figures for the leading model(s).",
        "- Supplementary material: full derived metrics, full tuner tables, detailed search-space tables, full threshold sweeps, and per-model explainability panels.",
        "",
        "## Recommended figure and table order",
        "",
        "1. `fig_balanced_ranking.png`",
        "2. `derived_metrics_summary_table.csv`",
        "3. `fig_accuracy_ranking.png`",
        "4. `fig_training_curves_accuracy.png` or `fig_training_curves_loss.png`",
        "5. `efficiency_summary.csv`",
        "6. `fig_calibration.png`",
        "7. `fig_failure_gallery.png`",
        "8. Supplementary: tuner and search-space figures, ROC/PR curves, threshold sweep, explainability panels",
    ]
    return "\n".join(lines)


def main() -> int:
    project_root = Path(__file__).resolve().parent
    output_dir = ensure_dir(project_root / "paper_ready_outputs")
    ctx = PipelineContext(project_root=project_root, output_dir=output_dir)
    ctx.log("Starting paper-ready results enhancement package.")

    extracted_dirs = extract_archives(ctx)
    inventory_df = build_inventory(ctx, extracted_dirs)
    priority1 = run_priority1(ctx)
    priority2 = run_priority2(ctx, priority1)

    final_summary = _final_summary(len(inventory_df), priority1, priority2)
    recommendations = _main_paper_recommendations(priority1)
    final_summary_path = write_text(project_root / "FINAL_SUMMARY.md", final_summary)
    recommendations_path = write_text(project_root / "MAIN_PAPER_RECOMMENDATIONS.md", recommendations)
    copy_text(final_summary_path, output_dir / "FINAL_SUMMARY.md")
    copy_text(recommendations_path, output_dir / "MAIN_PAPER_RECOMMENDATIONS.md")
    ctx.log("Package build complete.")
    ctx.finalize_logs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
