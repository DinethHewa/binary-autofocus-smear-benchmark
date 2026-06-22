# Final Summary

## Completed successfully

- Built `paper_ready_outputs/` and generated `4140` inventory rows.
- Generated Priority 1 tables, figures, and manuscript-ready draft text strictly from saved artifacts.
- Generated Priority 2 feasibility, threshold, calibration, failure-case, and explainability outputs without retraining.

## Not possible or limited

- All requested Priority 2 items were feasible from current artifacts.
- No TensorBoard event logs or notebooks were present in the project workspace, so convergence analysis relies on saved CSV histories only.
- Exact declared hyperparameter bounds were not fully recoverable for every family; the search-space report is based on observed completed trials plus config-level metadata.
- `matplotlib` was unavailable in the offline execution environment; chart-oriented figures were rendered with a Pillow fallback, so styling may differ slightly from a native matplotlib export.

## Artifacts used

- `reports/final/*.csv` and `reports/final/*.json`
- `runs/*/summary.json`, `best_hparams.json`, `tuning_results.csv`, `kt/*/history.csv`, and top-level `best_model.keras` files
- `runs/classical_ml/*.csv` and `runs/threshold_baselines/*.csv`
- `data/manifest_with_splits.csv` and the manifest-referenced image files

## Strongest publishable findings

- `cnn_attention`: balanced accuracy `0.9706`, MCC `0.8108`, AUC `0.9959`.
- `cnn`: balanced accuracy `0.9644`, MCC `0.8959`, AUC `0.9935`.
- `cnn_focus_hybrid`: balanced accuracy `0.9303`, MCC `0.8117`, AUC `0.9776`.

## Main paper vs supplementary material

- Main paper: the representative imbalance-aware benchmark table, the balanced-ranking figure, one convergence figure, the efficiency summary, and calibration / failure-case figures for the leading model(s).
- Supplementary material: full derived metrics, full tuner tables, detailed search-space tables, full threshold sweeps, and per-model explainability panels.

## Recommended figure and table order

1. `fig_balanced_ranking.png`
2. `derived_metrics_summary_table.csv`
3. `fig_accuracy_ranking.png`
4. `fig_training_curves_accuracy.png` or `fig_training_curves_loss.png`
5. `efficiency_summary.csv`
6. `fig_calibration.png`
7. `fig_failure_gallery.png`
8. Supplementary: tuner and search-space figures, ROC/PR curves, threshold sweep, explainability panels