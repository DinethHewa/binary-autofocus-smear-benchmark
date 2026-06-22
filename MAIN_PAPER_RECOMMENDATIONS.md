# Main Paper Recommendations

## Main-paper tables

- `derived_metrics_summary_table.csv`
- `ranking_comparison.csv`
- `efficiency_summary.csv`

## Main-paper figures

- `fig_balanced_ranking.png`
- `fig_accuracy_ranking.png`
- `fig_training_curves_loss.png` or `fig_training_curves_accuracy.png`
- `fig_calibration.png`

## Supplementary tables

- `derived_metrics_all_models.csv`
- `tuner_trials_cleaned.csv`
- `tuner_factor_effects.csv`
- `training_dynamics_summary.csv`
- `search_space_table.csv`
- `threshold_metrics.csv`

## Supplementary figures

- `fig_tuner_distribution.png`
- `fig_hparam_effects.png`
- `fig_search_space_summary.png`
- `fig_roc_curves.png`
- `fig_pr_curves.png`
- `fig_threshold_sweep.png`
- `fig_failure_gallery.png`
- `fig_gradcam_panel_cnn.png` / `fig_explanation_panel_cnn.png`
- `fig_gradcam_panel_cnn_attention.png` / `fig_explanation_panel_cnn_attention.png`
- `fig_gradcam_panel_transfer.png` / `fig_explanation_panel_transfer.png`

## Suggested Results section structure

1. Dataset composition and imbalance context
2. Comparative benchmark with imbalance-aware metrics
3. Hyperparameter search and model-selection evidence
4. Training dynamics / convergence
5. Deployment and calibration implications

## Suggested Discussion section structure

1. Why accuracy alone is insufficient for this benchmark
2. Which models remain trustworthy under balanced metrics
3. Search stability and overfitting risk
4. Deployment trade-offs
5. Limitations of the saved-artifact-only analysis

Strongest recommended paper claim:

- `cnn_attention` should anchor the main paper because it combines balanced accuracy `0.9706`, MCC `0.8108`, and AUC `0.9959`.