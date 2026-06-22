# BSPC Revision Execution Plan

Created before code changes after inspecting repository artifacts.

## Existing Artifact Inventory

- Manifest and splits:
  - `data/manifest_with_splits.csv`
  - `data/manifest_with_splits_with_splits.csv`
  - `data/manifest.csv`
  - `data/splits_meta.json`
- Existing pooled test predictions:
  - `reports/final/predictions.csv`
  - `reports/final/predictions_cnn.csv`
  - `reports/final/predictions_cnn_attention.csv`
  - `reports/final/predictions_cnn_focus_hybrid.csv`
  - `reports/final/predictions_transfer.csv`
  - `reports/final/predictions_hybrid_vit.csv`
  - `reports/final/predictions_focus_dnn.csv`
  - `reports/final/predictions_vit.csv`
- Existing validation/test prediction exports:
  - `journal2_gate_analysis/outputs/predictions/predictions_<model>_val.csv`
  - `journal2_gate_analysis/outputs/predictions/predictions_<model>_test.csv`
  - available for `cnn`, `cnn_attention`, `cnn_focus_hybrid`, `transfer`, `hybrid_vit`, `focus_dnn`, and `vit`
- Existing classical and threshold baseline predictions:
  - `runs/classical_ml/predictions.csv`
  - `runs/classical_ml/metrics.csv`
  - `runs/threshold_baselines/predictions.csv`
  - `runs/threshold_baselines/metrics.csv`
- Existing checkpoints:
  - `runs/<model>/best_model.keras` for the deep model families above
  - many Keras Tuner trial checkpoints under `runs/<model>/kt/`
- Existing reports/tables/figures:
  - `reports/final/leaderboard.csv`
  - `reports/final/per_dataset_metrics.csv`
  - `reports/final/confusion_matrices.json`
  - `reports/final/reliability.json`
  - `paper_ready_outputs/*.csv`, `paper_ready_outputs/*.png`, `paper_ready_outputs/*.md`

`git` is not available on PATH in this environment. Run metadata will record the commit hash as unavailable if it cannot be discovered.

## Global Rules

- New outputs only under `revision_outputs/`.
- No existing outputs will be deleted or overwritten outside `revision_outputs/`.
- Scripts will be idempotent and skip fresh outputs unless `--force` is passed.
- Metadata JSON sidecars will be written next to generated outputs.
- Missing artifacts will be appended to `revision_outputs/logs/missing_artifacts.log`.
- Training is disabled by default. The safe run will not pass `--allow-train` or `--allow-retrain`.

## Planned Stages

| Stage | Reuse existing artifacts | Evaluation only | Retraining required | Expected outputs | Exact command |
|---|---|---:|---:|---|---|
| Config and utilities | Use existing manifests, reports, predictions, checkpoints | No | No | `configs/bspc_revision.yaml`, `src/revision_utils.py` | No direct run |
| 01 anomaly audit | `reports/final/predictions*.csv`, `runs/classical_ml/predictions.csv`, `runs/threshold_baselines/predictions.csv`, manifest | Yes | No | `revision_outputs/audit/anomaly_audit_table.csv`, `revision_outputs/audit/anomaly_flags.csv`, `revision_outputs/audit/anomaly_audit_report.md` | `python scripts_revision/01_anomaly_audit.py --config configs/bspc_revision.yaml` |
| 02 label provenance audit | manifest and split metadata | Yes | No | `revision_outputs/audit/label_provenance_table.csv`, `revision_outputs/audit/label_provenance_report.md` | `python scripts_revision/02_label_provenance_audit.py --config configs/bspc_revision.yaml` |
| 03 architecture specification export | `runs/<model>/best_model.keras`, `runs/<model>/summary.json`, `runs/<model>/best_hparams.json`, model source files | Evaluation/introspection only | No | `revision_outputs/architecture/model_architecture_summary.md`, `revision_outputs/architecture/model_layers_<model>.csv`, `revision_outputs/tables/table_architecture_summary.csv` | `python scripts_revision/03_export_architecture_specs.py --config configs/bspc_revision.yaml` |
| 04 validation probability recovery | `journal2_gate_analysis/outputs/predictions/*_val.csv`, `runs/classical_ml/predictions.csv`, `runs/threshold_baselines/predictions.csv`; checkpoints only if needed | Yes | No by default | `revision_outputs/predictions/validation_predictions_<model>.csv`, `revision_outputs/predictions/validation_prediction_inventory.csv`, `revision_outputs/logs/validation_prediction_recovery.log` | `python scripts_revision/04_recover_validation_predictions.py --config configs/bspc_revision.yaml` |
| 05 validation-selected threshold analysis | recovered validation predictions and existing test predictions | Yes | No | `revision_outputs/thresholds/validation_threshold_grid_<model>.csv`, `revision_outputs/thresholds/selected_thresholds.csv`, `revision_outputs/thresholds/test_metrics_by_selected_threshold.csv`, `revision_outputs/tables/table_threshold_selection.csv`, `revision_outputs/figures/fig_threshold_selection_tradeoff.png` | `python scripts_revision/05_threshold_selection.py --config configs/bspc_revision.yaml` |
| 06 CAOPS | recovered validation predictions and existing test predictions | Yes | No | `revision_outputs/caops/caops_validation_grid.csv`, `revision_outputs/caops/caops_selected_thresholds.csv`, `revision_outputs/caops/caops_test_results.csv`, `revision_outputs/tables/table_caops_comparison.csv`, `revision_outputs/figures/fig_caops_tradeoff.png` | `python scripts_revision/06_caops.py --config configs/bspc_revision.yaml` |
| 07 multiseed planning | existing run folders, checkpoints, predictions, metrics | Inventory only | Only later, if user runs with `--run --allow-train` | `revision_outputs/multiseed/multiseed_run_plan.csv`, `revision_outputs/multiseed/multiseed_status.csv` | `python scripts_revision/07_multiseed_runner.py --config configs/bspc_revision.yaml --plan-only` |
| 08 LOSO planning | existing run folders and any LOSO outputs if present | Inventory only | Only later, if user runs with `--run --allow-train` | `revision_outputs/loso/loso_run_plan.csv`, `revision_outputs/loso/loso_status.csv` | `python scripts_revision/08_loso_runner.py --config configs/bspc_revision.yaml --plan-only` |
| 09 statistical analysis | existing per-sample test predictions | Yes | No | `revision_outputs/statistics/bootstrap_ci_main_metrics.csv`, `revision_outputs/statistics/paired_tests.csv`, `revision_outputs/tables/table_statistical_comparison.csv`, `revision_outputs/figures/fig_main_metrics_with_ci.png` | `python scripts_revision/09_statistical_analysis.py --config configs/bspc_revision.yaml` |
| 10 metric equations export | no model artifacts needed | No | No | `revision_outputs/paper_exports/metric_equations.tex` | `python scripts_revision/10_export_metric_equations.py --config configs/bspc_revision.yaml` |
| 11 paper-ready assets | revised outputs from stages 01-10 plus existing predictions/reports | Yes | No | `revision_outputs/tables/*.csv`, `revision_outputs/figures/*.png`, `revision_outputs/figures/*.pdf`, `revision_outputs/paper_exports/table_blocks_latex.tex`, `revision_outputs/paper_exports/figure_blocks_latex.tex` | `python scripts_revision/11_make_paper_assets.py --config configs/bspc_revision.yaml` |
| 12 master runner | orchestrates the scripts above | Depends on selected stage | No unless `--allow-train --confirm` are used | `revision_outputs/run_summary.md`, `revision_outputs/run_status.json` | See safe commands below |

## Safe Commands To Run Now

1. `python scripts_revision/run_bspc_revision.py --config configs/bspc_revision.yaml --stage audit`
2. `python scripts_revision/run_bspc_revision.py --config configs/bspc_revision.yaml --stage no_retrain`
3. `python scripts_revision/run_bspc_revision.py --config configs/bspc_revision.yaml --stage plan_multiseed`
4. `python scripts_revision/run_bspc_revision.py --config configs/bspc_revision.yaml --stage plan_loso`

## Commands Reserved For Later Training

- Multi-seed missing jobs only:
  - `python scripts_revision/07_multiseed_runner.py --config configs/bspc_revision.yaml --run --allow-train`
- LOSO missing jobs only:
  - `python scripts_revision/08_loso_runner.py --config configs/bspc_revision.yaml --run --allow-train`
- Master missing-training stage:
  - `python scripts_revision/run_bspc_revision.py --config configs/bspc_revision.yaml --stage train_missing --allow-train --confirm`

These later commands must not be run during the first safe/no-retrain pass.
