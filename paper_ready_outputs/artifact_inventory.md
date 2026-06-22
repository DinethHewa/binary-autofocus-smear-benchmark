# Artifact Inventory

This inventory is scoped to the project workspace rooted at the benchmark repository, plus ZIP archives extracted into the results package.

## Counts By Category

- `csv`: 1080
- `image_folder`: 15
- `json`: 1063
- `saved_model`: 1982

## Key Artifact Summary

| relative_path | category | description | aggregate | per_epoch | per_sample | saved_model | explainability |
|---|---|---|---|---|---|---|---|
| data\manifest.csv | csv | Manifest linking image paths to labels and splits. | no | no | no | no | no |
| data\manifest.csv | csv | Manifest linking image paths to labels and splits. | no | no | no | no | no |
| data\manifest.csv | csv | Manifest linking image paths to labels and splits. | no | no | no | no | no |
| data\manifest_with_splits.csv | csv | Manifest linking image paths to labels and splits. | no | no | no | no | no |
| data\manifest_with_splits.csv | csv | Manifest linking image paths to labels and splits. | no | no | no | no | no |
| data\manifest_with_splits.csv | csv | Manifest linking image paths to labels and splits. | no | no | no | no | no |
| data\manifest_with_splits_with_splits.csv | csv | Manifest linking image paths to labels and splits. | no | no | no | no | no |
| data\manifest_with_splits_with_splits.csv | csv | Manifest linking image paths to labels and splits. | no | no | no | no | no |
| data\manifest_with_splits_with_splits.csv | csv | Manifest linking image paths to labels and splits. | no | no | no | no | no |
| paper_ready_outputs\artifact_inventory.csv | csv | CSV artifact. | no | no | no | no | no |
| paper_ready_outputs\calibration_summary.csv | csv | CSV artifact. | no | no | no | no | yes |
| paper_ready_outputs\derived_metrics_all_models.csv | csv | Aggregate evaluation metrics; may also include confusion counts and calibration fields. | yes | no | no | no | no |
| paper_ready_outputs\derived_metrics_summary_table.csv | csv | Aggregate evaluation metrics; may also include confusion counts and calibration fields. | yes | no | no | no | no |
| paper_ready_outputs\efficiency_summary.csv | csv | CSV artifact. | no | no | no | no | no |
| paper_ready_outputs\explainability_samples.csv | csv | CSV artifact. | no | no | yes | no | yes |
| paper_ready_outputs\extracted_archives\focus_binary_benchmark\data\manifest.csv | csv | Manifest linking image paths to labels and splits. | no | no | no | no | no |
| paper_ready_outputs\extracted_archives\focus_binary_benchmark\data\manifest_with_splits.csv | csv | Manifest linking image paths to labels and splits. | no | no | no | no | no |
| paper_ready_outputs\extracted_archives\focus_binary_benchmark\data\manifest_with_splits_with_splits.csv | csv | Manifest linking image paths to labels and splits. | no | no | no | no | no |
| paper_ready_outputs\extracted_archives\focus_binary_benchmark\reports\final\leaderboard.csv | csv | Aggregate benchmark leaderboard for saved family representatives. | yes | no | no | no | no |
| paper_ready_outputs\extracted_archives\focus_binary_benchmark\reports\final\per_dataset_metrics.csv | csv | Aggregate evaluation metrics; may also include confusion counts and calibration fields. | yes | no | no | no | no |

## Notes

- Detected `992` per-epoch history CSV files under tuner trial directories; these support the convergence analysis even though no TensorBoard event logs were saved.
- Detected `1982` saved model artifacts (`.keras`/`.h5`/`.pb`). The Priority 2 feasibility step will use only the top-level representative models, not every tuner checkpoint.
- No notebooks were detected inside this project workspace.
- No TensorBoard event logs were detected inside this project workspace.
- Full machine-readable inventory: `paper_ready_outputs\artifact_inventory.csv`.