# Derived Metrics Methods

All derived metrics were computed directly from saved confusion counts (`TN`, `FP`, `FN`, `TP`).

- Accuracy = `(TP + TN) / (TP + TN + FP + FN)`
- Precision = `TP / (TP + FP)`
- Recall / Sensitivity = `TP / (TP + FN)`
- Specificity = `TN / (TN + FP)`
- F1-score = `2 * Precision * Recall / (Precision + Recall)`
- Balanced accuracy = `(Recall + Specificity) / 2`
- MCC = `(TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))`
- False positive rate = `FP / (FP + TN)`
- False negative rate = `FN / (FN + TP)`
- Negative predictive value = `TN / (TN + FN)`
- Geometric mean = `sqrt(Recall * Specificity)`

Rules and safeguards:

- AUC was copied only when it already existed in saved artifacts; it was never estimated from confusion counts.
- `collapsed_positive_only` flags models whose predictions contained no negatives (`TN + FN = 0`).
- `collapsed_negative_only` flags models whose predictions contained no positives (`TP + FP = 0`).
- `specificity_zero_flag` marks models with zero specificity.
- `balanced_metric_warning` is raised when the gap between raw accuracy and balanced accuracy was at least 0.05 or when collapse / specificity failure was detected.
- Classical and threshold baseline representatives were validation-selected when a saved validation summary existed; otherwise a test-only fallback was reported explicitly.