# Anomaly Audit Report

Models audited: 9
Dataset/model rows audited: 45
Anomaly flags raised: 21

## Reused Artifacts

- `D:\Dineth\focus_binary_benchmark\focus_binary_benchmark\reports\final\predictions_cnn.csv`
- `D:\Dineth\focus_binary_benchmark\focus_binary_benchmark\reports\final\predictions_cnn_attention.csv`
- `D:\Dineth\focus_binary_benchmark\focus_binary_benchmark\reports\final\predictions_cnn_focus_hybrid.csv`
- `D:\Dineth\focus_binary_benchmark\focus_binary_benchmark\reports\final\predictions_focus_dnn.csv`
- `D:\Dineth\focus_binary_benchmark\focus_binary_benchmark\reports\final\predictions_hybrid_vit.csv`
- `D:\Dineth\focus_binary_benchmark\focus_binary_benchmark\reports\final\predictions_transfer.csv`
- `D:\Dineth\focus_binary_benchmark\focus_binary_benchmark\reports\final\predictions_vit.csv`
- `D:\Dineth\focus_binary_benchmark\focus_binary_benchmark\runs\classical_ml\predictions.csv`
- `D:\Dineth\focus_binary_benchmark\focus_binary_benchmark\runs\threshold_baselines\predictions.csv`

## Missing Artifacts

- None

## Flags

- Classical ML on TBF: AUC <= 0.55 (AUC=0.5000, F1=0.0000, accuracy=0.9167)
- Classical ML on TBF: F1 == 0 (AUC=0.5000, F1=0.0000, accuracy=0.9167)
- Classical ML on TBF: predicted_positive_count == 0 (AUC=0.5000, F1=0.0000, accuracy=0.9167)
- Focus-DNN on BMA: severe dataset-specific failure (AUC=0.8327, F1=0.2778, accuracy=0.5667)
- Focus-DNN on TBF: F1 == 0 (AUC=0.9395, F1=0.0000, accuracy=0.9167)
- Focus-DNN on TBF: predicted_positive_count == 0 (AUC=0.9395, F1=0.0000, accuracy=0.9167)
- Focus-DNN on TBSI: accuracy < 0.50 (AUC=0.9771, F1=0.2222, accuracy=0.1250)
- Focus-DNN on TBSI: predicted_negative_count == 0 (AUC=0.9771, F1=0.2222, accuracy=0.1250)
- Focus-DNN on TBSI: severe dataset-specific failure (AUC=0.9771, F1=0.2222, accuracy=0.1250)
- Threshold baselines on BMA: accuracy < 0.50 (AUC=0.7909, F1=0.2326, accuracy=0.4500)
- Threshold baselines on BMA: severe dataset-specific failure (AUC=0.7909, F1=0.2326, accuracy=0.4500)
- Threshold baselines on PBS: severe dataset-specific failure (AUC=0.8417, F1=0.2985, accuracy=0.6412)
- Threshold baselines on TBF: AUC <= 0.55 (AUC=0.5000, F1=0.0000, accuracy=0.9167)
- Threshold baselines on TBF: F1 == 0 (AUC=0.5000, F1=0.0000, accuracy=0.9167)
- Threshold baselines on TBF: predicted_positive_count == 0 (AUC=0.5000, F1=0.0000, accuracy=0.9167)
- ViT on BMA: F1 == 0 (AUC=0.9527, F1=0.0000, accuracy=0.9167)
- ViT on BMA: predicted_positive_count == 0 (AUC=0.9527, F1=0.0000, accuracy=0.9167)
- ViT on TBF: AUC <= 0.55 (AUC=0.4935, F1=0.0000, accuracy=0.9167)
- ViT on TBF: F1 == 0 (AUC=0.4935, F1=0.0000, accuracy=0.9167)
- ViT on TBF: predicted_positive_count == 0 (AUC=0.4935, F1=0.0000, accuracy=0.9167)
- ViT on TBF: probability orientation suspicion (AUC=0.4935, F1=0.0000, accuracy=0.9167)
