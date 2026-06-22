# Repo Status

## Model Registry
- cnn: default, minimal, tuned
- cnn_attention: default
- focus_dnn: default, mlp_focus_measures
- cnn_focus_hybrid: cnn_plus_focus, default
- transfer: default
- vit: default
- hybrid: default
- hybrid_vit: default
- efficient_vit: default

## Checks
01. **[PASS]** Model registry - Loaded registry families
02. **[PASS]** Required files - All required files present
03. **[FAIL]** Tuning config coverage - Missing tunable families in configs/tuning.yaml | files: cnn_focus_hybrid, focus_dnn
04. **[PASS]** Leaderboard ingestion (deep) - compare_best scans family run folders
05. **[PASS]** Leaderboard ingestion (classical) - classical_ml ingestion present
06. **[FAIL]** Leaderboard ingestion (threshold baseline) - threshold baseline ingestion missing | files: /home/dineth/focus_measure/journal/focus_binary_benchmark/src/focus_binary/scripts/compare_best.py