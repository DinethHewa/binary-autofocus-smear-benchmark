# BSPC Revision Run Summary

Stage: `no_retrain`
Config: `configs/bspc_revision.yaml`
Success: True

## Commands

- OK: `D:\Dineth\myenv\Scripts\python.exe -u D:\Dineth\focus_binary_benchmark\focus_binary_benchmark\scripts_revision\04_recover_validation_predictions.py --config configs/bspc_revision.yaml --force`
  - Log: `D:\Dineth\focus_binary_benchmark\focus_binary_benchmark\revision_outputs\logs\01_04_recover_validation_predictions.log`
- OK: `D:\Dineth\myenv\Scripts\python.exe -u D:\Dineth\focus_binary_benchmark\focus_binary_benchmark\scripts_revision\05_threshold_selection.py --config configs/bspc_revision.yaml --force`
  - Log: `D:\Dineth\focus_binary_benchmark\focus_binary_benchmark\revision_outputs\logs\02_05_threshold_selection.log`
- OK: `D:\Dineth\myenv\Scripts\python.exe -u D:\Dineth\focus_binary_benchmark\focus_binary_benchmark\scripts_revision\06_caops.py --config configs/bspc_revision.yaml --force`
  - Log: `D:\Dineth\focus_binary_benchmark\focus_binary_benchmark\revision_outputs\logs\03_06_caops.log`
- OK: `D:\Dineth\myenv\Scripts\python.exe -u D:\Dineth\focus_binary_benchmark\focus_binary_benchmark\scripts_revision\09_statistical_analysis.py --config configs/bspc_revision.yaml --force`
  - Log: `D:\Dineth\focus_binary_benchmark\focus_binary_benchmark\revision_outputs\logs\04_09_statistical_analysis.log`
- OK: `D:\Dineth\myenv\Scripts\python.exe -u D:\Dineth\focus_binary_benchmark\focus_binary_benchmark\scripts_revision\10_export_metric_equations.py --config configs/bspc_revision.yaml --force`
  - Log: `D:\Dineth\focus_binary_benchmark\focus_binary_benchmark\revision_outputs\logs\05_10_export_metric_equations.log`
- OK: `D:\Dineth\myenv\Scripts\python.exe -u D:\Dineth\focus_binary_benchmark\focus_binary_benchmark\scripts_revision\11_make_paper_assets.py --config configs/bspc_revision.yaml --force`
  - Log: `D:\Dineth\focus_binary_benchmark\focus_binary_benchmark\revision_outputs\logs\06_11_make_paper_assets.log`
