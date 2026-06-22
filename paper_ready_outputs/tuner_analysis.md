# Tuner Analysis

This section uses only saved trial tables from custom CNN / DNN families (`cnn`, `cnn_attention`, `focus_dnn`, `cnn_focus_hybrid`).

## Top 10 Trials

| family | trial_id | score |
|---|---:|---:|
| cnn | 7 | 0.9968 |
| cnn | 46 | 0.9955 |
| cnn_attention | 45 | 0.9950 |
| cnn | 34 | 0.9931 |
| cnn_attention | 35 | 0.9928 |
| cnn_attention | 44 | 0.9928 |
| cnn_attention | 4 | 0.9928 |
| cnn | 76 | 0.9928 |
| cnn_attention | 47 | 0.9925 |
| cnn | 64 | 0.9921 |

## Stability Assessment

- `cnn`: best score `0.9968`, median `0.9169`, `6` trials within 0.005 of the best score. This looks like a broader good-performing region rather than a single isolated outlier.
- `cnn_attention`: best score `0.9950`, median `0.8840`, `5` trials within 0.005 of the best score. This looks like a broader good-performing region rather than a single isolated outlier.
- `cnn_focus_hybrid`: best score `0.9820`, median `0.9480`, `4` trials within 0.005 of the best score. This looks like a broader good-performing region rather than a single isolated outlier.
- `focus_dnn`: best score `0.8871`, median `0.6792`, `2` trials within 0.005 of the best score. This looks fragile/outlier-driven.

## Factor-Level Summary

- Mean and median score summaries by saved hyperparameter category are provided in `tuner_factor_effects.csv`.
- Exact declared tuner bounds were not fully recoverable from the saved artifacts for every family, so this analysis reports observed completed-trial settings rather than claiming full search-space reconstruction.