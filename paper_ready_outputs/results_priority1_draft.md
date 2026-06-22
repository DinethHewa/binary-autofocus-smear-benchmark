# Results Draft (Priority 1)

## Comparative Benchmark Overview

The pooled saved test split contains `743` images with a positive prevalence of `0.0956`.

The strongest saved overall representative under imbalance-aware ranking was `cnn_attention` with balanced accuracy `0.9706`, MCC `0.8108`, and AUC `0.9959`.
Raw accuracy alone overstated some models: the largest rank shift between accuracy and balanced accuracy was observed for `threshold_baselines:composite` (-1 positions).

## Imbalance-Aware Interpretation

- Report balanced accuracy, specificity, MCC, and confusion-derived failure rates alongside accuracy.
- Explicitly flag any rows marked `balanced_metric_warning`, `collapsed_positive_only`, or `collapsed_negative_only` in the derived tables.

## Hyperparameter Search Insights

The custom-family tuner analysis recovered `314` completed saved trials across CNN/DNN families.
The top-trial table and factor-effect summaries should be interpreted as observed completed-trial evidence rather than a full declarative reconstruction of the search space.

## Training Dynamics And Convergence

Best-trial histories were recovered for `7` families. No TensorBoard event logs were available, so convergence claims should be limited to the saved CSV histories.

## Deployment-Relevant Implications

The efficiency summary shows that smaller custom models can remain competitive while materially reducing saved file size and/or CPU latency relative to heavier transfer and transformer families.