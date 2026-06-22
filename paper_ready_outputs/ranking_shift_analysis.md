# Ranking Shift Analysis

- Compared `9` representative models/families.
- Spearman correlation between accuracy rank and balanced-accuracy rank: `0.6667`.
- Spearman correlation between accuracy rank and MCC rank: `0.8833`.
- Largest balanced-vs-accuracy rank shift: `threshold_baselines:composite` (-1 positions).

Interpretation:

- Raw accuracy remained high for multiple models because the negative class dominated the pooled benchmark.
- Balanced accuracy and MCC penalized asymmetric error profiles more strongly, especially when recall and specificity were poorly matched.
- Any row marked with `balanced_metric_warning` should be discussed with imbalance-aware metrics rather than accuracy alone.