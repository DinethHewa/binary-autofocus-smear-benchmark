# Discussion Draft (Priority 1)

- Accuracy alone is inadequate for this benchmark because the negative class dominates the pooled evaluation set.
- Models with similar accuracy can have materially different specificity, MCC, and balanced accuracy; this affects clinical and deployment trustworthiness.
- Collapse-style failure modes should be discussed explicitly wherever specificity or recall approaches zero on any dataset slice.
- Trust should favor models that remain strong under balanced metrics, show non-fragile tuner behavior, and do not require disproportionate deployment cost.
- The saved artifact set supports strong comparative benchmarking, but not stronger claims about external generalization or threshold transfer without additional held-out evaluation.