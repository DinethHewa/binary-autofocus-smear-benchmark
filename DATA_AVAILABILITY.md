# Data availability

This repository does not redistribute the source microscopy images. It contains the locked manifest and split metadata needed to identify the benchmark composition and preserve the stack-aware partition.

The pooled benchmark uses five source groups identified in the manuscript as WBC, TBF, PBS, BMA, and TBSI. Obtain each source from its original provider under that provider's license and terms. Do not infer permission to redistribute an image dataset from the MIT license applied to this code.

After obtaining the datasets:

1. Preserve the `dataset`, `label`, `stack_id`, and `split` columns in `data/manifest_with_splits.csv`.
2. Replace only the machine-specific `image_path` values, or mount the files so those paths resolve.
3. Set `paths.data_root` in `configs/bspc_revision.yaml`.
4. Run the audit stage before any evaluation or retraining.

The saved CSV outputs can be inspected without the images. Image-dependent explainability, robustness, classical focus-measure extraction, and model inference require the source images.
