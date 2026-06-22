# Multiseed Leakage-Safe Manifest

Source manifest: `D:\Dineth\focus_binary_benchmark\focus_binary_benchmark\data\manifest_with_splits.csv`
Output manifest: `D:\Dineth\focus_binary_benchmark\focus_binary_benchmark\revision_outputs\multiseed\leakage_safe_manifest.csv`

The multiseed runner uses this derived manifest because the original split can place copied TBF stacks such as `FieldPos011` and `FieldPos011 (2)` in different splits. The derived manifest canonicalizes copy suffixes in `stack_id`, then creates a deterministic group split.

Rows: 4969
Rows with canonicalized stack IDs: 708
Split seed: 42

The downstream multiseed training command runs with `--leakage-check stack_sha1`, so stack overlap and exact duplicate image leakage still block training. Perceptual-hash-only similarity is not used as a hard blocker for this training path because it over-flags visually similar microscopy fields.

## Split Counts

```
 dataset split  label  count
    TBSI  test      0     35
    TBSI  test      1      5
    TBSI train      0    147
    TBSI train      1     21
    TBSI   val      0     28
    TBSI   val      1      4
     bma  test      0     55
     bma  test      1      5
     bma train      0    297
     bma train      1     27
     bma   val      0     66
     bma   val      1      6
pbs_imgs  test      0    114
pbs_imgs  test      1     11
pbs_imgs train      0    570
pbs_imgs train      1     54
pbs_imgs   val      0    120
pbs_imgs   val      1     12
tbf_imgs  test      0    297
tbf_imgs  test      1     27
tbf_imgs train      0   1353
tbf_imgs train      1    123
tbf_imgs   val      0    352
tbf_imgs   val      1     32
     wbc  test      0    154
     wbc  test      1     22
     wbc train      0    742
     wbc train      1    106
     wbc   val      0    161
     wbc   val      1     23
```
