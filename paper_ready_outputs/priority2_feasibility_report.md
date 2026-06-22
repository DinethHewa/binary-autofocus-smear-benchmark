# Priority 2 Feasibility Report

Status legend:

- `feasible_now`: all required saved artifacts are present and usable without retraining.
- `feasible_with_minor_fixes`: the core artifacts exist but extra glue code or minor recovery is required.
- `not_feasible_from_current_artifacts`: the required artifacts are absent or incomplete.

## Item Status

- `saved_models`: `feasible_now`
- `preprocessing_logic`: `feasible_now`
- `test_or_validation_images`: `feasible_now`
- `explainability_analysis`: `feasible_now`
- `threshold_analysis`: `feasible_now`
- `failure_case_gallery`: `feasible_now`
- `calibration_analysis`: `feasible_now`

Important caveat:

- Deep-family validation probabilities were not saved in the final report artifacts, so any threshold sweep for deep models is descriptive on the test set unless fresh inference is explicitly run.