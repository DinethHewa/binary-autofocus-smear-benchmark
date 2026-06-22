# Binary autofocus classification in smear-related microscopy

Software, configuration, derived results, and publication artifacts for the manuscript **“Binary autofocus classification in smear-related microscopy: an imbalance-aware benchmark with calibration, explainability, and deployment analysis.”**

The benchmark compares nine focused-versus-unfocused classifier families on five smear-related microscopy sources under stack-aware data splitting. It reports imbalance-aware discrimination, calibration, threshold sensitivity, explainability, robustness, statistical comparisons, and deployment-oriented efficiency.

## Repository contents

```text
configs/                 Experiment and BSPC revision configuration
data/                    Manifests and split metadata (no image files)
src/                     Reusable Python package
scripts/                 Benchmark and reporting entry points
scripts_revision/        Manuscript-revision audit and reporting pipeline
reports/                 Saved benchmark reports
paper_ready_outputs/     Tables, figures, and manuscript-ready exports
revision_outputs/        Audit, multiseed, LOSO, statistical, and paper outputs
results_enhancement/     Additional derived analyses
```

`PROCEDURE_README_ORIGINAL.md` preserves the original command reference. `ZENODO_FILE_MAP.md` explains the difference between this GitHub-scale package and the complete Zenodo archive.

## Install

Python 3.9 or later is required. A GPU is optional for analysis of saved results but is normally required for practical model retraining.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Inspect the archived results

The main paper outputs are in `paper_ready_outputs/` and the revised publication outputs are in `revision_outputs/`. The completed run summary is `revision_outputs/run_summary.md`.

To regenerate analyses that do not retrain models, first restore the complete Zenodo archive, mount/download the source microscopy datasets, and update `paths.data_root` plus any machine-specific paths in `configs/bspc_revision.yaml`. Then run:

```bash
python scripts_revision/run_bspc_revision.py \
  --config configs/bspc_revision.yaml \
  --stage audit

python scripts_revision/run_bspc_revision.py \
  --config configs/bspc_revision.yaml \
  --stage no_retrain
```

Training stages are deliberately guarded. Review the generated plans before allowing training:

```bash
python scripts_revision/run_bspc_revision.py --config configs/bspc_revision.yaml --stage plan_multiseed
python scripts_revision/run_bspc_revision.py --config configs/bspc_revision.yaml --stage plan_loso
```

## Data and reproducibility boundary

The archive includes manifests, split metadata, saved predictions, result tables, figures, and—only in the complete Zenodo package—trained run artifacts. It does **not** include the raw microscopy images. Manifest paths record the original Windows workstation layout and must be remapped before image-dependent stages can run. See `DATA_AVAILABILITY.md`.

The GitHub package intentionally omits the approximately 25 GB `runs/` directory. Use the Zenodo archive for complete saved model/tuner artifacts.

## License and citation

Code is released under the MIT License. Dataset files are not redistributed and remain subject to their original providers’ terms. Generated results and figures should be cited through the Zenodo record and associated manuscript. See `CITATION.cff`.

## Manuscript-file status

The specifically named source file `BSPC_binary_1_v1.docx` was not present in the supplied workspace and is therefore not silently substituted or redistributed in this package. The procedure-to-paper mapping was verified against the available manuscript PDF and the saved analysis outputs.
