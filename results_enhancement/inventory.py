from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from .common import PipelineContext, ensure_dir, read_json, safe_read_csv, write_dataframe, write_text


def extract_archives(ctx: PipelineContext) -> list[Path]:
    archive_dir = ensure_dir(ctx.output_dir / "extracted_archives")
    extracted_dirs: list[Path] = []
    archives = sorted(
        path
        for path in ctx.project_root.rglob("*.zip")
        if ctx.output_dir not in path.parents
    )
    if not archives:
        ctx.log("No ZIP archives were found inside the project workspace.")
        return extracted_dirs

    for archive in archives:
        target_dir = archive_dir / archive.stem
        try:
            if target_dir.exists() and any(target_dir.iterdir()):
                ctx.log(f"Archive already extracted, reusing existing directory: {archive.name}")
            else:
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(target_dir)
                ctx.log(f"Extracted archive: {archive.relative_to(ctx.project_root)} -> {target_dir.relative_to(ctx.project_root)}")
            extracted_dirs.append(target_dir)
        except Exception as exc:  # pragma: no cover - defensive
            ctx.warn(f"Failed to extract archive '{archive}': {exc}")
    return extracted_dirs


def _bool_to_text(value: bool) -> str:
    return "yes" if value else "no"


def _infer_csv_support(path: Path, ctx: PipelineContext) -> tuple[str, dict[str, bool], str]:
    df = safe_read_csv(path, ctx, nrows=10)
    columns = [str(col) for col in df.columns]
    lower_cols = {col.lower() for col in columns}
    aggregate = any(col in lower_cols for col in {"accuracy", "auc", "f1", "precision", "recall", "balanced_accuracy", "mcc"})
    per_epoch = "epoch" in lower_cols
    per_sample = ("image_path" in lower_cols or "path" in lower_cols) and any(col in lower_cols for col in {"y_prob", "pred", "prediction"})
    explainability = any(token in path.name.lower() for token in {"explain", "gradcam", "calibration", "reliability"})

    description = "CSV artifact."
    name = path.name.lower()
    if "tuning_results" in name:
        description = "Hyperparameter trial summary table with trial-level score and searched settings."
    elif "history" in name:
        description = "Per-epoch training history exported from a Keras Tuner trial."
    elif "leaderboard" in name:
        description = "Aggregate benchmark leaderboard for saved family representatives."
    elif "predictions" in name:
        description = "Per-sample prediction probabilities and labels."
    elif "metrics" in name:
        description = "Aggregate evaluation metrics; may also include confusion counts and calibration fields."
    elif "val_curves" in name:
        description = "Saved validation-curve artifact."
    elif "manifest" in name:
        description = "Manifest linking image paths to labels and splits."
    return description, {
        "aggregate_metrics_only": aggregate and not per_epoch and not per_sample,
        "per_epoch_metrics": per_epoch,
        "per_sample_predictions": per_sample,
        "saved_model_inference": False,
        "explainability_analyses": explainability,
    }, ", ".join(columns[:12]) if columns else ""


def _infer_json_support(path: Path, ctx: PipelineContext) -> tuple[str, dict[str, bool], str]:
    payload = read_json(path, ctx)
    keys = list(payload.keys()) if isinstance(payload, dict) else []
    key_text = ", ".join(str(key) for key in keys[:12])
    name = path.name.lower()

    description = "JSON artifact."
    aggregate = False
    per_epoch = False
    per_sample = False
    explainability = any(token in name for token in {"explain", "reliability", "calibration"})
    if "confusion" in name:
        description = "Saved confusion matrices for pooled and per-dataset evaluation."
        aggregate = True
    elif "summary" in name:
        description = "Run summary with model size, latency, and selected hyperparameters."
        aggregate = True
    elif "reliability" in name:
        description = "Calibration/reliability-bin summary."
        aggregate = True
        explainability = True
    elif "repo_status" in name:
        description = "Repository audit summary documenting registry and evaluation coverage."
    elif "best_hparams" in name:
        description = "Best hyperparameters recovered from the tuner state."
    return description, {
        "aggregate_metrics_only": aggregate,
        "per_epoch_metrics": per_epoch,
        "per_sample_predictions": per_sample,
        "saved_model_inference": False,
        "explainability_analyses": explainability,
    }, key_text


def _artifact_description(path: Path, root: Path, ctx: PipelineContext) -> dict[str, Any]:
    suffix = path.suffix.lower()
    rel = str(path.relative_to(root))
    category = "other"
    description = "Artifact."
    support = {
        "aggregate_metrics_only": False,
        "per_epoch_metrics": False,
        "per_sample_predictions": False,
        "saved_model_inference": False,
        "explainability_analyses": False,
    }
    schema_preview = ""

    if suffix == ".csv":
        category = "csv"
        description, support, schema_preview = _infer_csv_support(path, ctx)
    elif suffix == ".json":
        category = "json"
        description, support, schema_preview = _infer_json_support(path, ctx)
    elif suffix == ".ipynb":
        category = "notebook"
        description = "Notebook artifact; useful for code provenance but no notebooks were detected in this workspace."
    elif path.name.startswith("events.out.tfevents"):
        category = "tensorboard_event"
        description = "TensorBoard event log supporting per-epoch training curves."
        support["per_epoch_metrics"] = True
    elif suffix in {".keras", ".h5"}:
        category = "saved_model"
        description = "Saved Keras/TensorFlow model; supports inference and explainability post-processing."
        support["saved_model_inference"] = True
    elif suffix == ".pb":
        category = "saved_model"
        description = "Saved TensorFlow graph/protobuf artifact."
        support["saved_model_inference"] = True

    return {
        "origin_root": str(root),
        "relative_path": rel,
        "path": str(path),
        "category": category,
        "size_bytes": path.stat().st_size,
        "description": description,
        "schema_preview": schema_preview,
        **support,
    }


def _manifest_image_rows(ctx: PipelineContext) -> list[dict[str, Any]]:
    manifest_path = ctx.project_root / "data" / "manifest_with_splits.csv"
    if not manifest_path.exists():
        return []
    manifest = safe_read_csv(manifest_path, ctx)
    if manifest.empty or "image_path" not in manifest.columns or "split" not in manifest.columns:
        return []

    rows: list[dict[str, Any]] = []
    grouped = manifest.groupby(["split", "dataset"], dropna=False)
    for (split, dataset), subset in grouped:
        image_paths = subset["image_path"].astype(str).tolist()
        common_root = ctx.project_root
        try:
            import os

            common_root = Path(os.path.commonpath(image_paths))
        except Exception:
            pass
        rows.append(
            {
                "origin_root": str(ctx.project_root),
                "relative_path": f"manifest::{split}::{dataset}",
                "path": str(common_root),
                "category": "image_folder",
                "size_bytes": "",
                "description": f"Manifest-referenced {split} image pool for dataset '{dataset}' ({len(subset)} images).",
                "schema_preview": "image_path, label, split",
                "aggregate_metrics_only": False,
                "per_epoch_metrics": False,
                "per_sample_predictions": False,
                "saved_model_inference": False,
                "explainability_analyses": False,
            }
        )
    return rows


def build_inventory(ctx: PipelineContext, extracted_dirs: list[Path]) -> pd.DataFrame:
    roots = [ctx.project_root] + extracted_dirs
    rows: list[dict[str, Any]] = []
    for root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if ctx.output_dir in path.parents and root == ctx.project_root:
                continue
            suffix = path.suffix.lower()
            if suffix in {".csv", ".json", ".ipynb", ".keras", ".h5", ".pb"} or path.name.startswith("events.out.tfevents"):
                rows.append(_artifact_description(path, root, ctx))
    rows.extend(_manifest_image_rows(ctx))

    inventory_df = pd.DataFrame(rows).sort_values(["category", "relative_path"]).reset_index(drop=True)
    write_dataframe(inventory_df, ctx.output_dir / "artifact_inventory.csv")

    counts = inventory_df["category"].value_counts().sort_index()
    lines = [
        "# Artifact Inventory",
        "",
        "This inventory is scoped to the project workspace rooted at the benchmark repository, plus ZIP archives extracted into the results package.",
        "",
        "## Counts By Category",
        "",
    ]
    for category, count in counts.items():
        lines.append(f"- `{category}`: {count}")

    lines.extend(
        [
            "",
            "## Key Artifact Summary",
            "",
            "| relative_path | category | description | aggregate | per_epoch | per_sample | saved_model | explainability |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )

    key_patterns = [
        "reports/final/leaderboard.csv",
        "reports/final/per_dataset_metrics.csv",
        "reports/final/predictions.csv",
        "reports/final/confusion_matrices.json",
        "runs/classical_ml/metrics.csv",
        "runs/classical_ml/predictions.csv",
        "runs/threshold_baselines/metrics.csv",
        "runs/threshold_baselines/predictions.csv",
    ]
    key_rows = inventory_df[inventory_df["relative_path"].isin(key_patterns)]
    if key_rows.empty:
        key_rows = inventory_df.head(20)

    for _, row in key_rows.iterrows():
        lines.append(
            "| {relative_path} | {category} | {description} | {aggregate} | {per_epoch} | {per_sample} | {saved_model} | {explainability} |".format(
                relative_path=row["relative_path"],
                category=row["category"],
                description=row["description"],
                aggregate=_bool_to_text(bool(row["aggregate_metrics_only"])),
                per_epoch=_bool_to_text(bool(row["per_epoch_metrics"])),
                per_sample=_bool_to_text(bool(row["per_sample_predictions"])),
                saved_model=_bool_to_text(bool(row["saved_model_inference"])),
                explainability=_bool_to_text(bool(row["explainability_analyses"])),
            )
        )

    history_count = int(
        inventory_df["relative_path"].astype(str).str.contains(r"history\.csv$", regex=True).sum()
    )
    model_count = int(inventory_df["category"].eq("saved_model").sum())
    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- Detected `{history_count}` per-epoch history CSV files under tuner trial directories; these support the convergence analysis even though no TensorBoard event logs were saved.",
            f"- Detected `{model_count}` saved model artifacts (`.keras`/`.h5`/`.pb`). The Priority 2 feasibility step will use only the top-level representative models, not every tuner checkpoint.",
            "- No notebooks were detected inside this project workspace.",
            "- No TensorBoard event logs were detected inside this project workspace.",
            f"- Full machine-readable inventory: `{(ctx.output_dir / 'artifact_inventory.csv').relative_to(ctx.project_root)}`.",
        ]
    )
    write_text(ctx.output_dir / "artifact_inventory.md", "\n".join(lines))
    ctx.log(f"Built artifact inventory with {len(inventory_df)} rows.")
    return inventory_df
