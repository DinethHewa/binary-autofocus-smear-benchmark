from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from revision_utils import (  # noqa: E402
    append_missing,
    best_baseline_model,
    build_metadata,
    compute_binary_metrics,
    config_hash,
    display_model_name,
    ensure_dir,
    fresh_all,
    load_config,
    load_manifest,
    load_predictions,
    repo_path,
    safe_write_csv,
    safe_write_text,
    save_metadata_for_outputs,
    standardize_model_name,
)


SCRIPT_NAME = Path(__file__).name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit suspicious per-dataset binary autofocus results.")
    parser.add_argument("--config", default="configs/bspc_revision.yaml")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _prediction_path_for_family(config: dict, family: str) -> Path | None:
    sources = config.get("prediction_sources") or {}
    if family in {"classical_ml", "threshold_baselines"}:
        key = "classical_predictions" if family == "classical_ml" else "threshold_predictions"
        p = repo_path(sources.get(key))
        return p if p and p.exists() else None
    final_pattern = sources.get("final_test_pattern", "reports/final/predictions_{model}.csv")
    journal_pattern = sources.get("deep_test_pattern", "journal2_gate_analysis/outputs/predictions/predictions_{model}_test.csv")
    for pattern in [final_pattern, journal_pattern]:
        p = repo_path(str(pattern).format(model=family))
        if p and p.exists():
            return p
    return None


def _load_family_predictions(config: dict, manifest: pd.DataFrame, family: str) -> tuple[pd.DataFrame, Path | None, str | None]:
    path = _prediction_path_for_family(config, family)
    if path is None:
        return pd.DataFrame(), None, f"Missing test predictions for {family}"

    df = load_predictions(path, model_code_name=family, split="test", config=config, manifest=manifest)
    if family == "classical_ml":
        model_name = best_baseline_model((config.get("prediction_sources") or {}).get("classical_metrics"), split="test")
        if model_name and "source_model" in df.columns:
            df = df[df["source_model"] == model_name].copy()
            df["selected_submodel"] = model_name
    elif family == "threshold_baselines":
        model_name = best_baseline_model((config.get("prediction_sources") or {}).get("threshold_metrics"), split="val")
        if model_name and "source_model" in df.columns:
            df = df[df["source_model"] == model_name].copy()
            df["selected_submodel"] = model_name
    if df.empty:
        return df, path, f"No usable test rows for {family} in {path}"
    return df, path, None


def _audit_rows_for_model(df: pd.DataFrame, family: str, config: dict) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    flags: list[dict] = []
    for dataset, group in df.groupby("dataset", dropna=False):
        y_true = group["true_label"].astype(int).to_numpy()
        y_prob = group["probability_focused"].astype(float).to_numpy()
        metrics = compute_binary_metrics(y_true, y_prob, threshold=float(config.get("default_threshold", 0.5)))
        n_focused = int((y_true == 1).sum())
        n_unfocused = int((y_true == 0).sum())
        majority = max(n_focused, n_unfocused) / max(len(y_true), 1)
        auc = float(metrics.get("AUC", np.nan))
        flipped_auc = compute_binary_metrics(y_true, 1.0 - y_prob)["AUC"] if not np.isnan(auc) and auc < 0.5 else np.nan
        row = {
            "dataset": dataset,
            "model_code_name": family,
            "model_display_name": display_model_name(family, config),
            "n_samples": int(len(group)),
            "n_focused": n_focused,
            "n_unfocused": n_unfocused,
            "TP": metrics["TP"],
            "TN": metrics["TN"],
            "FP": metrics["FP"],
            "FN": metrics["FN"],
            "predicted_positive_count": metrics["predicted_positive_count"],
            "predicted_negative_count": metrics["predicted_negative_count"],
            "mean_probability_focused": float(np.nanmean(y_prob)) if len(y_prob) else np.nan,
            "min_probability_focused": float(np.nanmin(y_prob)) if len(y_prob) else np.nan,
            "max_probability_focused": float(np.nanmax(y_prob)) if len(y_prob) else np.nan,
            "AUC": auc,
            "flipped_AUC": flipped_auc,
            "F1": metrics["F1"],
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "recall": metrics["recall"],
            "specificity": metrics["specificity"],
            "MCC": metrics["MCC"],
            "majority_class_baseline": float(majority),
        }
        rows.append(row)

        checks = [
            ("AUC <= 0.55", not np.isnan(auc) and auc <= 0.55),
            ("F1 == 0", abs(float(metrics["F1"])) < 1e-12),
            ("accuracy < 0.50", float(metrics["accuracy"]) < 0.50),
            ("predicted_positive_count == 0", int(metrics["predicted_positive_count"]) == 0),
            ("predicted_negative_count == 0", int(metrics["predicted_negative_count"]) == 0),
            ("probability orientation suspicion", not np.isnan(auc) and auc < 0.5),
            (
                "severe dataset-specific failure",
                float(metrics["accuracy"]) <= float(majority) - 0.20,
            ),
        ]
        for flag_name, triggered in checks:
            if triggered:
                flags.append(
                    {
                        "dataset": dataset,
                        "model_code_name": family,
                        "model_display_name": display_model_name(family, config),
                        "flag": flag_name,
                        "AUC": auc,
                        "flipped_AUC": flipped_auc,
                        "F1": metrics["F1"],
                        "accuracy": metrics["accuracy"],
                        "majority_class_baseline": float(majority),
                    }
                )
    return rows, flags


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    cfg_hash = config_hash(config)
    out_dir = ensure_dir((config.get("subdirs") or {}).get("audit", "revision_outputs/audit"))
    logs_dir = ensure_dir((config.get("subdirs") or {}).get("logs", "revision_outputs/logs"))

    outputs = [
        out_dir / "anomaly_audit_table.csv",
        out_dir / "anomaly_flags.csv",
        out_dir / "anomaly_audit_report.md",
    ]
    manifest_path = repo_path((config.get("paths") or {}).get("manifest"))
    model_families = [standardize_model_name(m) for m in config.get("model_families", [])]
    input_files = [args.config, manifest_path]
    for family in model_families:
        p = _prediction_path_for_family(config, family)
        if p:
            input_files.append(p)

    if not args.force and fresh_all(outputs, input_files, cfg_hash):
        print("anomaly audit outputs are fresh; skipping")
        return 0

    manifest = load_manifest(config=config)
    rows: list[dict] = []
    flags: list[dict] = []
    reused: list[str] = []
    missing: list[str] = []

    for family in model_families:
        preds, source, message = _load_family_predictions(config, manifest, family)
        if source:
            reused.append(str(source))
        if message:
            missing.append(message)
            append_missing(message, config)
        if preds.empty:
            continue
        model_rows, model_flags = _audit_rows_for_model(preds, family, config)
        rows.extend(model_rows)
        flags.extend(model_flags)

    audit_df = pd.DataFrame(rows).sort_values(["model_code_name", "dataset"]) if rows else pd.DataFrame()
    flags_df = pd.DataFrame(flags).sort_values(["model_code_name", "dataset", "flag"]) if flags else pd.DataFrame(
        columns=["dataset", "model_code_name", "model_display_name", "flag"]
    )
    safe_write_csv(audit_df, outputs[0])
    safe_write_csv(flags_df, outputs[1])

    lines = [
        "# Anomaly Audit Report",
        "",
        f"Models audited: {audit_df['model_code_name'].nunique() if not audit_df.empty else 0}",
        f"Dataset/model rows audited: {len(audit_df)}",
        f"Anomaly flags raised: {len(flags_df)}",
        "",
        "## Reused Artifacts",
        "",
    ]
    lines.extend([f"- `{p}`" for p in sorted(set(reused))] or ["- None"])
    lines.extend(["", "## Missing Artifacts", ""])
    lines.extend([f"- {m}" for m in missing] or ["- None"])
    lines.extend(["", "## Flags", ""])
    if flags_df.empty:
        lines.append("No anomaly flags were triggered.")
    else:
        for _, row in flags_df.iterrows():
            lines.append(
                f"- {row['model_display_name']} on {row['dataset']}: {row['flag']} "
                f"(AUC={row.get('AUC', np.nan):.4f}, F1={row.get('F1', np.nan):.4f}, "
                f"accuracy={row.get('accuracy', np.nan):.4f})"
            )
    safe_write_text("\n".join(lines) + "\n", outputs[2])

    metadata = build_metadata(script_name=SCRIPT_NAME, input_files=input_files, cfg_hash=cfg_hash, args=args)
    metadata["reused_artifacts"] = sorted(set(reused))
    metadata["missing_artifacts"] = missing
    save_metadata_for_outputs(outputs, metadata)
    (logs_dir / "anomaly_audit.log").write_text("Completed anomaly audit.\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
