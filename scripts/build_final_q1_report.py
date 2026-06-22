#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from focus_binary import paths
from focus_binary.utils.io import load_json, load_yaml, save_json
from focus_binary.utils.logging import get_logger

logger = get_logger(__name__)


def _str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    val = value.strip().lower()
    if val in {"true", "1", "yes", "y"}:
        return True
    if val in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {value}")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final Q1 report aggregator.")
    parser.add_argument("--manifest", required=True, help="Manifest CSV with splits")
    parser.add_argument("--runs-dir", default="./runs", help="Runs directory")
    parser.add_argument("--out-dir", default="./reports/final_q1", help="Output directory")
    parser.add_argument(
        "--families",
        default="cnn,cnn_attention,transfer,vit,hybrid_vit,focus_dnn,cnn_focus_hybrid,classical_ml,threshold_baselines",
        help="Comma-separated families",
    )
    parser.add_argument("--require-lodo", type=_str2bool, default=True)
    parser.add_argument("--require-explain", type=_str2bool, default=True)
    parser.add_argument("--require-robustness", type=_str2bool, default=True)
    parser.add_argument("--require-stats", type=_str2bool, default=True)
    return parser.parse_args(argv)


def _resolve_path(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = paths.PROJECT_ROOT / p
    return p


def _check_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path)
    required = {"dataset", "image_path", "label", "stack_id", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    return df


def _find_threshold_run(runs_dir: Path) -> Path | None:
    base = runs_dir / "threshold_baselines"
    if not base.exists():
        return None
    candidates = sorted(base.glob("*/metrics.csv"))
    if not candidates:
        return None
    return candidates[-1].parent


def _find_classical_run(runs_dir: Path) -> Path | None:
    base = runs_dir / "classical_ml"
    if not base.exists():
        return None
    if (base / "metrics.csv").exists():
        return base
    return None


def _check_artifacts(runs_dir: Path, families: List[str]) -> None:
    deep = {f for f in families if f not in {"classical_ml", "threshold_baselines"}}
    for family in deep:
        family_dir = runs_dir / family
        best_model = family_dir / "best_model.keras"
        best_hparams = family_dir / "best_hparams.json"
        if not best_hparams.exists():
            raise FileNotFoundError(f"Missing best_hparams for {family}: {best_hparams}")
        if not best_model.exists() and not (family_dir / "best_model.h5").exists():
            raise FileNotFoundError(f"Missing best_model for {family}: {best_model}")

    if "classical_ml" in families:
        if _find_classical_run(runs_dir) is None:
            raise FileNotFoundError("Missing classical_ml outputs under runs/classical_ml")

    if "threshold_baselines" in families:
        if _find_threshold_run(runs_dir) is None:
            raise FileNotFoundError("Missing threshold_baselines outputs under runs/threshold_baselines")


def _aggregate_lodo(lodo_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        lodo_df.groupby("family")[["auc", "f1", "ece", "brier"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    columns = ["family", "auc_mean", "auc_std", "f1_mean", "f1_std", "ece_mean", "ece_std", "brier_mean", "brier_std", "n"]
    data = []
    for _, row in agg.iterrows():
        data.append(
            {
                "family": row["family"],
                "auc_mean": float(row[("auc", "mean")]),
                "auc_std": float(row[("auc", "std")]) if not np.isnan(row[("auc", "std")]) else 0.0,
                "f1_mean": float(row[("f1", "mean")]),
                "f1_std": float(row[("f1", "std")]) if not np.isnan(row[("f1", "std")]) else 0.0,
                "ece_mean": float(row[("ece", "mean")]),
                "ece_std": float(row[("ece", "std")]) if not np.isnan(row[("ece", "std")]) else 0.0,
                "brier_mean": float(row[("brier", "mean")]),
                "brier_std": float(row[("brier", "std")]) if not np.isnan(row[("brier", "std")]) else 0.0,
                "n": int(row[("auc", "count")]),
            }
        )
    return pd.DataFrame(data, columns=columns)


def _load_summary(runs_dir: Path, family: str) -> Dict[str, object]:
    path = runs_dir / family / "summary.json"
    if path.exists():
        return load_json(path)
    return {}


def _leaderboard_md(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# LODO Leaderboard",
        "",
        "Ranking: mean LODO AUC desc, mean LODO F1 desc.",
        "",
        "| rank | family | lodo_auc | lodo_f1 | params_count | latency_ms_mean |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['rank']} | {row['family']} | {row['auc_mean']:.4f}±{row['auc_std']:.4f} | "
            f"{row['f1_mean']:.4f}±{row['f1_std']:.4f} | {row.get('params_count','NA')} | {row.get('latency_ms_mean','NA')} |"
        )
    path.write_text("\n".join(lines))


def _robustness_summary(family_dir: Path) -> Dict[str, float]:
    if not family_dir.exists():
        return {"delta_auc": float("nan"), "delta_f1": float("nan")}
    df = pd.read_csv(family_dir / "robustness_curves.csv")
    if df.empty:
        return {"delta_auc": float("nan"), "delta_f1": float("nan")}
    pooled = df[df["dataset"] == "all"] if "dataset" in df.columns else df
    baseline = pooled[pooled["perturb"] == "clean"]
    if baseline.empty:
        baseline_auc = float("nan")
        baseline_f1 = float("nan")
    else:
        baseline_auc = float(baseline["auc"].iloc[0])
        baseline_f1 = float(baseline["f1"].iloc[0])
    perturbed = pooled[pooled["perturb"] != "clean"]
    if perturbed.empty:
        return {"delta_auc": float("nan"), "delta_f1": float("nan")}
    min_auc = float(perturbed["auc"].min())
    min_f1 = float(perturbed["f1"].min())
    delta_auc = baseline_auc - min_auc if baseline_auc == baseline_auc else float("nan")
    delta_f1 = baseline_f1 - min_f1 if baseline_f1 == baseline_f1 else float("nan")
    return {"delta_auc": float(delta_auc), "delta_f1": float(delta_f1)}


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    runs_dir = _resolve_path(args.runs_dir)
    out_dir = _resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import focus_binary.robust.leakage  # noqa: F401
    except Exception as exc:  # pragma: no cover
        logger.error("Leakage module import failed", extra={"error": str(exc)})
        return 2

    try:
        _check_manifest(_resolve_path(args.manifest))
    except Exception as exc:
        logger.error("Manifest check failed", extra={"error": str(exc)})
        return 2

    try:
        _check_artifacts(runs_dir, families)
    except Exception as exc:
        logger.error("Artifact check failed", extra={"error": str(exc)})
        return 2

    lodo_path = paths.PROJECT_ROOT / "reports" / "lodo" / "lodo_summary_all.csv"
    if not lodo_path.exists() and args.require_lodo:
        logger.error("Missing LODO summary", extra={"path": str(lodo_path)})
        return 2

    lodo_df = pd.read_csv(lodo_path) if lodo_path.exists() else pd.DataFrame()
    if args.require_lodo and lodo_df.empty:
        logger.error("LODO summary is empty", extra={"path": str(lodo_path)})
        return 2

    lodo_summary = _aggregate_lodo(lodo_df) if not lodo_df.empty else pd.DataFrame()
    if not lodo_summary.empty:
        lodo_summary.to_csv(out_dir / "lodo_summary_by_family.csv", index=False)

    leaderboard = lodo_summary.copy()
    if not leaderboard.empty:
        params_list = []
        latency_list = []
        for _, row in leaderboard.iterrows():
            summary = _load_summary(runs_dir, row["family"])
            params_list.append(summary.get("params_count"))
            latency_list.append(summary.get("latency_ms_mean"))
        leaderboard["params_count"] = params_list
        leaderboard["latency_ms_mean"] = latency_list
        leaderboard = leaderboard.sort_values(["auc_mean", "f1_mean"], ascending=False)
        leaderboard["rank"] = np.arange(1, len(leaderboard) + 1)
        leaderboard.to_csv(out_dir / "leaderboard.csv", index=False)
        _leaderboard_md(leaderboard, out_dir / "leaderboard.md")

    if not lodo_summary.empty:
        calib_cols = lodo_summary[["family", "ece_mean", "ece_std", "brier_mean", "brier_std"]]
        calib_cols.to_csv(out_dir / "calibration_summary.csv", index=False)

    explain_rows = []
    explain_root = paths.PROJECT_ROOT / "reports" / "explain"
    for family in families:
        summary_path = explain_root / family / "explainability_summary.json"
        if not summary_path.exists():
            if args.require_explain:
                logger.error("Missing explainability summary", extra={"family": family, "path": str(summary_path)})
                return 2
            continue
        summary = load_json(summary_path)
        explain_rows.append(
            {
                "family": family,
                "stability_mean": summary.get("stability_mean"),
                "deletion_auc_mean": summary.get("deletion_auc_mean"),
                "insertion_auc_mean": summary.get("insertion_auc_mean"),
                "deletion_auc_feature_mean": summary.get("deletion_auc_feature_mean"),
                "top_features": ",".join(summary.get("top_features", [])),
            }
        )
    pd.DataFrame(explain_rows).to_csv(out_dir / "explainability_summary.csv", index=False)

    robustness_rows = []
    robustness_root = paths.PROJECT_ROOT / "reports" / "robustness"
    for family in families:
        family_dir = robustness_root / family
        if not (family_dir / "robustness_curves.csv").exists():
            if args.require_robustness:
                logger.error("Missing robustness curves", extra={"family": family, "path": str(family_dir)})
                return 2
            continue
        deltas = _robustness_summary(family_dir)
        robustness_rows.append({"family": family, **deltas})
    pd.DataFrame(robustness_rows).to_csv(out_dir / "robustness_summary.csv", index=False)

    efficiency_rows = []
    for family in families:
        summary = _load_summary(runs_dir, family)
        efficiency_rows.append(
            {
                "family": family,
                "params_count": summary.get("params_count"),
                "latency_ms_mean": summary.get("latency_ms_mean"),
                "latency_ms_p95": summary.get("latency_ms_p95"),
            }
        )
    pd.DataFrame(efficiency_rows).to_csv(out_dir / "efficiency_summary.csv", index=False)

    # Stats outputs
    stats_auc = out_dir / "stats_auc_ranks.csv"
    if args.require_stats and not stats_auc.exists():
        from focus_binary.scripts.stats_compare_q1 import main as stats_main
        stats_main(["--lodo-summary", str(lodo_path), "--out-dir", str(out_dir)])

    if args.require_stats and not stats_auc.exists():
        logger.error("Missing stats outputs", extra={"path": str(stats_auc)})
        return 2

    report_lines = [
        "# Q1 Final Report",
        "",
        "## Protocol",
        "- Stack-level splitting + leakage audit enforced.",
        "- Thresholds selected on validation only; test never used for thresholding.",
        "- Multi-seed evaluation and LODO cross-dataset testing used for robustness.",
        "",
        "## Evaluation Axes",
        "- LODO generalization (main ranking).",
        "- Perturbation robustness (worst-case deltas).",
        "- Calibration (ECE/Brier).",
        "- Explainability faithfulness (deletion/insertion or feature ablation).",
        "- Statistical tests: Friedman + Nemenyi (alpha=0.05), bootstrap for top-2 AUC.",
        "",
        "## Artifacts",
        f"- Leaderboard: {out_dir / 'leaderboard.csv'}",
        f"- LODO summary: {out_dir / 'lodo_summary_by_family.csv'}",
        f"- Calibration: {out_dir / 'calibration_summary.csv'}",
        f"- Explainability: {out_dir / 'explainability_summary.csv'}",
        f"- Robustness: {out_dir / 'robustness_summary.csv'}",
        f"- Efficiency: {out_dir / 'efficiency_summary.csv'}",
    ]
    (out_dir / "final_q1_report.md").write_text("\n".join(report_lines))

    logger.info("final Q1 report complete", extra={"out_dir": str(out_dir)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
