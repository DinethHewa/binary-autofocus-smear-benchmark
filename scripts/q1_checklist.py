#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_IMPORTS = [
    "focus_binary.data.manifest",
    "focus_binary.data.splits",
    "focus_binary.data.tfdata",
    "focus_binary.data.tfdata_features",
    "focus_binary.features.focus_measures",
    "focus_binary.models.registry",
    "focus_binary.models.focus_dnn",
    "focus_binary.models.cnn_focus_hybrid",
    "focus_binary.tuning.tuner",
    "focus_binary.eval.evaluate",
    "focus_binary.eval.report",
    "focus_binary.robust.leakage",
    "focus_binary.calib.calibration",
    "focus_binary.stats.tests",
    "focus_binary.explain.gradcam",
    "focus_binary.explain.vit_rollout",
    "focus_binary.explain.feature_importance",
]

REQUIRED_SCRIPTS = [
    "scripts/make_manifest.py",
    "scripts/tune_family.py",
    "scripts/compare_best.py",
    "scripts/multiseed_eval.py",
    "scripts/robustness_suite.py",
    "scripts/explain_samples.py",
    "scripts/stats_compare.py",
]

REQUIRED_FILES = [
    "src/focus_binary/features/focus_measures.py",
    "src/focus_binary/data/tfdata_features.py",
    "src/focus_binary/models/focus_dnn.py",
    "src/focus_binary/models/cnn_focus_hybrid.py",
    "src/focus_binary/explain/feature_importance.py",
]


@dataclass
class CheckResult:
    idx: int
    name: str
    status: str
    rationale: str
    files: List[str]


def _str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    val = value.strip().lower()
    if val in {"true", "1", "yes", "y"}:
        return True
    if val in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {value}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q1 checklist audit")
    parser.add_argument("--manifest", required=True, help="Path to manifest_with_splits.csv")
    parser.add_argument("--runs-dir", default="./runs", help="Runs directory")
    parser.add_argument("--out-dir", default="./reports/q1_checklist", help="Checklist report output dir")
    parser.add_argument(
        "--families",
        default="cnn,cnn_attention,transfer,vit,hybrid_vit,focus_dnn,cnn_focus_hybrid,convnext,swin",
        help="Comma-separated families",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated seeds")
    parser.add_argument("--expect-datasets", type=int, default=5, help="Expected number of datasets")
    parser.add_argument("--require-explainability", type=_str2bool, default=True)
    parser.add_argument("--require-robustness", type=_str2bool, default=True)
    parser.add_argument("--require-stats", type=_str2bool, default=True)
    parser.add_argument("--require-calibration", type=_str2bool, default=True)
    parser.add_argument("--require-lodo", type=_str2bool, default=True)
    parser.add_argument("--require-uncertainty", type=_str2bool, default=True)
    parser.add_argument("--light-mode", type=_str2bool, default=True)
    parser.add_argument("--max-params", type=int, default=10_000_000)
    parser.add_argument("--min-seeds", type=int, default=5)
    parser.add_argument("--min-trials", type=int, default=10)
    return parser.parse_args(argv)


def _add_result(results: List[CheckResult], status: str, name: str, rationale: str, files: Optional[List[str]] = None) -> None:
    results.append(
        CheckResult(
            idx=len(results) + 1,
            name=name,
            status=status,
            rationale=rationale,
            files=files or [],
        )
    )


def _print_report(results: List[CheckResult]) -> None:
    for res in results:
        files = ", ".join(res.files) if res.files else ""
        files_str = f" | files: {files}" if files else ""
        print(f"{res.idx:02d}. [{res.status}] {res.name} - {res.rationale}{files_str}")


def _save_reports(results: List[CheckResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "checklist_report.json"
    md_path = out_dir / "checklist_report.md"

    payload = {
        "summary": {
            "pass": sum(1 for r in results if r.status == "PASS"),
            "warn": sum(1 for r in results if r.status == "WARN"),
            "fail": sum(1 for r in results if r.status == "FAIL"),
        },
        "checks": [
            {
                "idx": r.idx,
                "name": r.name,
                "status": r.status,
                "rationale": r.rationale,
                "files": r.files,
            }
            for r in results
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2))

    lines = ["# Q1 Checklist Report", ""]
    for r in results:
        files = ", ".join(r.files) if r.files else ""
        files_str = f" | files: {files}" if files else ""
        lines.append(f"{r.idx:02d}. **[{r.status}]** {r.name} - {r.rationale}{files_str}")
    md_path.write_text("\n".join(lines))


def _safe_import(module_name: str) -> Optional[str]:
    try:
        __import__(module_name)
        return None
    except Exception as exc:  # pragma: no cover
        return str(exc)


def _sha1(path: Path) -> str:
    hasher = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _find_multiseed_dir(runs_dir: Path, family: str) -> Optional[Path]:
    candidates = [
        runs_dir / family,
        runs_dir.parent / "reports" / "multiseed" / family,
        runs_dir.parent / "reports" / "final_q1" / family,
    ]
    for cand in candidates:
        if (cand / "multiseed_metrics.csv").exists():
            return cand
    return None


def _find_explainability_paths(project_root: Path, family: str) -> Tuple[Optional[Path], Optional[Path]]:
    summary_candidates = [
        project_root / "reports" / "explain" / family / "explainability_summary.json",
        project_root / "reports" / "final_q1" / "explainability_summary.json",
        project_root / "reports" / "final_q1" / family / "explainability_summary.json",
    ]
    for cand in summary_candidates:
        if cand.exists():
            return cand, cand.parent

    explain_candidates = [
        project_root / "reports" / "final_q1" / "explain_samples",
        project_root / "reports" / "explain" / family / "explain_samples",
    ]
    for cand in explain_candidates:
        if cand.exists():
            return None, cand
    return None, None


def _find_robustness_path(project_root: Path, runs_dir: Path, family: str) -> Optional[Path]:
    candidates = [
        project_root / "reports" / "robustness" / family / "robustness_curves.csv",
        project_root / "reports" / "final_q1" / "robustness" / family / "robustness_curves.csv",
        project_root / "reports" / "multiseed" / family / "robustness" / "robustness_curves.csv",
        runs_dir / family / "robustness" / "robustness_curves.csv",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _check_perturb_coverage(df: pd.DataFrame) -> List[str]:
    if "perturb" not in df.columns:
        return []
    perturb = df["perturb"].astype(str).str.lower().unique().tolist()
    required = {
        "noise": any("noise" in p for p in perturb),
        "jpeg": any("jpeg" in p for p in perturb),
        "brightness": any("bright" in p for p in perturb),
        "contrast": any("contrast" in p for p in perturb),
        "blur": any("blur" in p for p in perturb),
    }
    missing = [key for key, ok in required.items() if not ok]
    return missing


def _check_feature_perturb_coverage(df: pd.DataFrame) -> List[str]:
    if "perturb" not in df.columns:
        return ["feat_noise", "feat_dropout"]
    perturb = df["perturb"].astype(str).str.lower().unique().tolist()
    required = {
        "feat_noise": any("feat_noise" in p for p in perturb),
        "feat_dropout": any("feat_dropout" in p for p in perturb),
    }
    return [key for key, ok in required.items() if not ok]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    results: List[CheckResult] = []

    manifest_path = Path(args.manifest)
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    # A1: Imports
    import_errors = []
    for module in REQUIRED_IMPORTS:
        err = _safe_import(module)
        if err:
            import_errors.append((module, err))
    if import_errors:
        _add_result(
            results,
            "FAIL",
            "Required imports",
            "Import failures detected",
            [f"{mod}: {err}" for mod, err in import_errors],
        )
    else:
        _add_result(results, "PASS", "Required imports", "All required modules import")

    # A2: Scripts
    missing_scripts = []
    for script in REQUIRED_SCRIPTS:
        if not (PROJECT_ROOT / script).exists():
            missing_scripts.append(script)
    if missing_scripts:
        _add_result(
            results,
            "FAIL",
            "Required scripts",
            "Missing required script files",
            missing_scripts,
        )
    else:
        _add_result(results, "PASS", "Required scripts", "All required scripts present")

    missing_files = []
    for rel_path in REQUIRED_FILES:
        if not (PROJECT_ROOT / rel_path).exists():
            missing_files.append(rel_path)
    if missing_files:
        _add_result(
            results,
            "FAIL",
            "Required modules",
            "Missing required module files",
            missing_files,
        )
    else:
        _add_result(results, "PASS", "Required modules", "All required module files present")

    # B3: Manifest load + columns
    df = None
    required_cols = {"dataset", "image_path", "label", "stack_id", "split"}
    if not manifest_path.exists():
        _add_result(results, "FAIL", "Manifest presence", "Manifest file not found", [str(manifest_path)])
    else:
        try:
            df = pd.read_csv(manifest_path)
            missing_cols = required_cols - set(df.columns)
            if missing_cols:
                _add_result(
                    results,
                    "FAIL",
                    "Manifest columns",
                    f"Missing columns: {sorted(missing_cols)}",
                    [str(manifest_path)],
                )
            else:
                _add_result(results, "PASS", "Manifest columns", "Required columns present", [str(manifest_path)])
        except Exception as exc:
            _add_result(results, "FAIL", "Manifest load", f"Failed to read manifest: {exc}", [str(manifest_path)])

    # Stop early if manifest missing
    if df is None or not required_cols.issubset(set(df.columns)):
        _print_report(results)
        _save_reports(results, out_dir)
        return 2

    # B4: Label validity
    labels = df["label"]
    if labels.isna().any():
        _add_result(results, "FAIL", "Label validity", "Labels contain NaN", [str(manifest_path)])
    elif not set(labels.unique()).issubset({0, 1}):
        _add_result(
            results,
            "FAIL",
            "Label validity",
            f"Unexpected label values: {sorted(set(labels.unique()))}",
            [str(manifest_path)],
        )
    else:
        _add_result(results, "PASS", "Label validity", "Labels are binary")

    # B5: File existence sample
    sample_n = min(200, len(df))
    sample_df = df.sample(n=sample_n, random_state=42)
    missing_files = [p for p in sample_df["image_path"].astype(str) if not Path(p).exists()]
    if missing_files:
        missing_ratio = len(missing_files) / sample_n
        status = "FAIL" if missing_ratio > 0.02 else "WARN"
        _add_result(
            results,
            status,
            "File existence (sample)",
            f"{len(missing_files)}/{sample_n} files missing ({missing_ratio:.1%})",
            missing_files[:10],
        )
    else:
        _add_result(results, "PASS", "File existence (sample)", "All sampled files exist")

    # B6: Dataset count
    dataset_count = df["dataset"].nunique()
    if dataset_count <= 1:
        _add_result(
            results,
            "FAIL",
            "Dataset count",
            f"Found {dataset_count} datasets",
            [str(manifest_path)],
        )
    elif dataset_count != args.expect_datasets:
        _add_result(
            results,
            "WARN",
            "Dataset count",
            f"Found {dataset_count}, expected {args.expect_datasets}",
            [str(manifest_path)],
        )
    else:
        _add_result(results, "PASS", "Dataset count", f"Found {dataset_count} datasets")

    # B7: Split proportions
    split_status = "PASS"
    split_notes = []
    for dataset, group in df.groupby("dataset"):
        split_counts = group["split"].value_counts()
        for split in ["train", "val", "test"]:
            if split_counts.get(split, 0) == 0:
                split_status = "FAIL"
                split_notes.append(f"{dataset}:{split} empty")
            else:
                ratio = split_counts[split] / len(group)
                if split in {"val", "test"} and ratio < 0.05:
                    split_status = "WARN" if split_status == "PASS" else split_status
                    split_notes.append(f"{dataset}:{split} ratio {ratio:.2%}")
    if split_status == "PASS":
        _add_result(results, "PASS", "Split proportions", "All datasets have train/val/test")
    else:
        _add_result(results, split_status, "Split proportions", "; ".join(split_notes), [str(manifest_path)])

    # B8: Stack leakage
    leak_pairs = []
    splits = df["split"].unique().tolist()
    groups_by_split = {split: set(df[df["split"] == split]["stack_id"].unique()) for split in splits}
    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1 :]:
            overlap = groups_by_split[s1] & groups_by_split[s2]
            if overlap:
                leak_pairs.append(f"{s1}∩{s2}={len(overlap)}")
    if leak_pairs:
        _add_result(results, "FAIL", "Stack leakage", "Overlap detected: " + ", ".join(leak_pairs))
    else:
        _add_result(results, "PASS", "Stack leakage", "No stack_id overlaps")

    # B9: Duplicate leakage via SHA1 (sample)
    if args.require_robustness:
        sample_n = min(3000, len(df))
        sample_df = df.sample(n=sample_n, random_state=123)
        seen = {}
        cross_split = set()
        within_split = set()
        for _, row in sample_df.iterrows():
            path = Path(str(row["image_path"]))
            split = str(row["split"])
            if not path.exists():
                continue
            digest = _sha1(path)
            prior = seen.get(digest)
            if prior is None:
                seen[digest] = split
            else:
                if prior != split:
                    cross_split.add(str(path))
                else:
                    within_split.add(str(path))
        if cross_split:
            _add_result(
                results,
                "FAIL",
                "Duplicate leakage (SHA1)",
                f"Duplicates across splits in sample ({len(cross_split)} files)",
                list(cross_split)[:10],
            )
        elif within_split:
            _add_result(
                results,
                "WARN",
                "Duplicate leakage (SHA1)",
                f"Duplicates within split in sample ({len(within_split)} files)",
                list(within_split)[:10],
            )
        else:
            _add_result(results, "PASS", "Duplicate leakage (SHA1)", "No duplicates found in sample")
    else:
        _add_result(results, "WARN", "Duplicate leakage (SHA1)", "Skipped (require_robustness=false)")

    # B10: Class balance
    warnings = []
    for (dataset, split), group in df.groupby(["dataset", "split"]):
        pos_rate = group["label"].mean()
        if pos_rate < 0.1 or pos_rate > 0.9:
            warnings.append(f"{dataset}:{split} pos_rate={pos_rate:.2f}")
    if warnings:
        _add_result(results, "WARN", "Class balance", "; ".join(warnings))
    else:
        _add_result(results, "PASS", "Class balance", "No extreme class imbalance")

    # B11: LODO summary
    lodo_summary_df = None
    lodo_summary_path = PROJECT_ROOT / "reports" / "lodo" / "lodo_summary_all.csv"
    if args.require_lodo:
        if not lodo_summary_path.exists():
            _add_result(
                results,
                "FAIL",
                "LODO summary",
                "Missing reports/lodo/lodo_summary_all.csv",
                [str(lodo_summary_path)],
            )
        else:
            try:
                lodo_summary_df = pd.read_csv(lodo_summary_path)
                if lodo_summary_df.empty:
                    _add_result(results, "FAIL", "LODO summary", "LODO summary is empty", [str(lodo_summary_path)])
                else:
                    _add_result(results, "PASS", "LODO summary", "LODO summary present", [str(lodo_summary_path)])
            except Exception as exc:
                _add_result(
                    results,
                    "FAIL",
                    "LODO summary",
                    f"Failed to read LODO summary: {exc}",
                    [str(lodo_summary_path)],
                )

    # C11/C12: Run artifacts
    for family in families:
        family_dir = runs_dir / family
        if not family_dir.exists():
            _add_result(
                results,
                "WARN",
                f"Artifacts ({family})",
                "Family directory missing; skipping checks",
                [str(family_dir)],
            )
            continue

        missing = []
        if not (family_dir / "best_hparams.json").exists():
            missing.append("best_hparams.json")
        if not ((family_dir / "best_model.keras").exists() or (family_dir / "best_model.h5").exists()):
            missing.append("best_model.keras/.h5")
        if not (family_dir / "summary.json").exists():
            missing.append("summary.json")
        if missing:
            _add_result(
                results,
                "FAIL",
                f"Tuning artifacts ({family})",
                "Missing tuning artifacts",
                [str(family_dir / item) for item in missing],
            )
        else:
            _add_result(results, "PASS", f"Tuning artifacts ({family})", "Required artifacts present")

        tuning_csv = family_dir / "tuning_results.csv"
        if tuning_csv.exists():
            try:
                trials = pd.read_csv(tuning_csv)
                if len(trials) < args.min_trials:
                    _add_result(
                        results,
                        "WARN",
                        f"Tuning trials ({family})",
                        f"Only {len(trials)} trials (< {args.min_trials})",
                        [str(tuning_csv)],
                    )
                else:
                    _add_result(
                        results,
                        "PASS",
                        f"Tuning trials ({family})",
                        f"{len(trials)} trials",
                        [str(tuning_csv)],
                    )
            except Exception as exc:
                _add_result(
                    results,
                    "WARN",
                    f"Tuning trials ({family})",
                    f"Failed to read tuning_results.csv: {exc}",
                    [str(tuning_csv)],
                )
        else:
            _add_result(results, "WARN", f"Tuning trials ({family})", "tuning_results.csv missing")

        multiseed_dir = _find_multiseed_dir(runs_dir, family)
        if args.min_seeds > 1 and args.require_robustness:
            if multiseed_dir is None:
                _add_result(
                    results,
                    "FAIL",
                    f"Multiseed outputs ({family})",
                    "Missing multiseed_metrics.csv",
                )
            else:
                metrics_path = multiseed_dir / "multiseed_metrics.csv"
                try:
                    metrics_df = pd.read_csv(metrics_path)
                    seed_count = metrics_df["seed"].nunique() if "seed" in metrics_df.columns else 0
                    if seed_count < args.min_seeds:
                        _add_result(
                            results,
                            "FAIL",
                            f"Multiseed outputs ({family})",
                            f"Only {seed_count} seeds (< {args.min_seeds})",
                            [str(metrics_path)],
                        )
                    else:
                        _add_result(
                            results,
                            "PASS",
                            f"Multiseed outputs ({family})",
                            f"{seed_count} seeds",
                            [str(metrics_path)],
                        )
                except Exception as exc:
                    _add_result(
                        results,
                        "FAIL",
                        f"Multiseed outputs ({family})",
                        f"Failed to read multiseed_metrics.csv: {exc}",
                        [str(metrics_path)],
                    )
        else:
            if multiseed_dir is None:
                _add_result(
                    results,
                    "WARN",
                    f"Multiseed outputs ({family})",
                    "Missing multiseed metrics (not required)",
                )
            else:
                _add_result(
                    results,
                    "PASS",
                    f"Multiseed outputs ({family})",
                    "Multiseed metrics present",
                    [str(multiseed_dir / "multiseed_metrics.csv")],
                )

        # D: Explainability
        if args.require_explainability:
            summary_path, explain_dir = _find_explainability_paths(PROJECT_ROOT, family)
            if summary_path is None and explain_dir is None:
                _add_result(
                    results,
                    "FAIL",
                    f"Explainability ({family})",
                    "Missing explainability outputs",
                )
            else:
                if summary_path is not None:
                    try:
                        summary = json.loads(summary_path.read_text())
                        has_stability = any(
                            key in summary
                            for key in ("stability_mean", "stability_score_mean", "stability_std", "stability_score_std")
                        )
                        if not has_stability:
                            _add_result(
                                results,
                                "WARN",
                                f"Explainability stability ({family})",
                                "Missing stability metrics in summary",
                                [str(summary_path)],
                            )
                        else:
                            _add_result(
                                results,
                                "PASS",
                                f"Explainability ({family})",
                                "Explainability summary present",
                                [str(summary_path)],
                            )
                        has_deletion = any(
                            key in summary
                            for key in ("deletion_auc_mean", "deletion_auc_feature_mean")
                        )
                        if not has_deletion:
                            _add_result(
                                results,
                                "FAIL",
                                f"Explainability faithfulness ({family})",
                                "Missing deletion metrics in summary",
                                [str(summary_path)],
                            )
                        else:
                            _add_result(
                                results,
                                "PASS",
                                f"Explainability faithfulness ({family})",
                                "Deletion metrics present",
                                [str(summary_path)],
                            )
                        if family in {"focus_dnn", "cnn_focus_hybrid"}:
                            has_features = any(
                                key in summary
                                for key in ("feature_importance", "feature_importance_csv", "top_features")
                            )
                            if not has_features:
                                _add_result(
                                    results,
                                    "FAIL",
                                    f"Feature importance ({family})",
                                    "Missing feature importance outputs",
                                    [str(summary_path)],
                                )
                            else:
                                _add_result(
                                    results,
                                    "PASS",
                                    f"Feature importance ({family})",
                                    "Feature importance outputs present",
                                    [str(summary_path)],
                                )
                    except Exception as exc:
                        _add_result(
                            results,
                            "WARN",
                            f"Explainability ({family})",
                            f"Failed to read explainability summary: {exc}",
                            [str(summary_path)],
                        )
                else:
                    _add_result(
                        results,
                        "PASS",
                        f"Explainability ({family})",
                        "Explainability assets present",
                        [str(explain_dir)],
                    )
        else:
            _add_result(results, "WARN", f"Explainability ({family})", "Skipped (require_explainability=false)")

        # E: Calibration
        if args.require_calibration:
            calib_ok = False
            calib_paths = []
            multiseed_dir = _find_multiseed_dir(runs_dir, family)
            if multiseed_dir is not None:
                metrics_path = multiseed_dir / "multiseed_metrics.csv"
                if metrics_path.exists():
                    try:
                        metrics_df = pd.read_csv(metrics_path)
                        if {"ece", "brier"}.issubset(metrics_df.columns):
                            calib_ok = True
                            calib_paths.append(str(metrics_path))
                    except Exception:
                        pass
            final_calib = PROJECT_ROOT / "reports" / "final_q1" / "calibration_summary.csv"
            if final_calib.exists():
                try:
                    calib_df = pd.read_csv(final_calib)
                    if "family" in calib_df.columns and {"ece_mean", "brier_mean"}.issubset(calib_df.columns):
                        if family in calib_df["family"].astype(str).tolist():
                            calib_ok = True
                            calib_paths.append(str(final_calib))
                except Exception:
                    pass

            if calib_ok:
                _add_result(results, "PASS", f"Calibration ({family})", "ECE/Brier present", calib_paths)
            else:
                _add_result(results, "FAIL", f"Calibration ({family})", "Missing ECE/Brier outputs")
        else:
            _add_result(results, "WARN", f"Calibration ({family})", "Skipped (require_calibration=false)")

        # E2: Uncertainty artifacts
        if args.require_uncertainty:
            multiseed_dir = _find_multiseed_dir(runs_dir, family)
            if multiseed_dir is None:
                _add_result(
                    results,
                    "FAIL",
                    f"Uncertainty artifacts ({family})",
                    "Missing multiseed outputs for uncertainty checks",
                )
            else:
                cal_path = multiseed_dir / "calibration_before_after.csv"
                sel_path = multiseed_dir / "selective_metrics.csv"
                missing = [str(p) for p in (cal_path, sel_path) if not p.exists()]
                if missing:
                    _add_result(
                        results,
                        "FAIL",
                        f"Uncertainty artifacts ({family})",
                        "Missing calibration_before_after.csv or selective_metrics.csv",
                        missing,
                    )
                else:
                    _add_result(
                        results,
                        "PASS",
                        f"Uncertainty artifacts ({family})",
                        "calibration_before_after.csv and selective_metrics.csv present",
                        [str(cal_path), str(sel_path)],
                    )
        else:
            _add_result(results, "WARN", f"Uncertainty artifacts ({family})", "Skipped (require_uncertainty=false)")

        # F: Robustness
        if args.require_robustness:
            robustness_path = _find_robustness_path(PROJECT_ROOT, runs_dir, family)
            if robustness_path is None:
                _add_result(results, "FAIL", f"Robustness ({family})", "robustness_curves.csv missing")
            else:
                try:
                    rob_df = pd.read_csv(robustness_path)
                    missing = _check_perturb_coverage(rob_df)
                    if missing:
                        _add_result(
                            results,
                            "WARN",
                            f"Robustness coverage ({family})",
                            f"Missing perturbations: {', '.join(missing)}",
                            [str(robustness_path)],
                        )
                    else:
                        _add_result(
                            results,
                            "PASS",
                            f"Robustness ({family})",
                            "robustness_curves.csv present",
                            [str(robustness_path)],
                        )
                    if family in {"focus_dnn", "cnn_focus_hybrid"}:
                        missing_feat = _check_feature_perturb_coverage(rob_df)
                        if missing_feat:
                            _add_result(
                                results,
                                "FAIL",
                                f"Feature robustness ({family})",
                                f"Missing feature perturbations: {', '.join(missing_feat)}",
                                [str(robustness_path)],
                            )
                        else:
                            _add_result(
                                results,
                                "PASS",
                                f"Feature robustness ({family})",
                                "Feature perturbations present",
                                [str(robustness_path)],
                            )
                except Exception as exc:
                    _add_result(
                        results,
                        "FAIL",
                        f"Robustness ({family})",
                        f"Failed to read robustness_curves.csv: {exc}",
                        [str(robustness_path)],
                    )
        else:
            _add_result(results, "WARN", f"Robustness ({family})", "Skipped (require_robustness=false)")

        # F2: LODO summaries
        if args.require_lodo:
            if lodo_summary_df is None or lodo_summary_df.empty:
                _add_result(
                    results,
                    "FAIL",
                    f"LODO summary ({family})",
                    "Missing or empty reports/lodo/lodo_summary_all.csv",
                )
            else:
                if family not in lodo_summary_df["family"].astype(str).tolist():
                    _add_result(
                        results,
                        "FAIL",
                        f"LODO summary ({family})",
                        "Family missing from lodo_summary_all.csv",
                        [str(PROJECT_ROOT / "reports" / "lodo" / "lodo_summary_all.csv")],
                    )
                else:
                    _add_result(
                        results,
                        "PASS",
                        f"LODO summary ({family})",
                        "Family present in lodo_summary_all.csv",
                        [str(PROJECT_ROOT / "reports" / "lodo" / "lodo_summary_all.csv")],
                    )
        else:
            _add_result(results, "WARN", f"LODO summary ({family})", "Skipped (require_lodo=false)")

        # H: Light-mode constraints
        if args.light_mode:
            summary_path = family_dir / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                params = summary.get("params_count")
                if params is None:
                    _add_result(results, "WARN", f"Params cap ({family})", "params_count missing", [str(summary_path)])
                elif int(params) > args.max_params:
                    _add_result(
                        results,
                        "WARN",
                        f"Params cap ({family})",
                        f"params_count {params} exceeds {args.max_params}",
                        [str(summary_path)],
                    )
                else:
                    _add_result(results, "PASS", f"Params cap ({family})", f"params_count {params}")

                latency = summary.get("latency_ms_mean")
                if latency is None or float(latency) <= 0:
                    _add_result(
                        results,
                        "WARN",
                        f"Latency ({family})",
                        "latency_ms_mean missing or invalid",
                        [str(summary_path)],
                    )
                else:
                    _add_result(results, "PASS", f"Latency ({family})", f"latency_ms_mean {latency:.2f}")
            else:
                _add_result(results, "WARN", f"Light-mode checks ({family})", "summary.json missing")
        else:
            _add_result(results, "WARN", f"Light-mode checks ({family})", "Skipped (light_mode=false)")

        # I: Reproducibility metadata
        summary_path = family_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            config_keys = {"input_size", "batch_size", "epochs"}
            has_config = any(key in summary for key in config_keys)
            has_seed = any(key in summary for key in ("seed", "seeds", "seed_count"))
            has_commit = any(key in summary for key in ("git_commit", "commit", "commit_hash"))

            if not has_config and not has_seed:
                _add_result(
                    results,
                    "FAIL",
                    f"Repro metadata ({family})",
                    "Missing seed/config metadata",
                    [str(summary_path)],
                )
            else:
                note = "Missing git commit hash" if not has_commit else "Metadata present"
                status = "WARN" if not has_commit else "PASS"
                _add_result(results, status, f"Repro metadata ({family})", note, [str(summary_path)])
        else:
            _add_result(results, "FAIL", f"Repro metadata ({family})", "summary.json missing", [str(summary_path)])

    # G: Statistical testing
    if args.require_stats:
        stats_dir = PROJECT_ROOT / "reports" / "final_q1"
        ranks_path = stats_dir / "stats_auc_ranks.csv"
        sig_path = stats_dir / "stats_auc_significance.csv"
        missing = [p for p in [ranks_path, sig_path] if not p.exists()]
        if missing:
            _add_result(
                results,
                "FAIL",
                "Statistical testing",
                "Missing stats outputs",
                [str(p) for p in missing],
            )
        else:
            _add_result(
                results,
                "PASS",
                "Statistical testing",
                "stats_auc_ranks.csv and stats_auc_significance.csv present",
            )
    else:
        _add_result(results, "WARN", "Statistical testing", "Skipped (require_stats=false)")

    # J: Final leaderboard
    leaderboard_dir = PROJECT_ROOT / "reports" / "final_q1"
    leaderboard_csv = leaderboard_dir / "leaderboard.csv"
    leaderboard_md = leaderboard_dir / "leaderboard.md"
    if not leaderboard_csv.exists() or not leaderboard_md.exists():
        missing = [str(p) for p in [leaderboard_csv, leaderboard_md] if not p.exists()]
        _add_result(results, "FAIL", "Final leaderboard", "Missing leaderboard outputs", missing)
    else:
        _add_result(results, "PASS", "Final leaderboard", "leaderboard.csv and leaderboard.md present")

    _print_report(results)
    _save_reports(results, out_dir)

    has_fail = any(r.status == "FAIL" for r in results)
    has_warn = any(r.status == "WARN" for r in results)
    if has_fail:
        return 2
    if has_warn:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
