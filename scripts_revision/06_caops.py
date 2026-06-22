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
    build_metadata,
    compute_binary_metrics,
    compute_brier,
    config_hash,
    display_model_name,
    ensure_dir,
    fresh_all,
    load_config,
    repo_path,
    safe_write_csv,
    save_metadata_for_outputs,
    standardize_model_name,
    threshold_grid,
)


SCRIPT_NAME = Path(__file__).name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibration-Aware Operating-Point Selection (CAOPS).")
    parser.add_argument("--config", default="configs/bspc_revision.yaml")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def operating_point_ece(y_true, y_prob, threshold: float, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    pred = (y_prob >= threshold).astype(int)
    confidence = np.where(pred == 1, y_prob, 1.0 - y_prob)
    correctness = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidence >= lo) & (confidence <= hi) if i == n_bins - 1 else (confidence >= lo) & (confidence < hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correctness[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def _metrics_at(df: pd.DataFrame, threshold: float) -> dict:
    y_true = df["true_label"].astype(int).to_numpy()
    y_prob = df["probability_focused"].astype(float).to_numpy()
    m = compute_binary_metrics(y_true, y_prob, threshold=threshold)
    return {
        **m,
        "ECE": operating_point_ece(y_true, y_prob, threshold),
        "Brier score": compute_brier(y_true, y_prob),
    }


def _validation_grid(df: pd.DataFrame, grid, model: str, config: dict) -> pd.DataFrame:
    rows = []
    for t in grid:
        m = _metrics_at(df, float(t))
        rows.append(
            {
                "model_code_name": model,
                "model_display_name": display_model_name(model, config),
                "threshold": float(t),
                "validation_balanced_accuracy": m["balanced_accuracy"],
                "validation_ECE": m["ECE"],
                "validation_MCC": m["MCC"],
                "validation_F1": m["F1"],
                "validation_recall": m["recall"],
                "validation_specificity": m["specificity"],
                "validation_Youden_J": m["recall"] + m["specificity"] - 1.0,
            }
        )
    return pd.DataFrame(rows)


def _base_selectors(grid_df: pd.DataFrame, default_threshold: float) -> list[dict]:
    selectors = []
    fixed = grid_df.iloc[(grid_df["threshold"] - default_threshold).abs().argsort()].iloc[0]
    selectors.append(("fixed_0p50", fixed, {"selection_family": "reference"}))
    for name, col in [
        ("Youden_J", "validation_Youden_J"),
        ("max_F1", "validation_F1"),
        ("max_balanced_accuracy", "validation_balanced_accuracy"),
        ("max_MCC", "validation_MCC"),
    ]:
        row = grid_df.sort_values([col, "threshold"], ascending=[False, True]).iloc[0]
        selectors.append((name, row, {"selection_family": "reference"}))
    return [
        {
            "selection_rule": name,
            "threshold": float(row["threshold"]),
            "validation_balanced_accuracy": float(row["validation_balanced_accuracy"]),
            "validation_ECE": float(row["validation_ECE"]),
            "validation_MCC": float(row["validation_MCC"]),
            "selection_family": extra["selection_family"],
            "constraint_satisfied": "",
            "lambda": "",
            "delta": "",
        }
        for name, row, extra in selectors
    ]


def _caops_selectors(grid_df: pd.DataFrame, lambdas: list[float], deltas: list[float]) -> list[dict]:
    rows = []
    for lam in lambdas:
        work = grid_df.copy()
        work["score"] = work["validation_balanced_accuracy"] - float(lam) * work["validation_ECE"]
        row = work.sort_values(["score", "validation_balanced_accuracy", "threshold"], ascending=[False, False, True]).iloc[0]
        rows.append(
            {
                "selection_rule": f"CAOPS_penalty_lambda_{lam}",
                "threshold": float(row["threshold"]),
                "validation_balanced_accuracy": float(row["validation_balanced_accuracy"]),
                "validation_ECE": float(row["validation_ECE"]),
                "validation_MCC": float(row["validation_MCC"]),
                "selection_family": "CAOPS_penalty",
                "constraint_satisfied": "",
                "lambda": float(lam),
                "delta": "",
                "CAOPS_score": float(row["score"]),
            }
        )
    for delta in deltas:
        feasible = grid_df[grid_df["validation_ECE"] <= float(delta)].copy()
        constraint_satisfied = True
        if feasible.empty:
            feasible = grid_df.sort_values(["validation_ECE", "threshold"], ascending=[True, True]).head(1).copy()
            constraint_satisfied = False
        row = feasible.sort_values(["validation_balanced_accuracy", "threshold"], ascending=[False, True]).iloc[0]
        rows.append(
            {
                "selection_rule": f"CAOPS_constraint_delta_{delta}",
                "threshold": float(row["threshold"]),
                "validation_balanced_accuracy": float(row["validation_balanced_accuracy"]),
                "validation_ECE": float(row["validation_ECE"]),
                "validation_MCC": float(row["validation_MCC"]),
                "selection_family": "CAOPS_constraint",
                "constraint_satisfied": bool(constraint_satisfied),
                "lambda": "",
                "delta": float(delta),
                "CAOPS_score": "",
            }
        )
    return rows


def _test_result(test_df: pd.DataFrame, model: str, selected: dict, config: dict) -> dict:
    m = _metrics_at(test_df, float(selected["threshold"]))
    return {
        "model_code_name": model,
        "model_display_name": display_model_name(model, config),
        **selected,
        "test_balanced_accuracy": m["balanced_accuracy"],
        "test_MCC": m["MCC"],
        "test_F1": m["F1"],
        "test_recall": m["recall"],
        "test_specificity": m["specificity"],
        "test_ECE": m["ECE"],
        "test_Brier score": m["Brier score"],
        "false_acceptance_rate": m["false_acceptance_rate"],
        "false_rejection_rate": m["false_rejection_rate"],
        "test_TP": m["TP"],
        "test_TN": m["TN"],
        "test_FP": m["FP"],
        "test_FN": m["FN"],
    }


def _make_figure(grid_df: pd.DataFrame, selected_df: pd.DataFrame, fig_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if grid_df.empty:
        return
    plt.figure(figsize=(8, 5))
    for model, group in grid_df.groupby("model_display_name"):
        plt.plot(group["validation_ECE"], group["validation_balanced_accuracy"], marker=".", linewidth=1, markersize=2, label=model)
    if not selected_df.empty:
        caops = selected_df[selected_df["selection_family"].astype(str).str.startswith("CAOPS")]
        plt.scatter(caops["validation_ECE"], caops["validation_balanced_accuracy"], s=18, c="black", alpha=0.65, label="CAOPS selections")
    plt.xlabel("Validation operating-point ECE")
    plt.ylabel("Validation balanced accuracy")
    plt.title("CAOPS Calibration-Accuracy Trade-off")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=300)
    plt.close()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    cfg_hash = config_hash(config)
    pred_dir = repo_path((config.get("subdirs") or {}).get("predictions", "revision_outputs/predictions"))
    out_dir = ensure_dir((config.get("subdirs") or {}).get("caops", "revision_outputs/caops"))
    table_dir = ensure_dir((config.get("subdirs") or {}).get("tables", "revision_outputs/tables"))
    fig_dir = ensure_dir((config.get("subdirs") or {}).get("figures", "revision_outputs/figures"))
    outputs = [
        out_dir / "caops_validation_grid.csv",
        out_dir / "caops_selected_thresholds.csv",
        out_dir / "caops_test_results.csv",
        table_dir / "table_caops_comparison.csv",
        fig_dir / "fig_caops_tradeoff.png",
    ]
    families = [standardize_model_name(m) for m in config.get("model_families", [])]
    input_files = [args.config]
    for model in families:
        input_files.extend([pred_dir / f"validation_predictions_{model}.csv", pred_dir / f"test_predictions_{model}.csv"])
    if not args.force and fresh_all(outputs, input_files, cfg_hash):
        print("CAOPS outputs are fresh; skipping")
        return 0

    lambdas = [float(v) for v in ((config.get("caops") or {}).get("lambda_values") or [0.0, 0.1, 0.25, 0.5, 1.0])]
    deltas = [float(v) for v in ((config.get("caops") or {}).get("ece_constraint_values") or [0.025, 0.05, 0.075, 0.10])]
    grid = threshold_grid(config)
    default_threshold = float(config.get("default_threshold", 0.5))
    all_grid = []
    all_selected = []
    all_test = []

    for model in families:
        val_path = pred_dir / f"validation_predictions_{model}.csv"
        test_path = pred_dir / f"test_predictions_{model}.csv"
        if not val_path.exists() or not test_path.exists():
            append_missing(f"CAOPS skipped for {model}: validation or test predictions missing", config)
            continue
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        if val_df.empty or test_df.empty:
            append_missing(f"CAOPS skipped for {model}: empty validation or test predictions", config)
            continue
        grid_df = _validation_grid(val_df, grid, model, config)
        all_grid.append(grid_df)
        selected = _base_selectors(grid_df, default_threshold) + _caops_selectors(grid_df, lambdas, deltas)
        for row in selected:
            row.update({"model_code_name": model, "model_display_name": display_model_name(model, config)})
            all_selected.append(row)
            all_test.append(_test_result(test_df, model, row, config))

    grid_df = pd.concat(all_grid, ignore_index=True) if all_grid else pd.DataFrame()
    selected_df = pd.DataFrame(all_selected)
    test_results = pd.DataFrame(all_test)
    safe_write_csv(grid_df, outputs[0])
    safe_write_csv(selected_df, outputs[1])
    safe_write_csv(test_results, outputs[2])
    safe_write_csv(test_results, outputs[3])
    _make_figure(grid_df, selected_df, outputs[4])
    metadata = build_metadata(script_name=SCRIPT_NAME, input_files=input_files, cfg_hash=cfg_hash, args=args)
    save_metadata_for_outputs(outputs, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
