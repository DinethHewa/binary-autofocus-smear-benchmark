from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from revision_utils import (  # noqa: E402
    append_missing,
    build_metadata,
    compute_binary_metrics,
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
    parser = argparse.ArgumentParser(description="Validation-selected threshold analysis.")
    parser.add_argument("--config", default="configs/bspc_revision.yaml")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_prediction_pair(pred_dir: Path, model: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    val_path = pred_dir / f"validation_predictions_{model}.csv"
    test_path = pred_dir / f"test_predictions_{model}.csv"
    val = pd.read_csv(val_path) if val_path.exists() else None
    test = pd.read_csv(test_path) if test_path.exists() else None
    return val, test


def _grid_metrics(df: pd.DataFrame, grid) -> pd.DataFrame:
    y_true = df["true_label"].astype(int).to_numpy()
    y_prob = df["probability_focused"].astype(float).to_numpy()
    rows = []
    for t in grid:
        m = compute_binary_metrics(y_true, y_prob, threshold=float(t))
        rows.append(
            {
                "threshold": float(t),
                "TP": m["TP"],
                "TN": m["TN"],
                "FP": m["FP"],
                "FN": m["FN"],
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "specificity": m["specificity"],
                "balanced_accuracy": m["balanced_accuracy"],
                "MCC": m["MCC"],
                "AUC": m["AUC"],
                "F1": m["F1"],
                "Youden_J": m["recall"] + m["specificity"] - 1.0,
            }
        )
    return pd.DataFrame(rows)


def _select_thresholds(grid_df: pd.DataFrame, default_threshold: float) -> list[dict]:
    selectors = [
        ("fixed_0p50", "fixed", default_threshold),
        ("Youden_J", "max", "Youden_J"),
        ("max_F1", "max", "F1"),
        ("max_balanced_accuracy", "max", "balanced_accuracy"),
        ("max_MCC", "max", "MCC"),
    ]
    selected = []
    for name, mode, value in selectors:
        if mode == "fixed":
            threshold = float(value)
            row = grid_df.iloc[(grid_df["threshold"] - threshold).abs().argsort()].iloc[0]
        else:
            metric = str(value)
            work = grid_df.sort_values([metric, "threshold"], ascending=[False, True])
            row = work.iloc[0]
            threshold = float(row["threshold"])
        selected.append(
            {
                "selection_rule": name,
                "threshold": threshold,
                "validation_accuracy": row["accuracy"],
                "validation_balanced_accuracy": row["balanced_accuracy"],
                "validation_MCC": row["MCC"],
                "validation_F1": row["F1"],
                "validation_Youden_J": row["Youden_J"],
                "validation_recall": row["recall"],
                "validation_specificity": row["specificity"],
            }
        )
    return selected


def _test_row(test_df: pd.DataFrame, model: str, rule: dict, config: dict) -> dict:
    y_true = test_df["true_label"].astype(int).to_numpy()
    y_prob = test_df["probability_focused"].astype(float).to_numpy()
    m = compute_binary_metrics(y_true, y_prob, threshold=float(rule["threshold"]))
    return {
        "model_code_name": model,
        "model_display_name": display_model_name(model, config),
        "selection_rule": rule["selection_rule"],
        "selected_threshold": rule["threshold"],
        **rule,
        "test_accuracy": m["accuracy"],
        "test_precision": m["precision"],
        "test_recall": m["recall"],
        "test_specificity": m["specificity"],
        "test_balanced_accuracy": m["balanced_accuracy"],
        "test_MCC": m["MCC"],
        "test_AUC": m["AUC"],
        "test_F1": m["F1"],
        "test_TP": m["TP"],
        "test_TN": m["TN"],
        "test_FP": m["FP"],
        "test_FN": m["FN"],
    }


def _make_figure(all_grid_rows: list[pd.DataFrame], fig_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not all_grid_rows:
        return
    plt.figure(figsize=(8, 5))
    for grid in all_grid_rows:
        model = grid["model_display_name"].iloc[0]
        plt.plot(grid["threshold"], grid["balanced_accuracy"], label=model, linewidth=1.5)
    plt.xlabel("Threshold")
    plt.ylabel("Validation balanced accuracy")
    plt.title("Validation Threshold Trade-off")
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
    out_dir = ensure_dir((config.get("subdirs") or {}).get("thresholds", "revision_outputs/thresholds"))
    table_dir = ensure_dir((config.get("subdirs") or {}).get("tables", "revision_outputs/tables"))
    fig_dir = ensure_dir((config.get("subdirs") or {}).get("figures", "revision_outputs/figures"))
    families = [standardize_model_name(m) for m in config.get("model_families", [])]
    outputs = [
        out_dir / "selected_thresholds.csv",
        out_dir / "test_metrics_by_selected_threshold.csv",
        table_dir / "table_threshold_selection.csv",
        fig_dir / "fig_threshold_selection_tradeoff.png",
    ]
    outputs.extend([out_dir / f"validation_threshold_grid_{m}.csv" for m in families])
    input_files = [args.config]
    for model in families:
        input_files.extend([pred_dir / f"validation_predictions_{model}.csv", pred_dir / f"test_predictions_{model}.csv"])
    if not args.force and fresh_all(outputs, input_files, cfg_hash):
        print("threshold outputs are fresh; skipping")
        return 0

    all_selected = []
    all_test = []
    all_grid_for_plot = []
    grid = threshold_grid(config)
    default_threshold = float(config.get("default_threshold", 0.5))

    for model in families:
        val, test = _load_prediction_pair(pred_dir, model)
        if val is None or test is None or val.empty or test.empty:
            append_missing(f"Threshold selection skipped for {model}: validation or test predictions missing", config)
            continue
        grid_df = _grid_metrics(val, grid)
        grid_df.insert(0, "model_code_name", model)
        grid_df.insert(1, "model_display_name", display_model_name(model, config))
        safe_write_csv(grid_df, out_dir / f"validation_threshold_grid_{model}.csv")
        all_grid_for_plot.append(grid_df)
        selected = _select_thresholds(grid_df, default_threshold)
        for row in selected:
            row.update({"model_code_name": model, "model_display_name": display_model_name(model, config)})
            all_selected.append(row)
            all_test.append(_test_row(test, model, row, config))

    selected_df = pd.DataFrame(all_selected)
    test_df = pd.DataFrame(all_test)
    safe_write_csv(selected_df, out_dir / "selected_thresholds.csv")
    safe_write_csv(test_df, out_dir / "test_metrics_by_selected_threshold.csv")
    safe_write_csv(test_df, table_dir / "table_threshold_selection.csv")
    _make_figure(all_grid_for_plot, fig_dir / "fig_threshold_selection_tradeoff.png")

    metadata = build_metadata(script_name=SCRIPT_NAME, input_files=input_files, cfg_hash=cfg_hash, args=args)
    generated = outputs
    save_metadata_for_outputs(generated, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
