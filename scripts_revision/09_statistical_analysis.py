from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from revision_utils import (  # noqa: E402
    append_missing,
    auc_difference_bootstrap,
    bootstrap_ci,
    build_metadata,
    config_hash,
    display_model_name,
    ensure_dir,
    fresh_all,
    load_config,
    paired_mcnemar,
    repo_path,
    safe_write_csv,
    save_metadata_for_outputs,
    standardize_model_name,
)


SCRIPT_NAME = Path(__file__).name
METRICS = ["balanced_accuracy", "MCC", "AUC", "recall", "specificity", "F1", "ECE", "Brier score"]
COMPARISONS = [
    ("cnn_attention", "cnn"),
    ("cnn_attention", "cnn_focus_hybrid"),
    ("cnn", "cnn_focus_hybrid"),
    ("cnn_attention", "classical_ml"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap CIs and paired statistical tests.")
    parser.add_argument("--config", default="configs/bspc_revision.yaml")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_predictions(pred_dir: Path, model: str) -> pd.DataFrame | None:
    path = pred_dir / f"test_predictions_{model}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    key = "image_path" if "image_path" in df.columns else "sample_id"
    return df.drop_duplicates(key).copy()


def _align(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    key = "image_path" if "image_path" in a.columns and "image_path" in b.columns else "sample_id"
    cols_a = [key, "true_label", "probability_focused"]
    cols_b = [key, "true_label", "probability_focused"]
    merged = a[cols_a].merge(b[cols_b], on=key, suffixes=("_a", "_b"))
    merged = merged[merged["true_label_a"] == merged["true_label_b"]].copy()
    return merged


def _make_figure(ci_df: pd.DataFrame, fig_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if ci_df.empty:
        return
    metric = "balanced_accuracy"
    work = ci_df[ci_df["metric"] == metric].copy()
    if work.empty:
        return
    work = work.sort_values("estimate", ascending=True)
    y = range(len(work))
    lower = work["estimate"] - work["ci_low"]
    upper = work["ci_high"] - work["estimate"]
    plt.figure(figsize=(7, 4.5))
    plt.errorbar(work["estimate"], y, xerr=[lower, upper], fmt="o", capsize=3)
    plt.yticks(list(y), work["model_display_name"])
    plt.xlabel("Balanced accuracy with 95% bootstrap CI")
    plt.title("Main Test Metric Confidence Intervals")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=300)
    plt.close()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    cfg_hash = config_hash(config)
    pred_dir = repo_path((config.get("subdirs") or {}).get("predictions", "revision_outputs/predictions"))
    out_dir = ensure_dir((config.get("subdirs") or {}).get("statistics", "revision_outputs/statistics"))
    table_dir = ensure_dir((config.get("subdirs") or {}).get("tables", "revision_outputs/tables"))
    fig_dir = ensure_dir((config.get("subdirs") or {}).get("figures", "revision_outputs/figures"))
    outputs = [
        out_dir / "bootstrap_ci_main_metrics.csv",
        out_dir / "paired_tests.csv",
        table_dir / "table_statistical_comparison.csv",
        fig_dir / "fig_main_metrics_with_ci.png",
    ]
    families = [standardize_model_name(m) for m in config.get("model_families", [])]
    input_files = [args.config]
    input_files.extend([pred_dir / f"test_predictions_{m}.csv" for m in families])
    if not args.force and fresh_all(outputs, input_files, cfg_hash):
        print("statistical outputs are fresh; skipping")
        return 0

    n_boot = int((config.get("bootstrap") or {}).get("iterations", 2000))
    conf = float((config.get("bootstrap") or {}).get("confidence_level", 0.95))
    seed = int((config.get("bootstrap") or {}).get("seed", 42))
    predictions = {}
    ci_rows = []

    for model in families:
        df = _load_predictions(pred_dir, model)
        if df is None:
            append_missing(f"Statistical analysis skipped for {model}: test predictions missing", config)
            continue
        predictions[model] = df
        y_true = df["true_label"].astype(int).to_numpy()
        y_prob = df["probability_focused"].astype(float).to_numpy()
        for metric in METRICS:
            ci = bootstrap_ci(y_true, y_prob, metric, n_iterations=n_boot, confidence_level=conf, seed=seed)
            ci_rows.append(
                {
                    "model_code_name": model,
                    "model_display_name": display_model_name(model, config),
                    "metric": metric,
                    **ci,
                    "bootstrap_iterations": n_boot,
                }
            )

    paired_rows = []
    for model_a, model_b in COMPARISONS:
        if model_a not in predictions or model_b not in predictions:
            append_missing(f"Paired comparison skipped: {model_a} vs {model_b} predictions missing", config)
            continue
        merged = _align(predictions[model_a], predictions[model_b])
        if merged.empty:
            append_missing(f"Paired comparison skipped: {model_a} vs {model_b} has no aligned rows", config)
            continue
        y = merged["true_label_a"].astype(int).to_numpy()
        prob_a = merged["probability_focused_a"].astype(float).to_numpy()
        prob_b = merged["probability_focused_b"].astype(float).to_numpy()
        pred_a = (prob_a >= float(config.get("default_threshold", 0.5))).astype(int)
        pred_b = (prob_b >= float(config.get("default_threshold", 0.5))).astype(int)
        mc = paired_mcnemar(y, pred_a, pred_b)
        auc_diff = auc_difference_bootstrap(y, prob_a, prob_b, n_iterations=n_boot, confidence_level=conf, seed=seed)
        paired_rows.append(
            {
                "model_a_code_name": model_a,
                "model_a_display_name": display_model_name(model_a, config),
                "model_b_code_name": model_b,
                "model_b_display_name": display_model_name(model_b, config),
                "n_aligned": len(merged),
                "mcnemar_b": mc["b"],
                "mcnemar_c": mc["c"],
                "mcnemar_statistic": mc["statistic"],
                "mcnemar_p_value": mc["p_value"],
                "mcnemar_method": mc["method"],
                "auc_a": auc_diff["auc_a"],
                "auc_b": auc_diff["auc_b"],
                "auc_difference_a_minus_b": auc_diff["auc_difference"],
                "auc_difference_ci_low": auc_diff["ci_low"],
                "auc_difference_ci_high": auc_diff["ci_high"],
                "auc_difference_p_value": auc_diff["p_value"],
                "auc_difference_method": "paired bootstrap",
            }
        )

    ci_df = pd.DataFrame(ci_rows)
    paired_df = pd.DataFrame(paired_rows)
    safe_write_csv(ci_df, outputs[0])
    safe_write_csv(paired_df, outputs[1])
    safe_write_csv(paired_df, outputs[2])
    _make_figure(ci_df, outputs[3])
    metadata = build_metadata(script_name=SCRIPT_NAME, input_files=input_files, cfg_hash=cfg_hash, args=args)
    metadata["delong_used"] = False
    metadata["auc_difference_method"] = "paired bootstrap"
    save_metadata_for_outputs(outputs, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
